#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
病棟ごとに（Heads を除外した）n-gram 分布を作り、病棟×病棟の JS distance を比較する。

- 入力:
    past_shifts_dir   : *.lp が病棟ごとに置いてあるディレクトリ
    group_settings_dir: 病棟名/ 以下に設定があるディレクトリ（data_loader.load_staff_group_timeline が読める形）

- 期間: デフォルト 20190101..20251231
- Heads は除外（Heads セグメントの n-gram はカウントしない）
- Unknown は NonHeads 側として含める
- JS distance = sqrt(JSD), かつ ln（自然対数）
- ヒートマップの vmin/vmax は統一（デフォルト vmax = sqrt(ln 2)）
- 病棟名（日本語）はヒートマップ表示で崩れるので短い英語へ変換（WARD_ALIAS）

★追加：
- 各病棟の total_ngrams を横軸=病棟で棒グラフ出力

使い方:
  python exp/statistics/ngram/js_ward_heatmap.py \
    exp/2019-2025-data/past-shifts \
    exp/2019-2025-data/group-settings \
    --n 3 --alpha 1e-3 --outdir out/js_ward --dump-topk --topk 10

出力:
  outdir/
    heatmap_js_wards_n{n}.png
    ward_total_ngrams_n{n}.png        ★追加
    topk_ward_{WARD}_n{n}.txt         (dump-topk 指定時)
"""

import os
import sys
import math
import argparse
import re
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# import path: exp/statistics/ngram/*.py から exp/statistics/ を見えるように
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date


# -------------------------------------------------------------
# 有効な勤務記号（必要なら合わせて増やしてOK）
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"}

UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


# -------------------------------------------------------------
# 病棟名（日本語）→ 短い英語ラベル
# -------------------------------------------------------------
WARD_ALIAS = {
    "GCU": "GCU",
    "NICU": "NICU",
    "集中治療室": "ICU",
    "救急外来": "ER",
    "師長勤務表": "Chief",

    "2階西病棟": "2W",
    "2階東病棟": "2E",
    "3階西病棟": "3W",
    "3階東病棟": "3E",
    "4階西病棟": "4W",
    "4階南病棟": "4S",
    "4階北病棟": "4N",
    "5階西病棟": "5W",
    "5階南病棟": "5S",
    "5階北病棟": "5N",
    "6階西病棟": "6W",
    "6階南病棟": "6S",
    "6階北病棟": "6N",
    "7階西病棟": "7W",
    "7階南病棟": "7S",
    "7階北病棟": "7N",
}


def ward_to_label(ward: str) -> str:
    """
    1) WARD_ALIAS にあればそれを使用
    2) なければ「{階}{方角}」を抽出して短縮（例: 8階東病棟 → 8E）
    3) それでも無理なら ASCII のみ抽出して短縮（最悪でも壊れない）
    """
    if ward in WARD_ALIAS:
        return WARD_ALIAS[ward]

    m = re.search(r"(\d+)階.*?(東|西|南|北)", ward)
    if m:
        floor = m.group(1)
        dir_map = {"東": "E", "西": "W", "南": "S", "北": "N"}
        return f"{floor}{dir_map[m.group(2)]}"

    ascii_only = re.sub(r"[^A-Za-z0-9]", "", ward)
    return ascii_only[:6] if ascii_only else "WARD"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target_group: str) -> bool:
    tg = target_group.lower()
    return any(g.lower() == tg for g in group_set)


class Segment:
    __slots__ = ("groups", "seq")

    def __init__(self, groups: FrozenSet[str]):
        self.groups = groups
        self.seq: List[Tuple[int, str]] = []


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
) -> List["Segment"]:
    """
    所属グループ集合が変わるたびにセグメント分割。
    Unknown は UNKNOWN_GROUP を付与して保持（=NonHeads側として扱える）。
    """
    seq = normalize_seq(seq)
    if not seq:
        return []
    seq.sort(key=lambda t: t[0])

    segs: List[Segment] = []
    cur: Optional[Segment] = None

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)
        if not gs:
            gset = frozenset([UNKNOWN_GROUP])
        else:
            gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            segs.append(cur)

        cur.seq.append((d, s))

    return segs


def prebuild_all_segments(seqs: SeqDict, timeline: dict) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline)
    return out


def count_ngrams_excluding_heads_by_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Counter:
    """
    Heads セグメントを除外し、NonHeads(+Unknown) だけで n-gram を集約カウント。
    セグメント境界は跨がない。
    """
    c = Counter()

    for _, segs in segs_by_person.items():
        for seg in segs:
            if group_set_contains(seg.groups, heads_name):
                continue  # Heads除外

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            for i in range(len(sseq) - n + 1):
                gram = tuple(sseq[i + k][1] for k in range(n))
                c[gram] += 1

    return c


# ---------------------------
# JS distance (sqrt(JSD))  - ln
# ---------------------------
def build_vocab(c1: Counter, c2: Counter) -> List[Tuple[str, ...]]:
    total = Counter()
    total.update(c1)
    total.update(c2)
    vocab = list(total.keys())
    vocab.sort()
    return vocab


def to_prob_vector(counter: Counter, vocab: List[Tuple[str, ...]], alpha: float) -> List[float]:
    total = float(sum(counter.values()))
    denom = total + alpha * len(vocab)
    if denom <= 0:
        return ([1.0 / len(vocab)] * len(vocab)) if vocab else []
    return [(counter.get(g, 0) + alpha) / denom for g in vocab]


def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi)  # ln
    return s


def js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_distance_from_counters(c1: Counter, c2: Counter, alpha: float) -> float:
    vocab = build_vocab(c1, c2)
    if not vocab:
        return 0.0
    p = to_prob_vector(c1, vocab, alpha)
    q = to_prob_vector(c2, vocab, alpha)
    d = js_divergence(p, q)
    if d < 0:
        d = 0.0
    return math.sqrt(d)


def compute_js_matrix_wards(
    wards: List[str],
    counters_by_ward: Dict[str, Counter],
    alpha: float,
) -> List[List[float]]:
    W = len(wards)
    mat = [[0.0 for _ in range(W)] for _ in range(W)]
    for i in range(W):
        for j in range(W):
            if i == j:
                mat[i][j] = 0.0
            elif j < i:
                mat[i][j] = mat[j][i]
            else:
                w1, w2 = wards[i], wards[j]
                mat[i][j] = js_distance_from_counters(
                    counters_by_ward[w1], counters_by_ward[w2], alpha
                )
    return mat


# ---------------------------
# dump topk
# ---------------------------
def _format_gram(gram: Tuple[str, ...]) -> str:
    return " -> ".join(gram)


def dump_topk_ward(path: str, ward: str, n: int, counter: Counter, topk: int) -> None:
    total = sum(counter.values())
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(f"=== Top-{topk} n-gram probability (n={n}) : {ward} (Heads excluded) ===\n\n")
        fp.write(f"[{ward}] 20190101..20251231\n")
        if total <= 0:
            fp.write("  (no data)\n")
            fp.write("  [total ngrams] 0\n")
            return

        for rank, (gram, cnt) in enumerate(counter.most_common(topk), start=1):
            p = cnt / total
            fp.write(f"  {rank:2d}. {cnt:8d}  {p*100:6.2f}%  {_format_gram(gram)}\n")
        fp.write(f"  [total ngrams] {total}\n")


# ---------------------------
# heatmap
# ---------------------------
def plot_heatmap(
    out_png: str,
    title: str,
    labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
) -> None:
    L = len(labels)
    fig_w = max(8.0, L * 0.5)
    fig_h = max(7.0, L * 0.5)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im)

    ticks = list(range(L))
    plt.xticks(ticks, labels, rotation=60, ha="right")
    plt.yticks(ticks, labels)

    mid = (vmin + vmax) / 2.0
    for i in range(L):
        for j in range(L):
            val = mat[i][j]
            txt_color = "white" if val > mid else "black"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# ★追加：病棟ごとの ngram データ数（総n-gram数）棒グラフ
# ---------------------------
def plot_count_bars(
    out_png: str,
    title: str,
    labels: List[str],
    counts: List[int],
) -> None:
    # 病棟が多いとラベルが長くなるので横幅を自動調整
    L = len(labels)
    fig_w = max(10.0, L * 0.55)
    fig_h = 5.5

    plt.figure(figsize=(fig_w, fig_h))
    x = list(range(L))
    plt.bar(x, counts)

    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("total n-grams")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def list_ward_files(past_shifts_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(past_shifts_dir):
        if fn.endswith(".lp"):
            files.append(fn)
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts_dir", help="ディレクトリ: 病棟ごとの *.lp がある場所")
    ap.add_argument("group_settings_dir", help="ディレクトリ: 病棟名/ 以下に設定がある場所")

    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")

    ap.add_argument("--date-start", type=int, default=20190101)
    ap.add_argument("--date-end", type=int, default=20251231)

    ap.add_argument("--outdir", default="out/js_ward")
    ap.add_argument("--dump-topk", action="store_true")
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--global-scale", action="store_true",
                    help="カラースケール統一 (vmin=0, vmax=sqrt(ln2)) を強制")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    ward_files = list_ward_files(args.past_shifts_dir)
    if not ward_files:
        raise SystemExit(f"no *.lp in: {args.past_shifts_dir}")

    counters_by_ward: Dict[str, Counter] = {}
    wards: List[str] = []
    totals_by_ward: Dict[str, int] = {}   # ★追加：病棟ごとの総n-gram数

    for fn in ward_files:
        ward = os.path.splitext(fn)[0]
        past_path = os.path.join(args.past_shifts_dir, fn)
        setting_path = os.path.join(args.group_settings_dir, ward)

        if not os.path.exists(setting_path):
            print(f"# skip (no settings): {ward}  path={setting_path}")
            continue

        seqs = data_loader.load_past_shifts(past_path)
        timeline = data_loader.load_staff_group_timeline(setting_path)
        segs_by_person = prebuild_all_segments(seqs, timeline)

        c = count_ngrams_excluding_heads_by_range(
            segs_by_person,
            n=args.n,
            heads_name=args.heads_name,
            date_start=args.date_start,
            date_end=args.date_end,
        )

        counters_by_ward[ward] = c
        wards.append(ward)

        total = sum(c.values())
        totals_by_ward[ward] = total  # ★追加
        print(f"# ward={ward:>12s}  total_ngrams={total}")

        if args.dump_topk:
            out_txt = os.path.join(args.outdir, f"topk_ward_{ward}_n{args.n}.txt")
            dump_topk_ward(out_txt, ward, args.n, c, args.topk)
            print(f"# wrote: {out_txt}")

    if len(wards) < 2:
        raise SystemExit("need >= 2 wards with settings to compute JS matrix")

    wards.sort()
    mat = compute_js_matrix_wards(wards, counters_by_ward, args.alpha)

    # 表示ラベルだけ英語短縮に変換（内部キーは日本語のまま）
    labels = [ward_to_label(w) for w in wards]

    # ★追加：病棟ごとの総n-gram数を棒グラフに
    counts = [int(totals_by_ward.get(w, 0)) for w in wards]
    out_bar = os.path.join(args.outdir, f"ward_total_ngrams_n{args.n}.png")
    plot_count_bars(
        out_png=out_bar,
        title=f"Total n-grams by ward (Heads excluded) n={args.n}",
        labels=labels,
        counts=counts,
    )
    print(f"# wrote: {out_bar}")

    # カラースケール（統一）
    if args.global_scale:
        vmin = 0.0
        vmax = math.sqrt(math.log(2.0))  # ln の JS distance 理論上限
    else:
        vmin = 0.0
        vmax = max(max(row) for row in mat) if mat else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmax = min(vmax, math.sqrt(math.log(2.0)))

    out_png = os.path.join(args.outdir, f"heatmap_js_wards_n{args.n}.png")
    plot_heatmap(
        out_png=out_png,
        title=f"JS distance heatmap by ward (Heads excluded) n={args.n} [ln]",
        labels=labels,
        mat=mat,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"# wrote: {out_png}")


if __name__ == "__main__":
    main()
