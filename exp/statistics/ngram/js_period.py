#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period × (n=1..5 の「2019~2025 を基準にした JS距離」) のヒートマップを 1枚で出す。

縦軸: period
横軸: 1-gram (vs 2019~2025), 2-gram (vs 2019~2025), ..., N-gram (vs 2019~2025)

仕様:
  - グループ集合の変化でセグメント分割（境界は跨がない）
  - Unknown は NonHeads 側に含める
  - JS distance = sqrt(JSD), ln (natural log)
  - カラースケール統一: vmin=0, vmax=sqrt(ln 2)
  - 各セルに値（小数3桁）を表示

使い方:
  python exp/statistics/ngram/js_period_vs_total_heatmap.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --nmin 1 --nmax 5 --alpha 1e-3 --outdir out/period_vs_total
"""

import os
import sys
import math
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date


# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"}
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


# -------------------------------------------------------------
# periods (fixed)
# ※君が貼ってくれた最新版に合わせる
# -------------------------------------------------------------
# key -> (date_start, date_end)
PERIODS: List[Tuple[str, int, int]] = [
    ("2024_1",     20240101, 20240131),
    ("2024_1~3",   20240101, 20240331),
    ("2024_1~6",   20240101, 20240630),
    ("2024",       20240101, 20241231),
    ("2024~2025",  20240101, 20251231),
    ("2019~2025",  20190101, 20251231),
]

BASE_KEY = "2019~2025"   # 横軸の基準分布は常にこれ


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
    Unknown は UNKNOWN_GROUP として保持（→ NonHeads 側に入る）
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


def count_ngrams_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter]:
    """
    指定範囲の Heads / NonHeads(+Unknown) を集約カウント。
    セグメント境界は跨がない。
    """
    heads = Counter()
    nonheads = Counter()

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            for i in range(len(sseq) - n + 1):
                gram = tuple(sseq[i + k][1] for k in range(n))
                if is_heads:
                    heads[gram] += 1
                else:
                    nonheads[gram] += 1

    return heads, nonheads


# ---------------------------
# JS distance (sqrt(JSD)) [ln]
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


# ---------------------------
# plotting: period × n
# ---------------------------
def plot_heatmap_period_x_n(
    out_png: str,
    title: str,
    period_labels: List[str],
    n_labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
) -> None:
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(6.5, R * 1.1)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im)

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), period_labels)

    mid = (vmin + vmax) / 2.0
    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="*.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/period_vs_total")
    ap.add_argument("--only-nonheads", action="store_true", help="NonHeads(+Unknown) だけ出力（Headsを出さない）")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # load
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # period keys
    period_keys = [k for (k, _, _) in PERIODS]
    if BASE_KEY not in period_keys:
        raise RuntimeError(f'BASE_KEY="{BASE_KEY}" not found in PERIODS')

    period_labels = period_keys[:]  # 縦軸

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram (vs {BASE_KEY})" for n in ns]  # 横軸

    # global scale (ln): vmax = sqrt(ln2)
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

    # 各 n について「period -> counter」を作って、各 period vs BASE を計算
    def build_matrix_for_group(is_heads: bool) -> List[List[float]]:
        mat: List[List[float]] = []
        for (pkey, d1, d2) in PERIODS:
            row: List[float] = []
            for n in ns:
                # period dist
                h_p, n_p = count_ngrams_heads_nonheads_in_range(
                    segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                )
                c_p = h_p if is_heads else n_p

                # base dist (毎回数えるの無駄だけど、最小実装としてそのまま)
                # 速くしたければ n ごとに base_counter をキャッシュすればOK
                for (bk, bd1, bd2) in PERIODS:
                    if bk == BASE_KEY:
                        h_b, n_b = count_ngrams_heads_nonheads_in_range(
                            segs_by_person, n=n, heads_name=args.heads_name, date_start=bd1, date_end=bd2
                        )
                        c_b = h_b if is_heads else n_b
                        break

                row.append(js_distance_from_counters(c_p, c_b, args.alpha))
            mat.append(row)
        return mat

    if not args.only_nonheads:
        mat_h = build_matrix_for_group(is_heads=True)
        out_h = os.path.join(args.outdir, f"heatmap_period_x_ngram_heads_n{args.nmin}-{args.nmax}.png")
        plot_heatmap_period_x_n(
            out_h,
            title=f"Heads: JSdist(period, {BASE_KEY}) for n={args.nmin}..{args.nmax} [ln]",
            period_labels=period_labels,
            n_labels=n_labels,
            mat=mat_h,
            vmin=vmin,
            vmax=vmax,
        )
        print(f"# wrote: {out_h}")

    mat_n = build_matrix_for_group(is_heads=False)
    out_n = os.path.join(args.outdir, f"heatmap_period_x_ngram_nonheads_n{args.nmin}-{args.nmax}.png")
    plot_heatmap_period_x_n(
        out_n,
        title=f"NonHeads: JSdist(period, {BASE_KEY}) for n={args.nmin}..{args.nmax} [ln]",
        period_labels=period_labels,
        n_labels=n_labels,
        mat=mat_n,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"# wrote: {out_n}")


if __name__ == "__main__":
    main()
