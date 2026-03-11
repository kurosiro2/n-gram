#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全期間（start-year〜end-year）で、
「看護師グループ同士」を比較した n-gram 出現確率分布の
JS distance (sqrt(JSD), ln) を
グループ×グループのヒートマップとして出力する。

★仕様まとめ
  - Night / Other / ALL はプロットしない
  - Unknown は --unknown-as に寄せた後、Other として除外
  - セグメント境界は跨がない
  - JS distance = sqrt(JSD), ln
  - Laplace smoothing:
      observed_ab | all
  - カラースケール統一: [0, sqrt(ln2)]
  - n ごとに 1 枚 PNG
  - 図の文字は全体的に大きめ（論文・スライド向け）
"""

import os
import sys
import math
import argparse
from collections import Counter
from typing import Dict, Tuple, List, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# ★ FONT SETTINGS（ここが今回の主役）
# =============================================================
plt.rcParams.update({
    "font.size": 16,        # 全体ベース
    "axes.titlesize": 20,   # タイトル
    "axes.labelsize": 18,   # 軸ラベル
    "xtick.labelsize": 18,  # x軸目盛り
    "ytick.labelsize": 18,  # y軸目盛り
    "legend.fontsize": 18,
})

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"}
UNKNOWN_GROUP = "__UNKNOWN__"

# ★除外グループ
EXCLUDE_GROUPS = {"night", "other", "all"}

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq):
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def within_range(d, start, end):
    return start <= d <= end


def get_groups_for_day(name, nid, date, timeline):
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target: str) -> bool:
    t = target.lower().strip()
    return any(g.lower().strip() == t for g in group_set)


class Segment:
    __slots__ = ("groups", "seq")

    def __init__(self, groups):
        self.groups = groups
        self.seq = []


def build_segments_for_person(seq, name, nid, timeline):
    seq = normalize_seq(seq)
    if not seq:
        return []

    seq.sort(key=lambda x: x[0])
    segs = []
    cur = None

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)
        if not gs:
            gset = frozenset([UNKNOWN_GROUP])
        else:
            gset = frozenset(sorted(gs))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            segs.append(cur)

        cur.seq.append((d, s))

    return segs


def prebuild_all_segments(seqs, timeline):
    out = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline)
    return out


def collect_group_names(segs_by_person, unknown_as):
    groups = set()
    for segs in segs_by_person.values():
        for seg in segs:
            if seg.groups == frozenset([UNKNOWN_GROUP]):
                groups.add(unknown_as)
            else:
                for g in seg.groups:
                    groups.add(unknown_as if g == UNKNOWN_GROUP else g)
    return sorted(groups, key=str.lower)


def count_ngrams_by_group(segs_by_person, n, group, d1, d2, unknown_as):
    c = Counter()
    g_norm = group.lower().strip()
    unk_norm = unknown_as.lower().strip()

    for segs in segs_by_person.values():
        for seg in segs:
            if seg.groups == frozenset([UNKNOWN_GROUP]):
                belongs = (g_norm == unk_norm)
            else:
                belongs = group_set_contains(seg.groups, group)

            if not belongs:
                continue

            sseq = [(d, s) for d, s in seg.seq if within_range(d, d1, d2)]
            if len(sseq) < n:
                continue

            for i in range(len(sseq) - n + 1):
                gram = tuple(sseq[i + k][1] for k in range(n))
                c[gram] += 1

    return c


# ---------------------------
# JS distance
# ---------------------------
def js_distance(c1, c2, alpha, support, n):
    if support == "observed_ab":
        vocab = set(c1) | set(c2)
        V = len(vocab)
    else:
        V = len(VALID_SHIFTS) ** n

    if V == 0:
        return 0.0

    t1, t2 = sum(c1.values()), sum(c2.values())
    d1, d2 = t1 + alpha * V, t2 + alpha * V

    kl1 = kl2 = 0.0
    keys = set(c1) | set(c2)

    for k in keys:
        p = (c1.get(k, 0) + alpha) / d1
        q = (c2.get(k, 0) + alpha) / d2
        m = 0.5 * (p + q)
        kl1 += p * math.log(p / m)
        kl2 += q * math.log(q / m)

    if support == "all":
        rest = V - len(keys)
        if rest > 0:
            p0 = alpha / d1
            q0 = alpha / d2
            m0 = 0.5 * (p0 + q0)
            kl1 += rest * p0 * math.log(p0 / m0)
            kl2 += rest * q0 * math.log(q0 / m0)

    return math.sqrt(0.5 * (kl1 + kl2))


# ---------------------------
# plot
# ---------------------------
def plot_heatmap(out_png, title, labels, mat):
    R = len(labels)
    C = len(labels)

    # ★FONT: 図自体も大きく
    fig_w = max(10.0, C * 1.0)
    fig_h = max(8.0, R * 1.0)

    # ★カラー範囲（固定したいならここ）
    VMIN = 0.0
    VMAX = 0.5

    # ★「値域の半分以上」なら黒
    threshold = VMIN + 0.5 * (VMAX - VMIN)  # 例: [0,0.5] なら 0.25

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=VMIN, vmax=VMAX, aspect="equal")
    plt.colorbar(im)

    plt.xticks(range(C), labels, rotation=45, ha="right")
    plt.yticks(range(R), labels)

    for i in range(R):
        for j in range(C):
            v = float(mat[i][j])
            txt_color = "black" if v >= threshold else "white"
            plt.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=12,   # ★FONT: セル内数値
                color=txt_color,
            )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts")
    ap.add_argument("group_settings")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--laplace-support", choices=["observed_ab", "all"], default="observed_ab")
    ap.add_argument("--unknown-as", default="Other")
    ap.add_argument("--outdir", default="out/group_vs_group_total")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    d1 = args.start_year * 10000 + 101
    d2 = args.end_year * 10000 + 1231

    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    groups = collect_group_names(segs_by_person, args.unknown_as)
    groups = [g for g in groups if g.lower() not in EXCLUDE_GROUPS]

    totals = {}
    for g in groups:
        totals[g] = sum(count_ngrams_by_group(segs_by_person, 1, g, d1, d2, args.unknown_as).values())

    groups = [g for g in groups if totals[g] > 0]
    labels = [f"{g}({totals[g]})" for g in groups]

    for n in range(args.nmin, args.nmax + 1):
        counters = {g: count_ngrams_by_group(segs_by_person, n, g, d1, d2, args.unknown_as)
                    for g in groups}

        mat = []
        for gi in groups:
            row = []
            for gj in groups:
                row.append(js_distance(
                    counters[gi], counters[gj],
                    args.alpha, args.laplace_support, n
                ))
            mat.append(row)

        out = os.path.join(args.outdir, f"heatmap_group_x_group_{n}gram.png")
        title = f"P(gram) (n={n})"
        plot_heatmap(out, title, labels, mat)
        print(f"# wrote: {out}")


if __name__ == "__main__":
    main()
