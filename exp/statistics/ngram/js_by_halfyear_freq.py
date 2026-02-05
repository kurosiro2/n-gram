#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(半年) × (n=1..N の「2019~2025 を基準にした JS距離」) のヒートマップを出す。

追加:
  - y軸ラベルに「その period の 1-gram 総数」を埋め込む

★追加:
  - Laplace support 切替:
      --laplace-support observed_ab | all
  - nごとの箱ひげ図:
      --boxplot / --boxplot-only
  - 外れ値(変な〇)を消す:
      showfliers=False
  - ★今回: 箱ひげ図に最小値/最大値を表示
      - 最小値: 「〇」(open circle)
      - 最大値: 「•」(dot)
      - 前面(zorder) + クリップ無効(clip_on=False) + ylim自動余白
      --boxplot-minmax-markers
  - フォント倍率:
      --font-scale
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
# matplotlib font sizing (global)
# -------------------------------------------------------------
def apply_font_scale(scale: float) -> None:
    if scale <= 0:
        scale = 1.0
    base = {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
    }
    for k, v in base.items():
        plt.rcParams[k] = v * scale


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
    tg = (target_group or "").strip().lower()
    if not tg:
        return False
    return any((g or "").strip().lower() == tg for g in group_set)


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
def _js_distance_smoothed(
    c1: Counter,
    c2: Counter,
    alpha: float,
    vocab_size: int,
    support_mode: str,
) -> float:
    if vocab_size <= 0:
        return 0.0

    tot1 = float(sum(c1.values()))
    tot2 = float(sum(c2.values()))
    denom1 = tot1 + alpha * float(vocab_size)
    denom2 = tot2 + alpha * float(vocab_size)
    if denom1 <= 0.0 or denom2 <= 0.0:
        return 0.0

    keys = set(c1.keys()) | set(c2.keys())

    kl_pm = 0.0
    kl_qm = 0.0

    for g in keys:
        p = (float(c1.get(g, 0)) + alpha) / denom1
        q = (float(c2.get(g, 0)) + alpha) / denom2
        m = 0.5 * (p + q)
        if p > 0.0 and m > 0.0:
            kl_pm += p * math.log(p / m)
        if q > 0.0 and m > 0.0:
            kl_qm += q * math.log(q / m)

    if support_mode == "all":
        observed_cnt = len(keys)
        rest = vocab_size - observed_cnt
        if rest > 0:
            p0 = alpha / denom1
            q0 = alpha / denom2
            m0 = 0.5 * (p0 + q0)
            if p0 > 0.0 and m0 > 0.0:
                kl_pm += rest * (p0 * math.log(p0 / m0))
            if q0 > 0.0 and m0 > 0.0:
                kl_qm += rest * (q0 * math.log(q0 / m0))

    jsd = 0.5 * (kl_pm + kl_qm)
    if jsd < 0.0:
        jsd = 0.0
    return math.sqrt(jsd)


def js_distance_from_counters(
    c1: Counter,
    c2: Counter,
    alpha: float,
    laplace_support: str,
    n: int,
) -> float:
    if laplace_support not in ("observed_ab", "all"):
        raise ValueError(f"laplace_support must be observed_ab|all, got {laplace_support}")

    if laplace_support == "observed_ab":
        vocab_size = len(set(c1.keys()) | set(c2.keys()))
        if vocab_size <= 0:
            return 0.0
        return _js_distance_smoothed(c1, c2, alpha, vocab_size=vocab_size, support_mode="observed_ab")

    vocab_size = (len(VALID_SHIFTS) ** n)
    return _js_distance_smoothed(c1, c2, alpha, vocab_size=vocab_size, support_mode="all")


# ---------------------------
# periods: half-year
# ---------------------------
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101,  y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701,  y * 10000 + 1231))
    return periods


# ---------------------------
# plotting
# ---------------------------
def plot_heatmap_period_x_n(
    out_png: str,
    title: str,
    period_labels: List[str],
    n_labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
    cell_fontsize: Optional[float] = None,
) -> None:
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(7.0, R * 0.55)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), period_labels)

    mid = (vmin + vmax) / 2.0
    fs = cell_fontsize if cell_fontsize is not None else (plt.rcParams["font.size"] * 0.85)

    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=fs, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _finite_only(vals: List[float]) -> List[float]:
    return [v for v in vals if (v is not None and math.isfinite(v))]


def plot_boxplot_by_n(
    out_png: str,
    title: str,
    n_labels: List[str],
    data_by_n: List[List[float]],
    ymin: float,
    ymax: float,
    show_minmax_markers: bool = False,
    minmax_marker_size: Optional[float] = None,
) -> None:
    """
    showfliers=False で外れ値マーカーは出さない。
    show_minmax_markers=True で min/max を表示する。
      - min: 「〇」(open circle)
      - max: 「•」(dot)

    ★確実に出すための対策:
      - data/ min/max を有限値で掃除（nan/inf 対策）
      - zorder=50 で前面
      - clip_on=False
      - ylim を min/max も考慮して余白付きで広げる
    """
    C = len(n_labels)
    fig_w = max(9.0, C * 1.8)
    fig_h = 6.5

    clean_data: List[List[float]] = [_finite_only(vs) for vs in data_by_n]

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    ax.boxplot(
        clean_data,
        tick_labels=n_labels,
        showfliers=False,
    )

    xs = list(range(1, C + 1))
    mins: List[Optional[float]] = []
    maxs: List[Optional[float]] = []
    finite_vals_all: List[float] = []

    for vals in clean_data:
        if not vals:
            mins.append(None)
            maxs.append(None)
            continue
        mn = min(vals)
        mx = max(vals)
        mins.append(mn)
        maxs.append(mx)
        finite_vals_all.append(mn)
        finite_vals_all.append(mx)

    y_lo = float(ymin)
    y_hi = float(ymax)

    if finite_vals_all:
        mn_all = min(finite_vals_all)
        mx_all = max(finite_vals_all)

        base_span = (y_hi - y_lo) if (y_hi > y_lo) else max(1e-6, (mx_all - mn_all))
        pad = 0.05 * base_span

        y_lo2 = min(y_lo, mn_all - pad)
        y_hi2 = max(y_hi, mx_all + pad)
        ax.set_ylim(y_lo2, y_hi2)
    else:
        ax.set_ylim(y_lo, y_hi)

    ax.set_xticklabels(n_labels, rotation=45, ha="right")
    ax.set_ylabel("JSDistance")
    ax.set_title(title)

    if show_minmax_markers:
        ms = float(minmax_marker_size) if (minmax_marker_size is not None) else float(plt.rcParams["font.size"] * 1.3)
        s_area_open = ms * ms
        s_area_dot = (ms * 2.0) * (ms * 2.0)

        x_min: List[int] = []
        y_min: List[float] = []
        x_max: List[int] = []
        y_max: List[float] = []

        for x, mn, mx in zip(xs, mins, maxs):
            if mn is not None and math.isfinite(mn):
                x_min.append(x); y_min.append(mn)
            if mx is not None and math.isfinite(mx):
                x_max.append(x); y_max.append(mx)

        # min: open circle
        sc_min = ax.scatter(
            x_min, y_min,
            marker="o",
            s=s_area_open,
            facecolors="none",
            edgecolors="black",
            linewidths=2.2,
            zorder=50,
            clip_on=False,
            label="min",
        )

        # max: dot (black)
        sc_max = ax.scatter(
            x_max, y_max,
            marker=".",
            s=s_area_dot,
            color="black",
            zorder=50,
            clip_on=False,
            label="max",
        )

        # ★凡例の表示順を「逆」に固定（max を先、min を後）
        ax.legend(handles=[sc_max, sc_min], labels=["max", "min"], loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="*.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings (setting.lp path)")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/halfyear_vs_total")
    ap.add_argument("--only-nonheads", action="store_true")

    ap.add_argument("--laplace-support", choices=["observed_ab", "all"], default="observed_ab")

    ap.add_argument("--boxplot", action="store_true")
    ap.add_argument("--boxplot-only", action="store_true")

    ap.add_argument("--font-scale", type=float, default=1.35)

    ap.add_argument("--boxplot-minmax-markers", action="store_true",
                    help="Overlay min/max markers on boxplot (min=open circle, max=dot).")
    ap.add_argument("--minmax-marker-size", type=float, default=None,
                    help="Marker size base (roughly). If omitted, uses rcParams font.size.")

    args = ap.parse_args()

    apply_font_scale(args.font_scale)
    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")
    if args.boxplot_only:
        args.boxplot = True

    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    periods = half_periods + [(base_key, base_start, base_end)]
    halfK = len(half_periods)

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels_heat = [f"{n}-gram (vs {base_key})" for n in ns]

    base_heads_by_n: Dict[int, Counter] = {}
    base_non_by_n: Dict[int, Counter] = {}
    for n in ns:
        h_b, non_b = count_ngrams_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_heads_by_n[n] = h_b
        base_non_by_n[n] = non_b

    base_labels = [k for (k, _, _) in periods]
    heads_1gram_totals: List[int] = []
    non_1gram_totals: List[int] = []
    for (_k, d1, d2) in periods:
        h1, n1 = count_ngrams_heads_nonheads_in_range(
            segs_by_person, n=1, heads_name=args.heads_name, date_start=d1, date_end=d2
        )
        heads_1gram_totals.append(int(sum(h1.values())))
        non_1gram_totals.append(int(sum(n1.values())))

    period_labels_heads = [f"{lbl}({t})" for lbl, t in zip(base_labels, heads_1gram_totals)]
    period_labels_non = [f"{lbl}({t})" for lbl, t in zip(base_labels, non_1gram_totals)]

    vmin = 0.0
    vmax = 0.4

    def build_matrix(is_heads: bool) -> List[List[float]]:
        mat: List[List[float]] = []
        for (_pkey, d1, d2) in periods:
            row: List[float] = []
            for n in ns:
                h_p, non_p = count_ngrams_heads_nonheads_in_range(
                    segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                )
                c_p = h_p if is_heads else non_p
                c_b = base_heads_by_n[n] if is_heads else base_non_by_n[n]
                row.append(
                    js_distance_from_counters(
                        c_p, c_b,
                        alpha=args.alpha,
                        laplace_support=args.laplace_support,
                        n=n,
                    )
                )
            mat.append(row)
        return mat

    def mat_halfyears_to_boxdata(mat: List[List[float]]) -> List[List[float]]:
        cols: List[List[float]] = [[] for _ in ns]
        for i in range(min(halfK, len(mat))):
            for j in range(len(ns)):
                cols[j].append(mat[i][j])
        return cols

    suffix = f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{args.laplace_support}_a{args.alpha}_fs{args.font_scale}"

    if not args.only_nonheads:
        mat_h = build_matrix(is_heads=True)

        if not args.boxplot_only:
            out_h = os.path.join(args.outdir, f"heatmap_halfyear_x_ngram_heads_{suffix}.png")
            plot_heatmap_period_x_n(
                out_h,
                title=f"Heads: JSdist(half-year, {base_key}) for n={args.nmin}..{args.nmax} [ln] (laplace-support={args.laplace_support})",
                period_labels=period_labels_heads,
                n_labels=n_labels_heat,
                mat=mat_h,
                vmin=vmin,
                vmax=vmax,
                cell_fontsize=plt.rcParams["font.size"] * 0.8,
            )
            print(f"# wrote: {out_h}")

        if args.boxplot:
            box_h = mat_halfyears_to_boxdata(mat_h)
            out_bh = os.path.join(args.outdir, f"boxplot_halfyear_jsdist_heads_{suffix}.png")
            plot_boxplot_by_n(
                out_bh,
                title=f"Heads: half-year JSdist vs {base_key} (n={args.nmin}..{args.nmax}) [ln] (laplace-support={args.laplace_support})",
                n_labels=[f"{n}-gram" for n in ns],
                data_by_n=box_h,
                ymin=vmin,
                ymax=vmax,
                show_minmax_markers=args.boxplot_minmax_markers,
                minmax_marker_size=args.minmax_marker_size,
            )
            print(f"# wrote: {out_bh}")

    mat_n = build_matrix(is_heads=False)

    if not args.boxplot_only:
        out_n = os.path.join(args.outdir, f"heatmap_halfyear_x_ngram_nonheads_{suffix}.png")
        plot_heatmap_period_x_n(
            out_n,
            title=f"NonHeads: JSdist(half-year, {base_key}) for n={args.nmin}..{args.nmax} [ln] (laplace-support={args.laplace_support})",
            period_labels=period_labels_non,
            n_labels=n_labels_heat,
            mat=mat_n,
            vmin=vmin,
            vmax=vmax,
            cell_fontsize=plt.rcParams["font.size"] * 0.8,
        )
        print(f"# wrote: {out_n}")

    if args.boxplot:
        box_n = mat_halfyears_to_boxdata(mat_n)
        out_bn = os.path.join(args.outdir, f"boxplot_halfyear_jsdist_nonheads_{suffix}.png")
        plot_boxplot_by_n(
            out_bn,
            title=f"JSDistance: half-year vs {base_key}",
            n_labels=[f"{n}-gram" for n in ns],
            data_by_n=box_n,
            ymin=vmin,
            ymax=vmax,
            show_minmax_markers=args.boxplot_minmax_markers,
            minmax_marker_size=args.minmax_marker_size,
        )
        print(f"# wrote: {out_bn}")


if __name__ == "__main__":
    main()
