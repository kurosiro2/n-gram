#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(任意: 月単位で区切り) × (n=1..N の「2019~2025 を基準にした JS距離」) のヒートマップを出す。

追加:
  - y軸ラベルに「その period の 1-gram 総数」を埋め込む
    例: 202401-202403(12345)

★period を柔軟化（今回）:
  - 半年固定ではなく、任意の「月数」で区切れる
    --period-months 1  : 1ヶ月ごと
    --period-months 3  : 3ヶ月ごと
    --period-months 6  : 半年ごと（従来相当）
    --period-months 12 : 1年ごと

★Laplace の支持集合X（=語彙/vocab）を切替可能（ngram頻度分布用）
  --laplace-support all:
      vocab_size = |VALID_SHIFTS|^n を仮定（全組合せが語彙にあるのと同等のLaplace）
      ただし巨大な語彙列挙はせず、辞書計算でJSDを求める
  --laplace-support observed_ab:
      vocab = observed in (A ∪ B)

★箱ひげ図
  --boxplot:
      heatmapに加えて boxplot を出す（Heads/NonHeadsそれぞれ）
  --boxplot-only:
      boxplot のみ出す（heatmapは出さない）

★箱ひげ図の外れ値マーカーを消す
  -> plt.boxplot(..., showfliers=False)

★フォント倍率
  --font-scale 1.0 を基準に 1.2, 1.4... で全体を拡大

仕様:
  - グループ集合の変化でセグメント分割（境界は跨がない）
  - Unknown は NonHeads 側に含める
  - JS distance = sqrt(JSD), ln (natural log)
  - vmax は sqrt(ln 2)
  - 各セルに値（小数3桁）を表示
"""

import os
import sys
import math
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set, FrozenSet
from datetime import date as _date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# matplotlib font sizing (global)
# -------------------------------------------------------------
def apply_font_scale(scale: float) -> None:
    """
    全図に効くフォント倍率設定。
    scale=1.0 を基準に、1.2, 1.4, 1.6... で全体的にデカくする。
    """
    if scale <= 0:
        scale = 1.0

    base = {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
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
def _js_distance_smoothed(
    c1: Counter,
    c2: Counter,
    alpha: float,
    vocab_size: int,
    support_mode: str,
) -> float:
    """
    Laplace(alpha)付きのJSDを「辞書計算」で求める（巨大vocabを明示構築しない）。

    support_mode:
      - observed_ab: vocab = keys(c1) ∪ keys(c2), vocab_size = len(vocab)
      - all        : vocab = 全組合せ(|VALID_SHIFTS|^n)を仮定
                     ただし keys以外の未観測要素は「両方とも alpha」を持つ想定でまとめて扱う
    """
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
    """
    laplace_support:
      - observed_ab: vocab = observed union
      - all        : vocab_size = |VALID_SHIFTS|^n
    """
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
# periods: generic by months
# ---------------------------
def _int_to_date(d: int) -> _date:
    y = d // 10000
    m = (d // 100) % 100
    dd = d % 100
    return _date(y, m, dd)


def _date_to_int(dt: _date) -> int:
    return dt.year * 10000 + dt.month * 100 + dt.day


def _add_months(y: int, m: int, k: int) -> Tuple[int, int]:
    """
    (y,m) に k months 足した (y2,m2) を返す（dayは扱わない）
    m: 1..12
    """
    idx = (y * 12 + (m - 1)) + k
    y2 = idx // 12
    m2 = (idx % 12) + 1
    return y2, m2


def _last_day_of_month(y: int, m: int) -> _date:
    """
    その月の最終日
    """
    if m == 12:
        nxt = _date(y + 1, 1, 1)
    else:
        nxt = _date(y, m + 1, 1)
    return nxt - __import__("datetime").timedelta(days=1)


def make_month_periods(
    start_year: int,
    end_year: int,
    period_months: int,
) -> List[Tuple[str, int, int]]:
    """
    [start_year-01-01 .. end_year-12-31] を、period_months で区切る。
    ラベルは "YYYYMM-YYYYMM"（月単位）にする。
    """
    if period_months <= 0:
        raise ValueError("period_months must be > 0")

    start_dt = _date(start_year, 1, 1)
    end_dt = _date(end_year, 12, 31)

    periods: List[Tuple[str, int, int]] = []

    y, m = start_dt.year, start_dt.month
    cur_start = start_dt

    while cur_start <= end_dt:
        y2, m2 = _add_months(y, m, period_months - 1)  # 期間末の月
        end_month_last = _last_day_of_month(y2, m2)
        cur_end = end_month_last if end_month_last <= end_dt else end_dt

        label = f"{cur_start.year:04d}{cur_start.month:02d}-{y2:04d}{m2:02d}"

        periods.append((label, _date_to_int(cur_start), _date_to_int(cur_end)))

        # 次の開始（月初）
        y, m = _add_months(y, m, period_months)
        cur_start = _date(y, m, 1)

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


def plot_boxplot_by_n(
    out_png: str,
    title: str,
    n_labels: List[str],
    data_by_n: List[List[float]],
    ymin: float,
    ymax: float,
) -> None:
    """
    ※「変な〇」は外れ値(fliers)なので showfliers=False で消す
    """
    C = len(n_labels)
    fig_w = max(9.0, C * 1.8)
    fig_h = 6.5

    plt.figure(figsize=(fig_w, fig_h))
    plt.boxplot(
        data_by_n,
        labels=n_labels,
        showfliers=False,
    )
    plt.ylim(ymin, ymax)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("JSDistance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="*.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)

    # ★追加: periodを月数で指定（1,3,6,12 など）
    ap.add_argument(
        "--period-months",
        type=int,
        default=6,
        help="Split periods by this many months (1=monthly, 3=quarter, 6=half-year, 12=year). Default=6",
    )

    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/periodmonths_vs_total")
    ap.add_argument("--only-nonheads", action="store_true")

    # Laplace support
    ap.add_argument(
        "--laplace-support",
        choices=["observed_ab", "all"],
        default="observed_ab",
        help=(
            "Laplace vocab support for n-gram distribution. "
            "observed_ab: vocab = observed grams in (A ∪ B). "
            "all: vocab_size = |VALID_SHIFTS|^n."
        ),
    )

    # boxplot
    ap.add_argument("--boxplot", action="store_true")
    ap.add_argument("--boxplot-only", action="store_true")

    # font scale
    ap.add_argument("--font-scale", type=float, default=1.35)

    args = ap.parse_args()

    apply_font_scale(args.font_scale)
    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.period_months <= 0:
        raise ValueError("--period-months must be > 0")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0 for Laplace smoothing support option")
    if args.boxplot_only:
        args.boxplot = True

    # load
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # periods (by months) + base(total)
    periods_var = make_month_periods(args.start_year, args.end_year, args.period_months)

    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    periods = periods_var + [(base_key, base_start, base_end)]
    varK = len(periods_var)  # boxplotは base除きでここまで

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram (vs {base_key})" for n in ns]

    # base counters cache
    base_heads_by_n: Dict[int, Counter] = {}
    base_non_by_n: Dict[int, Counter] = {}
    for n in ns:
        h_b, non_b = count_ngrams_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_heads_by_n[n] = h_b
        base_non_by_n[n] = non_b

    # labels: append 1-gram totals
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

    # global scale (ln)
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

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

    def mat_periods_to_boxdata(mat: List[List[float]]) -> List[List[float]]:
        cols: List[List[float]] = [[] for _ in ns]
        if not mat:
            return cols
        for i in range(min(varK, len(mat))):
            for j in range(len(ns)):
                cols[j].append(mat[i][j])
        return cols

    suffix = (
        f"{args.start_year}-{args.end_year}"
        f"_pm{args.period_months}"
        f"_n{args.nmin}-{args.nmax}"
        f"_{args.laplace_support}_a{args.alpha}"
        f"_fs{args.font_scale}"
    )

    # Heads outputs
    if not args.only_nonheads:
        mat_h = build_matrix(is_heads=True)

        if not args.boxplot_only:
            out_h = os.path.join(args.outdir, f"heatmap_periodmonths_x_ngram_heads_{suffix}.png")
            plot_heatmap_period_x_n(
                out_h,
                title=(
                    f"Heads: JSdist(period({args.period_months}mo), {base_key}) "
                    f"n={args.nmin}..{args.nmax} [ln] (laplace-support={args.laplace_support})"
                ),
                period_labels=period_labels_heads,
                n_labels=n_labels,
                mat=mat_h,
                vmin=vmin,
                vmax=vmax,
                cell_fontsize=plt.rcParams["font.size"] * 0.8,
            )
            print(f"# wrote: {out_h}")

        if args.boxplot:
            box_h = mat_periods_to_boxdata(mat_h)
            out_bh = os.path.join(args.outdir, f"boxplot_periodmonths_jsdist_heads_{suffix}.png")
            plot_boxplot_by_n(
                out_bh,
                title=(
                    f"Heads: period({args.period_months}mo) JSdist vs {base_key} "
                    f"(n={args.nmin}..{args.nmax}) [ln] (laplace-support={args.laplace_support})"
                ),
                n_labels=[f"{n}-gram" for n in ns],
                data_by_n=box_h,
                ymin=vmin,
                ymax=vmax,
            )
            print(f"# wrote: {out_bh}")

    # NonHeads outputs
    mat_n = build_matrix(is_heads=False)

    if not args.boxplot_only:
        out_n = os.path.join(args.outdir, f"heatmap_periodmonths_x_ngram_nonheads_{suffix}.png")
        plot_heatmap_period_x_n(
            out_n,
            title=(
                f"NonHeads: JSdist(period({args.period_months}mo), {base_key}) "
                f"n={args.nmin}..{args.nmax} [ln] (laplace-support={args.laplace_support})"
            ),
            period_labels=period_labels_non,
            n_labels=n_labels,
            mat=mat_n,
            vmin=vmin,
            vmax=vmax,
            cell_fontsize=plt.rcParams["font.size"] * 0.8,
        )
        print(f"# wrote: {out_n}")

    if args.boxplot:
        box_n = mat_periods_to_boxdata(mat_n)
        out_bn = os.path.join(args.outdir, f"boxplot_periodmonths_jsdist_nonheads_{suffix}.png")
        plot_boxplot_by_n(
            out_bn,
            title=f"NonHeads: period({args.period_months}mo) JSdist vs {base_key}",
            n_labels=[f"{n}-gram" for n in ns],
            data_by_n=box_n,
            ymin=vmin,
            ymax=vmax,
        )
        print(f"# wrote: {out_bn}")


if __name__ == "__main__":
    main()
