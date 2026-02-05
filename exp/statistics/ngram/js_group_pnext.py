#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【全期間】Group×Group の P(next|prefix) を比較した JS distance ヒートマップ（sqrt(JSD), ln）

- 対象: 条件付き確率 P(next|prefix)（末尾シフト next）
- 比較: グループ同士（Group×Group）
- 期間: start-year..end-year を全期間で集計
- n: nmin..nmax（n=1 は prefix=EMPTY で通常の 1-gram 分布と同値）
- Unknown は --unknown-as に寄せる（デフォ Other）
- Night / Other / ALL は除外（大小文字無視）
- avg-mode:
    weighted:  prefixごとのJSDを w(y)=cA(y)+cB(y) で重み付け平均 → sqrt
    uniform :  prefixごとのJSDを等重み平均 → sqrt
    iqr     :  prefixごとの sqrt(JSD) の Q1-Q3 を表示、色は median
- laplace-support:
    all        : support=VALID_SHIFTS(10) で add-k
    observed_ab: prefixごとに「A/Bで観測された next の union」だけに add-k
                （※表示ベクトルは常に10次元だが、非supportは add-kしない）
- セグメント境界（グループ集合変化）を跨がない
"""

import os
import sys
import math
import argparse
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# ★ フォント大きめ設定（全体）
# =============================================================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
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
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
X_SIZE = 10  # 表示ベクトルは常に10次元

UNKNOWN_GROUP = "__UNKNOWN__"
EXCLUDE_GROUPS = {"night", "other", "all"}  # ★除外（大小文字無視）

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


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


def build_segments_for_person(seq, name, nid, timeline) -> List["Segment"]:
    """
    所属グループ集合が変わるたびにセグメント分割。境界は跨がない。
    Unknown は UNKNOWN_GROUP として保持（後で unknown_as に寄せる）。
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


def collect_group_names(segs_by_person: Dict[PersonKey, List[Segment]], unknown_as: str) -> List[str]:
    """
    全データから出現したグループ名を集める（UNKNOWN_GROUP は unknown_as に集約）。
    """
    groups: Set[str] = set()
    for segs in segs_by_person.values():
        for seg in segs:
            if seg.groups == frozenset([UNKNOWN_GROUP]):
                groups.add(unknown_as)
            else:
                for g in seg.groups:
                    groups.add(unknown_as if g == UNKNOWN_GROUP else str(g))
    return sorted(groups, key=lambda x: x.lower())


# -------------------------------------------------------------
# count conditional P(next|prefix) per group
# -------------------------------------------------------------
def count_conditional_by_group_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    group_name: str,
    date_start: int,
    date_end: int,
    unknown_as: str,
) -> Tuple[Dict[Prefix, Counter], Counter]:
    """
    戻り値:
      cond[prefix][next] = count
      prefN[prefix]      = total count of events for that prefix (=sum_next cond)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    cond: Dict[Prefix, Counter] = defaultdict(Counter)
    prefN: Counter = Counter()
    pref_len = n - 1

    g_norm = group_name.strip().lower()
    unk_norm = unknown_as.strip().lower()

    for (_nid, _name), segs in segs_by_person.items():
        for seg in segs:
            # belongs?
            if seg.groups == frozenset([UNKNOWN_GROUP]):
                belongs = (g_norm == unk_norm)
            else:
                belongs = group_set_contains(seg.groups, group_name)
            if not belongs:
                continue

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for (_d, s) in sseq]

            if n == 1:
                pfx: Prefix = tuple()
                for x in shifts:
                    cond[pfx][x] += 1
                    prefN[pfx] += 1
                continue

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i : i + pref_len])
                nxt = shifts[i + pref_len]
                cond[pfx][nxt] += 1
                prefN[pfx] += 1

    return cond, prefN


# -------------------------------------------------------------
# probability vectors + JS distance
# -------------------------------------------------------------
def laplace_pnext_vector(
    cond: Dict[Prefix, Counter],
    prefN: Counter,
    prefix: Prefix,
    k: float,
    support: Optional[Set[str]] = None,
) -> List[float]:
    """
    10次元（VALID_SHIFTS順）の確率ベクトルを返す。
    support=None なら VALID_SHIFTS 全部に add-k。
    support がある場合:
      - x in support だけ add-k
      - x not in support は add-k しない（分母は Ny + k*|support|）
    """
    Ny = float(prefN.get(prefix, 0))
    cxy = cond.get(prefix, Counter())

    if support is None:
        support = VALID_SHIFTS_SET
    if not support:
        support = VALID_SHIFTS_SET

    denom = Ny + k * len(support)
    if denom <= 0.0:
        return [1.0 / X_SIZE] * X_SIZE

    out: List[float] = []
    for x in VALID_SHIFTS:
        if x in support:
            out.append((cxy.get(x, 0) + k) / denom)
        else:
            out.append(cxy.get(x, 0) / denom)
    return out


def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi)
    return s


def js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_distance_vec(p: List[float], q: List[float]) -> float:
    d = js_divergence(p, q)
    if d < 0.0:
        d = 0.0
    return math.sqrt(d)


# -------------------------------------------------------------
# quantiles (no numpy)
# -------------------------------------------------------------
def _quantile_sorted(xs_sorted: List[float], q: float) -> float:
    n = len(xs_sorted)
    if n <= 0:
        return 0.0
    if n == 1:
        return xs_sorted[0]
    if q <= 0.0:
        return xs_sorted[0]
    if q >= 1.0:
        return xs_sorted[-1]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    frac = pos - lo
    return xs_sorted[lo] * (1.0 - frac) + xs_sorted[hi] * frac


def _q1_med_q3(xs: List[float]) -> Tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    s = sorted(xs)
    return (
        _quantile_sorted(s, 0.25),
        _quantile_sorted(s, 0.50),
        _quantile_sorted(s, 0.75),
    )


def js_distance_pnext_aggregate(
    cond_a: Dict[Prefix, Counter],
    prefN_a: Counter,
    cond_b: Dict[Prefix, Counter],
    prefN_b: Counter,
    laplace_k: float,
    avg_mode: str,
    laplace_support: str,
) -> Tuple[float, str]:
    """
    戻り値:
      color_value: ヒートマップの色（float）
      text_value : セル内表示文字列

    avg_mode:
      weighted/uniform:
        セル値= sqrt( avg JSD ) の1値
      iqr:
        セル値(表示)= "Q1-Q3"
        色= median（prefixごとの sqrt(JSD) の中央値）
    """
    prefixes = set(prefN_a.keys()) | set(prefN_b.keys())
    if not prefixes:
        return 0.0, "0.000"

    if avg_mode not in ("weighted", "uniform", "iqr"):
        raise ValueError(f"avg_mode must be weighted|uniform|iqr, got {avg_mode}")
    if laplace_support not in ("all", "observed_ab"):
        raise ValueError(f"laplace_support must be all|observed_ab, got {laplace_support}")

    def _support_for_prefix(y: Prefix) -> Optional[Set[str]]:
        if laplace_support == "all":
            return None
        ca = cond_a.get(y, Counter())
        cb = cond_b.get(y, Counter())
        return set(ca.keys()) | set(cb.keys())

    # iqr mode: per-prefix sqrt(JSD) → Q1/med/Q3
    if avg_mode == "iqr":
        dists: List[float] = []
        for y in prefixes:
            sup = _support_for_prefix(y)
            pvec = laplace_pnext_vector(cond_a, prefN_a, y, laplace_k, support=sup)
            qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
            dists.append(js_distance_vec(pvec, qvec))
        q1, med, q3 = _q1_med_q3(dists)
        return med, f"{q1:.3f}-{q3:.3f}"

    # uniform: avg JSD then sqrt
    if avg_mode == "uniform":
        jsd_sum = 0.0
        mcnt = 0
        for y in prefixes:
            sup = _support_for_prefix(y)
            pvec = laplace_pnext_vector(cond_a, prefN_a, y, laplace_k, support=sup)
            qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
            jsd = js_divergence(pvec, qvec)
            if jsd < 0.0:
                jsd = 0.0
            jsd_sum += jsd
            mcnt += 1
        if mcnt <= 0:
            return 0.0, "0.000"
        val = math.sqrt(jsd_sum / float(mcnt))
        return val, f"{val:.3f}"

    # weighted: weights w(y)=cA(y)+cB(y) on JSD then sqrt
    wsum = 0.0
    weights: Dict[Prefix, float] = {}
    for y in prefixes:
        w = float(prefN_a.get(y, 0) + prefN_b.get(y, 0))
        weights[y] = w
        wsum += w
    if wsum <= 0.0:
        wsum = float(len(prefixes))
        for y in prefixes:
            weights[y] = 1.0

    jsd_agg = 0.0
    for y in prefixes:
        sup = _support_for_prefix(y)
        pvec = laplace_pnext_vector(cond_a, prefN_a, y, laplace_k, support=sup)
        qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
        jsd = js_divergence(pvec, qvec)
        if jsd < 0.0:
            jsd = 0.0
        jsd_agg += (weights[y] / wsum) * jsd

    val = math.sqrt(jsd_agg)
    return val, f"{val:.3f}"


# -------------------------------------------------------------
# plot
# -------------------------------------------------------------
def plot_heatmap_group_x_group(
    out_png: str,
    title: str,
    labels: List[str],
    color_mat: List[List[float]],
    text_mat: List[List[str]],
    vmin: float,
    vmax: float,
) -> None:
    R = len(labels)
    C = len(labels)

    # ★図自体も大きめ（文字が潰れないように）
    fig_w = max(10.0, C * 1.00)
    fig_h = max(8.0,  R * 1.00)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(color_mat, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im)

    plt.xticks(list(range(C)), labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), labels)

    mid = (vmin + vmax) / 2.0
    for i in range(R):
        for j in range(C):
            val = color_mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(
                j, i, text_mat[i][j],
                ha="center", va="center",
                fontsize=12,  # ★セル内文字
                color=txt_color
            )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="*.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)

    ap.add_argument("--laplace-k", type=float, default=1.0)
    ap.add_argument(
        "--laplace-support",
        choices=["all", "observed_ab"],
        default="all",
        help="all=10固定add-k, observed_ab=prefixごとにA/B観測next unionだけadd-k",
    )
    ap.add_argument(
        "--avg-mode",
        choices=["weighted", "uniform", "iqr"],
        default="weighted",
        help="weighted/uniform/iqr (iqrは表示Q1-Q3, 色はmedian)",
    )

    ap.add_argument("--unknown-as", default="Other")
    ap.add_argument("--min-1gram-total", type=int, default=1)
    ap.add_argument("--outdir", default="out/group_vs_group_total_pnext")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")
    if args.laplace_k <= 0:
        raise ValueError("--laplace-k must be > 0")

    date_start = args.start_year * 10000 + 101
    date_end   = args.end_year   * 10000 + 1231
    ns = list(range(args.nmin, args.nmax + 1))

    # load
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # groups + exclude Night/Other/ALL
    groups = collect_group_names(segs_by_person, unknown_as=args.unknown_as)
    groups = [g for g in groups if g.strip().lower() not in EXCLUDE_GROUPS]

    # totals for label (n=1, prefix=EMPTY)
    totals_1: Dict[str, int] = {}
    kept: List[str] = []
    for g in groups:
        cond1, prefN1 = count_conditional_by_group_in_range(
            segs_by_person, n=1, group_name=g,
            date_start=date_start, date_end=date_end,
            unknown_as=args.unknown_as,
        )
        t = int(sum(prefN1.values()))
        totals_1[g] = t
        if t >= args.min_1gram_total:
            kept.append(g)

    groups = kept
    if len(groups) <= 1:
        raise RuntimeError(
            f"Not enough groups after filtering. exclude={sorted(EXCLUDE_GROUPS)} "
            f"min-1gram-total={args.min_1gram_total} groups={groups}"
        )

    labels = [f"{g}({totals_1.get(g, 0)})" for g in groups]

    # global scale (ln): vmax = sqrt(ln2)
    vmin = 0.0
    vmax = 0.5

    # cache: (n, group) -> (cond, prefN)
    cache: Dict[Tuple[int, str], Tuple[Dict[Prefix, Counter], Counter]] = {}
    for n in ns:
        for g in groups:
            cache[(n, g)] = count_conditional_by_group_in_range(
                segs_by_person, n=n, group_name=g,
                date_start=date_start, date_end=date_end,
                unknown_as=args.unknown_as,
            )

    suffix = (
        f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_"
        f"k{args.laplace_k}_{args.avg_mode}_{args.laplace_support}_"
        f"unk-{args.unknown_as}_exclude-{'-'.join(sorted(EXCLUDE_GROUPS))}"
    )

    for n in ns:
        color_mat: List[List[float]] = []
        text_mat: List[List[str]] = []

        for gi in groups:
            row_c: List[float] = []
            row_t: List[str] = []
            cond_i, prefN_i = cache[(n, gi)]

            for gj in groups:
                cond_j, prefN_j = cache[(n, gj)]
                color_val, text = js_distance_pnext_aggregate(
                    cond_i, prefN_i, cond_j, prefN_j,
                    laplace_k=args.laplace_k,
                    avg_mode=args.avg_mode,
                    laplace_support=args.laplace_support,
                )
                row_c.append(color_val)
                row_t.append(text)

            color_mat.append(row_c)
            text_mat.append(row_t)

        out_png = os.path.join(args.outdir, f"heatmap_group_x_group_pnext_{n}gram_{suffix}.png")
        title = (
            f"P(next|(n-1)gram)  (n={n}) "
        )
        plot_heatmap_group_x_group(
            out_png=out_png,
            title=title,
            labels=labels,
            color_mat=color_mat,
            text_mat=text_mat,
            vmin=vmin,
            vmax=vmax,
        )
        print(f"# wrote: {out_png}")


if __name__ == "__main__":
    main()
