#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【全期間】Ward×Ward の P(next|prefix) を比較した JS distance ヒートマップ（sqrt(JSD), ln）

- 対象: 条件付き確率 P(next|prefix)（末尾シフト next）
- 比較: 病棟同士（Ward×Ward）
- 期間: start-year..end-year を全期間で集計
- n: nmin..nmax（n=1 は prefix=EMPTY で通常の 1-gram 分布と同値）
- avg-mode:
    weighted:  prefixごとのJSDを w(y)=cA(y)+cB(y) で重み付け平均 → sqrt
    uniform :  prefixごとのJSDを等重み平均 → sqrt
    iqr     :  prefixごとの sqrt(JSD) の Q1-Q3 を表示、色は median
- laplace-support:
    all        : support=VALID_SHIFTS(10) で add-k
    observed_ab: prefixごとに「A/Bで観測された next の union」だけに add-k
                （※表示ベクトルは常に10次元だが、非supportは add-kしない）
- 病棟分割は past-shifts の「ファイル単位」
  → group-settings による所属判定は使わない（引数としては互換のため残す）

表示:
- 病棟名は短い英語ラベルに変換（例: 4階南病棟 -> 4S, 7階北病棟 -> 7N）
- --plot-wards で表示する病棟をラベルで指定できる（例: "ICU,NICU,GCU,4S,7N"）
- vmax デフォルト 0.5
- セル文字色は値域の半分以上なら黒、未満なら白
"""

import os
import sys
import math
import argparse
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set

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
# import path（data_loader.py は statistics/ にある想定）
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
STAT_DIR = os.path.dirname(CURRENT_DIR)  # .../exp/statistics
if STAT_DIR not in sys.path:
    sys.path.insert(0, STAT_DIR)

import data_loader

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
X_SIZE = 10  # 表示ベクトルは常に10次元

Prefix = Tuple[str, ...]  # length = n-1

# =============================================================
# Ward label mapping (basename without .lp -> label)
# =============================================================
WARD_LABEL_MAP: Dict[str, str] = {
    # 一般病棟（階＋方角）
    "1階西病棟": "1W",
    "2階西病棟": "2W",
    "3階西病棟": "3W",
    "4階北病棟": "4N",
    "4階南病棟": "4S",
    "4階西病棟": "4W",
    "5階北病棟": "5N",
    "5階南病棟": "5S",
    "5階西病棟": "5W",
    "6階北病棟": "6N",
    "6階南病棟": "6S",
    "6階西病棟": "6W",
    "7階北病棟": "7N",
    "7階南病棟": "7S",
    "7階西病棟": "7W",

    # クリティカル系
    "集中治療室": "ICU",
    "救急外来": "ER",
    "手術部": "OR",
    "GCU": "GCU",
    "NICU": "NICU",

    # 外来・中央部門
    "外来": "OPD",
    "外来受付": "OPD-Front",
    "材料部": "CSSD",
    "感染制御部": "ICT",
    "医療の質・安全管理部": "QPS",
    "医療チームセンター": "MTC",
    "総合がん診療部": "CCU-Onc",
    "治験センター": "CRC",

    # 教育・管理
    "臨床教育部": "Clinical-Edu",
    "特定行為研修センター": "Spec-Train",
    "看護部管理室": "Nursing-Admin",
    "看護部管理室A": "Nursing-Admin-A",
    "看護部管理室B": "Nursing-Admin-B",
    "看護部管理室C": "Nursing-Admin-C",
    "師長勤務表": "Head-Nurse",
    "総合支援部A": "Support-A",
    "総合支援部B": "Support-B",
    "総合支援部C": "Support-C",

    # テスト
    "富士通テスト病棟": "Test-Ward",
}


# =============================================================
# small utils
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


def parse_csv_set(s: Optional[str]) -> Optional[Set[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: Set[str] = set()
    for tok in s.split(","):
        t = tok.strip()
        if t:
            out.add(t)
    return out or None


# =============================================================
# past-shifts dir loader (ward(label) -> seqs)
# =============================================================
def list_past_shift_files(past_dir: str) -> List[str]:
    if os.path.isfile(past_dir):
        return [past_dir]

    if not os.path.isdir(past_dir):
        raise FileNotFoundError(f"past-shifts path not found: {past_dir}")

    files: List[str] = []
    for name in sorted(os.listdir(past_dir)):
        p = os.path.join(past_dir, name)
        if os.path.isfile(p) and name.lower().endswith(".lp"):
            files.append(p)

    if not files:
        raise FileNotFoundError(f"No *.lp found under: {past_dir}")
    return files


def load_past_shifts_by_ward(
    past_dir: str,
    plot_wards: Optional[Set[str]],
    strict_label: bool,
) -> Tuple[Dict[str, Dict[Tuple[str, str], List[Tuple[int, str]]]], Dict[str, str]]:
    """
    戻り値:
      ward_to_seqs: {label: seqs}
      label_to_raw: {label: raw_basename}
    """
    ward_to_seqs: Dict[str, Dict[Tuple[str, str], List[Tuple[int, str]]]] = {}
    label_to_raw: Dict[str, str] = {}

    for f in list_past_shift_files(past_dir):
        raw = os.path.splitext(os.path.basename(f))[0]

        if raw in WARD_LABEL_MAP:
            label = WARD_LABEL_MAP[raw]
        else:
            if strict_label:
                print(f"# skip (no label mapping): {raw}")
                continue
            label = f"RAW:{raw}"

        if plot_wards is not None and label not in plot_wards:
            continue

        seqs = data_loader.load_past_shifts(f)
        if not seqs:
            continue

        ward_to_seqs[label] = seqs
        label_to_raw[label] = raw

    if not ward_to_seqs:
        if plot_wards is None:
            raise SystemExit("No wards loaded. Check mapping or past-shifts directory.")
        raise SystemExit(f"No wards loaded. Check --plot-wards={sorted(plot_wards)} and mapping.")

    return ward_to_seqs, label_to_raw


# =============================================================
# count conditional P(next|prefix) per ward (ALL within ward)
# =============================================================
def count_conditional_by_ward_in_range(
    ward_seqs: Dict[Tuple[str, str], List[Tuple[int, str]]],
    n: int,
    date_start: int,
    date_end: int,
) -> Tuple[Dict[Prefix, Counter], Counter]:
    """
    戻り値:
      cond[prefix][next] = count
      prefN[prefix]      = total count for that prefix
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    cond: Dict[Prefix, Counter] = defaultdict(Counter)
    prefN: Counter = Counter()
    pref_len = n - 1

    for (_nid, _name), seq in ward_seqs.items():
        seq = normalize_seq(seq)
        if not seq:
            continue
        seq.sort(key=lambda t: t[0])

        sseq = [(d, s) for (d, s) in seq if within_range(d, date_start, date_end)]
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


# =============================================================
# probability vectors + JS distance
# =============================================================
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


# =============================================================
# quantiles (no numpy)
# =============================================================
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


# =============================================================
# plot
# =============================================================
def plot_heatmap_ward_x_ward(
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

    fig_w = max(10.0, C * 1.00)
    fig_h = max(8.0,  R * 1.00)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(color_mat, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im)

    plt.xticks(list(range(C)), labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), labels)

    # ★「値域の半分以上」なら黒
    threshold = vmin + 0.5 * (vmax - vmin)

    for i in range(R):
        for j in range(C):
            val = float(color_mat[i][j])
            txt_color = "black" if val >= threshold else "white"
            plt.text(
                j, i, text_mat[i][j],
                ha="center", va="center",
                fontsize=12,
                color=txt_color
            )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts_dir", help="past-shifts dir (contains many *.lp) OR a single *.lp")
    ap.add_argument("group_settings_dir", help="group-settings dir (not used; kept for compatibility)")

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

    ap.add_argument("--min-1gram-total", type=int, default=1)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--outdir", default="out/ward_vs_ward_total_pnext")

    # ★追加: 表示する病棟をラベルで指定（例: ICU,NICU,GCU,4S,7N）
    ap.add_argument(
        "--plot-wards",
        default="",
        help="Comma-separated ward labels to include (e.g. 'ICU,NICU,GCU,4S,7N'). Empty means include all mapped wards.",
    )
    ap.add_argument(
        "--strict-label",
        action="store_true",
        help="If set, wards without label mapping are skipped instead of using 'RAW:<name>'.",
    )

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

    plot_wards = parse_csv_set(args.plot_wards)

    # 病棟ごとに past-shifts を読む（ラベル化 + フィルタ）
    ward_to_seqs, label_to_raw = load_past_shifts_by_ward(
        args.past_shifts_dir,
        plot_wards=plot_wards,
        strict_label=args.strict_label,
    )

    wards = sorted(ward_to_seqs.keys(), key=lambda x: x.lower())

    # totals for label (n=1, prefix=EMPTY)
    totals_1: Dict[str, int] = {}
    kept: List[str] = []
    for w in wards:
        cond1, prefN1 = count_conditional_by_ward_in_range(
            ward_to_seqs[w], n=1, date_start=date_start, date_end=date_end
        )
        t = int(sum(prefN1.values()))
        totals_1[w] = t
        if t >= args.min_1gram_total:
            kept.append(w)

    wards = kept
    if len(wards) <= 1:
        raise RuntimeError(
            f"Not enough wards after filtering. "
            f"min-1gram-total={args.min_1gram_total} wards={wards}"
        )

    labels = [f"{w}({totals_1.get(w, 0)})" for w in wards]

    # debug: loaded wards
    print("# loaded wards (label -> raw_basename) :")
    for w in wards:
        print(f"#   {w:12s} <- {label_to_raw.get(w, '?')}  total_1gram={totals_1.get(w, 0)}")

    # global scale (you requested fixed vmax style)
    vmin = 0.0
    vmax = float(args.vmax)

    # cache: (n, ward) -> (cond, prefN)
    cache: Dict[Tuple[int, str], Tuple[Dict[Prefix, Counter], Counter]] = {}
    for n in ns:
        for w in wards:
            cache[(n, w)] = count_conditional_by_ward_in_range(
                ward_to_seqs[w], n=n, date_start=date_start, date_end=date_end
            )

    suffix = (
        f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_"
        f"k{args.laplace_k}_{args.avg_mode}_{args.laplace_support}_"
        f"min1-{args.min_1gram_total}_vmax{args.vmax}"
    )

    for n in ns:
        color_mat: List[List[float]] = []
        text_mat: List[List[str]] = []

        for wi in wards:
            row_c: List[float] = []
            row_t: List[str] = []
            cond_i, prefN_i = cache[(n, wi)]

            for wj in wards:
                cond_j, prefN_j = cache[(n, wj)]
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

        out_png = os.path.join(args.outdir, f"heatmap_ward_x_ward_pnext_{n}gram_{suffix}.png")
        title = f"P(next|(n-1)gram)  (n={n})"
        plot_heatmap_ward_x_ward(
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
