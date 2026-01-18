#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
half-year period × (n=1..5 の「2019~2025 を基準にした 条件付き確率P(next|prefix) の JS距離」) ヒートマップ
+ デバッグログ出力版

デバッグで見たいもの:
  - prefix→next のカウント (c(y,x)) と prefix総数 (c(y))
  - （参考）n-gram（列）の頻度（tuple長n）
  - 円滑化前（MLE）と円滑化後（Laplace）の確率
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
X_SIZE = 10  # 必ず10固定

UNKNOWN_GROUP = "__UNKNOWN__"

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
    tg = target_group.lower()
    return any(g.lower() == tg for g in group_set)


class Segment:
    __slots__ = ("groups", "seq")

    def __init__(self, groups: FrozenSet[str]):
        self.groups = groups
        self.seq: List[Tuple[int, str]] = []


def build_segments_for_person(seq, name, nid, timeline) -> List["Segment"]:
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


# -------------------------------------------------------------
# counts for P(next|prefix)
# -------------------------------------------------------------
def count_conditional_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Dict[Prefix, Counter], Counter, Dict[Prefix, Counter], Counter]:
    if n < 1:
        raise ValueError("n must be >= 1")

    heads_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    non_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    heads_prefN: Counter = Counter()
    non_prefN: Counter = Counter()

    pref_len = n - 1

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for (_, s) in sseq]

            if n == 1:
                pfx: Prefix = tuple()
                for x in shifts:
                    if is_heads:
                        heads_cond[pfx][x] += 1
                        heads_prefN[pfx] += 1
                    else:
                        non_cond[pfx][x] += 1
                        non_prefN[pfx] += 1
                continue

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i : i + pref_len])
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# -------------------------------------------------------------
# (debug) n-gram frequency counts (tuple length n)
# -------------------------------------------------------------
def count_ngram_freq_heads_nonheads_in_range(
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

            shifts = [s for (_, s) in sseq]
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i : i + n])
                if is_heads:
                    heads[gram] += 1
                else:
                    nonheads[gram] += 1
    return heads, nonheads


def laplace_pnext_vector(cond: Dict[Prefix, Counter], prefN: Counter, prefix: Prefix, k: float) -> List[float]:
    Ny = float(prefN.get(prefix, 0))
    denom = Ny + k * X_SIZE
    cxy = cond.get(prefix, Counter())
    return [(cxy.get(x, 0) + k) / denom for x in VALID_SHIFTS]


def mle_pnext_vector(cond: Dict[Prefix, Counter], prefN: Counter, prefix: Prefix) -> List[float]:
    Ny = float(prefN.get(prefix, 0))
    if Ny <= 0:
        # prefixが存在しない場合は一様にしておく（ログ用）
        return [1.0 / X_SIZE] * X_SIZE
    cxy = cond.get(prefix, Counter())
    return [cxy.get(x, 0) / Ny for x in VALID_SHIFTS]


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


def js_distance_pnext_weighted(
    cond_p: Dict[Prefix, Counter],
    prefN_p: Counter,
    cond_b: Dict[Prefix, Counter],
    prefN_b: Counter,
    laplace_k: float,
) -> float:
    prefixes = set(prefN_p.keys()) | set(prefN_b.keys())
    if not prefixes:
        return 0.0

    wsum = 0.0
    weights = {}
    for y in prefixes:
        w = float(prefN_p.get(y, 0) + prefN_b.get(y, 0))
        weights[y] = w
        wsum += w
    if wsum <= 0.0:
        wsum = float(len(prefixes))
        for y in prefixes:
            weights[y] = 1.0

    jsd_agg = 0.0
    for y in prefixes:
        pvec = laplace_pnext_vector(cond_p, prefN_p, y, laplace_k)
        qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k)
        jsd = js_divergence(pvec, qvec)
        if jsd < 0:
            jsd = 0.0
        jsd_agg += (weights[y] / wsum) * jsd

    return math.sqrt(jsd_agg)


# ---------------------------
# periods
# ---------------------------
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101,  y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701,  y * 10000 + 1231))
    return periods


def plot_heatmap_period_x_n(out_png, title, period_labels, n_labels, mat, vmin, vmax) -> None:
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(7.0, R * 0.55)

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
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# DEBUG PRINTERS
# ---------------------------
def _fmt_prefix(pfx: Prefix) -> str:
    return ",".join(pfx) if pfx else "(EMPTY)"


def debug_dump_pnext(
    tag: str,
    cond: Dict[Prefix, Counter],
    prefN: Counter,
    n: int,
    laplace_k: float,
    topk_prefix: int,
    focus_prefix: Optional[Prefix],
) -> None:
    """
    - prefix上位(topk_prefix)を表示
    - focus_prefix があればそれも必ず表示
    - 各 prefix で: c(y), top next counts, MLE確率, Laplace確率（各上位だけ）
    """
    print("")
    print("=" * 80)
    print(f"[DEBUG] {tag}  n={n}  (|X|=10 fixed)  laplace_k={laplace_k}")
    print(f"[DEBUG] prefixes: {len(prefN)}  total_prefix_events(sum c(y))={sum(prefN.values())}")
    print("=" * 80)

    # candidate prefixes
    cand = [p for p, _ in prefN.most_common(topk_prefix)]
    if focus_prefix is not None and focus_prefix not in prefN:
        print(f"[DEBUG] focus_prefix={_fmt_prefix(focus_prefix)} is NOT observed in this range.")
        # still show it (Ny=0)
        cand = [focus_prefix] + cand
    elif focus_prefix is not None:
        if focus_prefix not in cand:
            cand = [focus_prefix] + cand

    seen = set()
    for pfx in cand:
        if pfx in seen:
            continue
        seen.add(pfx)

        Ny = prefN.get(pfx, 0)
        cxy = cond.get(pfx, Counter())
        top_next = cxy.most_common(5)

        print(f"\n[DEBUG] prefix={_fmt_prefix(pfx)}  c(y)={Ny}  distinct_next={len(cxy)}")
        if top_next:
            print("[DEBUG]   top next counts:", ", ".join([f"{nx}:{cnt}" for nx, cnt in top_next]))
        else:
            print("[DEBUG]   top next counts: (none)")

        # probabilities (print only for top next + a couple)
        mle = mle_pnext_vector(cond, prefN, pfx)
        lap = laplace_pnext_vector(cond, prefN, pfx, laplace_k)

        # show probs for top next + also show the largest laplace probs
        keys = [nx for nx, _ in top_next]
        # ensure unique
        keys = list(dict.fromkeys(keys))
        # if Ny==0, show all? (too long) -> show top by laplace
        if not keys:
            # show top 5 by laplace
            idxs = sorted(range(len(VALID_SHIFTS)), key=lambda i: lap[i], reverse=True)[:5]
            keys = [VALID_SHIFTS[i] for i in idxs]

        print("[DEBUG]   probs (MLE vs Laplace):")
        for x in keys:
            i = VALID_SHIFTS.index(x)
            print(f"          {x:>2}:  mle={mle[i]:.6f}   lap={lap[i]:.6f}")


def debug_dump_ngram_freq(
    tag: str,
    ngram_counter: Counter,
    topk: int,
) -> None:
    print("")
    print("-" * 80)
    print(f"[DEBUG] {tag}  n-gram frequency top{topk}")
    print(f"[DEBUG] total_ngrams={sum(ngram_counter.values())}  distinct={len(ngram_counter)}")
    for gram, cnt in ngram_counter.most_common(topk):
        print(f"  {cnt:>8}  {'-'.join(gram)}")
    print("-" * 80)


def parse_prefix_arg(s: str) -> Prefix:
    """
    "SE,SN" -> ("SE","SN")
    "" or "EMPTY" -> ()
    """
    s = s.strip()
    if s == "" or s.upper() == "EMPTY":
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if p not in VALID_SHIFTS_SET:
            raise ValueError(f"--debug-prefix contains invalid shift: {p}")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts")
    ap.add_argument("group_settings")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--laplace-k", type=float, default=1.0)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/halfyear_vs_total_pnext")
    ap.add_argument("--only-nonheads", action="store_true")

    # --- debug options ---
    ap.add_argument("--debug", action="store_true", help="中間ログを出す")
    ap.add_argument("--debug-period", default=None, help="例: 2019H1 / 2020H2 / 2019~2025 (base)")
    ap.add_argument("--debug-n", type=int, default=None, help="例: 3 (指定したnだけログ)")
    ap.add_argument("--debug-prefix", default=None, help='例: "SE,SN" (n=3ならprefix長2)')
    ap.add_argument("--debug-topk", type=int, default=10, help="prefix上位を何件表示するか")
    ap.add_argument("--debug-ngram-topk", type=int, default=15, help="n-gram頻度上位を何件表示するか（参考）")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.laplace_k <= 0:
        raise ValueError("--laplace-k must be > 0")

    # load
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # periods + base
    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231
    periods = half_periods + [(base_key, base_start, base_end)]
    period_labels = [k for (k, _, _) in periods]

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram P(next|prefix) (vs {base_key})" for n in ns]

    # base cache
    base_heads_cond_by_n = {}
    base_heads_prefN_by_n = {}
    base_non_cond_by_n = {}
    base_non_prefN_by_n = {}

    for n in ns:
        h_cond, h_prefN, non_cond, non_prefN = count_conditional_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_heads_cond_by_n[n] = h_cond
        base_heads_prefN_by_n[n] = h_prefN
        base_non_cond_by_n[n] = non_cond
        base_non_prefN_by_n[n] = non_prefN

    # DEBUG: dump selected period/n
    focus_prefix: Optional[Prefix] = None
    if args.debug_prefix is not None:
        focus_prefix = parse_prefix_arg(args.debug_prefix)

    if args.debug:
        dbg_period = args.debug_period if args.debug_period is not None else period_labels[0]
        dbg_n = args.debug_n if args.debug_n is not None else ns[0]

        # find period range
        found = [t for t in periods if t[0] == dbg_period]
        if not found:
            raise ValueError(f"--debug-period '{dbg_period}' not found. choices: {', '.join(period_labels[:6])} ...")
        pkey, d1, d2 = found[0]

        print("\n" + "#" * 90)
        print(f"[DEBUG] period={pkey} range={d1}..{d2}  n={dbg_n}")
        print("#" * 90)

        # count for that period
        h_cond_p, h_prefN_p, non_cond_p, non_prefN_p = count_conditional_heads_nonheads_in_range(
            segs_by_person, n=dbg_n, heads_name=args.heads_name, date_start=d1, date_end=d2
        )

        # also compute ngram freq (reference)
        h_ng, non_ng = count_ngram_freq_heads_nonheads_in_range(
            segs_by_person, n=dbg_n, heads_name=args.heads_name, date_start=d1, date_end=d2
        )

        if not args.only_nonheads:
            debug_dump_pnext(
                tag=f"{pkey} Heads",
                cond=h_cond_p,
                prefN=h_prefN_p,
                n=dbg_n,
                laplace_k=args.laplace_k,
                topk_prefix=args.debug_topk,
                focus_prefix=focus_prefix,
            )
            debug_dump_ngram_freq(f"{pkey} Heads", h_ng, args.debug_ngram_topk)

        debug_dump_pnext(
            tag=f"{pkey} NonHeads",
            cond=non_cond_p,
            prefN=non_prefN_p,
            n=dbg_n,
            laplace_k=args.laplace_k,
            topk_prefix=args.debug_topk,
            focus_prefix=focus_prefix,
        )
        debug_dump_ngram_freq(f"{pkey} NonHeads", non_ng, args.debug_ngram_topk)

        # compare with base for same n (optional)
        print("\n" + "#" * 90)
        print(f"[DEBUG] compare {pkey} vs BASE={base_key} (same n={dbg_n})  Laplace-k={args.laplace_k}")
        print("#" * 90)

        if not args.only_nonheads:
            dist_h = js_distance_pnext_weighted(
                h_cond_p, h_prefN_p,
                base_heads_cond_by_n[dbg_n], base_heads_prefN_by_n[dbg_n],
                laplace_k=args.laplace_k
            )
            print(f"[DEBUG] Heads JSdist(P(next|prefix)) = {dist_h:.6f}")

        dist_n = js_distance_pnext_weighted(
            non_cond_p, non_prefN_p,
            base_non_cond_by_n[dbg_n], base_non_prefN_by_n[dbg_n],
            laplace_k=args.laplace_k
        )
        print(f"[DEBUG] NonHeads JSdist(P(next|prefix)) = {dist_n:.6f}")
        print("#" * 90 + "\n")

    # heatmap
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

    def build_matrix(is_heads: bool) -> List[List[float]]:
        mat = []
        for (pkey, d1, d2) in periods:
            row = []
            for n in ns:
                h_cond_p, h_prefN_p, non_cond_p, non_prefN_p = count_conditional_heads_nonheads_in_range(
                    segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                )
                if is_heads:
                    cond_p, prefN_p = h_cond_p, h_prefN_p
                    cond_b, prefN_b = base_heads_cond_by_n[n], base_heads_prefN_by_n[n]
                else:
                    cond_p, prefN_p = non_cond_p, non_prefN_p
                    cond_b, prefN_b = base_non_cond_by_n[n], base_non_prefN_by_n[n]

                row.append(js_distance_pnext_weighted(cond_p, prefN_p, cond_b, prefN_b, laplace_k=args.laplace_k))
            mat.append(row)
        return mat

    suffix = f"laplacek{args.laplace_k}".replace(".", "p")

    if not args.only_nonheads:
        mat_h = build_matrix(is_heads=True)
        out_h = os.path.join(
            args.outdir,
            f"heatmap_halfyear_x_pnext_heads_{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{suffix}.png",
        )
        plot_heatmap_period_x_n(
            out_h,
            title=f"Heads: JSdist(half-year, {base_key}) on P(next|prefix) n={args.nmin}..{args.nmax} [ln] ({suffix})",
            period_labels=[k for (k, _, _) in periods],
            n_labels=n_labels,
            mat=mat_h,
            vmin=vmin,
            vmax=vmax,
        )
        print(f"# wrote: {out_h}")

    mat_n = build_matrix(is_heads=False)
    out_n = os.path.join(
        args.outdir,
        f"heatmap_halfyear_x_pnext_nonheads_{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{suffix}.png",
    )
    plot_heatmap_period_x_n(
        out_n,
        title=f"NonHeads: JSdist(half-year, {base_key}) on P(next|prefix) n={args.nmin}..{args.nmax} [ln] ({suffix})",
        period_labels=[k for (k, _, _) in periods],
        n_labels=n_labels,
        mat=mat_n,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"# wrote: {out_n}")


if __name__ == "__main__":
    main()
