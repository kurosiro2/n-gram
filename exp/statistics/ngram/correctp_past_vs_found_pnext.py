#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conditional probability (P(next|prefix)) version (n=2..N):
Extract prefixes to fix by comparing JSD distances:

For each n (2..nmax):
  - prefix length = n-1
  - For each prefix y:
      manual_jsd_list[y] = [ JSdist( P_halfyear(.|y), P_base(.|y) ) for each half-year ]
      manual_IQR[y]      = Q1–Q3 of that list
      found_jsd[y]       = JSdist( P_found(.|y), P_base(.|y) )
      diff               = distance from found_jsd to [Q1,Q3] (0 if inside)

Output (topK per n):
  n, prefix->next,
  base_prefix_mean_count, found_prefix_count,
  manual_jsd_Q1–Q3, found_jsd, diff

Evidence filter:
  --min-prefix-count : keep prefixes with (base_sum(prefix) + found_count(prefix)) >= this (per bucket, per n)
    * base is computed as MANUAL half-year mean count, but filtering uses base SUM (not mean) to keep scale comparable.

Print formatting:
  - Stdout uses fixed-width columns (no tab) to avoid broken alignment.
  - CSV keeps "–" in manual_jsd_q1_q3.
  - Stdout can use ASCII "-" via --print-ascii-dash (recommended if terminal width issues).
  - Stdout prefix column width can be adjusted by --print-prefix-width.
  - --print-full-prefix shows full prefix (no truncation); may wrap.
"""

import os
import sys
import argparse
import glob
import re
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

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
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
X_SIZE = 10
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]


# =============================================================
# helpers
# =============================================================
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


def fmt_prefix(pfx: Prefix) -> str:
    return ",".join(pfx) if pfx else "(EMPTY)"


def interval_distance(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# =============================================================
# MANUAL: segment builder (group-timeline aware)
# =============================================================
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


# =============================================================
# MANUAL: conditional counts for P(next|prefix)
# =============================================================
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
                pfx = tuple(shifts[i: i + pref_len])
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# =============================================================
# FOUND: load + conditional counts (no timeline)
# =============================================================
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_OUT = re.compile(r'^out_assigned\(\s*(\d+)\s*,\s*(\d{8})\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')
PAT_GROUP2 = re.compile(r'^group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')


def is_head_group_found(g: str) -> bool:
    if not g:
        return False
    gl = g.lower()
    return ("head" in gl) or ("師長" in g) or ("主任" in g)


def bucket_found(groups: Set[str], heads_name: str) -> str:
    hn = (heads_name or "").strip().lower()
    for g in groups:
        if (g or "").strip().lower() == hn:
            return "Heads"
        if is_head_group_found(g):
            return "Heads"
    return "NonHeads"


def _pick_lp_files_in_dir(d: str) -> List[str]:
    fs = sorted(glob.glob(os.path.join(d, "found-model*.lp")))
    if fs:
        return fs
    return sorted(glob.glob(os.path.join(d, "*.lp")))


def collect_found_entries(found_path: str) -> List[Tuple[str, List[str]]]:
    if os.path.isfile(found_path):
        label = os.path.basename(os.path.normpath(found_path))
        return [(label, [found_path])]

    if not os.path.isdir(found_path):
        return []

    found_path = os.path.normpath(found_path)

    direct = _pick_lp_files_in_dir(found_path)
    if direct:
        label = os.path.basename(found_path)
        return [(label, direct)]

    entries: List[Tuple[str, List[str]]] = []
    try:
        for name in sorted(os.listdir(found_path)):
            sub = os.path.join(found_path, name)
            if not os.path.isdir(sub):
                continue
            fs = _pick_lp_files_in_dir(sub)
            if fs:
                entries.append((os.path.basename(sub), fs))
    except OSError:
        pass

    if entries:
        return entries

    cand = sorted(glob.glob(os.path.join(found_path, "**", "found-model*.lp"), recursive=True))
    if not cand:
        cand = sorted(glob.glob(os.path.join(found_path, "**", "*.lp"), recursive=True))
    if not cand:
        return []

    by_dir: Dict[str, List[str]] = defaultdict(list)
    for fp in cand:
        by_dir[os.path.dirname(fp)].append(fp)

    out: List[Tuple[str, List[str]]] = []
    for d in sorted(by_dir.keys()):
        files = sorted(by_dir[d])
        out.append((os.path.basename(d), files))
    return out


def load_found_model(path: str) -> Tuple[Dict[int, List[str]], Dict[int, Set[str]]]:
    seqs_by_staff: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    groups_by_staff: Dict[int, Set[str]] = defaultdict(set)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue

            m = PAT_EXT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))
                sh = m.group(3)
                if sh in VALID_SHIFTS_SET:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_OUT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))
                sh = m.group(3)
                if sh in VALID_SHIFTS_SET:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

            m = PAT_GROUP2.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

    shifts_by_staff: Dict[int, List[str]] = {}
    for sid, pairs in seqs_by_staff.items():
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        shifts_by_staff[sid] = [s for _, s in pairs_sorted]

    return shifts_by_staff, groups_by_staff


def count_conditional_found_heads_nonheads(
    found_files: List[str],
    n: int,
    heads_name: str,
) -> Tuple[Dict[Prefix, Counter], Counter, Dict[Prefix, Counter], Counter]:
    if n < 1:
        raise ValueError("n must be >= 1")

    heads_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    non_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    heads_prefN: Counter = Counter()
    non_prefN: Counter = Counter()

    pref_len = n - 1

    for fp in found_files:
        shifts_by_staff, groups_by_staff = load_found_model(fp)

        for sid, shifts in shifts_by_staff.items():
            if len(shifts) < n:
                continue

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)

            if n == 1:
                pfx: Prefix = tuple()
                for x in shifts:
                    if bucket == "Heads":
                        heads_cond[pfx][x] += 1
                        heads_prefN[pfx] += 1
                    else:
                        non_cond[pfx][x] += 1
                        non_prefN[pfx] += 1
                continue

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i: i + pref_len])
                nxt = shifts[i + pref_len]
                if bucket == "Heads":
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# =============================================================
# JSD (sqrt(JSD)) [ln] + Laplace for P(next|prefix)
# =============================================================
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


def pnext_vec_for_two(
    condA: Dict[Prefix, Counter],
    prefNA: Counter,
    condB: Dict[Prefix, Counter],
    prefNB: Counter,
    prefix: Prefix,
    k: float,
    laplace_support: str,
) -> Tuple[List[float], List[float]]:
    NyA = float(prefNA.get(prefix, 0))
    NyB = float(prefNB.get(prefix, 0))
    cA = condA.get(prefix, Counter())
    cB = condB.get(prefix, Counter())

    # k=0: MLE
    if k <= 0.0:
        if NyA <= 0.0:
            pA = [1.0 / X_SIZE] * X_SIZE
        else:
            pA = [cA.get(x, 0) / NyA for x in VALID_SHIFTS]
        if NyB <= 0.0:
            pB = [1.0 / X_SIZE] * X_SIZE
        else:
            pB = [cB.get(x, 0) / NyB for x in VALID_SHIFTS]
        return pA, pB

    # k>0: Laplace
    if laplace_support == "observed_ab":
        support = set(cA.keys()) | set(cB.keys())
        if not support:
            support = VALID_SHIFTS_SET
        denomA = NyA + k * len(support)
        denomB = NyB + k * len(support)
        if denomA <= 0.0:
            denomA = k * len(support)
        if denomB <= 0.0:
            denomB = k * len(support)

        pA: List[float] = []
        pB: List[float] = []
        for x in VALID_SHIFTS:
            if x in support:
                pA.append((cA.get(x, 0) + k) / denomA)
                pB.append((cB.get(x, 0) + k) / denomB)
            else:
                pA.append(cA.get(x, 0) / denomA)
                pB.append(cB.get(x, 0) / denomB)
        return pA, pB

    # laplace_support == "all"
    denomA = NyA + k * X_SIZE
    denomB = NyB + k * X_SIZE
    if denomA <= 0.0:
        denomA = k * X_SIZE
    if denomB <= 0.0:
        denomB = k * X_SIZE

    pA = [(cA.get(x, 0) + k) / denomA for x in VALID_SHIFTS]
    pB = [(cB.get(x, 0) + k) / denomB for x in VALID_SHIFTS]
    return pA, pB


# =============================================================
# half-year periods
# =============================================================
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101,  y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701,  y * 10000 + 1231))
    return periods


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


def q1_q3(xs: List[float]) -> Tuple[float, float]:
    s = sorted(xs)
    return _quantile_sorted(s, 0.25), _quantile_sorted(s, 0.75)


# =============================================================
# pretty print (fixed width)
# =============================================================
def _make_dash(ascii_dash: bool) -> str:
    return "-" if ascii_dash else "–"


def _fmt_iqr(q1: float, q3: float, ascii_dash: bool) -> str:
    d = _make_dash(ascii_dash)
    return f"{q1:.6f}{d}{q3:.6f}"


def _truncate(s: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: width - 1] + "…"


def print_table(
    rows: List[Tuple[int, str, float, int, str, float, float]],
    title: str,
    prefix_width: int,
    ascii_dash: bool,
    full_prefix: bool,
) -> None:
    # rows: (rank, prefix_to_next, base_mean_cnt, found_cnt, manual_iqr_str, found_jsd, diff)
    prefix_width = _clamp_int(prefix_width, 10, 200)

    h_rank = "rank"
    h_pfx = "prefix->next"
    h_base = "manual(mean)"
    h_found = "auto"
    h_iqr = "manual_JSD Q1-Q3" if ascii_dash else "manual_JSD Q1–Q3"
    h_fjsd = "auto_JSD"
    h_diff = "diff"

    print(title)
    print(
        f"{h_rank:>4}  "
        f"{h_pfx:<{prefix_width}}  "
        f"{h_base:>10}  "
        f"{h_found:>8}  "
        f"{h_iqr:<20}  "
        f"{h_fjsd:>10}  "
        f"{h_diff:>10}"
    )

    for rank, pfx, base_mean_cnt, found_cnt, iqr, found_jsd, diff in rows:
        pfx_show = pfx if full_prefix else _truncate(pfx, prefix_width)
        print(
            f"{rank:>4}  "
            f"{pfx_show:<{prefix_width}}  "
            f"{base_mean_cnt:>10.3f}  "
            f"{found_cnt:>8}  "
            f"{iqr:<20}  "
            f"{found_jsd:>10.6f}  "
            f"{diff:>10.6f}"
        )


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts *.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("found_path", help="found dir OR found-model.lp")

    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)

    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=5)

    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-prefix-count", type=int, default=0,
                    help="keep prefixes with (base_sum(prefix) + found_count(prefix)) >= this (per bucket, per n)")
    ap.add_argument("--laplace-k", type=float, default=1.0,
                    help="k>=0 (k=0 means MLE)")
    ap.add_argument("--laplace-support", choices=["all", "observed_ab"], default="all")

    ap.add_argument("--outdir", default="out/fix_candidates_pnext_jsd")
    ap.add_argument("--only-nonheads", action="store_true")
    ap.add_argument("--print", dest="do_print", action="store_true")

    # ---- print formatting options
    ap.add_argument("--print-prefix-width", type=int, default=28,
                    help="stdout prefix->next column width (ignored if --print-full-prefix)")
    ap.add_argument("--print-full-prefix", action="store_true",
                    help="do not truncate prefix->next in stdout (may wrap)")
    ap.add_argument("--print-ascii-dash", action="store_true",
                    help="use ASCII '-' instead of '–' in stdout (CSV stays with '–')")

    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin < 2 or args.nmax < 2 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 2 <= nmin <= nmax")
    if args.topk <= 0:
        raise ValueError("--topk must be >= 1")
    if args.min_prefix_count < 0:
        raise ValueError("--min-prefix-count must be >= 0")
    if args.laplace_k < 0.0:
        raise ValueError("--laplace-k must be >= 0 (k=0 means MLE)")

    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    found_entries = collect_found_entries(args.found_path)
    if not found_entries:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

    # ---- load MANUAL + timeline segments
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # ---- periods and base range
    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    ns = list(range(args.nmin, args.nmax + 1))

    # ---- base caches (per n) for P_base(.|y)
    base_by_n_bucket: Dict[int, Dict[str, Tuple[Dict[Prefix, Counter], Counter]]] = {}
    for n in ns:
        hB_cond, hB_prefN, nB_cond, nB_prefN = count_conditional_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_by_n_bucket[n] = {
            "Heads": (hB_cond, hB_prefN),
            "NonHeads": (nB_cond, nB_prefN),
        }

    # ---- manual halfyear caches (per n, per bucket, per period idx)
    manual_period_counts: Dict[int, Dict[str, List[Tuple[Dict[Prefix, Counter], Counter]]]] = {}
    # ---- manual halfyear totals of "prefix occurrences" (per n, per bucket, per period idx)
    manual_period_totals: Dict[int, Dict[str, List[int]]] = {}

    for n in ns:
        manual_period_counts[n] = {"Heads": [], "NonHeads": []}
        manual_period_totals[n] = {"Heads": [], "NonHeads": []}

        for (_k, d1, d2) in half_periods:
            hC, hN, nC, nN = count_conditional_heads_nonheads_in_range(
                segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
            )
            manual_period_counts[n]["Heads"].append((hC, hN))
            manual_period_counts[n]["NonHeads"].append((nC, nN))

            # total prefix occurrences in that half-year = sum(prefN.values())
            manual_period_totals[n]["Heads"].append(int(sum(hN.values())))
            manual_period_totals[n]["NonHeads"].append(int(sum(nN.values())))

    for found_label, found_files in found_entries:
        # found caches per n
        found_by_n_bucket: Dict[int, Dict[str, Tuple[Dict[Prefix, Counter], Counter]]] = {}
        for n in ns:
            hF_cond, hF_prefN, nF_cond, nF_prefN = count_conditional_found_heads_nonheads(
                found_files, n=n, heads_name=args.heads_name
            )
            found_by_n_bucket[n] = {
                "Heads": (hF_cond, hF_prefN),
                "NonHeads": (nF_cond, nF_prefN),
            }

        buckets = ["NonHeads"] if args.only_nonheads else ["Heads", "NonHeads"]

        for bucket in buckets:
            for n in ns:
                base_cond, base_prefN = base_by_n_bucket[n][bucket]
                found_cond, found_prefN = found_by_n_bucket[n][bucket]

                prefixes: Set[Prefix] = set(base_prefN.keys()) | set(found_prefN.keys())
                for (condP, prefNP) in manual_period_counts[n][bucket]:
                    prefixes |= set(prefNP.keys())
                    prefixes |= set(condP.keys())

                # valid half-years for "base mean count": periods where total prefix occurrences > 0
                valid_periods = sum(1 for t in manual_period_totals[n][bucket] if t > 0)
                if valid_periods <= 0:
                    valid_periods = 1

                manual_jsd_list: Dict[Prefix, List[float]] = defaultdict(list)

                for y in prefixes:
                    for idx in range(len(half_periods)):
                        condP, prefNP = manual_period_counts[n][bucket][idx]
                        # if this half-year has no prefix occurrences at all, skip (no information)
                        if manual_period_totals[n][bucket][idx] <= 0:
                            continue
                        # also skip if both period and base have zero evidence for y
                        if prefNP.get(y, 0) <= 0 and base_prefN.get(y, 0) <= 0:
                            continue

                        pP, pB = pnext_vec_for_two(
                            condP, prefNP, base_cond, base_prefN,
                            prefix=y, k=args.laplace_k, laplace_support=args.laplace_support
                        )
                        manual_jsd_list[y].append(js_distance_vec(pP, pB))

                # scored: (diff, found_jsd, base_mean_cnt, base_sum_cnt, found_cnt, prefix, q1, q3)
                scored: List[Tuple[float, float, float, int, int, Prefix, float, float]] = []

                for y, xs in manual_jsd_list.items():
                    if not xs:
                        continue

                    # base mean count: half-year mean of prefix occurrences (exclude total==0 half-years)
                    base_sum = 0
                    for idx in range(len(half_periods)):
                        if manual_period_totals[n][bucket][idx] <= 0:
                            continue
                        _condP, prefNP = manual_period_counts[n][bucket][idx]
                        base_sum += int(prefNP.get(y, 0))
                    base_mean_cnt = base_sum / float(valid_periods)

                    found_cnt = int(found_prefN.get(y, 0))

                    # evidence filter uses base SUM + found count (scale-consistent)
                    if (base_sum + found_cnt) < args.min_prefix_count:
                        continue

                    q1, q3 = q1_q3(xs)

                    # compute found_jsd against base distribution
                    if found_cnt <= 0 and base_prefN.get(y, 0) <= 0:
                        continue

                    pF, pB = pnext_vec_for_two(
                        found_cond, found_prefN, base_cond, base_prefN,
                        prefix=y, k=args.laplace_k, laplace_support=args.laplace_support
                    )
                    found_jsd = js_distance_vec(pF, pB)
                    diff = interval_distance(found_jsd, q1, q3)

                    scored.append((diff, found_jsd, base_mean_cnt, base_sum, found_cnt, y, q1, q3))

                scored.sort(key=lambda t: (t[0], t[1], t[3] + t[4]), reverse=True)
                top = scored[:args.topk]

                out_csv = os.path.join(
                    args.outdir,
                    f"fix_candidates_pnextJSD_{bucket}_found-{found_label}_n{n}_k{args.laplace_k}_{args.laplace_support}.csv"
                )

                rows: List[List[str]] = []
                for diff, found_jsd, base_mean_cnt, _base_sum, found_cnt, y, q1, q3 in top:
                    rows.append([
                        str(n),
                        f"{fmt_prefix(y)}->next",
                        f"{base_mean_cnt:.6f}",
                        str(found_cnt),
                        f"{q1:.6f}–{q3:.6f}",
                        f"{found_jsd:.6f}",
                        f"{diff:.6f}",
                    ])

                write_csv(
                    out_csv,
                    ["n", "prefix_to_next", "base_prefix_mean_count", "found_prefix_count",
                     "manual_jsd_q1_q3", "found_jsd", "diff"],
                    rows,
                )
                print(f"# wrote: {out_csv}")

                if args.do_print:
                    dash_ascii = bool(args.print_ascii_dash)
                    prefix_width = int(args.print_prefix_width)

                    table_rows: List[Tuple[int, str, float, int, str, float, float]] = []
                    for i, (diff, found_jsd, base_mean_cnt, _base_sum, found_cnt, y, q1, q3) in enumerate(top, start=1):
                        pfx_str = f"{fmt_prefix(y)}->next"
                        iqr_str = _fmt_iqr(q1, q3, ascii_dash=dash_ascii)
                        table_rows.append((i, pfx_str, base_mean_cnt, found_cnt, iqr_str, found_jsd, diff))

                    print_table(
                        table_rows,
                        title=f"\n# {bucket} / FOUND={found_label} / n={n}  (top {args.topk})",
                        prefix_width=prefix_width,
                        ascii_dash=dash_ascii,
                        full_prefix=bool(args.print_full_prefix),
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
