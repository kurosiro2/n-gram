#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
distance のヒートマップ版（range A vs range B）

縦軸:  prefix（例: "SE,SN" / n=1なら "(EMPTY)"）
横軸:  smoothing k（例: 1, 1e-3, 1e-9, 0）

各セル:
  JSdist( sqrt(JSD) ) on P(next|prefix)  （ln）
  ※prefixごとに10次元分布 P(next|prefix) を作り、その分布同士でJSD。

重要:
- k=0 は「スムージング無し（MLE）」として扱う（Ny=0 のときは一様）
- 内部の count は timeline-aware segments と同じ
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


# =============================================================
# segments (timeline aware)
# =============================================================
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


# =============================================================
# counts for P(next|prefix)
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
                pfx = tuple(shifts[i : i + pref_len])
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# =============================================================
# probabilities + JSD
# =============================================================
def pnext_vector(cond: Dict[Prefix, Counter], prefN: Counter, prefix: Prefix, k: float) -> List[float]:
    """
    k>0: Laplace(k)
    k=0: MLE（Ny=0 のときは一様）
    """
    Ny = float(prefN.get(prefix, 0))
    cxy = cond.get(prefix, Counter())

    if k <= 0.0:
        if Ny <= 0.0:
            return [1.0 / X_SIZE] * X_SIZE
        return [cxy.get(x, 0) / Ny for x in VALID_SHIFTS]

    denom = Ny + k * X_SIZE
    return [(cxy.get(x, 0) + k) / denom for x in VALID_SHIFTS]


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
# heatmap plot
# =============================================================
def plot_heatmap(out_png: str, title: str, row_labels: List[str], col_labels: List[str], mat: List[List[float]]) -> None:
    R = len(row_labels)
    C = len(col_labels)

    fig_w = max(10.0, C * 2.0)
    fig_h = max(8.0, R * 0.40)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=0.0, vmax=math.sqrt(math.log(2.0)), aspect="auto")
    plt.colorbar(im)

    plt.xticks(list(range(C)), col_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), row_labels)

    mid = (0.0 + math.sqrt(math.log(2.0))) / 2.0
    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =============================================================
# CLI parsing utils
# =============================================================
def parse_k_list(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--k-list is empty")
    ks: List[float] = []
    for p in parts:
        try:
            ks.append(float(p))
        except ValueError:
            raise ValueError(f"invalid k in --k-list: {p}")
    return ks


def fmt_k(k: float) -> str:
    if k == 0.0:
        return "k=0"
    if 0 < abs(k) < 1e-2:
        return f"k={k:.0e}"
    return f"k={k:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts *.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")

    ap.add_argument("--n", type=int, required=True, help="ngram length n (prefix length n-1). n=1 allowed.")
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--bucket", choices=["Heads", "NonHeads"], default="NonHeads")

    ap.add_argument("--A-start", type=int, required=True, help="range A start yyyymmdd")
    ap.add_argument("--A-end", type=int, required=True, help="range A end   yyyymmdd")
    ap.add_argument("--B-start", type=int, required=True, help="range B start yyyymmdd")
    ap.add_argument("--B-end", type=int, required=True, help="range B end   yyyymmdd")

    ap.add_argument("--k-list", default="1,1e-3,1e-9,0", help='comma list, e.g. "1,1e-3,1e-9,0"')

    # ★変更: prefix（行）は prefix 単位なので top-prefixes にする
    ap.add_argument("--top-prefixes", type=int, default=120,
                    help="how many prefixes to show (by total prefix count in A+B). n=1 -> always 1 prefix.")
    ap.add_argument("--min-prefix-count", type=int, default=1,
                    help="drop prefixes whose total count (A+B) is less than this")

    ap.add_argument("--out", default="out/pnext_prefix_x_k_heatmap.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.n < 1:
        raise ValueError("--n must be >= 1")
    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    ensure_dir(os.path.dirname(args.out) or ".")

    ks = parse_k_list(args.k_list)
    col_labels = [fmt_k(k) for k in ks]

    # load + segments
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # count for A and B
    hA_cond, hA_prefN, nA_cond, nA_prefN = count_conditional_heads_nonheads_in_range(
        segs_by_person, n=args.n, heads_name=args.heads_name, date_start=args.A_start, date_end=args.A_end
    )
    hB_cond, hB_prefN, nB_cond, nB_prefN = count_conditional_heads_nonheads_in_range(
        segs_by_person, n=args.n, heads_name=args.heads_name, date_start=args.B_start, date_end=args.B_end
    )

    if args.bucket == "Heads":
        condA, prefNA = hA_cond, hA_prefN
        condB, prefNB = hB_cond, hB_prefN
    else:
        condA, prefNA = nA_cond, nA_prefN
        condB, prefNB = nB_cond, nB_prefN

    # ---------------------------------------------------------
    # build row candidates: prefix only
    # score = cA(y)+cB(y) で上位を取る
    # ---------------------------------------------------------
    prefixes: List[Tuple[Prefix, int, int, int]] = []  # (prefix, total, cA, cB)

    if args.n == 1:
        pfx = tuple()
        cA = int(prefNA.get(pfx, 0))
        cB = int(prefNB.get(pfx, 0))
        total = cA + cB
        if total >= args.min_prefix_count:
            prefixes.append((pfx, total, cA, cB))
    else:
        all_prefixes = set(prefNA.keys()) | set(prefNB.keys()) | set(condA.keys()) | set(condB.keys())
        for pfx in all_prefixes:
            cA = int(prefNA.get(pfx, 0))
            cB = int(prefNB.get(pfx, 0))
            total = cA + cB
            if total >= args.min_prefix_count:
                prefixes.append((pfx, total, cA, cB))

    if not prefixes:
        raise RuntimeError("No prefixes found. Try lowering --min-prefix-count or check ranges/bucket/n.")

    prefixes.sort(key=lambda t: t[1], reverse=True)
    if args.n != 1:
        prefixes = prefixes[: max(1, args.top_prefixes)]

    row_labels = [f"{fmt_prefix(pfx)}->next  (tot={tot})" for (pfx, tot, cA, cB) in prefixes]

    # ---------------------------------------------------------
    # compute matrix: per prefix, for each k
    # ---------------------------------------------------------
    mat: List[List[float]] = []

    for (pfx, _tot, _cA, _cB) in prefixes:
        row_vals: List[float] = []
        for k in ks:
            pA = pnext_vector(condA, prefNA, pfx, k)
            pB = pnext_vector(condB, prefNB, pfx, k)
            row_vals.append(js_distance_vec(pA, pB))
        mat.append(row_vals)

    if args.title is not None:
        title = args.title
    else:
        title = (
            f"{args.bucket}: JSdist on P(next|prefix) per-prefix  "
            f"A={args.A_start}..{args.A_end}  B={args.B_start}..{args.B_end}  n={args.n}"
        )

    plot_heatmap(args.out, title, row_labels, col_labels, mat)
    print(f"# wrote: {args.out}")


if __name__ == "__main__":
    main()
