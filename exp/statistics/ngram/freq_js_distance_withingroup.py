#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(セグメント対応) グループ内の個人同士JS距離の「平均」を n ごとに計算し、
All + 各グループの折れ線を 1枚(summary_mean.png) に重ねて出す。

- All:
    全看護師（個人）を対象に、全セグメント（= グループ不問）で n-gram 分布を作る
    -> 個人同士JS距離の平均
- Group g:
    g に属するセグメントだけで n-gram 分布を作る
    -> 個人同士JS距離の平均

使い方例:
  python exp/statistics/ngram/js_within_group_summary.py \
      exp/2019-2025-data/past-shifts/GCU.lp \
      exp/2019-2025-data/group-settings/GCU/ \
      --nmin 1 --nmax 10 --min-total 50 --alpha 1e-3 --topk 0 \
      --outdir out/js_within_group --include-unknown

出力:
  outdir/summary_mean.png
  outdir/summary_mean.csv
"""

import os
import sys
import math
import csv
import argparse
from collections import defaultdict, Counter
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
# settings
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH"
}

PersonKey = Tuple[int, str]   # (nurse_id, name)
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_group_dirname(g: str) -> str:
    return g.replace("/", "_").replace("\\", "_").strip()


def filter_seqs_by_date(seqs: SeqDict, date_start: Optional[int], date_end: Optional[int]) -> SeqDict:
    if date_start is None and date_end is None:
        return seqs
    out: SeqDict = {}
    for k, seq in seqs.items():
        sub = []
        for d, s in seq:
            if date_start is not None and d < date_start:
                continue
            if date_end is not None and d > date_end:
                continue
            sub.append((d, s))
        if sub:
            sub.sort(key=lambda t: t[0])
            out[k] = sub
    return out


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def norm(s: str) -> str:
    return str(s).lower()


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
    __slots__ = ("groups", "seq", "start", "end")
    def __init__(self, groups: FrozenSet[str]):
        self.groups: FrozenSet[str] = groups
        self.seq: List[Tuple[int, str]] = []
        self.start: Optional[int] = None
        self.end: Optional[int] = None


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
    *,
    include_unknown: bool,
) -> List[Segment]:
    """
    その人の (date,shift) を、所属グループ集合が変わるたびに分割する。

    - include_unknown=False: groupが取れない日はスキップ（従来挙動）
    - include_unknown=True : groupが取れない日は groups={"Unknown"} として扱う
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
            if not include_unknown:
                continue
            gs = {"Unknown"}

        gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            cur.start = d
            segs.append(cur)

        cur.seq.append((d, s))
        cur.end = d

    return segs


def prebuild_all_segments(seqs: SeqDict, timeline: dict, *, include_unknown: bool) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline, include_unknown=include_unknown)
    return out


def discover_groups_from_segments(
    segs_by_person: Dict[PersonKey, List[Segment]],
    *,
    include_unknown: bool
) -> List[str]:
    groups: Set[str] = set()
    for segs in segs_by_person.values():
        for seg in segs:
            groups.update(seg.groups)

    # ★ここがポイント：All は捨てずに、自前で先頭に追加する（Allは後で特別扱い）
    groups.discard("All")  # 設定内に "All" があっても、重複や誤解を避けるため一旦除外
    if not include_unknown:
        groups.discard("Unknown")

    out = sorted(groups, key=lambda x: x.lower())
    return out


def count_ngrams_person_all_segmented(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
) -> Dict[PersonKey, Counter]:
    """All用：グループ条件なしで全セグメントを数える"""
    assert n >= 1
    out: Dict[PersonKey, Counter] = defaultdict(Counter)
    for pk, segs in segs_by_person.items():
        for seg in segs:
            sseq = seg.seq
            if len(sseq) < n:
                continue
            for i in range(len(sseq) - n + 1):
                gram = tuple(sseq[i + k][1] for k in range(n))
                out[pk][gram] += 1
    return out


def count_ngrams_person_group_segmented(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    target_group: str,
) -> Dict[PersonKey, Counter]:
    """グループg用：gに属するセグメントだけ数える"""
    assert n >= 1
    out: Dict[PersonKey, Counter] = defaultdict(Counter)
    for pk, segs in segs_by_person.items():
        for seg in segs:
            if not group_set_contains(seg.groups, target_group):
                continue
            sseq = seg.seq
            if len(sseq) < n:
                continue
            for i in range(len(sseq) - n + 1):
                gram = tuple(sseq[i + k][1] for k in range(n))
                out[pk][gram] += 1
    return out


def build_vocab(person_counts: Dict[PersonKey, Counter], topk: int = 0) -> List[Tuple[str, ...]]:
    total = Counter()
    for c in person_counts.values():
        total.update(c)
    if topk and topk > 0:
        grams = [g for (g, _) in total.most_common(topk)]
    else:
        grams = list(total.keys())
    grams.sort()
    return grams


def to_prob_vector(counter: Counter, vocab: List[Tuple[str, ...]], alpha: float) -> List[float]:
    total = 0.0
    for g in vocab:
        total += counter.get(g, 0)
    denom = total + alpha * len(vocab)
    if denom <= 0:
        return [1.0 / len(vocab)] * len(vocab)
    return [(counter.get(g, 0) + alpha) / denom for g in vocab]


def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        # smoothing前提なので qi=0 はほぼ出ないが、安全のため
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi, 2)
    return s


def js_distance(p: List[float], q: List[float]) -> float:
    """JS distance = sqrt(JSD). log base2 のとき理論上 [0,1]。"""
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    if jsd < 0:
        jsd = 0.0
    return math.sqrt(jsd)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def flatten_pairwise(dist_mat: List[List[float]]) -> List[float]:
    xs = []
    for i in range(len(dist_mat)):
        for j in range(i + 1, len(dist_mat)):
            xs.append(dist_mat[i][j])
    return xs


def compute_mean_pairwise_for_group(
    segs_by_person: Dict[PersonKey, List[Segment]],
    *,
    group: str,
    n: int,
    alpha: float,
    topk: int,
    min_total: int,
) -> Tuple[Optional[float], int, int]:
    """
    戻り値: (mean_pairwise or None, n_person_used, vocab_size)
    """
    if group.lower() == "all":
        person_counts = count_ngrams_person_all_segmented(segs_by_person, n)
    else:
        person_counts = count_ngrams_person_group_segmented(segs_by_person, n, group)

    persons: List[PersonKey] = []
    filtered: Dict[PersonKey, Counter] = {}
    for pk, c in person_counts.items():
        if sum(c.values()) >= min_total:
            persons.append(pk)
            filtered[pk] = c

    if len(persons) < 2:
        return None, len(persons), 0

    vocab = build_vocab(filtered, topk=topk)
    if len(vocab) == 0:
        return None, len(persons), 0

    vecs = [to_prob_vector(filtered[pk], vocab, alpha=alpha) for pk in persons]

    dist_mat = [[0.0] * len(persons) for _ in range(len(persons))]
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            d = js_distance(vecs[i], vecs[j])
            dist_mat[i][j] = d
            dist_mat[j][i] = d

    pairwise = flatten_pairwise(dist_mat)
    if not pairwise:
        return None, len(persons), len(vocab)

    return mean(pairwise), len(persons), len(vocab)


def save_summary_mean_plot(path: str, ns: List[int], series: Dict[str, List[Optional[float]]]) -> None:
    plt.figure(figsize=(10, 5.5))

    for g, ys in series.items():
        # None は欠損として飛ばしつつ、線が途切れる形で描く
        xs_plot, ys_plot = [], []
        for x, y in zip(ns, ys):
            if y is None or (isinstance(y, float) and math.isnan(y)):
                xs_plot.append(x)
                ys_plot.append(float("nan"))
            else:
                xs_plot.append(x)
                ys_plot.append(y)
        plt.plot(xs_plot, ys_plot, marker="o", label=g)

    plt.xticks(ns)
    plt.xlabel("n (n-gram length)")
    plt.ylabel("Mean pairwise JS distance (within group)")
    plt.title("Within-group mean JS distance (All + each group)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="例: .../past-shifts/GCU.lp")
    ap.add_argument("group_settings", help="例: .../group-settings/GCU/")
    ap.add_argument("--group", action="append",
                    help="対象グループ名（省略時は自動で全グループ）。複数指定可。All も指定可。")
    ap.add_argument("--include-unknown", action="store_true",
                    help="groupが取れない日を Unknown として扱う（AllにもUnknown日が入る）")
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--date-start", type=int, default=None)
    ap.add_argument("--date-end", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=0)
    ap.add_argument("--outdir", default="out/js_within_group")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    seqs = filter_seqs_by_date(seqs, args.date_start, args.date_end)

    segs_by_person = prebuild_all_segments(seqs, timeline, include_unknown=args.include_unknown)

    # groups to run
    if args.group and len(args.group) > 0:
        groups = args.group[:]
        # ★All を先頭に持ってくる（指定されていれば）
        # 何回も入るのを防ぐ
        normed = []
        for g in groups:
            if g is None:
                continue
            if g.lower() == "all":
                continue
            normed.append(g)
        if any(g.lower() == "all" for g in groups):
            groups_to_run = ["All"] + normed
        else:
            groups_to_run = normed
    else:
        discovered = discover_groups_from_segments(segs_by_person, include_unknown=args.include_unknown)
        groups_to_run = ["All"] + discovered

    ns = list(range(args.nmin, args.nmax + 1))

    # series[group] = [mean(nmin), mean(nmin+1), ...]
    series: Dict[str, List[Optional[float]]] = {g: [] for g in groups_to_run}

    # csv
    summary_csv = os.path.join(args.outdir, "summary_mean.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["group", "n", "mean_pairwise", "n_person_used", "vocab_size",
                    "alpha", "topk", "min_total", "date_start", "date_end", "include_unknown"])

        for n in ns:
            print(f"# n={n}")
            for g in groups_to_run:
                m, n_person, vocab_size = compute_mean_pairwise_for_group(
                    segs_by_person,
                    group=g, n=n,
                    alpha=args.alpha, topk=args.topk, min_total=args.min_total
                )
                series[g].append(m)
                w.writerow([
                    g, n,
                    "" if m is None or (isinstance(m, float) and math.isnan(m)) else f"{m:.10f}",
                    n_person, vocab_size,
                    args.alpha, args.topk, args.min_total,
                    args.date_start, args.date_end, int(args.include_unknown)
                ])

                if m is None:
                    print(f"  - {g}: skipped (persons={n_person}, vocab={vocab_size})")
                else:
                    print(f"  - {g}: mean={m:.6f} (persons={n_person}, vocab={vocab_size})")

    out_png = os.path.join(args.outdir, "summary_mean.png")
    save_summary_mean_plot(out_png, ns, series)

    print(f"# summary csv  -> {summary_csv}")
    print(f"# summary plot -> {out_png}")


if __name__ == "__main__":
    main()
