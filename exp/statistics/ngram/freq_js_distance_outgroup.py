#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(セグメント対応) グループごとの「平均」n-gram確率分布を作り、
グループ間の JS distance (= sqrt(JSD)) を計算してプロットする。

追加:
  - outdir/summary_mean.png
    => 各グループ g について
         mean_to_others(g, n) = 平均_{h!=g} JSdist(centroid_g, centroid_h)
       を n ごとに折れ線で重ね描きする。

  - --include-all
    => 疑似グループ "All"（= グループ関係なく、groupが取れた日だけを全てごちゃまぜ集計）
       を labels の先頭に追加する。

使い方例:
  python exp/statistics/ngram/js_group_centroid.py \
      exp/2019-2025-data/past-shifts/GCU.lp \
      exp/2019-2025-data/group-settings/GCU/ \
      --nmin 1 --nmax 15 --min-total 50 --alpha 1e-3 \
      --outdir out/js_centroid \
      --include-all

デバッグ例（日ごとの group 出力）:
  python exp/statistics/ngram/js_group_centroid.py \
      exp/2019-2025-data/past-shifts/GCU.lp \
      exp/2019-2025-data/group-settings/GCU/ \
      --nmin 1 --nmax 1 --min-total 50 --alpha 1e-3 \
      --debug-person-name "name" \
      --debug-date-start 20241013 --debug-date-end 20241020 \
      --debug-print-daily
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

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date


# -----------------------------
# settings
# -----------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH"
}

PersonKey = Tuple[int, str]   # (nurse_id, name)
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


class PastShiftSource:
    def __init__(self, past_shifts_path: str, setting_path: str):
        self.past_shifts_path = past_shifts_path
        self.setting_path = setting_path

    def load(self) -> Tuple[SeqDict, dict]:
        seqs = data_loader.load_past_shifts(self.past_shifts_path)
        timeline = data_loader.load_staff_group_timeline(self.setting_path)
        return seqs, timeline


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


# -----------------------------
# Debug helper
# -----------------------------
def _debug_should_print_daily(
    debug_print_daily: bool,
    debug_person_name: Optional[str],
    debug_person_id: Optional[int],
    name: str,
    nid: int,
    d: int,
    debug_date_start: Optional[int],
    debug_date_end: Optional[int],
) -> bool:
    if not debug_print_daily:
        return False
    if debug_person_name is not None and name != debug_person_name:
        return False
    if debug_person_id is not None and nid != debug_person_id:
        return False
    if debug_date_start is not None and d < debug_date_start:
        return False
    if debug_date_end is not None and d > debug_date_end:
        return False
    return True


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
    # debug params
    debug_print_daily: bool = False,
    debug_person_name: Optional[str] = None,
    debug_person_id: Optional[int] = None,
    debug_date_start: Optional[int] = None,
    debug_date_end: Optional[int] = None,
    debug_print_skip_summary: bool = False,
) -> List[Segment]:
    """
    その人の (date, shift) を、所属グループ集合が変わるたびに分割する。
    groupが取れない日はスキップ（Unknownを含めたいなら get_groups_for_day 側を調整）

    追加:
      --debug-print-daily を有効にすると、日ごとに
        name  YYYYMMDD  group(s) or None
      を出す。
    """
    seq = normalize_seq(seq)
    if not seq:
        return []
    seq.sort(key=lambda t: t[0])

    segs: List[Segment] = []
    cur: Optional[Segment] = None

    skipped_days = 0
    used_days = 0

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)

        # --- debug print (daily) ---
        if _debug_should_print_daily(
            debug_print_daily, debug_person_name, debug_person_id,
            name, nid, d, debug_date_start, debug_date_end
        ):
            gstr = "None" if (not gs) else ",".join(sorted(gs, key=lambda x: x.lower()))
            print(f"{name}\t{d}\t{gstr}", flush=True)

        if not gs:
            skipped_days += 1
            continue

        used_days += 1
        gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            cur.start = d
            segs.append(cur)

        cur.seq.append((d, s))
        cur.end = d

    # --- optional summary per person (only when debug target matches) ---
    if debug_print_skip_summary:
        ok_person = True
        if debug_person_name is not None and name != debug_person_name:
            ok_person = False
        if debug_person_id is not None and nid != debug_person_id:
            ok_person = False
        if ok_person:
            total = skipped_days + used_days
            print(
                f"# [debug] person={name}({nid}) total_days={total} used={used_days} "
                f"skipped(no-group)={skipped_days}",
                flush=True
            )

    return segs


def prebuild_all_segments(
    seqs: SeqDict,
    timeline: dict,
    # debug args
    debug_print_daily: bool = False,
    debug_person_name: Optional[str] = None,
    debug_person_id: Optional[int] = None,
    debug_date_start: Optional[int] = None,
    debug_date_end: Optional[int] = None,
    debug_print_skip_summary: bool = False,
) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(
            seq, name, nid, timeline,
            debug_print_daily=debug_print_daily,
            debug_person_name=debug_person_name,
            debug_person_id=debug_person_id,
            debug_date_start=debug_date_start,
            debug_date_end=debug_date_end,
            debug_print_skip_summary=debug_print_skip_summary,
        )
    return out


def discover_groups_from_segments(
    segs_by_person: Dict[PersonKey, List[Segment]],
    include_unknown: bool = False
) -> List[str]:
    groups: Set[str] = set()
    for segs in segs_by_person.values():
        for seg in segs:
            groups.update(seg.groups)

    # ★ 注意: ここでは「設定上の All」を除外したままでOK。
    # --include-all で作る疑似All（ごちゃまぜ）は labels 側で追加する。
    groups.discard("All")
    if not include_unknown:
        groups.discard("Unknown")

    out = sorted(groups, key=lambda x: x.lower())
    print(f"# [discover_groups] groups={out}")
    return out


def count_ngrams_person_group_segmented(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    target_group: str,
) -> Dict[PersonKey, Counter]:
    """対象グループに属する区間(segment)だけから n-gram を数える（個人別にCounter）。"""
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


def count_ngrams_person_all_segmented(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
) -> Dict[PersonKey, Counter]:
    """
    疑似グループ "All"（全グループごちゃまぜ）として、全segmentから n-gram を数える（個人別Counter）。

    ※ 現仕様では「groupが取れない日は build_segments で捨てられている」ため、
       All も “groupが取れた日だけ” のごちゃまぜになる。
    """
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


def build_vocab_from_counters(counters: List[Counter]) -> List[Tuple[str, ...]]:
    """語彙 = 全カウンタの和集合（出ない n-gram も 0 扱いするために必須）"""
    total = Counter()
    for c in counters:
        total.update(c)
    grams = list(total.keys())
    grams.sort()
    return grams


def to_prob_vector(counter: Counter, vocab: List[Tuple[str, ...]], alpha: float) -> List[float]:
    """vocab 上で確率ベクトル化（alpha 平滑）。"""
    total = 0.0
    for g in vocab:
        total += counter.get(g, 0)
    denom = total + alpha * len(vocab)
    if denom <= 0:
        if len(vocab) == 0:
            return []
        return [1.0 / len(vocab)] * len(vocab)
    return [(counter.get(g, 0) + alpha) / denom for g in vocab]


def mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    out = [0.0] * d
    for v in vectors:
        for i in range(d):
            out[i] += v[i]
    inv = 1.0 / len(vectors)
    return [x * inv for x in out]


def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi, 2)
    return s


def js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_distance(p: List[float], q: List[float]) -> float:
    """JS distance = sqrt(JSD). log base2 のとき [0,1] に収まる。"""
    d = js_divergence(p, q)
    if d < 0:
        d = 0.0
    return math.sqrt(d)


def save_heatmap(
    path: str,
    labels: List[str],
    dist_mat: List[List[float]],
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    plt.figure(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.45)))
    plt.imshow(dist_mat, vmin=vmin, vmax=vmax)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.colorbar()
    plt.title(title)

    # 数値を重ねる
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            v = dist_mat[i][j]
            plt.text(
                j, i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if (vmin is not None and vmax is not None and v > (vmin + vmax) / 2) else "white"
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_summary_plot_mean_max(path: str, ns: List[int], means: List[float], maxs: List[float]) -> None:
    """従来の summary_mean_max.png"""
    plt.figure(figsize=(8, 4.5))
    plt.plot(ns, means, marker="o", label="mean pairwise JS distance")
    plt.plot(ns, maxs, marker="o", label="max pairwise JS distance")
    for x, y in zip(ns, means):
        plt.text(x, y, f"{y:.3f}", fontsize=9, ha="left", va="bottom")
    for x, y in zip(ns, maxs):
        plt.text(x, y, f"{y:.3f}", fontsize=9, ha="left", va="bottom")
    plt.xticks(ns)
    plt.xlabel("n (n-gram length)")
    plt.ylabel("JS distance")
    plt.title("Group centroid JS distance summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_summary_plot_mean_by_group(path: str, ns: List[int], labels: List[str], mean_by_group: Dict[str, List[float]]) -> None:
    """summary_mean.png（全グループの mean-to-others を重ね描き）"""
    plt.figure(figsize=(9, 5))
    for g in labels:
        ys = mean_by_group.get(g, [])
        if not ys:
            continue
        plt.plot(ns, ys, marker="o", label=g)

    plt.xticks(ns)
    plt.xlabel("n (n-gram length)")
    plt.ylabel("Mean JS distance to other groups")
    plt.title("Mean JS distance by group (centroid vs others)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pairwise_stats(dist_mat: List[List[float]]) -> Tuple[float, float]:
    """対角以外の上三角だけを集計して mean/max を返す。"""
    vals = []
    k = len(dist_mat)
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(dist_mat[i][j])
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    mx = max(vals)
    return mean, mx


def mean_to_others(dist_mat: List[List[float]], idx: int) -> float:
    """dist_mat[idx][j] (j!=idx) の平均。k<2 のとき 0.0"""
    k = len(dist_mat)
    if k <= 1:
        return 0.0
    s = 0.0
    c = 0
    for j in range(k):
        if j == idx:
            continue
        s += dist_mat[idx][j]
        c += 1
    return (s / c) if c else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="例: ./past-shifts/GCU.lp")
    ap.add_argument("group_settings", help="例: ./group-settings/GCU/")
    ap.add_argument("--group", action="append",
                    help="対象グループ名（省略時は自動で全グループ）。複数指定可。")
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--date-start", type=int, default=None)
    ap.add_argument("--date-end", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--min-total", type=int, default=20,
                    help="個人の総出現回数がこれ未満なら、その人は平均との差分布の計算から除外")
    ap.add_argument("--outdir", default="out/js_centroid")
    ap.add_argument("--include-unknown", action="store_true")

    # ★ NEW: pseudo All
    ap.add_argument("--include-all", action="store_true",
                    help="疑似グループ All（全グループごちゃまぜ）を追加して描画する")

    # ---- debug options ----
    ap.add_argument("--debug-print-daily", action="store_true",
                    help="日ごとの group を name<TAB>date<TAB>groups/None で出力")
    ap.add_argument("--debug-person-name", default=None,
                    help="デバッグ出力する person の name を指定（省略可）")
    ap.add_argument("--debug-person-id", type=int, default=None,
                    help="デバッグ出力する nurse_id を指定（省略可）")
    ap.add_argument("--debug-date-start", type=int, default=None,
                    help="デバッグ出力する日付範囲（start, YYYYMMDD）")
    ap.add_argument("--debug-date-end", type=int, default=None,
                    help="デバッグ出力する日付範囲（end, YYYYMMDD）")
    ap.add_argument("--debug-skip-summary", action="store_true",
                    help="person ごとの skipped(no-group) の集計を出す（person指定推奨）")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    src = PastShiftSource(args.past_shifts, args.group_settings)
    seqs, timeline = src.load()
    seqs = filter_seqs_by_date(seqs, args.date_start, args.date_end)

    # segments (with debug)
    segs_by_person = prebuild_all_segments(
        seqs, timeline,
        debug_print_daily=args.debug_print_daily,
        debug_person_name=args.debug_person_name,
        debug_person_id=args.debug_person_id,
        debug_date_start=args.debug_date_start,
        debug_date_end=args.debug_date_end,
        debug_print_skip_summary=args.debug_skip_summary,
    )

    # groups to run
    if args.group and len(args.group) > 0:
        groups_to_run = args.group
        print(f"# groups (manual): {groups_to_run}")
    else:
        groups_to_run = discover_groups_from_segments(
            segs_by_person, include_unknown=args.include_unknown
        )
        print(f"# groups (auto): {groups_to_run}")

    labels = groups_to_run[:]
    if args.include_all and "All" not in labels:
        labels = ["All"] + labels

    # ここで labels が確定（All/Leader/Midlevel/...）
    print(f"# labels: {labels}")

    ns: List[int] = []
    mean_list: List[float] = []
    max_list: List[float] = []

    # group別 mean（他グループ平均との差）
    mean_by_group: Dict[str, List[float]] = {g: [] for g in labels}

    # 全nのdist_matを溜めて global vmin/vmax を揃える
    all_dist_mats: List[Tuple[int, List[List[float]]]] = []
    global_vals: List[float] = []

    summary_csv = os.path.join(args.outdir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["n", "mean_pairwise_jsdist", "max_pairwise_jsdist", "groups", "alpha", "min_total"])

        for n in range(args.nmin, args.nmax + 1):
            print(f"# n={n} ...")

            # ---- pass: 各グループで「個人カウンタ」を集める（min_totalで足切り） ----
            group_person_counters: Dict[str, List[Counter]] = {}
            for g in labels:
                if g == "All" and args.include_all:
                    person_counts = count_ngrams_person_all_segmented(segs_by_person, n)
                else:
                    person_counts = count_ngrams_person_group_segmented(segs_by_person, n, g)

                kept: List[Counter] = []
                for _, c in person_counts.items():
                    if sum(c.values()) >= args.min_total:
                        kept.append(c)
                group_person_counters[g] = kept
                print(f"  - group={g}: persons_kept={len(kept)}")

            # 語彙 = 全グループ・全個人の和集合（“出ない n-gram も 0 扱い”を保証）
            all_counters: List[Counter] = []
            for g in labels:
                all_counters.extend(group_person_counters[g])
            vocab = build_vocab_from_counters(all_counters)
            print(f"  - vocab_size={len(vocab)}")

            # 個人確率ベクトル -> グループ平均ベクトル（重心）
            centroids: Dict[str, List[float]] = {}
            for g in labels:
                vecs = [to_prob_vector(c, vocab, alpha=args.alpha) for c in group_person_counters[g]]
                if vecs:
                    centroids[g] = mean_vector(vecs)
                else:
                    centroids[g] = ([1.0 / len(vocab)] * len(vocab)) if len(vocab) > 0 else []

            # ---- dist matrix (group centroid) ----
            k = len(labels)
            dist_mat = [[0.0] * k for _ in range(k)]
            for i in range(k):
                for j in range(i + 1, k):
                    d = js_distance(centroids[labels[i]], centroids[labels[j]])
                    dist_mat[i][j] = d
                    dist_mat[j][i] = d

            # stats (pairwise among all groups)
            mean_d, max_d = pairwise_stats(dist_mat)
            ns.append(n)
            mean_list.append(mean_d)
            max_list.append(max_d)
            w.writerow([n, f"{mean_d:.6f}", f"{max_d:.6f}", "|".join(labels), args.alpha, args.min_total])

            # groupごとの mean-to-others
            for gi, g in enumerate(labels):
                mean_by_group[g].append(mean_to_others(dist_mat, gi))

            # store for global scale
            all_dist_mats.append((n, dist_mat))
            for row in dist_mat:
                global_vals.extend(row)

    print(f"# summary -> {summary_csv}")

    # global vmin/vmax（全n共通）
    if not global_vals:
        print("# no distance values; skip heatmaps.")
        return
    global_min = min(global_vals)
    global_max = max(global_vals)
    print(f"# global color scale: vmin={global_min:.6f}, vmax={global_max:.6f}")

    # save heatmaps (same scale)
    for n, dist_mat in all_dist_mats:
        png = os.path.join(args.outdir, f"heatmap_js_centroid_n{n}.png")
        title = f"Group centroid JS distance (segmented) n={n}"
        save_heatmap(png, labels, dist_mat, title, vmin=global_min, vmax=global_max)
        print(f"  -> {png}")

    # summary plot (mean+max) + mean-by-group
    if ns:
        summary_png = os.path.join(args.outdir, "summary_mean_max.png")
        save_summary_plot_mean_max(summary_png, ns, mean_list, max_list)
        print(f"# summary plot -> {summary_png}")

        summary_mean_png = os.path.join(args.outdir, "summary_mean.png")
        save_summary_plot_mean_by_group(summary_mean_png, ns, labels, mean_by_group)
        print(f"# summary mean-by-group plot -> {summary_mean_png}")


if __name__ == "__main__":
    main()
