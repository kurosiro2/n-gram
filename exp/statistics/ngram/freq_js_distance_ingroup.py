#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import csv
import argparse
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Iterable, Optional, Set, Any, FrozenSet

import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date

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


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
) -> List[Segment]:
    """
    その人の (date,shift) を、所属グループ集合が変わるたびに分割する。
    groupが取れない日はスキップ（Unknownを含めたいならここを変える）
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
            continue

        gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            cur.start = d
            segs.append(cur)

        cur.seq.append((d, s))
        cur.end = d

    return segs


def prebuild_all_segments(seqs: SeqDict, timeline: dict) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline)
    return out


def discover_groups_from_segments(
    segs_by_person: Dict[PersonKey, List[Segment]],
    include_unknown: bool = False
) -> List[str]:
    groups: Set[str] = set()
    for segs in segs_by_person.values():
        for seg in segs:
            groups.update(seg.groups)

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
        s += pi * math.log(pi / qi)
    return s


def js_distance(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return math.sqrt(js)


def mean_distribution(vecs: List[List[float]]) -> List[float]:
    m = [0.0] * len(vecs[0])
    for v in vecs:
        for i, x in enumerate(v):
            m[i] += x
    m = [x / len(vecs) for x in m]
    z = sum(m)
    if z <= 0:
        return [1.0 / len(m)] * len(m)
    return [x / z for x in m]


def flatten_pairwise(dist_mat: List[List[float]]) -> List[float]:
    xs = []
    for i in range(len(dist_mat)):
        for j in range(i + 1, len(dist_mat)):
            xs.append(dist_mat[i][j])
    return xs


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    m = len(ys)
    if m % 2 == 1:
        return ys[m // 2]
    return 0.5 * (ys[m // 2 - 1] + ys[m // 2])


def write_pairwise_csv(path: str, persons: List[PersonKey], dist_mat: List[List[float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["nurse_id_i", "name_i", "nurse_id_j", "name_j", "js_distance"])
        for i, (nid_i, name_i) in enumerate(persons):
            for j in range(i + 1, len(persons)):
                nid_j, name_j = persons[j]
                w.writerow([nid_i, name_i, nid_j, name_j, f"{dist_mat[i][j]:.10f}"])


def write_to_centroid_csv(path: str, persons: List[PersonKey], dists: List[float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["nurse_id", "name", "js_to_centroid"])
        for (nid, name), d in sorted(zip(persons, dists), key=lambda t: t[1]):
            w.writerow([nid, name, f"{d:.10f}"])


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

    # ★ 数値を重ねる
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


# =============================================================
# DEBUG: group changes
# =============================================================
def norm(s: str) -> str:
    return str(s).lower()


def format_group_set(gs: FrozenSet[str]) -> str:
    return "|".join(sorted(gs, key=lambda x: x.lower()))


def detect_group_changes(
    segs_by_person: Dict[PersonKey, List[Segment]],
) -> List[Dict[str, Any]]:
    """
    変化点を列挙:
      - person ごとに segment が 2つ以上ある => グループ変化あり
      - segment 境界で (date, from_groups, to_groups) を記録
    """
    rows: List[Dict[str, Any]] = []
    for (nid, name), segs in segs_by_person.items():
        if len(segs) <= 1:
            continue
        for i in range(1, len(segs)):
            prev = segs[i-1]
            cur = segs[i]
            rows.append({
                "nurse_id": nid,
                "name": name,
                "change_index": i,
                "from_start": prev.start,
                "from_end": prev.end,
                "from_groups": format_group_set(prev.groups),
                "to_start": cur.start,
                "to_end": cur.end,
                "to_groups": format_group_set(cur.groups),
            })
    return rows


def debug_print_changes(
    segs_by_person: Dict[PersonKey, List[Segment]],
    *,
    name_filters: List[str],
    min_changes: int,
    limit_people: int,
) -> None:
    # person -> number of changes
    people = []
    for pk, segs in segs_by_person.items():
        if len(segs) <= 1:
            continue
        nid, name = pk
        if name_filters:
            nm = norm(name)
            if not any(f in nm for f in name_filters):
                continue
        changes = len(segs) - 1
        if changes >= min_changes:
            people.append((changes, pk))

    people.sort(key=lambda t: (-t[0], norm(t[1][1]), t[1][0]))

    if limit_people and limit_people > 0:
        people = people[:limit_people]

    print("# ---- DEBUG: group changes (segment boundaries) ----")
    print(f"# people={len(people)}  min_changes={min_changes}  name_filters={name_filters if name_filters else 'None'}")

    for changes, (nid, name) in people:
        segs = segs_by_person[(nid, name)]
        print(f"{nid}\t{name}\tchanges={changes}\tsegments={len(segs)}")
        for i, seg in enumerate(segs):
            print(f"  seg{i}: {seg.start}..{seg.end}  groups={format_group_set(seg.groups)}")


def write_changes_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "nurse_id", "name", "change_index",
            "from_start", "from_end", "from_groups",
            "to_start", "to_end", "to_groups"
        ])
        for r in rows:
            w.writerow([
                r["nurse_id"], r["name"], r["change_index"],
                r["from_start"], r["from_end"], r["from_groups"],
                r["to_start"], r["to_end"], r["to_groups"]
            ])
    print(f"# debug changes csv -> {path}")


# =============================================================
# Representatives (optional)
# =============================================================
def find_persons_by_name_query(seqs: SeqDict, query: str) -> List[PersonKey]:
    q = norm(query)
    cands = []
    for (nid, name) in seqs.keys():
        if q in norm(name):
            cands.append((nid, name))
    cands.sort(key=lambda x: (norm(x[1]), x[0]))
    return cands


def groups_seen_from_segments(
    segs_by_person: Dict[PersonKey, List[Segment]],
    person: PersonKey
) -> List[str]:
    groups: Set[str] = set()
    for seg in segs_by_person.get(person, []):
        groups.update(seg.groups)
    return sorted(groups, key=lambda x: x.lower())


def count_ngrams_for_specific_person_segmented(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    person: PersonKey
) -> Counter:
    out = Counter()
    for seg in segs_by_person.get(person, []):
        sseq = seg.seq
        if len(sseq) < n:
            continue
        for i in range(len(sseq) - n + 1):
            gram = tuple(sseq[i + k][1] for k in range(n))
            out[gram] += 1
    return out


def compute_representatives_heatmaps(
    segs_by_person: Dict[PersonKey, List[Segment]],
    representatives: List[PersonKey],
    *,
    nmin: int,
    nmax: int,
    outdir: str,
    alpha: float,
    topk: int,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    reps_dir = os.path.join(outdir, "representatives")
    ensure_dir(reps_dir)

    rep_csv = os.path.join(outdir, "representatives.csv")
    with open(rep_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["nurse_id", "name", "groups_seen"])
        for pk in representatives:
            gs = groups_seen_from_segments(segs_by_person, pk)
            w.writerow([pk[0], pk[1], "|".join(gs)])
    print(f"# representatives -> {rep_csv}")

    labels = [pk[1] for pk in representatives]

    for n in range(nmin, nmax + 1):
        counts_by_rep: Dict[PersonKey, Counter] = {}
        for pk in representatives:
            counts_by_rep[pk] = count_ngrams_for_specific_person_segmented(segs_by_person, n, pk)

        vocab = build_vocab(counts_by_rep, topk=topk)
        vecs = [to_prob_vector(counts_by_rep[pk], vocab, alpha=alpha) for pk in representatives]

        dist_mat = [[0.0] * len(representatives) for _ in range(len(representatives))]
        for i in range(len(representatives)):
            for j in range(i + 1, len(representatives)):
                d = js_distance(vecs[i], vecs[j])
                dist_mat[i][j] = d
                dist_mat[j][i] = d

        png = os.path.join(reps_dir, f"heatmap_reps_js_n{n}.png")
        title = f"Representatives (segmented) JS distance n={n}"
        save_heatmap(png, labels, dist_mat, title, vmin=vmin, vmax=vmax)
        print(f"# reps heatmap -> {png}")


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="例: .../past-shifts/GCU.lp")
    ap.add_argument("group_settings", help="例: .../group-settings/GCU/")
    ap.add_argument("--group", action="append",
                    help="対象グループ名（省略時は自動で全グループ）。複数指定可。")
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--date-start", type=int, default=None)
    ap.add_argument("--date-end", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=0)
    ap.add_argument("--outdir", default="out/js_distance_segment")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--heatmap-scale", choices=["global", "per_image"], default="global")
    ap.add_argument("--include-unknown", action="store_true")

    # reps
    ap.add_argument("--rep", action="append",
                    help="代表者の名前（部分一致可）。複数指定可。例: --rep \"山田\"")
    ap.add_argument("--rep-heatmap", action="store_true",
                    help="指定代表者同士のヒートマップを n=1..5 で出す")

    # DEBUG changes
    ap.add_argument("--debug-changes", action="store_true",
                    help="途中でグループが変わる人（segmentが2つ以上）を表示")
    ap.add_argument("--debug-changes-csv", default=None,
                    help="変化点をCSVに保存（例: out/changes.csv）")
    ap.add_argument("--debug-min-changes", type=int, default=1,
                    help="この回数以上変化した人だけ表示（デフォルト1）")
    ap.add_argument("--debug-name", action="append", default=[],
                    help="デバッグ対象を名前で絞る（部分一致）。複数指定可。")
    ap.add_argument("--debug-limit", type=int, default=50,
                    help="デバッグ表示する人数上限（0で無制限）")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    src = PastShiftSource(args.past_shifts, args.group_settings)
    seqs, timeline = src.load()
    seqs = filter_seqs_by_date(seqs, args.date_start, args.date_end)

    # segments
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # DEBUG: show group changes
    if args.debug_changes:
        name_filters = [norm(x) for x in args.debug_name if x]
        debug_print_changes(
            segs_by_person,
            name_filters=name_filters,
            min_changes=args.debug_min_changes,
            limit_people=args.debug_limit,
        )

    if args.debug_changes_csv:
        rows = detect_group_changes(segs_by_person)
        # optional filter by name
        if args.debug_name:
            fs = [norm(x) for x in args.debug_name if x]
            rows = [r for r in rows if any(f in norm(r["name"]) for f in fs)]
        # optional filter by min_changes (people-level)
        if args.debug_min_changes > 1:
            # count per person
            cnt = defaultdict(int)
            for r in rows:
                cnt[(r["nurse_id"], r["name"])] += 1
            rows = [r for r in rows if cnt[(r["nurse_id"], r["name"])] >= args.debug_min_changes]

        write_changes_csv(args.debug_changes_csv, rows)

    # groups to run
    if args.group and len(args.group) > 0:
        groups_to_run = args.group
        print(f"# groups (manual): {groups_to_run}")
    else:
        groups_to_run = discover_groups_from_segments(segs_by_person, include_unknown=args.include_unknown)
        print(f"# groups (auto): {groups_to_run}")

    # two-pass for global min/max
    results: List[Dict[str, Any]] = []
    global_min = float("inf")
    global_max = float("-inf")

    print("# pass1: compute all distance matrices (segmented) ...")
    for g in groups_to_run:
        for n in range(args.nmin, args.nmax + 1):
            person_counts = count_ngrams_person_group_segmented(segs_by_person, n, g)

            persons: List[PersonKey] = []
            filtered_counts: Dict[PersonKey, Counter] = {}
            for pk, c in person_counts.items():
                if sum(c.values()) >= args.min_total:
                    persons.append(pk)
                    filtered_counts[pk] = c

            if len(persons) < 2:
                print(f"[group={g} n={n}] skipped (persons={len(persons)})")
                continue

            vocab = build_vocab(filtered_counts, topk=args.topk)
            vecs = [to_prob_vector(filtered_counts[pk], vocab, alpha=args.alpha) for pk in persons]

            dist_mat = [[0.0] * len(persons) for _ in range(len(persons))]
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    d = js_distance(vecs[i], vecs[j])
                    dist_mat[i][j] = d
                    dist_mat[j][i] = d

            pairwise = flatten_pairwise(dist_mat)
            if pairwise:
                global_min = min(global_min, min(pairwise))
                global_max = max(global_max, max(pairwise))

            centroid = mean_distribution(vecs)
            to_centroid = [js_distance(v, centroid) for v in vecs]

            results.append({
                "group": g, "n": n,
                "persons": persons,
                "dist_mat": dist_mat,
                "to_centroid": to_centroid,
                "vocab_size": len(vocab),
                "pairwise": pairwise,
            })

            print(f"[group={g} n={n}] persons={len(persons)} vocab={len(vocab)}")

    if not results:
        print("# No results produced.")
        return

    if args.heatmap and args.heatmap_scale == "global":
        print(f"# heatmap global scale: vmin={global_min:.6f}, vmax={global_max:.6f}")

    summary_path = os.path.join(args.outdir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fp_sum:
        wsum = csv.writer(fp_sum)
        wsum.writerow([
            "group", "n",
            "n_person",
            "vocab_size",
            "mean_pairwise", "median_pairwise", "max_pairwise",
            "mean_to_centroid", "median_to_centroid", "max_to_centroid"
        ])

        print("# pass2: write outputs ...")
        for r in results:
            g = r["group"]
            n = r["n"]
            persons = r["persons"]
            dist_mat = r["dist_mat"]
            to_centroid = r["to_centroid"]
            pairwise = r["pairwise"]
            vocab_size = r["vocab_size"]

            g_dir = os.path.join(args.outdir, safe_group_dirname(g))
            ensure_dir(g_dir)

            write_pairwise_csv(os.path.join(g_dir, f"pairwise_js_n{n}.csv"), persons, dist_mat)
            write_to_centroid_csv(os.path.join(g_dir, f"to_centroid_js_n{n}.csv"), persons, to_centroid)

            wsum.writerow([
                g, n, len(persons), vocab_size,
                f"{mean(pairwise):.10f}", f"{median(pairwise):.10f}", f"{max(pairwise):.10f}",
                f"{mean(to_centroid):.10f}", f"{median(to_centroid):.10f}", f"{max(to_centroid):.10f}",
            ])

            if args.heatmap:
                labels = [name for (_, name) in persons]
                png = os.path.join(g_dir, f"heatmap_js_n{n}.png")
                title = f"JS distance (segmented) (group={g}, n={n})"
                if args.heatmap_scale == "global":
                    save_heatmap(png, labels, dist_mat, title, vmin=global_min, vmax=global_max)
                else:
                    save_heatmap(png, labels, dist_mat, title)

    print(f"# summary -> {summary_path}")

    # reps
    if args.rep and args.rep_heatmap:
        reps: List[PersonKey] = []
        used: Set[PersonKey] = set()

        for q in args.rep:
            cands = find_persons_by_name_query(seqs, q)
            if not cands:
                print(f"[rep] not found: query='{q}'")
                continue

            if len(cands) == 1:
                pk = cands[0]
            else:
                print(f"[rep] query='{q}' matched {len(cands)}; selecting first: {cands[0]}")
                pk = cands[0]

            if pk not in used:
                reps.append(pk)
                used.add(pk)

        if len(reps) < 2:
            print("# rep-heatmap: need at least 2 representatives.")
            return

        rep_vmin = global_min if (args.heatmap_scale == "global") else None
        rep_vmax = global_max if (args.heatmap_scale == "global") else None

        compute_representatives_heatmaps(
            segs_by_person, reps,
            nmin=args.nmin, nmax=args.nmax,
            outdir=args.outdir,
            alpha=args.alpha,
            topk=args.topk,
            vmin=rep_vmin, vmax=rep_vmax,
        )


if __name__ == "__main__":
    main()
