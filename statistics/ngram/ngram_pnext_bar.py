#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【事前評価】past-shifts（期間A vs 期間B）で
Heads vs NonHeads の 2値だけ、
「末尾シフトに関する条件付き確率 P(next|prefix)」の freq上位Kを
同一グラフ(2本バー)で比較してPNG出力（＋printで表も出す）

比較方法:
- 期間A/B それぞれで n-gram をカウントして (prefix->next) の freq を得る
- "比較軸" となるイベント集合を選ぶ:
    * --basis=A なら 期間A の freq 上位K を採用
    * --basis=B なら 期間B の freq 上位K を採用
    * --basis=UNION なら A上位K ∪ B上位K（最大2K、見づらいので注意）
- 採用したイベント (prefix->next) について
    P_A(next|prefix), P_B(next|prefix) を計算
    （片方に存在しない prefix は 0 扱いにできる: --missing-policy zero）
- 棒グラフは A/B を横並び表示

追加（デバッグ/探索）:
  --list-nurses
      past-shifts 内の (nurse_id, name) 一覧を TSV で出して終了（id\tname）
  --find-id 74398
      id の文字列表現に '74398' を含む nurse_id を検索して表示（最大200件）
  --find-name 山田
      name に '山田' を含むものを検索して表示（最大200件）

使い方（単一病棟）:
  python ngram_pnext_compare_two_periods.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --n 5 --topk 20 \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --basis A \
    --heads-name Heads \
    --outdir out/pnext_compare

使い方（全病棟: past-shifts-dir × group-settings-root）:
  python ngram_pnext_compare_two_periods.py \
    /workspace/2025/past-shifts \
    /workspace/2025/group-settings \
    --n 5 --topk 20 \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --basis A \
    --outdir out/pnext_compare_all

※ group-settings は「病棟ディレクトリ（例: .../GCU/）」または root を渡す想定
"""

import os
import sys
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# NEW: global font sizes (bigger text everywhere in plots)
# -------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.titlesize": 16,     # title
    "axes.labelsize": 15,     # x/y label
    "xtick.labelsize": 12,    # x ticks
    "ytick.labelsize": 12,    # y ticks
    "legend.fontsize": 12,    # legend
    "figure.titlesize": 16,   # figure title
})
# もし日本語が□になる場合は、環境にあるフォント名に合わせて指定（例）
# plt.rcParams["font.family"] = "IPAexGothic"

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]
EventKey = Tuple[Prefix, str]  # (prefix, next)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


def within_range(d: int, start: Optional[int], end: Optional[int]) -> bool:
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


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


def count_ngram_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: Optional[int],
    date_end: Optional[int],
) -> Tuple[Counter, Counter]:
    if n < 2:
        raise ValueError("n>=2 を想定（prefix長が必要）")

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


def to_pnext_events(ng_counter: Counter) -> Tuple[Counter, Counter]:
    event = Counter()
    prefN = Counter()
    for gram, c in ng_counter.items():
        prefix = gram[:-1]
        nxt = gram[-1]
        event[(prefix, nxt)] += c
        prefN[prefix] += c
    return event, prefN


def pnext_of(event: Counter, prefN: Counter, key: EventKey, missing_policy: str) -> float:
    prefix, nxt = key
    Ny = prefN.get(prefix, 0)
    if Ny <= 0:
        if missing_policy == "zero":
            return 0.0
        elif missing_policy == "skip":
            return float("nan")
        else:
            return 0.0
    return event.get((prefix, nxt), 0) / Ny


def fmt_prefix(prefix: Prefix) -> str:
    return "-".join(prefix)


def fmt_event(prefix: Prefix, nxt: str) -> str:
    return f"{fmt_prefix(prefix)}→{nxt}"


def select_basis_events(eventA: Counter, eventB: Counter, topk: int, basis: str) -> List[EventKey]:
    if basis == "A":
        return [k for k, _ in eventA.most_common(topk)]
    if basis == "B":
        return [k for k, _ in eventB.most_common(topk)]
    if basis == "UNION":
        keys = []
        keys.extend([k for k, _ in eventA.most_common(topk)])
        keys.extend([k for k, _ in eventB.most_common(topk)])
        seen = set()
        out = []
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out
    raise ValueError("--basis must be A, B, or UNION")


def print_compare_table(
    tag: str,
    keys: List[EventKey],
    eventA: Counter,
    prefA: Counter,
    eventB: Counter,
    prefB: Counter,
    missing_policy: str,
):
    print("")
    print("=" * 120)
    print(tag)
    print("=" * 120)
    print(f"{'rank':>4}  {'freqA':>6}  {'freqB':>6}  {'event(prefix->next)':<35}  {'P_A':>10}  {'P_B':>10}")
    for i, (prefix, nxt) in enumerate(keys, 1):
        fa = eventA.get((prefix, nxt), 0)
        fb = eventB.get((prefix, nxt), 0)
        pa = pnext_of(eventA, prefA, (prefix, nxt), missing_policy)
        pb = pnext_of(eventB, prefB, (prefix, nxt), missing_policy)
        print(f"{i:>4}  {fa:>6}  {fb:>6}  {fmt_event(prefix, nxt):<35}  {pa:>10.6f}  {pb:>10.6f}")


def plot_compare_bar(
    tag: str,
    keys: List[EventKey],
    eventA: Counter,
    prefA: Counter,
    eventB: Counter,
    prefB: Counter,
    missing_policy: str,
    out_png: str,
    labelA: str,
    labelB: str,
):
    if not keys:
        print(f"# [WARN] no keys for plot: {tag}")
        return

    labels = [fmt_event(pfx, nxt) for (pfx, nxt) in keys]
    valsA = []
    valsB = []
    filtered_labels = []
    filtered_keys = []

    for k, lab in zip(keys, labels):
        pa = pnext_of(eventA, prefA, k, missing_policy)
        pb = pnext_of(eventB, prefB, k, missing_policy)
        if missing_policy == "skip" and (pa != pa or pb != pb):  # nan check
            continue
        filtered_keys.append(k)
        filtered_labels.append(lab)
        valsA.append(0.0 if pa != pa else pa)
        valsB.append(0.0 if pb != pb else pb)

    if not filtered_keys:
        print(f"# [WARN] all items skipped by missing_policy=skip: {tag}")
        return

    fig_w = max(12, len(filtered_labels) * 0.65)
    fig_h = 6.0
    plt.figure(figsize=(fig_w, fig_h))

    x = list(range(len(filtered_labels)))
    width = 0.40
    xa = [xi - width / 2 for xi in x]
    xb = [xi + width / 2 for xi in x]

    plt.bar(xa, valsA, width=width, label=labelA)
    plt.bar(xb, valsB, width=width, label=labelB)

    plt.xticks(x, filtered_labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(next|prefix) (MLE)")
    plt.xlabel("prefix → next (basis top-K by freq)")
    plt.title(tag)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


def compute_period(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    start: Optional[int],
    end: Optional[int],
):
    heads_ng, nonheads_ng = count_ngram_heads_nonheads_in_range(
        segs_by_person, n=n, heads_name=heads_name, date_start=start, date_end=end
    )
    h_event, h_pref = to_pnext_events(heads_ng)
    n_event, n_pref = to_pnext_events(nonheads_ng)
    return (h_event, h_pref, n_event, n_pref)


def dump_nurses(seqs: SeqDict, find_id: Optional[str], find_name: Optional[str], limit: int = 200) -> None:
    """
    past-shifts 内の nurse 一覧を出す（TSV: id\\tname）
    find_id / find_name があれば部分一致で絞る
    """
    keys = list(seqs.keys())

    if find_id is not None:
        q = str(find_id)
        keys = [k for k in keys if q in str(k[0])]

    if find_name is not None:
        q = str(find_name)
        keys = [k for k in keys if q in str(k[1])]

    keys.sort(key=lambda k: (str(k[0]), str(k[1])))

    if not keys:
        print("# no nurses matched.")
        return

    for (nid, name) in keys[:limit]:
        print(f"{nid}\t{name}")

    if len(keys) > limit:
        print(f"# ... truncated: {len(keys)} total matched, showed first {limit}")


def run_one(
    ward_name: str,
    past_shifts_file: str,
    group_settings_dir: str,
    n: int,
    topk: int,
    a_start: Optional[int],
    a_end: Optional[int],
    b_start: Optional[int],
    b_end: Optional[int],
    heads_name: str,
    outdir: str,
    basis: str,
    missing_policy: str,
):
    seqs = data_loader.load_past_shifts(past_shifts_file)
    timeline = data_loader.load_staff_group_timeline(group_settings_dir)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # Period A/B
    (hA, hpA, nA, npA) = compute_period(segs_by_person, n, heads_name, a_start, a_end)
    (hB, hpB, nB, npB) = compute_period(segs_by_person, n, heads_name, b_start, b_end)

    # basis keys
    keys_h = select_basis_events(hA, hB, topk=topk, basis=basis)
    keys_n = select_basis_events(nA, nB, topk=topk, basis=basis)

    # tags
    pa = f"{a_start if a_start is not None else 'MIN'}..{a_end if a_end is not None else 'MAX'}"
    pb = f"{b_start if b_start is not None else 'MIN'}..{b_end if b_end is not None else 'MAX'}"
    period_labelA = f"A({pa})"
    period_labelB = f"B({pb})"

    tag_h = f'Ward="{ward_name}" Heads  N={n}  basis={basis}  K={topk}  missing={missing_policy}'
    tag_n = f'Ward="{ward_name}" NonHeads  N={n}  basis={basis}  K={topk}  missing={missing_policy}'

    # print
    print_compare_table(tag_h, keys_h, hA, hpA, hB, hpB, missing_policy)
    print_compare_table(tag_n, keys_n, nA, npA, nB, npB, missing_policy)

    # plot
    safe_ward = ward_name.replace("/", "_")
    out_h = os.path.join(outdir, f"pnext_compare_heads_ward-{safe_ward}_N{n}_K{topk}_basis-{basis}.png")
    out_n = os.path.join(outdir, f"pnext_compare_nonheads_ward-{safe_ward}_N{n}_K{topk}_basis-{basis}.png")

    plot_compare_bar(
        tag=f"{tag_h}\n{period_labelA} vs {period_labelB}",
        keys=keys_h,
        eventA=hA, prefA=hpA,
        eventB=hB, prefB=hpB,
        missing_policy=missing_policy,
        out_png=out_h,
        labelA=period_labelA,
        labelB=period_labelB,
    )
    plot_compare_bar(
        tag=f"{tag_n}\n{period_labelA} vs {period_labelB}",
        keys=keys_n,
        eventA=nA, prefA=npA,
        eventB=nB, prefB=npB,
        missing_policy=missing_policy,
        out_png=out_n,
        labelA=period_labelA,
        labelB=period_labelB,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts の .lp か、.lpが並ぶディレクトリ")
    ap.add_argument("group_settings", help="病棟の group-settings ディレクトリ、または group-settings-root")

    ap.add_argument("--n", type=int, required=True, help="n-gram の N（>=2）")
    ap.add_argument("--topk", type=int, default=20, help="freq 上位K件")

    # Period A
    ap.add_argument("--a-start", type=int, default=None, help="期間A start YYYYMMDD")
    ap.add_argument("--a-end", type=int, default=None, help="期間A end YYYYMMDD")

    # Period B
    ap.add_argument("--b-start", type=int, default=None, help="期間B start YYYYMMDD")
    ap.add_argument("--b-end", type=int, default=None, help="期間B end YYYYMMDD")

    ap.add_argument("--heads-name", default="Heads", help='Heads 判定に使うグループ名（default "Heads"）')
    ap.add_argument("--outdir", default="out/pnext_compare", help="出力先ディレクトリ")

    ap.add_argument("--basis", default="A", choices=["A", "B", "UNION"],
                    help="比較軸: A=期間Aのfreq上位K / B=期間B / UNION=A∪B")
    ap.add_argument("--missing-policy", default="zero", choices=["zero", "skip"],
                    help="片方にprefixが存在しない場合: zero=0にする / skip=その項目を落とす")

    # ---- NEW: listing / search ----
    ap.add_argument("--list-nurses", action="store_true",
                    help="past-shifts 内の nurse id/name 一覧を TSV で出して終了")
    ap.add_argument("--find-id", type=str, default=None,
                    help="指定文字列を含む nurse_id を検索して表示（例: 74398）")
    ap.add_argument("--find-name", type=str, default=None,
                    help="指定文字列を含む name を検索して表示（例: 山田）")
    ap.add_argument("--list-limit", type=int, default=200,
                    help="--list-nurses/--find-id/--find-name の表示上限（default 200）")

    args = ap.parse_args()

    if args.n < 2:
        print("[ERROR] --n は 2 以上（prefixが必要）", file=sys.stderr)
        sys.exit(1)

    # 全病棟モード
    if os.path.isdir(args.past_shifts):
        past_dir = args.past_shifts
        settings_root = args.group_settings
        if not os.path.isdir(settings_root):
            print(f"[ERROR] group_settings-root がディレクトリではない: {settings_root}", file=sys.stderr)
            sys.exit(1)

        lp_files = [f for f in sorted(os.listdir(past_dir)) if f.endswith(".lp")]
        if not lp_files:
            print(f"[ERROR] .lp が見つからない: {past_dir}", file=sys.stderr)
            sys.exit(1)

        # 一覧/検索モード
        if args.list_nurses or args.find_id is not None or args.find_name is not None:
            for fname in lp_files:
                ward = os.path.splitext(fname)[0]
                past_file = os.path.join(past_dir, fname)
                print("\n" + "#" * 140)
                print(f'# LIST ward="{ward}"  past="{past_file}"')
                print("#" * 140)
                seqs = data_loader.load_past_shifts(past_file)
                dump_nurses(seqs, args.find_id, args.find_name, limit=args.list_limit)
            return

        ensure_dir(args.outdir)

        for fname in lp_files:
            ward = os.path.splitext(fname)[0]
            past_file = os.path.join(past_dir, fname)
            ward_settings_dir = os.path.join(settings_root, ward)
            if not os.path.isdir(ward_settings_dir):
                print(f'# [SKIP] ward="{ward}" settings not found: {ward_settings_dir}')
                continue

            print("\n" + "#" * 140)
            print(f'# RUN ward="{ward}"  past="{past_file}"  settings="{ward_settings_dir}"')
            print("#" * 140)

            run_one(
                ward_name=ward,
                past_shifts_file=past_file,
                group_settings_dir=ward_settings_dir,
                n=args.n,
                topk=args.topk,
                a_start=args.a_start, a_end=args.a_end,
                b_start=args.b_start, b_end=args.b_end,
                heads_name=args.heads_name,
                outdir=args.outdir,
                basis=args.basis,
                missing_policy=args.missing_policy,
            )
        return

    # 単一病棟モード
    past_file = args.past_shifts
    settings_dir = args.group_settings

    if not os.path.isfile(past_file):
        print(f"[ERROR] past_shifts がファイルじゃない: {past_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(settings_dir):
        print(f"[ERROR] group_settings は病棟ディレクトリを指定して: {settings_dir}", file=sys.stderr)
        sys.exit(1)

    # 一覧/検索モード（ここで終了）
    if args.list_nurses or args.find_id is not None or args.find_name is not None:
        seqs = data_loader.load_past_shifts(past_file)
        dump_nurses(seqs, args.find_id, args.find_name, limit=args.list_limit)
        return

    ensure_dir(args.outdir)

    ward = os.path.splitext(os.path.basename(past_file))[0]
    run_one(
        ward_name=ward,
        past_shifts_file=past_file,
        group_settings_dir=settings_dir,
        n=args.n,
        topk=args.topk,
        a_start=args.a_start, a_end=args.a_end,
        b_start=args.b_start, b_end=args.b_end,
        heads_name=args.heads_name,
        outdir=args.outdir,
        basis=args.basis,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
