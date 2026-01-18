#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【事前評価】prefix固定で、2期間の P(next|prefix) (MLE) を同じ棒グラフで比較（円滑化なし）

- past-shifts(.lp) を読み、group-settings（timeline）で「グループ集合が変わるたびに」セグメント分割
  → セグメント境界を跨ぐ n-gram は数えない
- Heads / NonHeads の2値のみ（Heads-name で判定）
- 指定prefix（長さ = N-1）に対する next 分布（10種）を出す
- 期間A / 期間B を横並びバーで比較（同一PNGに出力）
- MLEのみ（Laplace等の円滑化は一切しない）

使い方（単一病棟）:
  python pnext_prefix_compare_2periods.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --heads-name Heads \
    --outdir out/prefix_compare

使い方（全病棟: past-shifts-dir × group-settings-root）:
  python pnext_prefix_compare_2periods.py \
    /workspace/2025/past-shifts \
    /workspace/2025/group-settings \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --outdir out/prefix_compare_all
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
# import path (exp/statistics 配下で使う想定)
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
    """
    所属グループ集合が変わるたびにセグメント分割。
    境界を跨ぐ n-gram を数えないための前処理。
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


def parse_prefix(prefix_str: str, expected_len: int) -> Prefix:
    s = prefix_str.strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f'--prefix must have length {expected_len} (got {len(parts)}): "{prefix_str}"')
    for p in parts:
        if p not in VALID_SHIFTS_SET:
            raise ValueError(f"--prefix contains invalid shift: {p}")
    return tuple(parts)


def count_next_for_prefix_in_period(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter, int, int]:
    """
    期間内で、target_prefix に一致する (prefix->next) だけ数える
    戻り値:
      heads_next_counter, nonheads_next_counter, heads_Ny, nonheads_Ny
    """
    heads_next = Counter()
    non_next = Counter()
    Ny_h = 0
    Ny_n = 0

    pref_len = n - 1

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for (_, s) in sseq]
            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i : i + pref_len])
                if pfx != target_prefix:
                    continue
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_next[nxt] += 1
                    Ny_h += 1
                else:
                    non_next[nxt] += 1
                    Ny_n += 1

    return heads_next, non_next, Ny_h, Ny_n


def mle_probs(next_counter: Counter, Ny: int) -> List[float]:
    """
    MLE確率ベクトル（Ny=0なら“未定義”だが、図を出すため 0ベクトルで返す）
    """
    if Ny <= 0:
        return [0.0] * len(VALID_SHIFTS)
    return [next_counter.get(x, 0) / Ny for x in VALID_SHIFTS]


def print_debug(tag: str, prefix: Prefix, next_counter: Counter, Ny: int) -> None:
    print("")
    print("=" * 100)
    print(tag)
    print("=" * 100)
    print(f"prefix={','.join(prefix)}  Ny=c(prefix)={Ny}")
    top = next_counter.most_common(10)
    if top:
        print("top next counts:", ", ".join([f"{x}:{c}" for x, c in top]))
    else:
        print("top next counts: (none)")
    probs = mle_probs(next_counter, Ny)
    print(f"{'next':>4}  {'count':>6}  {'P_MLE':>10}")
    for x, p in zip(VALID_SHIFTS, probs):
        print(f"{x:>4}  {next_counter.get(x,0):>6}  {p:>10.6f}")


def plot_compare(
    out_png: str,
    title: str,
    probs_A: List[float],
    probs_B: List[float],
    label_A: str,
    label_B: str,
) -> None:
    xs = list(range(len(VALID_SHIFTS)))
    width = 0.38

    plt.figure(figsize=(12.5, 5.6))

    xs1 = [x - width/2 for x in xs]
    xs2 = [x + width/2 for x in xs]

    plt.bar(xs1, probs_A, width=width, label=label_A)
    plt.bar(xs2, probs_B, width=width, label=label_B)

    plt.xticks(xs, VALID_SHIFTS)
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(next|prefix)")
    plt.xlabel("next shift")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


def run_one_ward(
    ward_name: str,
    past_shifts_file: str,
    group_settings_dir: str,
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
    outdir: str,
):
    seqs = data_loader.load_past_shifts(past_shifts_file)
    timeline = data_loader.load_staff_group_timeline(group_settings_dir)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # ---- period A
    hA, nA, Ny_hA, Ny_nA = count_next_for_prefix_in_period(
        segs_by_person, n, target_prefix, heads_name, a_start, a_end
    )
    # ---- period B
    hB, nB, Ny_hB, Ny_nB = count_next_for_prefix_in_period(
        segs_by_person, n, target_prefix, heads_name, b_start, b_end
    )

    prefix_str = ",".join(target_prefix)
    labelA = f"A:{a_start}-{a_end}"
    labelB = f"B:{b_start}-{b_end}"

    # debug print
    print_debug(f'Ward="{ward_name}" Heads  {labelA}', target_prefix, hA, Ny_hA)
    print_debug(f'Ward="{ward_name}" Heads  {labelB}', target_prefix, hB, Ny_hB)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelA}', target_prefix, nA, Ny_nA)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelB}', target_prefix, nB, Ny_nB)

    # probs
    phA = mle_probs(hA, Ny_hA)
    phB = mle_probs(hB, Ny_hB)
    pnA = mle_probs(nA, Ny_nA)
    pnB = mle_probs(nB, Ny_nB)

    # title note (Ny=0 を明記したい)
    note_h = f"(Np A={Ny_hA}, Np B={Ny_hB})"
    note_n = f"(Np A={Ny_nA}, Np B={Ny_nB})"

    safe_ward = ward_name.replace("/", "_")
    safe_prefix = prefix_str.replace(",", "-")

    out_h = os.path.join(outdir, f"compare_prefix-{safe_prefix}_heads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}.png")
    out_n = os.path.join(outdir, f"compare_prefix-{safe_prefix}_nonheads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}.png")

    plot_compare(
        out_png=out_h,
        title=f"Ward={ward_name} Heads | prefix={prefix_str} | N={n} | {note_h}",
        probs_A=phA,
        probs_B=phB,
        label_A=labelA,
        label_B=labelB,
    )
    plot_compare(
        out_png=out_n,
        title=f"Ward={ward_name} NonHeads | prefix={prefix_str} | N={n} | {note_n}",
        probs_A=pnA,
        probs_B=pnB,
        label_A=labelA,
        label_B=labelB,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts の .lp か、.lpが並ぶディレクトリ")
    ap.add_argument("group_settings", help="病棟の group-settings ディレクトリ、または group-settings-root")
    ap.add_argument("--n", type=int, required=True, help="n-gram の N（>=2）")
    ap.add_argument("--prefix", required=True, help='例: "SE,SN"（N=3なら2個）')

    ap.add_argument("--a-start", type=int, required=True, help="期間A start YYYYMMDD")
    ap.add_argument("--a-end", type=int, required=True, help="期間A end YYYYMMDD")
    ap.add_argument("--b-start", type=int, required=True, help="期間B start YYYYMMDD")
    ap.add_argument("--b-end", type=int, required=True, help="期間B end YYYYMMDD")

    ap.add_argument("--heads-name", default="Heads", help='Heads 判定に使うグループ名（default "Heads"）')
    ap.add_argument("--outdir", default="out/prefix_compare", help="出力先ディレクトリ")
    args = ap.parse_args()

    if args.n < 2:
        print("[ERROR] --n は 2 以上（prefixが必要）", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.outdir)

    target_prefix = parse_prefix(args.prefix, expected_len=args.n - 1)

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

        for fname in lp_files:
            ward = os.path.splitext(fname)[0]
            past_file = os.path.join(past_dir, fname)
            ward_settings_dir = os.path.join(settings_root, ward)
            if not os.path.isdir(ward_settings_dir):
                print(f'# [SKIP] ward="{ward}" settings not found: {ward_settings_dir}')
                continue

            print("\n" + "#" * 120)
            print(f'# RUN ward="{ward}" prefix="{args.prefix}" N={args.n}  A={args.a_start}-{args.a_end}  B={args.b_start}-{args.b_end}')
            print("#" * 120)

            run_one_ward(
                ward_name=ward,
                past_shifts_file=past_file,
                group_settings_dir=ward_settings_dir,
                n=args.n,
                target_prefix=target_prefix,
                heads_name=args.heads_name,
                a_start=args.a_start,
                a_end=args.a_end,
                b_start=args.b_start,
                b_end=args.b_end,
                outdir=args.outdir,
            )
        return

    # 単一病棟モード
    past_file = args.past_shifts
    settings_dir = args.group_settings

    if not os.path.isfile(past_file):
        print(f"[ERROR] past_shifts がファイルじゃない: {past_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(settings_dir):
        print(f"[ERROR] group_settings は病棟のディレクトリを指定して: {settings_dir}", file=sys.stderr)
        sys.exit(1)

    ward = os.path.splitext(os.path.basename(past_file))[0]
    run_one_ward(
        ward_name=ward,
        past_shifts_file=past_file,
        group_settings_dir=settings_dir,
        n=args.n,
        target_prefix=target_prefix,
        heads_name=args.heads_name,
        a_start=args.a_start,
        a_end=args.a_end,
        b_start=args.b_start,
        b_end=args.b_end,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
