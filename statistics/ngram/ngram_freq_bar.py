#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【事前評価】past-shifts（看護師A vs 看護師B）で
「頻度分布（確率）」を棒グラフ(2本バー)で比較してPNG出力（＋printで表も出す）

- n=1:
    1-gram（勤務記号）の確率分布 P(shift) を比較
- n>=2:
    条件付き確率 P(next|prefix) を、freq上位Kの (prefix->next) で比較

比較方法（n>=2）:
- A/B それぞれで n-gram をカウントして (prefix->next) の freq を得る
- "比較軸" となるイベント集合を選ぶ:
    * --basis=A なら 看護師A の freq 上位K を採用
    * --basis=B なら 看護師B の freq 上位K を採用
    * --basis=UNION なら A上位K ∪ B上位K（最大2K）
- 採用イベントについて P_A(next|prefix), P_B(next|prefix) を計算
    （片方に存在しない prefix は 0 扱いにできる: --missing-policy zero）

追加（デバッグ/探索）:
  --list-nurses
      past-shifts 内の (nurse_id, name) 一覧を TSV で出して終了（id\tname）
  --find-id 74398
      id の文字列表現に '74398' を含む nurse_id を検索して表示（最大200件）
  --find-name 山田
      name に '山田' を含むものを検索して表示（最大200件）

使い方例（単一病棟）:
  python ngram_pdist_compare_two_nurses.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --n 5 --topk 20 \
    --start 20240101 --end 20241231 \
    --nurse-a-id 101 --nurse-b-id 202 \
    --basis UNION \
    --outdir out/pdist_nurse_compare

※ nurse指定は id優先:
   --nurse-a-id / --nurse-b-id
   ない場合は --nurse-a-name / --nurse-b-name を使う（完全一致）
"""

import os
import sys
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# NEW: global font sizes (bigger text everywhere in plots)
# -------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,          # base
    "axes.titlesize": 18,     # title
    "axes.labelsize": 17,     # x/y label
    "xtick.labelsize": 14,    # x ticks
    "ytick.labelsize": 14,    # y ticks
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 18,   # figure title
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

import data_loader  # load_past_shifts

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)

PersonKey = Tuple[int, str]     # (nurse_id, name)
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]
EventKey = Tuple[Prefix, str]   # (prefix, next)


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


def find_person_key(
    seqs: SeqDict,
    nurse_id: Optional[int],
    nurse_name: Optional[str],
) -> PersonKey:
    """
    nurse_id は int で受け取るが、past-shifts 側の key が str/ゼロ埋めの可能性もあるので
    念のため str 化して比較もする（ただし基本は完全一致）。
    """
    keys = list(seqs.keys())

    if nurse_id is not None:
        nid_int = int(nurse_id)
        nid_str = str(nurse_id)

        # 1) int 完全一致
        cands = [k for k in keys if isinstance(k[0], int) and k[0] == nid_int]
        if cands:
            return cands[0]

        # 2) str 完全一致（k[0] が文字列/別型のとき）
        cands = [k for k in keys if str(k[0]) == nid_str]
        if cands:
            return cands[0]

        raise ValueError(f"nurse_id={nurse_id} が past-shifts に見つからない")

    if nurse_name is not None:
        cands = [k for k in keys if k[1] == nurse_name]
        if not cands:
            raise ValueError(f'nurse_name="{nurse_name}" が past-shifts に見つからない（完全一致）')
        return cands[0]

    raise ValueError("nurse指定がない: --nurse-a-id/--nurse-a-name 等を指定して")


def count_ngrams_for_person(
    seq: List[Tuple[int, str]],
    n: int,
    date_start: Optional[int],
    date_end: Optional[int],
) -> Counter:
    seq = normalize_seq(seq)
    seq = [(d, s) for (d, s) in seq if within_range(d, date_start, date_end)]
    seq.sort(key=lambda t: t[0])
    shifts = [s for (_, s) in seq]

    cnt = Counter()
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(shifts) < n:
        return cnt

    for i in range(len(shifts) - n + 1):
        gram = tuple(shifts[i:i+n])
        cnt[gram] += 1
    return cnt


def to_event_and_prefix(ng_counter: Counter) -> Tuple[Counter, Counter]:
    """
    n=1 でも扱えるようにする:
      gram=(X,) のとき prefix=() next=X
    """
    event = Counter()
    prefN = Counter()
    for gram, c in ng_counter.items():
        if len(gram) == 1:
            prefix = tuple()
            nxt = gram[0]
        else:
            prefix = gram[:-1]
            nxt = gram[-1]
        event[(prefix, nxt)] += c
        prefN[prefix] += c
    return event, prefN


def p_of(event: Counter, prefN: Counter, key: EventKey, missing_policy: str) -> float:
    """
    P(next|prefix) （n=1 の場合 prefix=() なので P(shift) になる）
    """
    prefix, nxt = key
    Ny = prefN.get(prefix, 0)
    if Ny <= 0:
        if missing_policy == "zero":
            return 0.0
        if missing_policy == "skip":
            return float("nan")
        return 0.0
    return event.get((prefix, nxt), 0) / Ny


def fmt_prefix(prefix: Prefix) -> str:
    if not prefix:
        return "∅"
    return "-".join(prefix)


def fmt_event(prefix: Prefix, nxt: str) -> str:
    if not prefix:
        return f"{nxt}"
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
    eventA: Counter, prefA: Counter,
    eventB: Counter, prefB: Counter,
    missing_policy: str,
):
    print("")
    print("=" * 120)
    print(tag)
    print("=" * 120)
    print(f"{'rank':>4}  {'freqA':>6}  {'freqB':>6}  {'event':<38}  {'P_A':>10}  {'P_B':>10}")
    for i, (prefix, nxt) in enumerate(keys, 1):
        fa = eventA.get((prefix, nxt), 0)
        fb = eventB.get((prefix, nxt), 0)
        pa = p_of(eventA, prefA, (prefix, nxt), missing_policy)
        pb = p_of(eventB, prefB, (prefix, nxt), missing_policy)
        print(f"{i:>4}  {fa:>6}  {fb:>6}  {fmt_event(prefix, nxt):<38}  {pa:>10.6f}  {pb:>10.6f}")


def plot_compare_bar(
    tag: str,
    keys: List[EventKey],
    eventA: Counter, prefA: Counter,
    eventB: Counter, prefB: Counter,
    missing_policy: str,
    out_png: str,
    labelA: str,
    labelB: str,
    ylabel: str,
):
    if not keys:
        print(f"# [WARN] no keys for plot: {tag}")
        return

    labels = [fmt_event(pfx, nxt) for (pfx, nxt) in keys]
    valsA, valsB = [], []
    filtered_labels = []

    for k, lab in zip(keys, labels):
        pa = p_of(eventA, prefA, k, missing_policy)
        pb = p_of(eventB, prefB, k, missing_policy)
        if missing_policy == "skip" and (pa != pa or pb != pb):  # nan check
            continue
        filtered_labels.append(lab)
        valsA.append(0.0 if pa != pa else pa)
        valsB.append(0.0 if pb != pb else pb)

    if not filtered_labels:
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

    # x tick が長くなりがちなので、ここだけさらに調整したければ fontsize を指定してもOK
    plt.xticks(x, filtered_labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.xlabel("gram")
    plt.title(tag)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


def run_one(
    ward_name: str,
    past_shifts_file: str,
    n: int,
    topk: int,
    start: Optional[int],
    end: Optional[int],
    nurseA_id: Optional[int],
    nurseA_name: Optional[str],
    nurseB_id: Optional[int],
    nurseB_name: Optional[str],
    outdir: str,
    basis: str,
    missing_policy: str,
):
    seqs = data_loader.load_past_shifts(past_shifts_file)

    keyA = find_person_key(seqs, nurseA_id, nurseA_name)
    keyB = find_person_key(seqs, nurseB_id, nurseB_name)

    seqA = seqs.get(keyA, [])
    seqB = seqs.get(keyB, [])

    ngA = count_ngrams_for_person(seqA, n=n, date_start=start, date_end=end)
    ngB = count_ngrams_for_person(seqB, n=n, date_start=start, date_end=end)

    eventA, prefA = to_event_and_prefix(ngA)
    eventB, prefB = to_event_and_prefix(ngB)

    keys = select_basis_events(eventA, eventB, topk=topk, basis=basis)

    # labels
    pr = f"{start if start is not None else 'MIN'}..{end if end is not None else 'MAX'}"
    personA = f"A({keyA[1]})"
    personB = f"B({keyB[1]})"

    safe_ward = ward_name.replace("/", "_")
    out_png = os.path.join(
        outdir,
        f"pdist_compare_n{n}_ward-{safe_ward}_K{topk}_basis-{basis}_A{keyA[0]}_B{keyB[0]}.png"
    )

    # tag
    if n == 1:
        tag = f'P(gram) : n=1'
        ylabel = "P(gram)"
    else:
        tag = f'Ward="{ward_name}"  P(next|prefix)  n={n}  range={pr}  basis={basis}  K={topk}  missing={missing_policy}'
        ylabel = "P(next|prefix) (MLE)"

    # print + plot
    print_compare_table(tag, keys, eventA, prefA, eventB, prefB, missing_policy)
    plot_compare_bar(
        tag=f"{tag}",
        keys=keys,
        eventA=eventA, prefA=prefA,
        eventB=eventB, prefB=prefB,
        missing_policy=missing_policy,
        out_png=out_png,
        labelA=personA,
        labelB=personB,
        ylabel=ylabel,
    )


def dump_nurses(seqs: SeqDict, find_id: Optional[str], find_name: Optional[str], limit: int = 200) -> None:
    """
    past-shifts 内の nurse 一覧を出す（TSV: id\tname）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts の .lp か、.lpが並ぶディレクトリ")
    ap.add_argument("group_settings", help="(互換のため残す) 未使用。指定されても無視します。")

    ap.add_argument("--n", type=int, required=True, help="n-gram の N（>=1） n=1 は P(shift)")
    ap.add_argument("--topk", type=int, default=20, help="freq 上位K件")

    ap.add_argument("--start", type=int, default=None, help="対象期間 start YYYYMMDD")
    ap.add_argument("--end", type=int, default=None, help="対象期間 end YYYYMMDD")

    # nurse A/B
    ap.add_argument("--nurse-a-id", type=int, default=None, help="看護師A nurse_id（id優先）")
    ap.add_argument("--nurse-b-id", type=int, default=None, help="看護師B nurse_id（id優先）")
    ap.add_argument("--nurse-a-name", default=None, help="看護師A name（完全一致）")
    ap.add_argument("--nurse-b-name", default=None, help="看護師B name（完全一致）")

    ap.add_argument("--outdir", default="out/pdist_nurse_compare", help="出力先ディレクトリ")

    ap.add_argument("--basis", default="A", choices=["A", "B", "UNION"],
                    help="比較軸: A=看護師Aのfreq上位K / B=看護師B / UNION=A∪B")
    ap.add_argument("--missing-policy", default="zero", choices=["zero", "skip"],
                    help="片方にprefixが存在しない場合: zero=0にする / skip=その項目を落とす")

    # ---- listing / search ----
    ap.add_argument("--list-nurses", action="store_true",
                    help="past-shifts 内の nurse id/name 一覧を TSV で出して終了")
    ap.add_argument("--find-id", type=str, default=None,
                    help="指定文字列を含む nurse_id を検索して表示（例: 74398）")
    ap.add_argument("--find-name", type=str, default=None,
                    help="指定文字列を含む name を検索して表示（例: 山田）")
    ap.add_argument("--list-limit", type=int, default=200,
                    help="--list-nurses/--find-id/--find-name の表示上限（default 200）")

    args = ap.parse_args()

    if args.n < 1:
        print("[ERROR] --n は 1 以上", file=sys.stderr)
        sys.exit(1)

    # 全病棟モード
    if os.path.isdir(args.past_shifts):
        past_dir = args.past_shifts
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

        # 通常モード
        ensure_dir(args.outdir)
        for fname in lp_files:
            ward = os.path.splitext(fname)[0]
            past_file = os.path.join(past_dir, fname)

            print("\n" + "#" * 140)
            print(f'# RUN ward="{ward}"  past="{past_file}"')
            print("#" * 140)

            run_one(
                ward_name=ward,
                past_shifts_file=past_file,
                n=args.n,
                topk=args.topk,
                start=args.start,
                end=args.end,
                nurseA_id=args.nurse_a_id,
                nurseA_name=args.nurse_a_name,
                nurseB_id=args.nurse_b_id,
                nurseB_name=args.nurse_b_name,
                outdir=args.outdir,
                basis=args.basis,
                missing_policy=args.missing_policy,
            )
        return

    # 単一病棟モード
    past_file = args.past_shifts
    if not os.path.isfile(past_file):
        print(f"[ERROR] past_shifts がファイルじゃない: {past_file}", file=sys.stderr)
        sys.exit(1)

    # 一覧/検索モード（ここで終了）
    if args.list_nurses or args.find_id is not None or args.find_name is not None:
        seqs = data_loader.load_past_shifts(past_file)
        dump_nurses(seqs, args.find_id, args.find_name, limit=args.list_limit)
        return

    # 通常モード
    ensure_dir(args.outdir)
    ward = os.path.splitext(os.path.basename(past_file))[0]
    run_one(
        ward_name=ward,
        past_shifts_file=past_file,
        n=args.n,
        topk=args.topk,
        start=args.start,
        end=args.end,
        nurseA_id=args.nurse_a_id,
        nurseA_name=args.nurse_a_name,
        nurseB_id=args.nurse_b_id,
        nurseB_name=args.nurse_b_name,
        outdir=args.outdir,
        basis=args.basis,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
