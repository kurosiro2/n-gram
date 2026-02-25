#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
決定的後続パターン (P(next|prefix)=1.0) を出力し，
そのパターンを持つ看護師IDも併せて表示する。

★仕様:
  - VALID_SHIFTS のみ有効
  - それ以外のシフトが出たら、その時点でいったん ngram カウントを止める
    (= そのシフトを境界として区切り、次の有効シフトから再開)
  - 境界をまたぐ n-gram は数えない

出力形式（各 N ごと）:
  freq  prefix -> next   [nurse_ids]

使い方:
  python det_next_pairs_with_nurses_validseg.py past-shifts.lp setting.lp N_min N_max [--year YYYY]
"""

import sys
import os
from collections import Counter, defaultdict

# -------------------------------------------------------------
# import data_loader
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader


# -------------------------------------------------------------
# 勤務シフト（8種）+ 休暇シフト（2種）だけを有効とする
#   → ngram_past_shifts_group.py と揃えておく
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH",
}


def filter_seqs_by_year(seqs_dict, year=None):
    if year is None:
        return seqs_dict
    start = year * 10000 + 101
    end   = year * 10000 + 1231
    out = {}
    for k, seq in seqs_dict.items():
        sub = [(d, s) for (d, s) in seq if start <= d <= end]
        if sub:
            sub.sort(key=lambda x: x[0])
            out[k] = sub
    return out


def split_into_valid_segments(shifts):
    """
    shifts: ["D","WR",...]
    return: list[list[str]]  # VALID_SHIFTS だけからなる連続区間の列
    """
    segs = []
    cur = []
    for s in shifts:
        if s in VALID_SHIFTS:
            cur.append(s)
        else:
            # 無効シフトが来たら区切る
            if cur:
                segs.append(cur)
                cur = []
    if cur:
        segs.append(cur)
    return segs


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python det_next_pairs_with_nurses_validseg.py "
            "past-shifts.lp setting.lp N_min N_max [--year YYYY]"
        )
        sys.exit(1)

    past_shifts_lp = sys.argv[1]
    setting_lp     = sys.argv[2]   # 互換のため受け取る（今回は未使用）
    N_min          = int(sys.argv[3])
    N_max          = int(sys.argv[4])

    year = None
    args = sys.argv[5:]
    i = 0
    while i < len(args):
        if args[i] == "--year" and i + 1 < len(args):
            year = int(args[i + 1])
            i += 2
        else:
            print(f"Unknown arg: {args[i]}")
            sys.exit(1)

    if N_min < 2:
        print("N_min must be >= 2")
        sys.exit(1)

    seqs = data_loader.load_past_shifts(past_shifts_lp)
    seqs = filter_seqs_by_year(seqs, year)

    if not seqs:
        print("No data.")
        return

    print("# VALID_SHIFTS =", sorted(VALID_SHIFTS))
    if year is None:
        print("# Target: all dates")
    else:
        print(f"# Target: year={year}")
    print("# Output: freq  prefix -> next   [nurse_ids]")
    print()

    # ---------------------------------------------------------
    # N ごとに処理
    # ---------------------------------------------------------
    for N in range(N_min, N_max + 1):
        ctrN   = Counter()
        ctrNm1 = Counter()

        # (prefix, next) -> set(nurse_id)
        nurses_by_pair = defaultdict(set)

        for (nurse_id, name), seq in seqs.items():
            if len(seq) < 1:
                continue

            shifts_all = [s for (_, s) in seq]
            segments = split_into_valid_segments(shifts_all)
            if not segments:
                continue

            for shifts in segments:
                if len(shifts) < (N - 1):
                    continue

                # (N-1)-gram
                if N - 1 >= 1:
                    for i2 in range(len(shifts) - (N - 1) + 1):
                        ctrNm1[tuple(shifts[i2:i2 + (N - 1)])] += 1

                # N-gram
                if len(shifts) >= N:
                    for i2 in range(len(shifts) - N + 1):
                        gram = tuple(shifts[i2:i2 + N])
                        prefix = gram[:-1]
                        nxt = gram[-1]

                        ctrN[gram] += 1
                        nurses_by_pair[(prefix, nxt)].add(nurse_id)

        # prefix -> Counter(next)
        next_by_prefix = defaultdict(Counter)
        for gram, c in ctrN.items():
            next_by_prefix[gram[:-1]][gram[-1]] += c

        det_rows = []
        for prefix, next_ctr in next_by_prefix.items():
            base = ctrNm1.get(prefix, 0)
            if base <= 0:
                continue
            if len(next_ctr) != 1:
                continue

            (nxt, c) = next(iter(next_ctr.items()))
            # 決定的: P=1.0 <=> c == base
            if c == base:
                nurse_ids = sorted(nurses_by_pair[(prefix, nxt)])
                det_rows.append((c, prefix, nxt, nurse_ids))

        det_rows.sort(key=lambda x: (-x[0], x[1], x[2]))

        print(f"----- N={N} deterministic next patterns (count={len(det_rows)}) -----")
        for c, prefix, nxt, nurse_ids in det_rows:
            prefix_str = "-".join(prefix)
            ids_str = ", ".join(str(i) for i in nurse_ids)
            print(f"{c:6d}  {prefix_str} -> {nxt}   [{ids_str}]")
        print()


if __name__ == "__main__":
    main()
