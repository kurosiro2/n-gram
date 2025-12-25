#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全病棟での3-gram確率 上位10件を決め、そのTop10を横軸に固定して
指定した病棟だけを並べて比較（グループ化棒グラフ1枚）する。

★仕様:
- VALID_SHIFTS 以外が出たらそこで区切り、次の有効シフトから再開
  （区切りをまたぐ n-gram は数えない）

使い方:
  python ward_compare_top10_3gram.py [past_shifts_dir] [output_png] [--year YYYY]

例:
  python ward_compare_top10_3gram.py \
    exp/data/real-name/past-shifts-2019-2025/ \
    out_top10_3gram_compare.png \
    --year 2025
"""

import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# data_loader import（このファイルを exp/statistics/ngram/ 付近に置く想定）
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader


# -------------------------------------------------------------
# 有効シフト（ngram_past_shifts_group.py と同じ仕様）
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH",
}

# 比較したい病棟（ユーザ指定）
TARGET_WARDS = [
    "2階西病棟",
    "3階西病棟",
    "5階北病棟",
    "7階南病棟",
    "GCU",
]

TOP_K = 10
N = 3


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
    segs = []
    cur = []
    for s in shifts:
        if s in VALID_SHIFTS:
            cur.append(s)
        else:
            if cur:
                segs.append(cur)
                cur = []
    if cur:
        segs.append(cur)
    return segs


def count_ngrams_with_segmentation(seqs_dict, n):
    ctr = Counter()
    total = 0
    for (nid, name), seq in seqs_dict.items():
        if not seq:
            continue
        shifts_all = [s for (_, s) in seq]
        segments = split_into_valid_segments(shifts_all)
        for seg in segments:
            if len(seg) < n:
                continue
            for i in range(len(seg) - n + 1):
                gram = tuple(seg[i:i+n])
                ctr[gram] += 1
                total += 1
    return ctr, total


def list_ward_lp_files(past_shifts_dir):
    ward_files = {}
    for fname in sorted(os.listdir(past_shifts_dir)):
        if not fname.endswith(".lp"):
            continue
        ward = os.path.splitext(fname)[0]
        ward_files[ward] = os.path.join(past_shifts_dir, fname)
    return ward_files


def gram_to_label(gram):
    return "-".join(gram)


def main():
    if len(sys.argv) < 3:
        print("Usage: python ward_compare_top10_3gram.py [past_shifts_dir] [output_png] [--year YYYY]")
        sys.exit(1)

    past_shifts_dir = sys.argv[1]
    output_png      = sys.argv[2]

    year = None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--year" and i + 1 < len(args):
            year = int(args[i+1]); i += 2
        else:
            print(f"Unknown arg: {args[i]}")
            sys.exit(1)

    ward_files = list_ward_lp_files(past_shifts_dir)

    # --- 全病棟で Top10 を決める ---
    global_ctr = Counter()
    global_total = 0

    # 病棟ごとの結果も保存しておく（あとで TARGET_WARDS を描く）
    ward_ctr = {}
    ward_total = {}

    for ward, path in ward_files.items():
        seqs = data_loader.load_past_shifts(path)
        seqs = filter_seqs_by_year(seqs, year)
        ctr, total = count_ngrams_with_segmentation(seqs, N)

        ward_ctr[ward] = ctr
        ward_total[ward] = total

        global_ctr.update(ctr)
        global_total += total

    if global_total == 0:
        print("No 3-grams found (after filtering/segmentation).")
        sys.exit(0)

    global_items = []
    for gram, c in global_ctr.items():
        p = c / global_total
        global_items.append((p, c, gram))
    global_items.sort(key=lambda x: (-x[0], -x[1], x[2]))
    top_items = global_items[:TOP_K]
    top_grams = [gram for (p, c, gram) in top_items]

    # 確認用txt（pngと同じ場所に置く）
    txt_path = os.path.splitext(output_png)[0] + "_GLOBAL_top10.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# N={N}, TOP_K={TOP_K}\n")
        f.write(f"# year={year if year is not None else 'all'}\n")
        f.write(f"# global_total_ngrams={global_total}\n")
        f.write(f"# VALID_SHIFTS={sorted(VALID_SHIFTS)}\n\n")
        for rank, (p, c, gram) in enumerate(top_items, 1):
            f.write(f"{rank:2d}. {p:.6f}  (count={c})  {gram_to_label(gram)}\n")

    # --- 指定5病棟だけ並べて描く（1枚） ---
    # 存在チェック
    missing = [w for w in TARGET_WARDS if w not in ward_files]
    if missing:
        print("# [WARN] These wards not found in directory (.lp name mismatch?):", ", ".join(missing))

    wards = [w for w in TARGET_WARDS if w in ward_files]

    if not wards:
        print("No target wards found. Check .lp filenames.")
        sys.exit(0)

    x_labels = [gram_to_label(g) for g in top_grams]
    x = list(range(len(top_grams)))

    # 病棟ごとの確率ベクトル
    ys = {}
    for w in wards:
        total = ward_total.get(w, 0)
        ctr = ward_ctr.get(w, Counter())
        if total <= 0:
            ys[w] = [0.0] * len(top_grams)
        else:
            ys[w] = [ctr.get(g, 0) / total for g in top_grams]

    # グループ化棒グラフ
    plt.figure(figsize=(max(12, len(x_labels) * 0.9), 6))
    m = len(wards)
    width = 0.8 / m

    for j, w in enumerate(wards):
        xj = [xi - 0.4 + width/2 + j*width for xi in x]
        plt.bar(xj, ys[w], width=width, label=w)

    plt.xticks(x, x_labels, rotation=55, ha="right")
    plt.xlabel("Global top10 3-grams (fixed axis)")
    plt.ylabel("Probability in ward")
    title_year = "all" if year is None else str(year)
    plt.title(f"Compare wards on global top10 3-gram probabilities (year={title_year})")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"# Saved: {output_png}")
    print(f"# Saved: {txt_path}")


if __name__ == "__main__":
    main()
