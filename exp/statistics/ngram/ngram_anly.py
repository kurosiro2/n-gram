#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
past-shifts と found-model の n-gram 分布 (freq_share) と
P(next|prefix) を比較して棒グラフを描くスクリプト。

使い方:
  python ngram_freq_anly.py past-shifts.lp setting.lp found-model.lp N output.png

  - past-shifts.lp : data_loader.load_past_shifts で読む LP
  - setting.lp     : data_loader.load_staff_group_timeline で読む設定 LP
  - found-model.lp : ext_assigned / staff_group / staff を含む clingo の解
  - N              : n-gram の N
  - output.png     : freq_share の PNG ファイルパス
                     P(next|prefix) は <output>_pnext.png に保存する

※ グループは "All" 固定で比較
"""

import sys
import os
import re
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 親ディレクトリを import パスに追加して data_loader を読めるようにする
#   このファイル: exp/statistics/ngram/ngram_freq_anly.py
#   data_loader:   exp/statistics/data_loader.py
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline

# -------------------------------------------------------------
# ★ 期間指定（past-shifts 側にだけ適用したい場合に使う）
#   - None にするとその側の制限なし
#   - 例: 2025年だけ → DATE_START = 20250101, DATE_END = 20251231
# -------------------------------------------------------------
DATE_START = 20240101  # 例: 20250101
DATE_END   = 20241231  # 例: 20251231


# -------------------------------------------------------------
# 勤務シフト（8種）+ 休暇シフト（2種）だけを有効とする
#   → ngram_past_shifts_group.py / ngram_found_shifts_group.py と揃える
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH",
}

# -------------------------------------------------------------
# found-model 用の正規表現（ngram_found_shifts_group.py と同じ）
# -------------------------------------------------------------
PAT_EXT   = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\("([^"]+)"\s*,\s*(\d+)\)\.')
PAT_STAFF = re.compile(r'^staff\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*"[^"]*"\s*,\s*"([^"]+)"')


# -------------------------------------------------------------
# ★ 日付フィルタ（past-shifts 側）
# -------------------------------------------------------------
def filter_seqs_by_date(seqs_dict, start_date, end_date):
    """
    seqs_dict: {(nurse_id, name): [(date, shift), ...]}
    start_date, end_date: int (YYYYMMDD) or None

    指定された日付範囲 [start_date, end_date] に入るシフトだけを残す。
    1人のシフトが全て範囲外なら、その人自体を削除する。
    """
    if start_date is None and end_date is None:
        # 期間指定なし → そのまま返す
        return seqs_dict

    filtered = {}
    for key, seq in seqs_dict.items():
        sub = []
        for d, s in seq:
            if start_date is not None and d < start_date:
                continue
            if end_date is not None and d > end_date:
                continue
            sub.append((d, s))
        if sub:
            sub.sort(key=lambda t: t[0])
            filtered[key] = sub
    return filtered


# -------------------------------------------------------------
# n-gram 集計ロジック（past-shifts 側）
# -------------------------------------------------------------
def ngram_counts_by_group_past(seqs_dict, group_timeline, n):
    """
    past-shifts 側: n-gram 出現回数をグループ別にカウント（タイムライン対応版）。

    - seqs_dict: {(nurse_id, name): [(date, shift), ...]}
      * data_loader.load_past_shifts() の戻り値を想定
    - group_timeline:
        load_staff_group_timeline() の戻り値
        name キー・(name, nurse_id) キーの両方を含みうる
    - n: n-gram の n
    """
    from data_loader import get_groups_for_date  # 明示 import（読みやすさ用）

    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    # 名前順にソート（見やすさ用）
    for (nid, name), seq in sorted(seqs_dict.items(), key=lambda kv: kv[0][1]):
        if len(seq) < n:
            continue

        # seq: [(date, shift), ...]
        for i in range(len(seq) - n + 1):
            window = seq[i: i + n]
            dates = [d for (d, _) in window]
            shifts = [s for (_, s) in window]

            # 無効なシフト記号を含む n-gram はスキップ
            if any(s not in VALID_SHIFTS for s in shifts):
                continue

            ref_date = dates[-1]  # 「最後の日」のグループを使う
            gram = tuple(shifts)

            # ★ name だけでなく nurse_id も渡す
            groups = get_groups_for_date(name, ref_date, group_timeline, nurse_id=nid)
            if not groups:
                groups = {"Unknown"}

            gset = set(groups)
            gset.add("All")

            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


# -------------------------------------------------------------
# n-gram 集計ロジック（found-model 側）
# -------------------------------------------------------------
def ngram_counts_by_group_found(series_by_staff, groups_by_staff, n):
    """
    found-model 側: ext_assigned から作ったシフト列を使って、
    n-gram 出現回数をグループ別にカウントする。
    """
    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    for sidx, series in series_by_staff.items():
        if len(series) < n:
            continue

        for i in range(len(series) - n + 1):
            gram = tuple(series[i:i + n])

            # 無効なシフト記号が含まれていたらスキップ
            if any(s not in VALID_SHIFTS for s in gram):
                continue

            gset = set(groups_by_staff.get(sidx, {"Unknown"}))
            gset.add("All")

            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


def group_sort_key(g):
    if g == "All":
        return (0, "")
    if g == "Unknown":
        return (2, "")
    return (1, g.lower())


# -------------------------------------------------------------
# freq_share / P(next|prefix) を計算
# -------------------------------------------------------------
def compute_freq_share(counter: Counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {gram: c / total for gram, c in counter.items()}


def compute_cond_prob(counter_N: Counter, counter_Nm1: Counter):
    """P(next|prefix) = c(gram) / c(prefix) を gram ごとに計算"""
    cond = {}
    for gram, c in counter_N.items():
        if len(gram) < 2:
            continue
        prefix = gram[:-1]
        base = counter_Nm1.get(prefix, 0)
        if base > 0:
            cond[gram] = c / base
    return cond


# -------------------------------------------------------------
# メイン
# -------------------------------------------------------------
def main():
    if len(sys.argv) != 6:
        print("Usage: python ngram_freq_anly.py past-shifts.lp setting.lp found-model.lp N output.png")
        sys.exit(1)

    past_shifts_file = sys.argv[1]
    setting_file = sys.argv[2]
    found_model_file = sys.argv[3]
    N = int(sys.argv[4])
    output_png = sys.argv[5]

    if not os.path.isfile(past_shifts_file):
        print(f"[ERROR] past-shifts ファイルが見つかりません: {past_shifts_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(setting_file):
        print(f"[ERROR] setting ファイルが見つかりません: {setting_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(found_model_file):
        print(f"[ERROR] found-model ファイルが見つかりません: {found_model_file}", file=sys.stderr)
        sys.exit(1)

    if N < 1:
        print("N は 1 以上を指定してください。", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # 1) past-shifts 側の n-gram 集計
    # ---------------------------------------------------------
    print("# Loading past-shifts...")
    seqs = data_loader.load_past_shifts(past_shifts_file)

    # 期間フィルタ
    before = len(seqs)
    seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
    after = len(seqs)
    print(f"# [past-shifts] date filter: nurses {before} -> {after}")
    if not seqs:
        print("[ERROR] No shifts in specified period (past-shifts). Abort.")
        sys.exit(1)

    group_timeline = data_loader.load_staff_group_timeline(setting_file)

    counters_past_N = ngram_counts_by_group_past(seqs, group_timeline, N)
    counter_past_all = counters_past_N.get("All", Counter())
    if not counter_past_all:
        print('[ERROR] past-shifts 側で group="All" の n-gram が得られませんでした。', file=sys.stderr)
        sys.exit(1)
    freq_past = compute_freq_share(counter_past_all)
    print(f"# past-shifts: {len(counter_past_all)} distinct {N}-grams for group=All")

    # ★ N>=2 のときは N-1gram も取って P(next|prefix) を計算する
    if N >= 2:
        counters_past_Nm1 = ngram_counts_by_group_past(seqs, group_timeline, N - 1)
        counter_past_nm1_all = counters_past_Nm1.get("All", Counter())
        cond_past = compute_cond_prob(counter_past_all, counter_past_nm1_all)
        print(f"# past-shifts: {len(cond_past)} {N}-grams with defined P(next|prefix)")
    else:
        cond_past = {}

    # ---------------------------------------------------------
    # 2) found-model 側の n-gram 集計
    # ---------------------------------------------------------
    print("# Loading found-model...")
    ext_by_staff = defaultdict(list)    # staffIdx -> [(dayIdx, "SHIFT"), ...]
    groups_by_staff = defaultdict(set)  # staffIdx -> {group1, group2, ...}
    staff_info = {}                     # staffIdx -> (name, nurse_id)

    with open(found_model_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = PAT_EXT.match(line)
            if m:
                sidx, didx, shift = int(m[1]), int(m[2]), m[3]
                ext_by_staff[sidx].append((didx, shift))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname, sidx = m[1], int(m[2])
                groups_by_staff[sidx].add(gname)
                continue

            m = PAT_STAFF.match(line)
            if m:
                sidx, name, nurse_id = int(m[1]), m[2], m[3]
                staff_info[sidx] = (name, nurse_id)
                continue

    # dayIdx でソートしてシフト列だけ取り出す
    series_by_staff = {}
    for sidx, pairs in ext_by_staff.items():
        pairs.sort(key=lambda t: t[0])
        series_by_staff[sidx] = [s for _, s in pairs]

    counters_found_N = ngram_counts_by_group_found(series_by_staff, groups_by_staff, N)
    counter_found_all = counters_found_N.get("All", Counter())
    if not counter_found_all:
        print('[WARN] found-model 側で group="All" の n-gram が空です（全て 0 として扱います）。')
    freq_found = compute_freq_share(counter_found_all)
    print(f"# found-model: {len(counter_found_all)} distinct {N}-grams for group=All")

    if N >= 2:
        counters_found_Nm1 = ngram_counts_by_group_found(series_by_staff, groups_by_staff, N - 1)
        counter_found_nm1_all = counters_found_Nm1.get("All", Counter())
        cond_found = compute_cond_prob(counter_found_all, counter_found_nm1_all)
        print(f"# found-model: {len(cond_found)} {N}-grams with defined P(next|prefix)")
    else:
        cond_found = {}

    # ---------------------------------------------------------
    # 3) past-shifts の n-gram を横軸にして freq_share の棒グラフを作成
    # ---------------------------------------------------------
    TOP_K = 10
    grams_sorted = [gram for gram, _ in sorted(
        counter_past_all.items(), key=lambda kv: -kv[1]
    )[:TOP_K]]

    labels = ["-".join(g) for g in grams_sorted]
    x = list(range(len(grams_sorted)))
    width = 0.4

    y_past = [freq_past.get(g, 0.0) for g in grams_sorted]
    y_found = [freq_found.get(g, 0.0) for g in grams_sorted]

    plt.figure(figsize=(max(8, len(grams_sorted) * 0.4), 6))
    plt.bar([xi - width/2 for xi in x], y_past, width=width, label="manual-shifts")
    plt.bar([xi + width/2 for xi in x], y_found, width=width, label="automatic-shifts")

    plt.xticks(x, labels, rotation=90,fontsize=14)
    plt.ylabel("freq_share")
    plt.title(f"N={N} freq_share")
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    print(f"# Figure (freq_share) saved to: {output_png}")

    # ---------------------------------------------------------
    # 4) P(next|prefix) の棒グラフ
    # ---------------------------------------------------------
    if N >= 2:
        y_past_cond = [cond_past.get(g, 0.0) for g in grams_sorted]
        y_found_cond = [cond_found.get(g, 0.0) for g in grams_sorted]

        root, ext = os.path.splitext(output_png)
        if not ext:
            ext = ".png"
        output_cond_png = f"{root}_pnext{ext}"

        plt.figure(figsize=(max(8, len(grams_sorted) * 0.4), 6))
        plt.bar([xi - width/2 for xi in x], y_past_cond, width=width, label="manual-shifts")
        plt.bar([xi + width/2 for xi in x], y_found_cond, width=width, label="automatic-shifts")

        plt.xticks(x, labels, rotation=90,fontsize=14)
        plt.ylabel(r"$P(\mathrm{s_n}\mid s_1, s_2, \dots, s_{n-1})$")
        plt.title("N=3 "+ r"$P(\mathrm{s_n}\mid s_1, s_2, \dots, s_{n-1})$")
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.savefig(output_cond_png)
        plt.close()

        print(f"# Figure (P(next|prefix)) saved to: {output_cond_png}")
    else:
        print("# N=1 のため P(next|prefix) グラフは作成しません。")


if __name__ == "__main__":
    main()
