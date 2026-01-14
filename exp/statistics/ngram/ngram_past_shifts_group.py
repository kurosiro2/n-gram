#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【統合版】past-shifts を読み、n-gram を「看護師グループ別」に集計して表示する。

✅ 対応モード
  1) 単一病棟モード:
       python ngram_past_shifts_group.py <past-shifts.lp> <setting.lp or setting-dir> <N> [csv_path]

  2) 全病棟モード（ディレクトリ入力）:
       python ngram_past_shifts_group.py <past-shifts-dir/> <group-settings-root-dir/> <N> [csv_path]

  - past-shifts-dir は直下の *.lp を列挙
    ファイル名（拡張子除く）を ward 名として扱う
      例: past-shifts/2階西病棟.lp → ward="2階西病棟"
  - group-settings-root-dir は
      group-settings/<ward>/YYYY-MM-DD/setting.lp
    を読む前提で、<ward> ディレクトリを setting_path として data_loader に渡す
    （load_staff_group_timeline はディレクトリ対応してるため）

出力:
  - N=1: 1-gram（勤務記号の割合）
  - N=2: 2-gram の freq_share と P(next|prefix)
  - N>=3: N-gram の freq_share と P(next|prefix)
  - CSV（任意）: group, N, gram, count, freq_share, cond_prob

注意:
  - data_loader は exp/statistics/data_loader.py を想定
  - get_groups_for_date の呼び方は data_loader に合わせている
"""

import sys
import os
import csv
from collections import defaultdict, Counter

# -------------------------------------------------------------
# 親ディレクトリを import パスに追加して data_loader を読めるようにする
#   このファイル: exp/statistics/ngram/ngram_past_shifts_group.py
#   data_loader:   exp/statistics/data_loader.py
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, load_staff_groups, get_groups_for_date


# -------------------------------------------------------------
# ★ 期間指定（ここを書き換えて使う）
#   - None にするとその側の制限なし
# -------------------------------------------------------------
DATE_START = 20240101  # 例: 20250101
DATE_END = 20240631    # 例: 20251231


# -------------------------------------------------------------
# 勤務シフト（8種）+ 休暇シフト（2種）だけを有効とする
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH"
}


# =============================================================
# helpers
# =============================================================
def filter_seqs_by_date(seqs_dict, start_date, end_date):
    """
    seqs_dict: {(nid,name): [(date,shift), ...]}
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


def collect_past_shift_files(past_dir: str):
    """past-shifts ディレクトリ直下の *.lp を列挙"""
    result = []
    for fn in sorted(os.listdir(past_dir)):
        p = os.path.join(past_dir, fn)
        if os.path.isfile(p) and fn.endswith(".lp"):
            result.append(p)
    return result


def merge_group_counters(dst, src):
    """defaultdict(Counter) 同士を加算マージ"""
    for g, c in src.items():
        dst[g].update(c)


def _stable_most_common(counter: Counter):
    """頻度降順・語順安定の並べ替え"""
    return sorted(counter.items(), key=lambda kv: (-kv[1], tuple(kv[0])))


# =============================================================
# n-gram 集計ロジック（あなたの data_loader に完全に合わせる）
# =============================================================
def ngram_counts_by_group(seqs_dict, group_timeline, n):
    """
    n-gram 出現回数をグループ別にカウント（タイムライン対応版）。

    - seqs_dict: {(nurse_id, name): [(date, shift), .]}
      * data_loader.load_past_shifts() の戻り値を想定
    - group_timeline:
        load_staff_group_timeline() の戻り値
        name キー・(name, nurse_id) キーの両方を含みうる
    - n: n-gram の n

    仕様:
      - 各 n-gram について、その n-gram の「最後の日付」を基準に
        その時点でのグループを
          get_groups_for_date(name, date, group_timeline, nurse_id=nid)
        で引く。
      - グループが見つからなければ { "Unknown" } として扱う。
      - すべての n-gram は "All" グループにもカウントする。
      - ★ 勤務シフト8種 + 休暇シフト2種以外のシフトを含む n-gram は無視する。
    """
    from data_loader import get_groups_for_date

    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    for (nid, name), seq in sorted(seqs_dict.items(), key=lambda kv: kv[0][1]):
        if len(seq) < n:
            continue

        for i in range(len(seq) - n + 1):
            window = seq[i: i + n]
            dates = [d for (d, _) in window]
            shifts = [s for (_, s) in window]

            if any(s not in VALID_SHIFTS for s in shifts):
                continue

            ref_date = dates[-1]
            gram = tuple(shifts)

            # ★ここが重要：data_loader のシグネチャ通りに呼ぶ
            groups = get_groups_for_date(name, ref_date, group_timeline, nurse_id=nid)
            if not groups:
                groups = {"Unknown"}

            gset = set(groups)
            gset.add("All")

            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


# =============================================================
# 出力
# =============================================================
def print_unigram_share(group: str, uni_counter: Counter, csv_rows=None):
    total = sum(uni_counter.values())
    print(f'\n----- Group="{group}" | 1-gram（勤務記号の割合） -----')
    if total == 0:
        print("  (no data)")
        return

    for (gram, c) in _stable_most_common(uni_counter):
        s = gram[0]
        share = c / total
        print(f" {c:6d}  {s:<3}   {share*100:6.2f}%")
        if csv_rows is not None:
            csv_rows.append({
                "group": group,
                "N": 1,
                "gram": s,
                "count": c,
                "freq_share": share,
                "cond_prob": "",
            })


def print_bigram_score(group: str, uni_counter: Counter, bi_counter: Counter, csv_rows=None):
    total_bi = sum(bi_counter.values())
    print(f'\n----- Group="{group}" | 2-gram（freq_share と P(next|prefix)） -----')
    print("  freq   prefix -> next         freq_share    P(next|prefix)")
    if total_bi == 0:
        print("  (no data)")
        return

    rows = []
    for gram, c in bi_counter.items():
        prefix = (gram[0],)
        base_prefix = uni_counter.get(prefix, 0)
        if base_prefix <= 0:
            continue
        freq_share = c / total_bi
        cond_prob = c / base_prefix
        rows.append((freq_share, c, gram, cond_prob))

    rows.sort(key=lambda x: (-x[0], -x[1], tuple(x[2])))

    for (freq_share, c, gram, cond_prob) in rows:
        arrow = f"{gram[0]}->{gram[1]}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")
        if csv_rows is not None:
            csv_rows.append({
                "group": group,
                "N": 2,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def print_ngramN_score(group: str, n_counter: Counter, nm1_counter: Counter, N: int, csv_rows=None):
    total_N = sum(n_counter.values())
    print(f'\n----- Group="{group}" | {N}-gram（freq_share と P(next|prefix)） -----')
    print("  freq   prefix -> next         freq_share    P(next|prefix)")
    if total_N == 0:
        print("  (no data)")
        return

    rows = []
    for gram, c in n_counter.items():
        prefix = gram[:-1]
        base_prefix = nm1_counter.get(prefix, 0)
        if base_prefix <= 0:
            continue
        freq_share = c / total_N
        cond_prob = c / base_prefix
        rows.append((freq_share, c, gram, cond_prob))

    rows.sort(key=lambda x: (-x[0], -x[1], tuple(x[2])))

    for (freq_share, c, gram, cond_prob) in rows:
        prefix_str = "-".join(gram[:-1])
        nxt = gram[-1]
        arrow = f"{prefix_str}->{nxt}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")

        if csv_rows is not None:
            csv_rows.append({
                "group": group,
                "N": N,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def group_sort_key(g):
    if g == "All":
        return (0, "")
    if g == "Unknown":
        return (2, "")
    return (1, g.lower())


# =============================================================
# main
# =============================================================
def main():
    if len(sys.argv) < 4:
        print("python ngram_past_shifts_group.py [shift_file_or_dir] [setting_path_or_root] [N] [csv_path(optional)]")
        print("  単一病棟モード: past-shifts.lp と setting.lp(or setting dir) を指定")
        print("  全病棟モード: past-shifts-dir と group-settings-root-dir を指定")
        print(f"  ※ 期間指定: DATE_START={DATE_START}, DATE_END={DATE_END} をスクリプト内で編集")
        sys.exit(1)

    shift_arg = sys.argv[1]
    setting_arg = sys.argv[2]
    N = int(sys.argv[3])
    csv_path = sys.argv[4] if len(sys.argv) >= 5 else None
    csv_rows = []

    N_eff = max(1, N)

    # =========================================================
    # Single ward mode
    # =========================================================
    if os.path.isfile(shift_arg):
        shift_file = shift_arg
        setting_path = setting_arg

        if not os.path.isfile(shift_file):
            print(f"[ERROR] past-shifts ファイルが見つかりません: {shift_file}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(setting_path) and not os.path.isdir(setting_path):
            print(f"[ERROR] setting パスが見つかりません: {setting_path}", file=sys.stderr)
            sys.exit(1)

        seqs = data_loader.load_past_shifts(shift_file)
        before = len(seqs)
        seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
        after = len(seqs)
        print(f"# [Single ward] date filter: nurses {before} -> {after}")
        if not seqs:
            print("# No shifts in specified period. Abort.")
            return

        group_timeline = data_loader.load_staff_group_timeline(setting_path)

        counters_1 = ngram_counts_by_group(seqs, group_timeline, 1)
        counters_2 = ngram_counts_by_group(seqs, group_timeline, 2)
        counters_N = ngram_counts_by_group(seqs, group_timeline, N_eff)
        counters_Nm1 = ngram_counts_by_group(seqs, group_timeline, N_eff - 1) if N_eff >= 2 else {}

        for g in sorted(counters_N.keys(), key=group_sort_key):
            if N_eff == 1:
                print_unigram_share(g, counters_N[g], csv_rows=csv_rows)
            elif N_eff == 2:
                uni = counters_1.get(g, Counter())
                bi = counters_2.get(g, Counter())
                print_bigram_score(g, uni, bi, csv_rows=csv_rows)
            else:
                n_counter = counters_N.get(g, Counter())
                nm1_counter = counters_Nm1.get(g, Counter())
                print_ngramN_score(g, n_counter, nm1_counter, N_eff, csv_rows=csv_rows)

        if csv_path:
            fieldnames = ["group", "N", "gram", "count", "freq_share", "cond_prob"]
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"# CSV written to: {csv_path}")

        return

    # =========================================================
    # All wards mode
    # =========================================================
    if not os.path.isdir(shift_arg):
        print(f"[ERROR] past-shifts-dir が見つかりません: {shift_arg}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(setting_arg):
        print(f"[ERROR] group-settings-root-dir が見つかりません: {setting_arg}", file=sys.stderr)
        sys.exit(1)

    past_dir = shift_arg
    settings_root = setting_arg

    shift_files = collect_past_shift_files(past_dir)
    if not shift_files:
        print(f"[ERROR] No *.lp found under: {past_dir}", file=sys.stderr)
        sys.exit(1)

    # 全病棟を合算したカウンタ
    counters_1_all = defaultdict(Counter)
    counters_2_all = defaultdict(Counter)
    counters_N_all = defaultdict(Counter)
    counters_Nm1_all = defaultdict(Counter) if N_eff >= 2 else {}

    processed = 0
    skipped_no_setting = 0
    skipped_empty = 0

    for spath in shift_files:
        ward = os.path.splitext(os.path.basename(spath))[0]
        ward_setting_dir = os.path.join(settings_root, ward)

        if not os.path.isdir(ward_setting_dir):
            skipped_no_setting += 1
            continue

        seqs = data_loader.load_past_shifts(spath)
        before = len(seqs)
        seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
        after = len(seqs)
        if not seqs:
            skipped_empty += 1
            continue

        # ★重要：ディレクトリを渡す（YYYY-MM-DD/setting.lp を全部読み取って timeline 化）
        group_timeline = data_loader.load_staff_group_timeline(ward_setting_dir)

        c1 = ngram_counts_by_group(seqs, group_timeline, 1)
        c2 = ngram_counts_by_group(seqs, group_timeline, 2)
        cN = ngram_counts_by_group(seqs, group_timeline, N_eff)
        cNm1 = ngram_counts_by_group(seqs, group_timeline, N_eff - 1) if N_eff >= 2 else {}

        merge_group_counters(counters_1_all, c1)
        merge_group_counters(counters_2_all, c2)
        merge_group_counters(counters_N_all, cN)
        if N_eff >= 2:
            merge_group_counters(counters_Nm1_all, cNm1)

        processed += 1

    print(f"# [All wards] processed={processed}, skipped_no_setting={skipped_no_setting}, skipped_empty={skipped_empty}")
    if processed == 0:
        print("# No wards processed. (ward名が一致しているか確認して)", file=sys.stderr)
        sys.exit(1)

    # 全病棟合算の結果を表示
    for g in sorted(counters_N_all.keys(), key=group_sort_key):
        if N_eff == 1:
            print_unigram_share(g, counters_N_all[g], csv_rows=csv_rows)
        elif N_eff == 2:
            uni = counters_1_all.get(g, Counter())
            bi = counters_2_all.get(g, Counter())
            print_bigram_score(g, uni, bi, csv_rows=csv_rows)
        else:
            n_counter = counters_N_all.get(g, Counter())
            nm1_counter = counters_Nm1_all.get(g, Counter())
            print_ngramN_score(g, n_counter, nm1_counter, N_eff, csv_rows=csv_rows)

    if csv_path:
        fieldnames = ["group", "N", "gram", "count", "freq_share", "cond_prob"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"# CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
