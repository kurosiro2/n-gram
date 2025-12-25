#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
過去勤務シフト (past-shifts.lp) を読み込み、
看護師グループごとに n-gram 出現回数をカウントするスクリプト。

使い方:

  # 単一病棟モード（従来どおり）
  python ngram_past_shifts_group.py past-shifts.lp setting.lp N

  # 全病棟モード（ディレクトリ × ディレクトリ）
  #   past-shifts-dir 配下の *.lp を病棟ごとに処理し、
  #   group-settings-root からは data_loader_all.load_all_staff_group_timelines()
  #   を使って全病棟のグループタイムラインを読む。
  python ngram_past_shifts_group.py past-shifts-dir group-settings-root N

依存:
  - data_loader.load_past_shifts()
  - data_loader.load_staff_group_timeline()
  - data_loader_all.load_all_staff_group_timelines()

インターフェース:
  - 他のスクリプトからは
        from ngram_past_shifts_group import ngram_counts_by_group
    として利用できる。

  - ngram_counts_by_group(seqs_dict, group_timeline, n) は
    グループ名 -> Counter({ ngram(tuple[str]): count, ... }) を返す。

  - ここではグループ "All" も特別に作って、
    全グループ合算のカウンタを提供する。
"""

import sys
import os
from collections import defaultdict, Counter

# -------------------------------------------------------------
# 親ディレクトリを import パスに追加して data_loader 系を読めるように
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader       # load_past_shifts, load_staff_group_timeline
import data_loader_all   # load_all_staff_group_timelines


# -------------------------------------------------------------
# ユーティリティ: 指定した日付のグループ集合をタイムラインから取得
# -------------------------------------------------------------
def get_groups_for_date(timeline, date):
    """
    timeline: [(start_date, set(groups)), ...] が start_date 昇順で入っていることを想定。
    date: int (YYYYMMDD など)

    return: set(groups) もしくは空集合
    """
    if not timeline:
        return set()

    last_groups = set()
    for start_date, groups in timeline:
        if date < start_date:
            break
        last_groups = set(groups)
    return last_groups


# -------------------------------------------------------------
# n-gram 出現回数をグループ別にカウント（タイムライン対応版）
# -------------------------------------------------------------
def ngram_counts_by_group(seqs_dict, group_timeline, n):
    """
    n-gram 出現回数をグループ別にカウントする。

    Parameters
    ----------
    seqs_dict : dict
        {(nurse_id, name): [(date:int, shift:str), ...]}
        data_loader.load_past_shifts() の戻り値を想定。

    group_timeline : dict
        看護師ごとのグループ変遷タイムライン。
        data_loader.load_staff_group_timeline(setting.lp) あるいは
        data_loader_all.load_all_staff_group_timelines(root)[ward_name]
        の内側の dict を想定。
        典型的には
            { name: [(start_date:int, set(groups)), ...], ... }
        の形。

    n : int
        n-gram の n (>=1)。

    Returns
    -------
    dict[str, Counter]
        グループ名 -> Counter({ ngram(tuple[str]): count, ... })
        特別に "All" グループも含める。
    """
    counters = defaultdict(Counter)

    for (nurse_id, name), seq in seqs_dict.items():
        if not seq:
            continue

        # seq は [(date, shift), ...] を日付順にソート
        seq_sorted = sorted(seq, key=lambda t: t[0])
        dates = [d for d, _ in seq_sorted]
        shifts = [s for _, s in seq_sorted]

        # 名前（name）でタイムラインを引く前提
        timeline = group_timeline.get(name, [])

        # 日ごとのグループ集合を作る
        groups_each_day = [get_groups_for_date(timeline, d) for d in dates]

        # セグメント分割: グループ集合が変わるごとに系列を切る
        start_idx = 0
        while start_idx < len(shifts):
            groups0 = groups_each_day[start_idx]
            # グループ情報がなければ Unknown にする
            if not groups0:
                groups0 = {"Unknown"}

            end_idx = start_idx + 1
            while end_idx < len(shifts):
                g_next = groups_each_day[end_idx]
                if not g_next:
                    g_next = {"Unknown"}
                if g_next != groups0:
                    break
                end_idx += 1

            # [start_idx, end_idx) が同一グループ集合のセグメント
            seg_shifts = shifts[start_idx:end_idx]

            # n-gram をカウント
            if len(seg_shifts) >= n:
                for i in range(len(seg_shifts) - n + 1):
                    gram = tuple(seg_shifts[i:i + n])

                    # "All" グループ
                    counters["All"][gram] += 1

                    # 実グループ
                    for g in groups0:
                        counters[g][gram] += 1

            start_idx = end_idx

    return counters


# -------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  # 単一病棟モード（従来どおり）")
        print("  python ngram_past_shifts_group.py past-shifts.lp setting.lp N")
        print("")
        print("  # 全病棟モード（ディレクトリ × ディレクトリ）")
        print("  python ngram_past_shifts_group.py past-shifts-dir group-settings-root N")
        sys.exit(1)

    past_arg = sys.argv[1]     # ファイル or ディレクトリ
    setting_arg = sys.argv[2]  # setting.lp or group-settings root dir
    N = int(sys.argv[3])

    if N <= 0:
        print("N は 1 以上を指定してください。", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------
    # 1) ディレクトリ指定 → 全病棟モード
    # -------------------------------------------------
    if os.path.isdir(past_arg):
        past_shifts_dir = past_arg
        settings_root = setting_arg

        if not os.path.isdir(settings_root):
            print(f"[ERROR] settings_root として指定したパスがディレクトリではありません: {settings_root}",
                  file=sys.stderr)
            sys.exit(1)

        # 全病棟のグループタイムラインを読む
        # all_timelines: { ward_name: { name: [(start_date, set(groups)), ...] } }
        all_timelines = data_loader_all.load_all_staff_group_timelines(settings_root)
        print("# Wards found in settings_root:", ", ".join(sorted(all_timelines.keys())))

        # past-shifts ディレクトリ内の .lp を全部拾う
        ward_shift_files = {}
        for fname in sorted(os.listdir(past_shifts_dir)):
            if not fname.endswith(".lp"):
                continue
            ward_name = os.path.splitext(fname)[0]
            ward_shift_files[ward_name] = os.path.join(past_shifts_dir, fname)

        if not ward_shift_files:
            print(f"[ERROR] past-shifts-dir 内に .lp ファイルが見つかりません: {past_shifts_dir}",
                  file=sys.stderr)
            sys.exit(1)

        print("# Wards found in past_shifts_dir:", ", ".join(sorted(ward_shift_files.keys())))

        # 病棟ごとに処理
        for ward_name, shift_path in sorted(ward_shift_files.items()):
            if ward_name not in all_timelines:
                print(f"# [WARN] ward={ward_name} に対応する group-settings が見つからないのでスキップ")
                continue

            if not os.path.isfile(shift_path):
                print(f"# [WARN] shift_file not found for ward={ward_name}: {shift_path}")
                continue

            print(f"# --- Ward={ward_name} ---")
            seqs = data_loader.load_past_shifts(shift_path)
            group_timeline = all_timelines[ward_name]

            counters_by_group = ngram_counts_by_group(seqs, group_timeline, N)

            # 出力
            _print_ngram_counts(counters_by_group, N, header_prefix=f'Ward="{ward_name}" ')

        return

    # -------------------------------------------------
    # 2) 単一ファイル指定 → 従来モード
    # -------------------------------------------------
    shift_file = past_arg
    setting_file = setting_arg

    if not os.path.isfile(shift_file):
        print(f"[ERROR] past-shifts ファイルが見つかりません: {shift_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(setting_file):
        print(f"[ERROR] setting ファイルが見つかりません: {setting_file}", file=sys.stderr)
        sys.exit(1)

    # 1病棟分のシフトとタイムラインを読み込み
    seqs = data_loader.load_past_shifts(shift_file)
    group_timeline = data_loader.load_staff_group_timeline(setting_file)

    counters_by_group = ngram_counts_by_group(seqs, group_timeline, N)

    # 出力
    _print_ngram_counts(counters_by_group, N, header_prefix="")


# -------------------------------------------------------------
# 出力用ヘルパ
# -------------------------------------------------------------
def _print_ngram_counts(counters_by_group, N, header_prefix=""):
    """
    counters_by_group: { group_name: Counter({ ngram(tuple[str]): count, ... }) }
    N: n-gram の長さ
    header_prefix: "Ward=\"GCU\" " など、見出しに付けるプレフィックス
    """
    for g in sorted(counters_by_group.keys()):
        counter = counters_by_group[g]
        if not counter:
            continue

        if N == 1:
            # 1-gram のときは割合も出す
            total = sum(counter.values())
            print(f"\n----- {header_prefix}Group=\"{g}\" | 1-gram（勤務記号の割合） -----")
            # 1-gram は gram = (shift,) を想定
            for gram, c in counter.most_common():
                shift = gram[0] if isinstance(gram, tuple) and len(gram) == 1 else str(gram)
                ratio = (c / total * 100.0) if total > 0 else 0.0
                print(f"{c:7d}  {shift:<4}  {ratio:6.2f}%")
        else:
            # N>=2 のときは単純に頻度だけ出す
            print(f"\n----- {header_prefix}Group=\"{g}\" | {N}-gram -----")
            for gram, c in counter.most_common():
                if isinstance(gram, tuple):
                    s = "-".join(gram)
                else:
                    s = str(gram)
                print(f"{c:7d}  {s}")


if __name__ == "__main__":
    main()
