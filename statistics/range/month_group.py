#!/usr/bin/env python3
# 使い方:
#   PYTHONPATH=. python exp/statistics/range/month_group.py \
#       exp/2019-2025-data/real-name/past-shifts/GCU.lp \
#       exp/2019-2025-data/real-name/group-settings/GCU
#
# past_shifts と setting の紐づけは「名前のみ」で行う。
# setting はディレクトリを渡すと YYYY-MM-DD/setting.lp を時系列で読み取り、
# グループ変更にも対応する（data_loader 側で処理）。

import sys
import os
from datetime import date
from collections import defaultdict
from statistics import mean, pstdev

# -------------------------------------------------------------
# モジュールパス調整（exp/statistics/ から data_loader を読めるようにする）
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_loader import (
    load_past_shifts,           # past-shifts.lp → {(nid,name): [(yyyymmdd,shift),...]}
    load_staff_group_timeline,  # setting_dir/file → { name: [(start_date, groups), ...] }
    get_groups_for_date,        # (name, yyyymmdd_int, timeline) → set(groups)
)

# ==============================================================
# 月操作ユーティリティ
# ==============================================================

def first_day_of_month(d: date) -> date:
    return d.replace(day=1)

def add_months(d: date, k: int) -> date:
    """月加算（年繰り上がり対応）"""
    y = d.year + (d.month - 1 + k) // 12
    m = (d.month - 1 + k) % 12 + 1
    return date(y, m, 1)

def month_range_inclusive(start_m: date, end_m: date):
    """start_m, end_m を含む月初列を返す"""
    months = []
    cur = first_day_of_month(start_m)
    end = first_day_of_month(end_m)
    while cur <= end:
        months.append(cur)
        cur = add_months(cur, 1)
    return months

# ==============================================================
# 1) past-shifts から「人×月×シフト」のカウントを作る
# ==============================================================

def build_month_counts_from_seqs(seqs):
    """
    seqs: load_past_shifts() の戻り値
        {(nurse_id, name): [(date_int, shift), ...]}

    戻り値:
        per_person_month_counts: {(nid,name): {month_firstday(date): {shift: count}}}
        all_shifts: set of shift codes
        all_people: set of (nid,name)
        min_month, max_month: date オブジェクト（全体の最小・最大月） or None
    """
    per_person_month_counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    all_shifts = set()
    all_people = set()
    min_month = None
    max_month = None

    for (nid, name), records in seqs.items():
        all_people.add((nid, name))
        for ymd_int, shift in records:
            # yyyymmdd int → date
            y = ymd_int // 10000
            m = (ymd_int // 100) % 100
            d = ymd_int % 100
            dt = date(y, m, d)
            m0 = first_day_of_month(dt)

            per_person_month_counts[(nid, name)][m0][shift] += 1
            all_shifts.add(shift)

            if min_month is None or m0 < min_month:
                min_month = m0
            if max_month is None or m0 > max_month:
                max_month = m0

    return per_person_month_counts, all_shifts, all_people, min_month, max_month

# ==============================================================
# 2) タイムライン付きグループ情報を使って、グループ別に月集計
# ==============================================================

def compute_group_month_series(
    per_person_month_counts,
    all_shifts,
    group_timeline,
    propagate_unknown_to="Unknown",
):
    """
    タイムライン group_timeline を使って、月ごとに「どのグループに所属していたか」
    に応じてカウントを振り分ける。

    戻り値:
      group_person_shift_series:
        {group: { person: {shift: [month_count, month_count, ...] } } }

      group_members:
        {group: set(person)}  # 1回でもそのグループに属した人
    """
    group_person_shift_series = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    group_members = defaultdict(set)

    for (nid, name), month_map in per_person_month_counts.items():
        if not month_map:
            continue

        # この人の活動期間（最初の月〜最後の月）
        first_m = min(month_map.keys())
        last_m  = max(month_map.keys())
        months_span = month_range_inclusive(first_m, last_m)

        for m0 in months_span:
            ymd_int = m0.year * 10000 + m0.month * 100 + m0.day  # 月初日を代表日とする

            # この日付時点でのグループをタイムラインから取得（名前で引く）
            groups_raw = get_groups_for_date(name, ymd_int, group_timeline)

            if groups_raw:
                groups_for_stats = set(groups_raw)
                groups_for_stats.add("All")
            else:
                # setting に出てこない or 日付外 → Unknown と All 扱い
                groups_for_stats = {"All", propagate_unknown_to or "Unknown"}

            # この月の各シフト回数（無いシフトは 0）
            counts_this_month = month_map.get(m0, {})
            for g in groups_for_stats:
                group_members[g].add((nid, name))
                for s in all_shifts:
                    c = counts_this_month.get(s, 0)
                    group_person_shift_series[g][(nid, name)][s].append(c)

    return group_person_shift_series, group_members

def aggregate_group_month_stats(group_person_shift_series, group_members):
    """
    group_person_shift_series から、1人あたり月平均などを計算する。

    戻り値:
      stats_by_group[group] = {
        "per_person_means": {shift: [ per-person mean over months ]},
        "per_person_mins":  {shift: [ per-person min over months ]},
        "per_person_maxs":  {shift: [ per-person max over months ]},
      }
    """
    stats_by_group = {}

    for g, person_series_map in group_person_shift_series.items():
        per_person_means = defaultdict(list)
        per_person_mins  = defaultdict(list)
        per_person_maxs  = defaultdict(list)

        for person, shift_series_map in person_series_map.items():
            for s, series in shift_series_map.items():
                if not series:
                    continue
                avg = sum(series) / len(series)
                per_person_means[s].append(avg)
                per_person_mins[s].append(int(min(series)))
                per_person_maxs[s].append(int(max(series)))

        stats_by_group[g] = {
            "per_person_means": dict(per_person_means),
            "per_person_mins":  dict(per_person_mins),
            "per_person_maxs":  dict(per_person_maxs),
        }

    return stats_by_group

# ==============================================================
# 3) 出力
# ==============================================================

def print_group_table(group_name, stats, n_people_in_group):
    per_person_means = stats["per_person_means"]
    per_person_mins  = stats["per_person_mins"]
    per_person_maxs  = stats["per_person_maxs"]

    if not per_person_means:
        return

    print(f'\n===== Group="{group_name}" | Monthly per-person averages (timeline-aware) =====')
    print(f"Total unique staff (in group): {n_people_in_group}")
    print("shift      mean(shift)   sd(person)    min(shift)    max(shift) share_avg")

    # share_avg 用に v=mean(person) をまず計算
    mean_by_shift = {s: (mean(v) if v else 0.0) for s, v in per_person_means.items()}
    denom = sum(mean_by_shift.values()) if mean_by_shift else 0.0

    rows = []
    for s in sorted(per_person_means.keys()):
        vals = per_person_means[s]
        mval = mean(vals) if vals else 0.0
        sdv  = pstdev(vals) if len(vals) > 1 else 0.0

        mins = per_person_mins.get(s, [])
        maxs = per_person_maxs.get(s, [])
        mn   = float(min(mins)) if mins else 0.0
        mx   = float(max(maxs)) if maxs else 0.0

        share = (mval / denom) if denom > 0 else 0.0
        rows.append((s, mval, sdv, mn, mx, share))

    # mean 降順 → shift 昇順
    rows.sort(key=lambda x: (-x[1], x[0]))
    for s, mval, sdv, mn, mx, share in rows:
        print(f"{s:<6}  {mval:12.3f}  {sdv:11.3f}  {mn:11.0f}  {mx:11.0f}  {share:9.3f}")

# ==============================================================
# 4) main
# ==============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python exp/statistics/range/month_group.py <past_shifts.lp> <setting_dir_or_file>")
        sys.exit(1)

    shift_file   = sys.argv[1]
    setting_path = sys.argv[2]

    # 1) past_shifts 読み込み（共通 loader）
    seqs = load_past_shifts(shift_file)
    if not seqs:
        print("No shift_data in file.")
        return

    per_person_month_counts, all_shifts, all_people, min_month, max_month = \
        build_month_counts_from_seqs(seqs)

    if min_month is None:
        print("No monthly data.")
        return

    # 2) setting タイムライン読み込み（名前ベース）
    #    propagate_past=True にすると「最初に登場したグループを過去にも遡って適用」する。
    propagate_past = True  # ← 必要に応じてここだけ True/False を切り替える
    group_timeline = load_staff_group_timeline(setting_path, propagate_past=propagate_past)

    # 3) タイムラインに従って、グループ別の人×月×シフト系列を構築
    group_person_shift_series, group_members = compute_group_month_series(
        per_person_month_counts,
        all_shifts,
        group_timeline,
        propagate_unknown_to="Unknown",  # Unknown を使いたくなければ None でもOK
    )

    # 4) グループ別に per-person 指標を計算
    stats_by_group = aggregate_group_month_stats(group_person_shift_series, group_members)

    # 5) 出力順: All → その他昇順 → Unknown
    all_groups = set(stats_by_group.keys())
    if "All" not in all_groups:
        all_groups.add("All")
    if "Unknown" not in all_groups:
        all_groups.add("Unknown")

    group_order = sorted(
        all_groups,
        key=lambda g: (0, "") if g == "All" else (2, "") if g == "Unknown" else (1, g.lower())
    )

    for g in group_order:
        if g not in stats_by_group:
            continue
        n_people = len(group_members.get(g, set()))
        if n_people == 0:
            continue
        print_group_table(g, stats_by_group[g], n_people)

if __name__ == "__main__":
    main()
