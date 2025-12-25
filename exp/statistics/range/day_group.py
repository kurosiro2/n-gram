#!/usr/bin/env python3
# 使い方:
#   PYTHONPATH=. python exp/statistics/range/day_group.py \
#       exp/2019-2025-data/real-name/past-shifts/GCU.lp \
#       exp/2019-2025-data/real-name/group-settings/GCU
#
# past_shifts と setting の紐づけは「名前のみ」で行う。
# setting はディレクトリを渡すと YYYY-MM-DD/setting.lp を時系列で読み取り、
# グループ変更にも対応する（data_loader 側で処理）。

import sys
import os
from datetime import datetime, timedelta, date
from collections import defaultdict
from statistics import mean, pstdev

# パス調整（exp/statistics 直下の data_loader を import できるように）
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_loader import (
    load_past_shifts,           # past-shifts.lp → {(nid,name): [(yyyymmdd,shift),...]}
    load_staff_group_timeline,  # setting_dir/file → { name: [(start_date, groups), ...] }
    get_groups_for_date,        # (name, yyyymmdd_int, timeline) → set(groups)
)

WD_LABEL = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def monday_of(d: date):
    """指定日を含む週の月曜日を返す"""
    return d - timedelta(days=d.weekday())


def main():
    if len(sys.argv) < 3:
        print("Usage: python exp/statistics/range/day_group.py <past_shifts.lp> <setting_dir_or_file>")
        sys.exit(1)

    shift_file   = sys.argv[1]
    setting_path = sys.argv[2]

    # ---------------------------------------------------------
    # 1) past_shifts 読み込み（共通 loader）
    #    seqs: {(nid, name): [(date_int, shift), ...]}
    # ---------------------------------------------------------
    seqs = load_past_shifts(shift_file)
    if not seqs:
        print("No shift_data in file.")
        return

    # 同一人×同一日が複数あった場合は「最初勝ち」とする（元コードの挙動を踏襲）
    seen = {}           # key = (nid, name, ymd_int) -> shift
    all_people = set()  # (nid, name)
    all_shifts = set()

    for (nid, name), records in seqs.items():
        for ymd_int, shift in records:
            key = (nid, name, ymd_int)
            if key in seen:
                continue
            seen[key] = shift
            all_people.add((nid, name))
            all_shifts.add(shift)

    if not seen:
        print("No data after dedup.")
        return

    # ---------------------------------------------------------
    # 2) setting タイムライン読み込み（名前ベース）
    #    propagate_past=True:
    #      最初に登場したグループを、過去の勤務日にも遡って適用
    # ---------------------------------------------------------
    propagate_past = True  # 必要に応じて True/False 手動で切り替え
    group_timeline = load_staff_group_timeline(setting_path, propagate_past=propagate_past)

    # ---------------------------------------------------------
    # 3) グループ別 日別カウント & 在籍セット（タイムライン反映）
    # ---------------------------------------------------------
    # day_counts_by_group[g][date][shift] = count
    # day_staffs_by_group[g][date] = set(nurse_id)
    day_counts_by_group = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    day_staffs_by_group = defaultdict(lambda: defaultdict(set))
    all_days_by_group   = defaultdict(set)
    all_staff_by_group  = defaultdict(set)

    for (nid, name, ymd_int), shift in seen.items():
        # 日付オブジェクトに変換
        d_str = str(ymd_int)
        d = datetime.strptime(d_str, "%Y%m%d").date()

        # この日のこの人のグループをタイムラインから取得（name で引く）
        groups_raw = get_groups_for_date(name, ymd_int, group_timeline)
        if groups_raw:
            groups_for_stats = set(groups_raw)
            groups_for_stats.add("All")
        else:
            # setting に出てこない or 日付外 → Unknown と All 扱い
            groups_for_stats = {"All", "Unknown"}

        for g in groups_for_stats:
            day_counts_by_group[g][d][shift] += 1
            day_staffs_by_group[g][d].add(nid)
            all_days_by_group[g].add(d)
            all_staff_by_group[g].add(nid)

    # ---------------------------------------------------------
    # 4) グループごとに統計を計算して出力
    #    出力順: All → その他昇順 → Unknown
    # ---------------------------------------------------------
    all_groups = set(all_days_by_group.keys())
    if "All" not in all_groups:
        all_groups.add("All")
    if "Unknown" not in all_groups:
        all_groups.add("Unknown")

    group_order = sorted(
        all_groups,
        key=lambda g: (0, "") if g == "All" else (2, "") if g == "Unknown" else (1, g.lower())
    )

    for g in group_order:
        if not all_days_by_group[g]:
            continue

        # 日付範囲（グループ内での最小〜最大）をゼロ埋め
        min_day, max_day = min(all_days_by_group[g]), max(all_days_by_group[g])
        days = []
        cur = min_day
        while cur <= max_day:
            days.append(cur)
            cur += timedelta(days=1)

        # 週ごとの在籍人数平均（グループ内）
        week_people = defaultdict(set)
        for d in days:
            w = monday_of(d)
            for nid in day_staffs_by_group[g].get(d, ()):
                week_people[w].add(nid)
        avg_staff_per_week = mean(len(s) for s in week_people.values()) if week_people else 0.0

        # 曜日ごとの平均在籍人数（グループ内）
        ppl_by_wd = [[] for _ in range(7)]
        for d in days:
            wd = d.weekday()
            ppl_by_wd[wd].append(len(day_staffs_by_group[g].get(d, ())))
        avg_staff_per_weekday = [mean(v) if v else 0.0 for v in ppl_by_wd]

        print(f'\n===== Group="{g}" | Day stats by Weekday (timeline-aware) =====')
        print(f"Total calendar days: {len(days)}")
        print(f"Total staff (in group): {len(all_staff_by_group[g])}")
        print(f"Avg staff per week: {avg_staff_per_week:.2f}")

        # 曜日ごとの統計
        for wd in range(7):
            totals = defaultdict(list)  # shift -> 人数カウント系列
            for d in days:
                if d.weekday() != wd:
                    continue
                counts = day_counts_by_group[g].get(d, {})
                for shift in all_shifts:
                    totals[shift].append(counts.get(shift, 0))

            rows = []
            avg_staff_on_day = avg_staff_per_weekday[wd]
            for shift in sorted(all_shifts):
                ts = totals[shift]
                if not ts:
                    continue
                m_onday = mean(ts)
                s_onday = pstdev(ts) if len(ts) > 1 else 0.0
                mn = int(min(ts)) if ts else 0
                mx = int(max(ts)) if ts else 0
                share = (m_onday / avg_staff_on_day) if avg_staff_on_day > 0 else 0.0
                rows.append((shift, m_onday, s_onday, mn, mx, share))

            # 並び順：mean 降順 → shift 名昇順
            rows.sort(key=lambda x: (-x[1], x[0]))

            print(f"\n[{WD_LABEL[wd]}]  Avg staff on {WD_LABEL[wd]}: {avg_staff_on_day:.2f}")
            print("shift    mean(staff)  sd(weekday)    min(staff)    max(staff)     share_avg")
            for shift, m_onday, s_onday, mn, mx, share in rows:
                print(f"{shift:<6}  {m_onday:11.2f}  {s_onday:8.2f}  {mn:11d}  {mx:11d}   {share:14.4f}")


if __name__ == "__main__":
    main()
