# 使い方: python exp/range-evaluate/day.py <past_shifts.lp>

import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean, pstdev

PAT = re.compile(r'^shift_data\("([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d{8})\s*,\s*"([^"]+)"\)\.')
WD_LABEL = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def monday_of(d):
    """指定日を含む週の月曜日を返す"""
    return d - timedelta(days=d.weekday())

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp/range-evaluate/daily.py <past_shifts.lp>")
        sys.exit(1)
    infile = sys.argv[1]

    # --- 重複排除（同一人×同一日：最初勝ち） ---
    seen = {}
    with open(infile, "r", encoding="utf-8") as f:
        for raw in f:
            m = PAT.match(raw.strip())
            if not m:
                continue
            nurse_id, _name, ymd_str, shift = m.group(1), m.group(2), m.group(3), m.group(4)
            ymd = int(ymd_str)
            key = (nurse_id, ymd)
            if key not in seen:
                seen[key] = shift

    # --- 日別カウント & 在籍 ---
    day_counts = defaultdict(lambda: defaultdict(int))  # date -> shift -> count
    day_staffs = defaultdict(set)                       # date -> set(nurse_id)
    all_shifts, all_days, all_staff = set(), set(), set()

    for (nurse_id, ymd), shift in seen.items():
        d = datetime.strptime(str(ymd), "%Y%m%d").date()
        day_counts[d][shift] += 1
        day_staffs[d].add(nurse_id)
        all_shifts.add(shift)
        all_days.add(d)
        all_staff.add(nurse_id)

    if not all_days:
        print("No data.")
        return

    # --- 日付範囲をゼロ埋め ---
    min_day, max_day = min(all_days), max(all_days)
    days = []
    cur = min_day
    while cur <= max_day:
        days.append(cur)
        cur += timedelta(days=1)

    # --- 週ごとの在籍人数平均（参考） ---
    week_people = defaultdict(set)
    for d in days:
        w = monday_of(d)
        for nid in day_staffs.get(d, ()):
            week_people[w].add(nid)
    avg_staff_per_week = mean(len(s) for s in week_people.values()) if week_people else 0.0

    # --- 曜日ごとの平均在籍人数 ---
    ppl_by_wd = [[] for _ in range(7)]
    for d in days:
        wd = d.weekday()
        ppl_by_wd[wd].append(len(day_staffs.get(d, ())))
    avg_staff_per_weekday = [mean(v) if v else 0.0 for v in ppl_by_wd]

    print("----- Daily stats by Weekday  -----")
    print(f"Total calendar days: {len(days)}")
    print(f"Total staff (unique): {len(all_staff)}")
    print(f"Avg staff per week: {avg_staff_per_week:.2f}")

    # --- 曜日ごとの統計 ---
    for wd in range(7):
        totals = defaultdict(list)  # shift -> 人数カウント
        for d in days:
            if d.weekday() != wd:
                continue
            counts = day_counts[d]
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
            mx = int(max(ts)) if ts else 0
            share = (m_onday / avg_staff_on_day) if avg_staff_on_day > 0 else 0.0
            rows.append((shift, m_onday, s_onday, mx, share))

        
        rows.sort(key=lambda x: (-x[1], x[0]))

        print(f"\n[{WD_LABEL[wd]}]  Avg staff on {WD_LABEL[wd]}: {avg_staff_on_day:.2f}")
        print("shift   avg_staff  sd_staff   max_staff   share_avg")
        for shift, m_onday, s_onday, mx, share in rows:
            print(f"{shift:<6}  {m_onday:11.2f}  {s_onday:8.2f}  {mx:11d}   {share:14.4f}")

if __name__ == "__main__":
    main()
