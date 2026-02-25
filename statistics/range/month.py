# 使い方: python exp/range-evaluate/month.py <past_shifts.lp>
# 指標についてはweek.pyと同じ

import re
import sys
from datetime import date, datetime
from collections import defaultdict
from statistics import mean, pstdev

PAT = re.compile(r'^shift_data\("([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d{8})\s*,\s*"([^"]+)"\)\.')

def first_day_of_month(d: date) -> date:
    return d.replace(day=1)

def add_months(d: date, k: int) -> date:
    # 月加算（年繰り上がり対応）
    y = d.year + (d.month - 1 + k) // 12
    m = (d.month - 1 + k) % 12 + 1
    return date(y, m, 1)

def month_range_inclusive(start_m: date, end_m: date):
    months = []
    cur = first_day_of_month(start_m)
    end = first_day_of_month(end_m)
    while cur <= end:
        months.append(cur)
        cur = add_months(cur, 1)
    return months

def percentile_sorted(vals_sorted, p):
    
    if not vals_sorted:
        return 0.0
    k = (p/100) * len(vals_sorted)
    idx = int(k) - 1 if k.is_integer() else int(k)
    idx = max(0, min(idx, len(vals_sorted)-1))
    return float(vals_sorted[idx])

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp/range-evaluate/month.py <past_shifts.lp>")
        sys.exit(1)
    infile = sys.argv[1]

    # person_key = (nurse_id, name)
    # per_person_month_counts[person_key][month_firstday][shift] = count
    per_person_month_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    all_shifts = set()
    all_people = set()
    min_month = None
    max_month = None

    # 読み込み
    with open(infile, "r", encoding="utf-8") as f:
        for raw in f:
            m = PAT.match(raw.strip())
            if not m:
                continue
            nurse_id, name, ymd, shift = m.group(1), m.group(2), m.group(3), m.group(4)
            d = datetime.strptime(ymd, "%Y%m%d").date()
            m0 = first_day_of_month(d)

            person = (nurse_id, name)
            per_person_month_counts[person][m0][shift] += 1
            all_people.add(person)
            all_shifts.add(shift)

            if min_month is None or m0 < min_month:
                min_month = m0
            if max_month is None or m0 > max_month:
                max_month = m0

    if min_month is None:
        print("No data.")
        return

    # 各人の活動期間（初登場月～最終登場月）を抽出
    active_span_by_person = {}
    for person, month_map in per_person_month_counts.items():
        first_m = min(month_map.keys())
        last_m  = max(month_map.keys())
        active_span_by_person[person] = (first_m, last_m)

    # 各シフトについて「人ごとの月平均（活動期間ゼロ埋め）」を作る
    # per_person_means[shift] = [ personごとの {活動期間平均/月} のリスト ]
    per_person_means = defaultdict(list)

    for person, (first_m, last_m) in active_span_by_person.items():
        months = month_range_inclusive(first_m, last_m)
        month_counts = per_person_month_counts[person]

        # すべてのシフトを対象（出現ゼロでも0で埋める）
        for s in all_shifts:
            series = [month_counts.get(m, {}).get(s, 0) for m in months]
            # 活動期間の「月平均」
            avg_per_month = mean(series) if series else 0.0
            per_person_means[s].append(avg_per_month)

    # 出力
    print("----- Monthly per-person averages from past  -----")
    print(f"Total unique staff : {len(all_people)}")
    print("shift   mean(person)   sd(person)    p25     p50     p75     max(person)")

    rows = []
    for s in sorted(all_shifts):
        vals = per_person_means.get(s, [])
        m  = mean(vals) if vals else 0.0
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        vs = sorted(vals)
        p25 = percentile_sorted(vs, 25)
        p50 = percentile_sorted(vs, 50)
        p75 = percentile_sorted(vs, 75)
        mx  = max(vals) if vals else 0.0
        rows.append((s, m, sd, p25, p50, p75, mx))

    rows.sort(key=lambda x: (-x[1], x[0]))

    for s, m, sd, p25, p50, p75, mx in rows:
        print(f"{s:<6}  {m:12.3f}  {sd:11.3f}  {p25:6.1f}  {p50:6.1f}  {p75:6.1f}  {mx:11.1f}")

if __name__ == "__main__":
    main()
