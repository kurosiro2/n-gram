
# 使い方: python exp/range-evaluate/week.py <past_shifts.lp>
# 指標について
"""
| 指標               | 意味                       | 計算の仕方                    | 直感的な意味                           |
| ---------------- | ------------------------ | ------------------------ | -------------------------------- |
| **mean(person)** | 各人の週平均値の平均               | `sum(vals) / len(vals)`  | 「1人あたり、週に平均して何回そのシフトをしていたか」      |
| **sd(person)**   | 標準偏差（Standard Deviation） | `pstdev(vals)` （母集団標準偏差） | 人ごとのばらつきの大きさ。数値が大きいほど、スタッフ間で差がある |
| **p25**          | 第1四分位（25th percentile）   | 昇順に並べて下位25%の位置           | 下から25%の人がこの回数以下しかそのシフトをしていない     |
| **p50**          | 中央値（50th percentile）     | 昇順で真ん中の値                 | 全スタッフの真ん中の人がこのくらいの週平均            |
| **p75**          | 第3四分位（75th percentile）   | 上位25%の境界                 | 上位25%の人はこの回数以上シフトしている            |
| **max(person)**  | 最大値                      | `max(vals)`              | 週平均でもっとも多くそのシフトに入っていた人の値         |
| **n_people**     | サンプル数                    | `len(vals)`              | このシフトが1回でもあった人数                  |

"""



import re
import sys
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
from statistics import mean, pstdev

PAT = re.compile(r'^shift_data\("([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d{8})\s*,\s*"([^"]+)"\)\.')

def monday_of(d: date) -> date:
    """その日の属する週の月曜（日付）を返す"""
    return d - timedelta(days=d.weekday())

def week_range(monday_start: date, monday_end: date):
   
    w = monday_start
    while w <= monday_end:
        yield w
        w += timedelta(days=7)

def percentile_sorted(vals_sorted, p):
 
    if not vals_sorted:
        return 0.0
    k = (p/100) * len(vals_sorted)
    idx = int(k) - 1 if k.is_integer() else int(k)
    idx = max(0, min(idx, len(vals_sorted)-1))
    return float(vals_sorted[idx])

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp/range-evaluate/week.py <past_shifts.lp>")
        sys.exit(1)
    infile = sys.argv[1]

    # person_week_counts[(id,name)][week_monday][shift] = count
    person_week_counts: dict[tuple, dict[date, Counter]] = defaultdict(lambda: defaultdict(Counter))
    all_shifts = set()
    all_people = set()

    # 読み込み
    with open(infile, "r", encoding="utf-8") as f:
        for raw in f:
            m = PAT.match(raw.strip())
            if not m:
                continue
            nurse_id, name, ymd, shift = m.group(1), m.group(2), m.group(3), m.group(4)
            d = datetime.strptime(ymd, "%Y%m%d").date()
            w = monday_of(d)

            person_week_counts[(nurse_id, name)][w][shift] += 1
            all_shifts.add(shift)
            all_people.add((nurse_id, name))

    if not person_week_counts:
        print("No data.")
        return

    # 各人の在籍期間（最初に登場した週〜最後に登場した週）をゼロ埋めし、
    # 「人×週」ベースで各シフトの週平均を算出
    # per_person_means[shift] = [ personごとの(在籍週ゼロ埋め)週平均, ... ]
    per_person_means: dict[str, list[float]] = defaultdict(list)

    for person, week_map in person_week_counts.items():
        # その人が登場した週の最小・最大を取得
        weeks_of_person = sorted(week_map.keys())
        first_w, last_w = weeks_of_person[0], weeks_of_person[-1]

        # 在籍期間内の各週について「その週のシフト件数」を作成（未登場週は 0 扱い）
        # person_series[shift] = [w1_count, w2_count, ...]
        person_series: dict[str, list[int]] = defaultdict(list)

        for w in week_range(first_w, last_w):
            ctr = week_map.get(w, Counter())  # 未登場週は空カウンタ
            # その週に観測された全シフト（人グローバル）を対象に0埋め
            for s in all_shifts:
                person_series[s].append(ctr.get(s, 0))

        # 人ごとの週平均（在籍期間内の週数で割る）
        denom = max(1, len(person_series[next(iter(all_shifts))]))  # 在籍週数（安全策）
        for s in all_shifts:
            avg = sum(person_series[s]) / denom
            per_person_means[s].append(avg)

    # 出力：人平均の分布（人をサンプルとする）
    print("----- Weekly per-person averages from past (active-span zero-filled) -----")
    print(f"Total unique staff (ever observed): {len(all_people)}")
    print("shift   mean(person)   sd(person)    p25     p50     p75     max(person)")

    rows = []
    for s in sorted(all_shifts):
        vals = per_person_means[s]
        m = mean(vals) if vals else 0.0
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
