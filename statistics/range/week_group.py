# 使い方: python exp/statistics/range/week.py <past_shifts.lp> <setting_dir_or_file>
# 指標について
"""
| 指標               | 意味                                  | 計算の仕方                                 | 直感的な意味                                                |
| ---------------- | ------------------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **mean(shift)** | 各人の週平均値の平均                        | 各人の (週ごとの回数の平均) を平均                | 「1人あたり、週に平均して何回そのシフトをしていたか」                    |
| **sd(person)**   | 標準偏差（母集団）                         | `pstdev(各人の週平均)`                       | 人ごとのばらつきの大きさ。大きいほどスタッフ間で差がある                     |
| **min(shift)**  | 週ごとの回数の最小（グループ内、整数）        | min( 各人の min(週ごとの回数) )              | 例：WR で誰かが「ある週で休み0回」なら 0 になる                          |
| **max(shift)**  | 週ごとの回数の最大（グループ内、整数）        | max( 各人の max(週ごとの回数) )              | 例：D が多い週で誰かが「その週 D=5回」なら 5 になる                         |
| **share_avg**    | 平均シェア                              | mean(s)/Σ mean(·)                          | グループ内平均（mean(person)）に占めるそのシフトの割合（0〜1）           |
"""

#!/usr/bin/env python3
import re,os
import sys
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
from statistics import mean, pstdev

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, load_staff_groups, get_groups_for_date

# ★ data_loader からタイムライン系を使う（パスはプロジェクト構成に合わせて）
from data_loader import (
    load_staff_group_timeline,
    get_groups_for_date,
)

# shift_data("id","name",YYYYMMDD,"shift").
PAT = re.compile(
    r'^shift_data\("([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d{8})\s*,\s*"([^"]+)"\)\.'
)


def monday_of(d: date) -> date:
    """その日の属する週の月曜（日付）を返す"""
    return d - timedelta(days=d.weekday())


def week_range(monday_start: date, monday_end: date):
    """月曜ベースの週を [start, end] で順に返す"""
    w = monday_start
    while w <= monday_end:
        yield w
        w += timedelta(days=7)


def read_shifts(infile):
    """
    past_shifts.lp を読む
    person_week_counts[(id,name)][week_monday][shift] = count
    """
    person_week_counts = defaultdict(lambda: defaultdict(Counter))
    all_shifts = set()
    all_people = set()

    with open(infile, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = PAT.match(line)
            if not m:
                continue
            nurse_id, name, ymd, shift = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            d = datetime.strptime(ymd, "%Y%m%d").date()
            w = monday_of(d)
            person_week_counts[(nurse_id, name)][w][shift] += 1
            all_shifts.add(shift)
            all_people.add((nurse_id, name))

    return person_week_counts, all_shifts, all_people


def collect_group_order(group_timeline):
    """
    タイムラインから出てくるグループ名を集めて、
    All を先頭, Unknown を最後にした順序で返す。
    """
    groups = set()
    for snaps in group_timeline.values():
        for _, gset in snaps:
            groups.update(gset)

    groups.add("All")
    groups.add("Unknown")

    def key(g):
        if g == "All":
            return (0, "")
        if g == "Unknown":
            return (2, "")
        return (1, g.lower())

    return sorted(groups, key=key)


def per_group_stats(
    group_name,
    all_people,
    person_week_counts,
    all_shifts,
    group_timeline,
):
    """
    タイムラインに基づき「その週に group_name に属している週だけ」を使って
    週平均などを計算する。

    戻り値:
      per_person_means:       dict shift -> list[ per-person weekly mean (float) ]
      per_person_week_mins:   dict shift -> list[ per-person weekly min (int) ]
      per_person_week_maxs:   dict shift -> list[ per-person weekly max (int) ]
      n_people_in_group:      そのグループで少なくとも1週は在籍していた人数
    """
    per_person_means = defaultdict(list)
    per_person_week_mins = defaultdict(list)
    per_person_week_maxs = defaultdict(list)
    n_people_in_group = 0

    if not all_shifts:
        return per_person_means, per_person_week_mins, per_person_week_maxs, 0

    for person in all_people:
        nurse_id, name = person
        week_map = person_week_counts.get(person)
        if not week_map:
            continue

        weeks_of_person = sorted(week_map.keys())
        if not weeks_of_person:
            continue
        first_w, last_w = weeks_of_person[0], weeks_of_person[-1]

        # この人が「group_name に属していた週だけ」のシリーズを作る
        series_by_shift = defaultdict(list)
        used_weeks = 0

        for w in week_range(first_w, last_w):
            # その週の代表日として月曜の日付を使う（YYYYMMDD int）
            date_int = int(w.strftime("%Y%m%d"))

            if group_name == "All":
                in_group = True
            else:
                gset = get_groups_for_date(name, date_int, group_timeline)
                # タイムラインになければ Unknown とみなす
                if not gset:
                    gset = {"Unknown"}
                in_group = group_name in gset

            if not in_group:
                continue

            used_weeks += 1
            ctr = week_map.get(w, Counter())
            # その週の各シフト回数（0 を含めて全集計）
            for s in all_shifts:
                series_by_shift[s].append(ctr.get(s, 0))

        if used_weeks == 0:
            # このグループに一度も属していない
            continue

        n_people_in_group += 1

        # この人について shift ごとの weekly mean / min / max を計算
        for s in all_shifts:
            series = series_by_shift[s]
            # used_weeks > 0 のときは series の長さは used_weeks のはず
            if not series:
                # 念のため保険：全部 0 の週だったケース
                series = [0] * used_weeks

            avg = sum(series) / used_weeks
            per_person_means[s].append(avg)
            per_person_week_mins[s].append(int(min(series)))
            per_person_week_maxs[s].append(int(max(series)))

    return per_person_means, per_person_week_mins, per_person_week_maxs, n_people_in_group


def print_group_table(
    group_name,
    per_person_means,
    per_person_week_mins,
    per_person_week_maxs,
    n_people_in_group,
):
    print(f'\n===== Group="{group_name}" | Weekly per-person averages (timeline-aware) =====')
    print(f"Total staff (in group): {n_people_in_group}")
    print("shift      mean(shift)   sd(person)    min(shift)    max(shift)  share_avg")

    # share_avg の分母（各シフトの mean(person) の総和）
    mean_by_shift = {s: (mean(v) if v else 0.0) for s, v in per_person_means.items()}
    denom = sum(mean_by_shift.values()) if mean_by_shift else 0.0

    rows = []
    for s in sorted(per_person_means.keys()):
        vals = per_person_means[s]
        m_val = mean(vals) if vals else 0.0
        sd_val = pstdev(vals) if len(vals) > 1 else 0.0

        mins = per_person_week_mins.get(s, [])
        maxs = per_person_week_maxs.get(s, [])
        mn = float(min(mins)) if mins else 0.0
        mx = float(max(maxs)) if maxs else 0.0

        share = (m_val / denom) if denom > 0 else 0.0
        rows.append((s, m_val, sd_val, mn, mx, share))

    # mean 降順 → shift 名昇順
    rows.sort(key=lambda x: (-x[1], x[0]))
    for s, m_val, sd_val, mn, mx, share in rows:
        print(
            f"{s:<6}  {m_val:12.3f}  {sd_val:11.3f}  "
            f"{mn:11.0f}  {mx:11.0f}  {share:9.3f}"
        )


def main():
    if len(sys.argv) < 3:
        print("Usage: python exp/statistics/range/week.py <past_shifts.lp> <setting_dir_or_file>")
        sys.exit(1)

    shift_file = sys.argv[1]
    setting_path = sys.argv[2]

    # 1) 過去勤務データ（週単位に集計）
    person_week_counts, all_shifts, all_people = read_shifts(shift_file)
    if not person_week_counts:
        print("No data.")
        return

    # 2) グループのタイムライン（名前ベース）
    group_timeline = load_staff_group_timeline(setting_path)

    # 3) タイムラインから存在するグループ一覧を作る（All / Unknown も含む）
    group_order = collect_group_order(group_timeline)

    # 4) グループごとに集計して出力
    for g in group_order:
        per_person_means, per_person_week_mins, per_person_week_maxs, n_people = per_group_stats(
            g,
            all_people,
            person_week_counts,
            all_shifts,
            group_timeline,
        )
        if n_people == 0 or not per_person_means:
            # このグループに該当する在籍週が誰にもない
            continue

        # 観測ゼロのシフトを除外（全部表示したいならこのフィルタを外す）
        per_person_means = {s: v for s, v in per_person_means.items() if v}
        per_person_week_mins = {s: per_person_week_mins[s] for s in per_person_means.keys()}
        per_person_week_maxs = {s: per_person_week_maxs[s] for s in per_person_means.keys()}

        print_group_table(g, per_person_means, per_person_week_mins, per_person_week_maxs, n_people)


if __name__ == "__main__":
    main()
