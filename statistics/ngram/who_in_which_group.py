#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
who_in_which_group.py

past-shifts と group-settings(setting.lp) から、
「誰がどんな看護師グループに所属しているか」を一覧表示するツール。

例:
  python exp/statistics/tools/who_in_which_group.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --date-start 20250101 --date-end 20251231

名前で絞る:
  --name "山田"   (部分一致)

グループで絞る:
  --group "newcomer"

日付ごとのタイムラインも出す:
  --timeline

CSV保存:
  --csv out/groups.csv
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Set, Optional

# -------------------------------------------------------------
# import path: exp/statistics/data_loader.py を読む
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
# tools/ の1つ上が statistics/ ならさらに1つ上が exp/
# ここでは「このスクリプトをどこに置いても動く」ように2段上を入れる
PARENT_1 = os.path.dirname(CURRENT_DIR)
PARENT_2 = os.path.dirname(PARENT_1)
if PARENT_1 not in sys.path:
    sys.path.append(PARENT_1)
if PARENT_2 not in sys.path:
    sys.path.append(PARENT_2)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date

PersonKey = Tuple[int, str]  # (nurse_id, name)
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


def normalize_name(s: str) -> str:
    return str(s).strip().lower()


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="例: .../past-shifts/GCU.lp")
    ap.add_argument("group_settings", help="例: .../group-settings/GCU/ (setting ディレクトリ or setting.lp)")
    ap.add_argument("--date-start", type=int, default=None)
    ap.add_argument("--date-end", type=int, default=None)
    ap.add_argument("--name", action="append", default=[],
                    help="名前でフィルタ（部分一致）。複数指定可。例: --name 山田")
    ap.add_argument("--group", action="append", default=[],
                    help="グループでフィルタ（部分一致）。複数指定可。例: --group newcomer")
    ap.add_argument("--timeline", action="store_true",
                    help="日付ごとの所属グループも表示（長くなる）")
    ap.add_argument("--csv", default=None, help="CSV出力パス（例: out/groups.csv）")
    ap.add_argument("--max-days", type=int, default=0,
                    help="timeline表示で、各人の表示日数上限（0で無制限）")
    args = ap.parse_args()

    seqs: SeqDict = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)

    name_filters = [normalize_name(x) for x in args.name if x]
    group_filters = [normalize_name(x) for x in args.group if x]

    # nurse -> set(groups)
    groups_seen: Dict[PersonKey, Set[str]] = defaultdict(set)
    # nurse -> [(date, groups_str)]
    timeline_seen: Dict[PersonKey, List[Tuple[int, str]]] = defaultdict(list)

    for (nid, name), seq in seqs.items():
        # 日付で走査する（shiftは関係なく、所属グループ確認用）
        days = 0
        for d, _ in seq:
            if args.date_start is not None and d < args.date_start:
                continue
            if args.date_end is not None and d > args.date_end:
                continue

            gs = get_groups_for_day(name, nid, d, timeline)
            if not gs:
                continue

            groups_seen[(nid, name)].update(gs)

            if args.timeline:
                if args.max_days and days >= args.max_days:
                    continue
                gs_str = "|".join(sorted(gs, key=lambda x: x.lower()))
                timeline_seen[(nid, name)].append((d, gs_str))
                days += 1

    # apply name filter
    persons = list(groups_seen.keys())
    persons.sort(key=lambda x: (normalize_name(x[1]), x[0]))

    def pass_name(pk: PersonKey) -> bool:
        if not name_filters:
            return True
        nm = normalize_name(pk[1])
        return any(f in nm for f in name_filters)

    def pass_group(pk: PersonKey) -> bool:
        if not group_filters:
            return True
        gs = [normalize_name(x) for x in groups_seen.get(pk, set())]
        for f in group_filters:
            if any(f in g for g in gs):
                return True
        return False

    persons = [pk for pk in persons if pass_name(pk) and pass_group(pk)]

    # output to stdout
    print("# ---- Nurse -> Groups (unique) ----")
    print(f"# nurses_shown={len(persons)}  date={args.date_start}..{args.date_end}")
    if name_filters:
        print(f"# name_filter={args.name}")
    if group_filters:
        print(f"# group_filter={args.group}")

    for (nid, name) in persons:
        gs = sorted(groups_seen[(nid, name)], key=lambda x: x.lower())
        print(f"{nid}\t{name}\t{', '.join(gs)}")

        if args.timeline:
            rows = sorted(timeline_seen.get((nid, name), []), key=lambda t: t[0])
            for d, gstr in rows:
                print(f"  {d}: {gstr}")

    # CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["nurse_id", "name", "groups_seen"])
            for (nid, name) in persons:
                gs = sorted(groups_seen[(nid, name)], key=lambda x: x.lower())
                w.writerow([nid, name, "|".join(gs)])
        print(f"# csv -> {args.csv}")


if __name__ == "__main__":
    main()
