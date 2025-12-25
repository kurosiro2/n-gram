#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader_all.py

既存の data_loader.py を利用して、
「すべての病棟の看護師グループ情報」をまとめて読み込むためのユーティリティ。

使い方イメージ:

    import data_loader_all as dla

    root = "exp/2019-2025-data/real-name/group-settings"

    # 1) 病棟ごとの「名前 -> グループ集合（時系列潰した版）」
    all_groups = dla.load_all_staff_groups(root)
    print(all_groups["GCU"]["山田太郎"])

    # 2) 病棟ごとの「名前 -> タイムライン（start_date, groups）」
    all_tl = dla.load_all_staff_group_timelines(root)
    print(all_tl["4階南病棟"]["佐藤花子"])

    # 3) 病棟ごとの ignore_ids を読みたいとき
    all_ignore = dla.load_all_ignore_ids(root)
    print(all_ignore.get("GCU", set()))
"""

import os
from collections import defaultdict

import data_loader  # 同じディレクトリにある前提（import path は呼び出し元で調整）


def load_all_staff_groups(settings_root: str):
    """
    settings_root 配下にある「病棟ごとのディレクトリ」
      例: settings_root/
            GCU/
              2025-01-01/setting.lp
              2025-03-01/setting.lp
            4階南病棟/
              2025-02-10/setting.lp
            ...
    に対して、それぞれ data_loader.load_staff_groups() を呼び出し、

      { ward_name: { name: set(groups) } }

    を返す。

    ※ ward_name はディレクトリ名（例: "GCU", "4階南病棟"）。
    """
    all_wards = {}

    if not os.path.isdir(settings_root):
        raise FileNotFoundError(f"settings_root not found: {settings_root}")

    for ward_name in sorted(os.listdir(settings_root)):
        ward_dir = os.path.join(settings_root, ward_name)
        if not os.path.isdir(ward_dir):
            continue

        try:
            staff_groups = data_loader.load_staff_groups(ward_dir)
        except FileNotFoundError:
            # その病棟ディレクトリに setting.lp が見つからない場合はスキップ
            continue

        all_wards[ward_name] = staff_groups

    return all_wards


def load_all_staff_group_timelines(settings_root: str):
    """
    settings_root 配下にある「病棟ごとのディレクトリ」に対して、
    data_loader.load_staff_group_timeline() を呼び出し、

      {
        ward_name: {
          name: [ (start_date, set(groups)), ... ]
        },
        ...
      }

    の形で返す。

    例:
      settings_root = "exp/2019-2025-data/real-name/group-settings"

      -> load_all_staff_group_timelines(settings_root)["GCU"]["山田太郎"]
         = [(20250301, {"Leaders"}), (20251001, {"Leaders","Night"})]
         みたいなイメージ。
    """
    all_timelines = {}

    if not os.path.isdir(settings_root):
        raise FileNotFoundError(f"settings_root not found: {settings_root}")

    for ward_name in sorted(os.listdir(settings_root)):
        ward_dir = os.path.join(settings_root, ward_name)
        if not os.path.isdir(ward_dir):
            continue

        try:
            timeline = data_loader.load_staff_group_timeline(ward_dir)
        except FileNotFoundError:
            # 病棟ディレクトリ内に YYYY-MM-DD/setting.lp が無い場合など
            continue

        all_timelines[ward_name] = timeline

    return all_timelines


def _ignore_ids_dir_from_settings_root(settings_root: str) -> str:
    """
    settings_root から ignored-ids ディレクトリのパスを推定するヘルパ。

    例:
      settings_root = exp/2019-2025-data/real-name/group-settings
        -> realname_dir = exp/2019-2025-data/real-name
        -> ignore_dir   = exp/2019-2025-data/real-name/ignored-ids
    """
    realname_dir = os.path.dirname(settings_root)
    ignore_dir = os.path.join(realname_dir, "ignored-ids")
    return ignore_dir


def load_ignore_ids_for_ward(settings_root: str, ward_name: str):
    """
    1 病棟分の ignore_ids を読み込む。

    - settings_root:
        例) exp/2019-2025-data/real-name/group-settings
    - ward_name:
        例) "GCU", "4階南病棟" など（group-settings 配下のディレクトリ名と一致する想定）

    探すファイル:
        <real-name>/ignored-ids/<ward_name>.txt

    戻り値:
        set(normalize_id(id_str), ...)
        ファイルが無ければ空集合。
    """
    ignore_dir = _ignore_ids_dir_from_settings_root(settings_root)
    ignore_file = os.path.join(ignore_dir, f"{ward_name}.txt")

    if not os.path.exists(ignore_file):
        return set()

    ids = set()
    with open(ignore_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # カンマ/空白区切りどっちでもOKにする
            for token in line.replace(",", " ").split():
                nid = data_loader.normalize_id(token)
                ids.add(nid)

    # print(f"# [data_loader_all] loaded ignore_ids for ward={ward_name} from {ignore_file}: {sorted(ids)}")
    return ids


def load_all_ignore_ids(settings_root: str):
    """
    settings_root 配下のすべての病棟について、
    ward ごとの ignore_ids を読み込んで返す。

    戻り値:
        {
          "GCU": {"12345", "67890", ...},
          "4階南病棟": {...},
          ...
        }

    ※ ignore-ids/<ward>.txt が存在しない病棟は空集合 {}。
    """
    if not os.path.isdir(settings_root):
        raise FileNotFoundError(f"settings_root not found: {settings_root}")

    all_ignore = {}

    for ward_name in sorted(os.listdir(settings_root)):
        ward_dir = os.path.join(settings_root, ward_name)
        if not os.path.isdir(ward_dir):
            continue

        ids = load_ignore_ids_for_ward(settings_root, ward_name)
        all_ignore[ward_name] = ids

    return all_ignore


# -------------------------------------------------------------
# おまけ: ward + name でフラットに見たいとき用のヘルパ
# -------------------------------------------------------------

def flatten_all_staff_groups(all_wards_dict):
    """
    load_all_staff_groups() の戻り値

        { ward_name: { name: set(groups) } }

    をフラットにして

        { (ward_name, name): set(groups) }

    に変換するヘルパ（使いたければどうぞ）。
    """
    flat = {}
    for ward, name_map in all_wards_dict.items():
        for name, groups in name_map.items():
            flat[(ward, name)] = set(groups)
    return flat


def flatten_all_staff_group_timelines(all_tl_dict):
    """
    load_all_staff_group_timelines() の戻り値

        { ward_name: { name: [(start_date, set(groups)), ...] } }

    をフラットにして

        { (ward_name, name): [(start_date, set(groups)), ...] }

    に変換するヘルパ。
    """
    flat = {}
    for ward, name_map in all_tl_dict.items():
        for name, timeline in name_map.items():
            flat[(ward, name)] = list(timeline)
    return flat


if __name__ == "__main__":
    # 簡単なデバッグ用（必要なら使ってね）
    import sys

    if len(sys.argv) >= 2:
        root = sys.argv[1]
        print(f"# settings_root = {root}")

        print("\n# --- load_all_staff_groups ---")
        all_groups = load_all_staff_groups(root)
        print(f"wards: {list(all_groups.keys())}")

        print("\n# --- load_all_staff_group_timelines ---")
        all_tl = load_all_staff_group_timelines(root)
        print(f"wards (timeline): {list(all_tl.keys())}")

        print("\n# --- load_all_ignore_ids ---")
        all_ignore = load_all_ignore_ids(root)
        for w, ids in all_ignore.items():
            print(f"  ward={w}: {sorted(ids)}")
    else:
        print("Usage: python data_loader_all.py [settings_root]")
