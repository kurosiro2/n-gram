#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngram_past_shifts_group.py を使って、
グループ別 n-gram の「確率分布」をグラフ化して PNG 出力するスクリプト。

- 単一病棟モード:
    python ngram_prob_dist_plot.py past-shifts.lp setting.lp N out_dir

- 全病棟モード:
    python ngram_prob_dist_plot.py past-shifts-dir group-settings-root N out_dir

仕様:
  - ngram_past_shifts_group.py の
      - ngram_counts_by_group
      - filter_seqs_by_date
      - group_sort_key
      - DATE_START / DATE_END
    を再利用する。
  - n-gram 出現回数 counter を正規化して
        P(gram) = count / Σ count
    の確率分布として扱い、上位 TOP_K 個を棒グラフで描画。
  - グラフの x 軸: n-gram（例: "WR-D", "LD-SE-SN"）
           y 軸: P(gram)
"""

import sys
import os
from collections import Counter

# 親ディレクトリを import パスに追加
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader        # load_past_shifts, load_staff_group_timeline
import data_loader_all    # load_all_staff_group_timelines

# 既存スクリプトから必要な機能を再利用
from ngram_past_shifts_group import (
    ngram_counts_by_group,
    filter_seqs_by_date,
    group_sort_key,
    DATE_START,
    DATE_END,
)

# GUI なし環境でも保存できるように
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 上位いくつまで描画するか
TOP_K = 20


def make_prob_plot_for_group(
    group_name: str,
    n_counter: Counter,
    N: int,
    ward_name: str | None,
    out_dir: str,
):
    """
    1つのグループの n-gram Counter から確率分布グラフを作成し PNG 保存。

    - group_name: "Expert", "Novice", "All" など
    - n_counter : Counter({gram(tuple[str]): count, ...})
    - N         : n-gram の N
    - ward_name : 病棟名（全病棟モード時に "GCU" などを入れる）
    - out_dir   : PNG 出力先ディレクトリ
    """
    if not n_counter:
        return

    total = sum(n_counter.values())
    if total <= 0:
        return

    # 出現回数の多い順に並べて上位 TOP_K を取る
    items = n_counter.most_common(TOP_K)

    labels = []
    probs = []
    for gram, c in items:
        if isinstance(gram, tuple):
            label = "-".join(gram)
        else:
            label = str(gram)
        labels.append(label)
        probs.append(c / total)

    if not labels:
        return

    # ファイル名
    safe_group = group_name.replace("/", "_")
    if ward_name:
        filename = f"prob_N{N}_ward-{ward_name}_group-{safe_group}.png"
    else:
        filename = f"prob_N{N}_group-{safe_group}.png"
    out_path = os.path.join(out_dir, filename)

    # プロット
    plt.figure(figsize=(max(8, len(labels) * 0.4), 4))

    x = range(len(labels))
    plt.bar(x, probs)
    plt.xticks(x, labels, rotation=45, ha="right")

    title_parts = []
    if ward_name:
        title_parts.append(f"Ward={ward_name}")
    title_parts.append(f"Group={group_name}")
    title_parts.append(f"N={N}")
    title = " / ".join(title_parts)

    plt.title(title)
    plt.ylabel("Probability P(gram)")
    plt.xlabel("n-gram")
    plt.tight_layout()
    plt.grid(axis="y", linestyle=":", linewidth=0.5)

    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"# Saved: {out_path}")


def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print("  # 単一病棟モード")
        print("  python ngram_prob_dist_plot.py past-shifts.lp setting.lp N out_dir")
        print("")
        print("  # 全病棟モード（ディレクトリ × ディレクトリ）")
        print("  python ngram_prob_dist_plot.py past-shifts-dir group-settings-root N out_dir")
        print("")
        print(f"  ※ 期間指定は ngram_past_shifts_group.py 内の DATE_START={DATE_START}, DATE_END={DATE_END} に従う")
        sys.exit(1)

    shift_arg   = sys.argv[1]  # ファイル or ディレクトリ
    setting_arg = sys.argv[2]  # setting.lp or group-settings root dir
    N           = int(sys.argv[3])
    out_dir     = sys.argv[4]

    if N <= 0:
        print("N は 1 以上を指定してください。", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # 1) 全病棟モード（第1引数がディレクトリ）
    # -------------------------------------------------
    if os.path.isdir(shift_arg):
        past_shifts_dir = shift_arg
        settings_root   = setting_arg

        if not os.path.isdir(settings_root):
            print(f"[ERROR] settings_root として指定したパスがディレクトリではありません: {settings_root}",
                  file=sys.stderr)
            sys.exit(1)

        # 全病棟のグループタイムライン
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

        N_eff = max(1, N)

        # 病棟ごとに n-gram の確率分布を描画
        for ward_name, shift_file in sorted(ward_shift_files.items()):
            if ward_name not in all_timelines:
                print(f"# [WARN] ward={ward_name} に対応する group-settings が見つからないのでスキップ")
                continue
            if not os.path.isfile(shift_file):
                print(f"# [WARN] shift_file not found for ward={ward_name}: {shift_file}")
                continue

            print(f"\n========== Ward=\"{ward_name}\" ==========")

            seqs = data_loader.load_past_shifts(shift_file)
            # ★ 期間フィルタは ngram_past_shifts_group.py と同じ関数を使用
            before = len(seqs)
            seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
            after = len(seqs)
            print(f"# [Ward={ward_name}] date filter: nurses {before} -> {after}")
            if not seqs:
                print(f"# [Ward={ward_name}] no shifts in specified period; skip.")
                continue

            group_timeline = all_timelines[ward_name]

            counters_N = ngram_counts_by_group(seqs, group_timeline, N_eff)

            for g in sorted(counters_N.keys(), key=group_sort_key):
                # 不要なら Unknown をスキップするなど調整可能
                # if g == "Unknown":
                #     continue
                make_prob_plot_for_group(
                    group_name=g,
                    n_counter=counters_N[g],
                    N=N_eff,
                    ward_name=ward_name,
                    out_dir=out_dir,
                )

        return

    # -------------------------------------------------
    # 2) 単一病棟モード
    # -------------------------------------------------
    shift_file   = shift_arg
    setting_file = setting_arg

    if not os.path.isfile(shift_file):
        print(f"[ERROR] past-shifts ファイルが見つかりません: {shift_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(setting_file):
        print(f"[ERROR] setting ファイルが見つかりません: {setting_file}", file=sys.stderr)
        sys.exit(1)

    print("# Single-ward mode")

    seqs = data_loader.load_past_shifts(shift_file)
    before = len(seqs)
    seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
    after = len(seqs)
    print(f"# [Single ward] date filter: nurses {before} -> {after}")
    if not seqs:
        print("# No shifts in specified period. Abort.")
        return

    group_timeline = data_loader.load_staff_group_timeline(setting_file)

    N_eff = max(1, N)
    counters_N = ngram_counts_by_group(seqs, group_timeline, N_eff)

    # 単一病棟なので ward_name は None（ファイル名を入れたければここで変えてもOK）
    for g in sorted(counters_N.keys(), key=group_sort_key):
        make_prob_plot_for_group(
            group_name=g,
            n_counter=counters_N[g],
            N=N_eff,
            ward_name=None,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
