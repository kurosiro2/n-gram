#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年だけに絞って、病棟を無視して
「同じ看護師グループごとに」P(next|prefix) < 1.0 を満たす
n-gram の「種類数」の曲線を出すスクリプト。

・全病棟の past-shifts を読む（または特定病棟 1ファイルだけでもOK）
・各病棟ごとに ngram_counts_by_group で N-gram カウント
・グループごとにカウンタを合算 → 全病棟をまとめた結果
・グループごとに PNG を出力（All/Unknown は PNG からは除外）
・グローバル CSV には種類数と weighted の両方を出力

縦軸（メイン指標）:
  - positive_type_count: P(next|prefix) < 1.0 を満たす n-gram の「種類数」

上限カーブ:
  - total_type_count: そのグループで観測された n-gram 「種類数」（prefix が存在するもの）

サブ指標（CSV にのみ出力）:
  - total_weighted: 全 n-gram の出現回数の総和
  - positive_weighted: P(next|prefix) < 1.0 を満たす n-gram の出現回数の総和

PNG は各グループごとに 2 種類:
  1) <group>_2025_typecount.png
     - total_type（upper）
     - positive_type
  2) <group>_2025_cumulative_typecount.png
     - cumulative_total_type（1〜N までの total_type 累積和）
     - total_type（upper）
     - positive_type

依存:
    - data_loader.py
    - data_loader_all.py
    - ngram_past_shifts_group.py  (ngram_counts_by_group を利用)

使い方:
    python ngram_condprob_typecount_by_group_all.py \
        [past_shifts_dir_or_file] [settings_root] [N_max] [output_dir] [global_csv]
"""

import sys
import os
from collections import Counter, defaultdict

# -------------------------------------------------------------
# import path 調整（このファイルの親ディレクトリを sys.path に追加）
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader
import data_loader_all
from ngram_past_shifts_group import ngram_counts_by_group

# GUI なし環境でも保存できるように
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 設定: 対象とする日付範囲（2025 年のみ）
# -------------------------------------------------------------
YEAR_START = 20250101  # inclusive
YEAR_END   = 20251231  # inclusive


def positive_stats(n_counter: Counter, nm1_counter: Counter, N: int):
    """
    n_counter: N-gram -> freq のカウンタ
    nm1_counter: (N-1)-gram -> freq のカウンタ
    N: n-gram の長さ

    戻り値:
      total_weighted      : すべての N-gram の freq の総和
      positive_weighted   : P(next|prefix) < 1.0 を満たす N-gram の freq の総和
      total_type_count    : 対象 N-gram の「種類数」(prefix が存在するものの個数)
      positive_type_count : 条件 P(next|prefix) < 1.0 を満たす N-gram の「種類数」
    """
    total_weighted = sum(n_counter.values())

    # N=1 の場合は条件付き確率が定義できないので、
    # ここでは「全部 positive」とみなしつつ、
    # 種類数は単純に len(n_counter) とする。
    if N == 1:
        positive_weighted   = total_weighted
        total_type_count    = len(n_counter)
        positive_type_count = len(n_counter)
        return total_weighted, positive_weighted, total_type_count, positive_type_count

    positive_weighted   = 0
    total_type_count    = 0
    positive_type_count = 0

    for gram, c in n_counter.items():
        # N >= 2 のときだけ来るはずだが、一応 len チェック
        if len(gram) < 2:
            continue

        prefix = gram[:-1]
        base   = nm1_counter.get(prefix, 0)

        # 分母となる prefix の出現回数が 0 のものはスキップ
        if base <= 0:
            continue

        total_type_count += 1

        p = c / base
        if p < 1.0 - 1e-12:
            positive_weighted   += c
            positive_type_count += 1

    return total_weighted, positive_weighted, total_type_count, positive_type_count


def filter_seqs_by_date(seqs, start_date: int, end_date: int):
    """
    seqs: {(nurse_id, name): [(date, shift), ...]}
    から、指定された日付範囲 [start_date, end_date] に入るシフトだけを残す。

    1人分のシフトが全部範囲外なら、その人自体を削除する。
    """
    filtered = {}
    for key, seq in seqs.items():
        sub = [(d, s) for (d, s) in seq if start_date <= d <= end_date]
        if sub:
            sub.sort(key=lambda t: t[0])
            filtered[key] = sub
    return filtered


def group_sort_key(g: str):
    """
    グループ名のソート順:
      - "All" を先頭
      - "Unknown" を最後
      - その他はアルファベット順
    """
    if g == "All":
        return (0, "")
    if g == "Unknown":
        return (2, "")
    return (1, g.lower())


# -------------------------------------------------------------
# デバッグ用ヘルパ（どの病棟・看護師がその N-gram を出しているか）
# -------------------------------------------------------------
def find_one_owner_across_wards(ward_seqs_2025, gram):
    """
    全病棟のシフト列から、指定された N-gram gram を1件だけ探す。

    ward_seqs_2025: { ward_name: {(nid, name): [(date, shift), ...]} }
    gram: tuple of shifts

    return:
        (ward_name, nurse_id, name, start_date, end_date) or None
    """
    N = len(gram)
    gram_list = list(gram)

    for ward_name, seqs in ward_seqs_2025.items():
        for (nid, name), seq in seqs.items():
            if len(seq) < N:
                continue
            dates  = [d for (d, _) in seq]
            shifts = [s for (_, s) in seq]
            for i in range(len(seq) - N + 1):
                if shifts[i : i + N] == gram_list:
                    start_date = dates[i]
                    end_date   = dates[i + N - 1]
                    return ward_name, nid, name, start_date, end_date

    return None


def dump_positive_ngrams_for_group(
    ward_seqs_2025,
    group_name: str,
    N: int,
    n_counter: Counter,
    nm1_counter: Counter,
    max_rows: int = 20,
):
    """
    デバッグ用:
      P(next|prefix) < 1.0 の N-gram を上位 max_rows 件だけ出力し、
      それを実際に出している病棟 / 看護師 / 日付範囲も表示する。
    """
    rows = []

    for gram, c in n_counter.items():
        if len(gram) < 2:
            continue

        prefix = gram[:-1]
        base   = nm1_counter.get(prefix, 0)
        if base <= 0:
            continue

        p = c / base
        if p < 1.0 - 1e-12:
            rows.append((c, p, gram, base))

    if not rows:
        print(f"# [DEBUG] Group={group_name}, N={N}: P(next|prefix) < 1.0 の N-gram はありません")
        return

    # freq 降順、p 昇順でソート
    rows.sort(key=lambda x: (-x[0], x[1]))

    print(f"# ==== [DEBUG] Group={group_name}, N={N}: P(next|prefix) < 1.0 の N-gram 上位 {max_rows} 件 ====")
    print("# freq   P(next|prefix)   base_prefix   ward   nurse_id  name  [start_date,end_date]   prefix -> next")

    for c, p, gram, base in rows[:max_rows]:
        prefix = gram[:-1]
        last   = gram[-1]
        arrow  = "-".join(prefix) + " -> " + last

        owner = find_one_owner_across_wards(ward_seqs_2025, gram)
        if owner is None:
            owner_info = "?, ?, ?, [?,?]"
        else:
            ward, nid, name, start_date, end_date = owner
            owner_info = f"{ward}  {nid}  {name}  [{start_date},{end_date}]"

        print(
            f"{c:6d}  {p:14.6f}  {base:12d}   {owner_info}   {arrow}"
        )


# -------------------------------------------------------------
# main
# -------------------------------------------------------------
def main():
    if len(sys.argv) < 6:
        print(
            "python ngram_condprob_typecount_by_group_all.py "
            "[past_shifts_dir_or_file] [settings_root] [N_max] [output_dir] [global_csv]"
        )
        sys.exit(1)

    past_shifts_dir_or_file = sys.argv[1]  # 例: ディレクトリ or exp/.../GCU.lp
    settings_root           = sys.argv[2]  # 例: exp/2019-2025-data/real-name/group-settings
    N_max                   = int(sys.argv[3])
    output_dir              = sys.argv[4]
    global_csv_path         = sys.argv[5]

    os.makedirs(output_dir, exist_ok=True)

    # 1) 全病棟のタイムライン情報を読み込み
    all_timelines = data_loader_all.load_all_staff_group_timelines(settings_root)
    print("# Wards found in settings_root:", ", ".join(sorted(all_timelines.keys())))

    # 2) past-shifts（ディレクトリ or 単一ファイル）から ward ごとの .lp を探す
    ward_shift_files = {}

    if os.path.isdir(past_shifts_dir_or_file):
        for fname in sorted(os.listdir(past_shifts_dir_or_file)):
            if not fname.endswith(".lp"):
                continue
            ward_name  = os.path.splitext(fname)[0]
            shift_path = os.path.join(past_shifts_dir_or_file, fname)
            ward_shift_files[ward_name] = shift_path
    else:
        if not past_shifts_dir_or_file.endswith(".lp"):
            print(f"[ERROR] past_shifts_dir_or_file is not a directory and not a .lp file: {past_shifts_dir_or_file}")
            sys.exit(1)
        ward_name  = os.path.splitext(os.path.basename(past_shifts_dir_or_file))[0]
        ward_shift_files[ward_name] = past_shifts_dir_or_file

    print("# Wards found in past_shifts:", ", ".join(sorted(ward_shift_files.keys())))

    # 3) 各病棟のシフト（2025年だけ）を先に読み込んでおく
    ward_seqs_2025 = {}
    for ward_name, shift_file in sorted(ward_shift_files.items()):
        if ward_name not in all_timelines:
            print(f"# [WARN] ward={ward_name} has no settings timeline; skip.")
            continue

        if not os.path.isfile(shift_file):
            print(f"# [WARN] shift_file not found for ward={ward_name}: {shift_file}")
            continue

        seqs = data_loader.load_past_shifts(shift_file)
        before = len(seqs)
        seqs = filter_seqs_by_date(seqs, YEAR_START, YEAR_END)
        after = len(seqs)
        print(f"# [Ward={ward_name}] 2025 filter: nurses {before} -> {after}")
        if not seqs:
            print(f"# [Ward={ward_name}] no 2025 shifts; skip.")
            continue

        ward_seqs_2025[ward_name] = seqs

    if not ward_seqs_2025:
        print("# No ward has 2025 data. Abort.")
        return

    # 4) グローバル CSV をオープン
    with open(global_csv_path, "w", encoding="utf-8") as csv_fp:
        # 種類数と weighted の両方を書く
        print(
            "group,N,"
            "total_type,positive_type,ratio_type,"
            "total_weighted,positive_weighted,ratio_weighted",
            file=csv_fp
        )

        # グループごとの曲線を保持
        per_group_data = defaultdict(
            lambda: {
                "Ns": [],
                "total_type": [],
                "positive_type": [],
                "total_weighted": [],
                "positive_weighted": [],
                "ratio_type": [],
                "ratio_weighted": [],
            }
        )

        # 5) N = 1..N_max について、全病棟ぶんのカウンタを合算
        for N in range(1, N_max + 1):
            print(f"# ==== N={N} ====")

            aggregated_N   = defaultdict(Counter)
            aggregated_Nm1 = defaultdict(Counter)  # N>=2 のときだけ有効

            # (1) 各病棟で ngram_counts_by_group を計算し、グループごとに加算
            for ward_name, seqs in sorted(ward_seqs_2025.items()):
                group_timeline = all_timelines[ward_name]

                counters_N_ward = ngram_counts_by_group(seqs, group_timeline, N)
                for g, ctr in counters_N_ward.items():
                    aggregated_N[g].update(ctr)

                if N > 1:
                    counters_Nm1_ward = ngram_counts_by_group(seqs, group_timeline, N - 1)
                    for g, ctr in counters_Nm1_ward.items():
                        aggregated_Nm1[g].update(ctr)

            # (2) グループごとに positive_stats を計算
            for g in sorted(aggregated_N.keys(), key=group_sort_key):
                n_counter   = aggregated_N[g]
                nm1_counter = aggregated_Nm1[g] if N > 1 else Counter()

                total_w, pos_w, total_t, pos_t = positive_stats(n_counter, nm1_counter, N)

                ratio_t = (pos_t / total_t) if total_t > 0 else 0.0
                ratio_w = (pos_w / total_w) if total_w > 0 else 0.0

                per_group_data[g]["Ns"].append(N)
                per_group_data[g]["total_type"].append(total_t)
                per_group_data[g]["positive_type"].append(pos_t)
                per_group_data[g]["total_weighted"].append(total_w)
                per_group_data[g]["positive_weighted"].append(pos_w)
                per_group_data[g]["ratio_type"].append(ratio_t)
                per_group_data[g]["ratio_weighted"].append(ratio_w)

                # CSV: group, N, 種類数と weighted 両方
                print(
                    f"{g},{N},"
                    f"{total_t},{pos_t},{ratio_t:.6f},"
                    f"{total_w},{pos_w},{ratio_w:.6f}",
                    file=csv_fp
                )

                # デバッグ出力（weighted が残っているグループを観察したいとき）
                if N >= 20 and pos_w > 0:
                    dump_positive_ngrams_for_group(
                        ward_seqs_2025=ward_seqs_2025,
                        group_name=g,
                        N=N,
                        n_counter=n_counter,
                        nm1_counter=nm1_counter,
                        max_rows=20,
                    )

    # 6) グループごとの PNG: upper & positive（種類数）
    for g in sorted(per_group_data.keys(), key=group_sort_key):
        if g in ("All", "Unknown"):
            # 集計自体は CSV に残っているので、PNG だけスキップ
            continue

        data = per_group_data[g]
        Ns             = data["Ns"]
        total_type     = data["total_type"]     # 上限: distinct N-gram 数
        positive_type  = data["positive_type"]  # P<1.0 の distinct N-gram 数

        if not Ns:
            continue

        png_name   = f"{g}_2025_typecount.png"
        output_png = os.path.join(output_dir, png_name)

        plt.figure(figsize=(8, 4))

        # メイン：P<1.0 の distinct N-gram 数
        plt.plot(Ns, positive_type, marker=".", linestyle="-",
                 label="(P(next|prefix) < 1.0) N-gram types ")

        # 上限: 観測された distinct N-gram の種類数
        plt.plot(Ns, total_type, marker="x", linestyle="--",
                 label="upper : total N-gram types")

        plt.xlabel("N (n-gram length)")
        plt.ylabel("Number of N-gram types")
        plt.title(f"Group={g} (2025)")
        plt.grid(True)

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f"# [Group={g}] Saved plot to: {output_png}")

    # 6.5) All グループの upper & positive
    if "All" in per_group_data:
        data_all = per_group_data["All"]
        Ns_all            = data_all["Ns"]
        total_type_all    = data_all["total_type"]
        positive_type_all = data_all["positive_type"]

        if Ns_all:
            png_name_all   = "All_2025_typecount.png"
            output_png_all = os.path.join(output_dir, png_name_all)

            plt.figure(figsize=(8, 4))

            plt.plot(Ns_all, positive_type_all, marker=".",
                     linestyle="-", label="(P(next|prefix) < 1.0) N-gram types")
            plt.plot(Ns_all, total_type_all, marker="x",
                     linestyle="--", label="upper: total N-gram types")

            plt.xlabel("N (n-gram length)")
            plt.ylabel("Number of N-gram types")
            plt.title("Group=All (2025)")
            plt.grid(True)

            plt.legend()
            plt.tight_layout()
            plt.savefig(output_png_all, dpi=200)
            plt.close()
            print(f"# [Group=All] Saved plot to: {output_png_all}")

    # 6.8) 累積種類数 PNG（累積 + upper + positive）
    for g in sorted(per_group_data.keys(), key=group_sort_key):
        if g in ("All", "Unknown"):
            continue

        data = per_group_data[g]
        Ns             = data["Ns"]
        total_type     = data["total_type"]
        positive_type  = data["positive_type"]

        if not Ns:
            continue

        # 累積 total_type
        cumulative = []
        acc = 0
        for t in total_type:
            acc += t
            cumulative.append(acc)

        png_name   = f"{g}_2025_cumulative_typecount.png"
        output_png = os.path.join(output_dir, png_name)

        plt.figure(figsize=(8, 4))

        # 累積
        plt.plot(
            Ns,
            cumulative,
            marker="o",
            linestyle="-",
            label="cumulative total N-gram types"
        )

        # 各 N の upper（total_type）
        plt.plot(
            Ns,
            total_type,
            marker="x",
            linestyle="--",
            label="upper: total N-grams types"
        )

        # 各 N の positive_type
        plt.plot(
            Ns,
            positive_type,
            marker=".",
            linestyle="-.",
            label="(P(next|prefix) < 1.0) N-gram types "
        )

        plt.xlabel("N (n-gram length)")
        plt.ylabel("cumulative total N-gram types")
        plt.title(f"Group={g} / Type Counts (2025)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f"# [Group={g}] Saved cumulative plot to: {output_png}")

    # 6.9) ALL グループの累積 PNG（累積 + upper + positive）
    if "All" in per_group_data:
        data_all = per_group_data["All"]
        Ns_all            = data_all["Ns"]
        total_type_all    = data_all["total_type"]
        positive_type_all = data_all["positive_type"]

        if Ns_all:
            cumulative_all = []
            acc_all = 0
            for t in total_type_all:
                acc_all += t
                cumulative_all.append(acc_all)

            png_name_all   = "All_2025_cumulative_typecount.png"
            output_png_all = os.path.join(output_dir, png_name_all)

            plt.figure(figsize=(8, 4))

            plt.plot(
                Ns_all,
                cumulative_all,
                marker="o",
                linestyle="-",
                label="cumulative total N-gram types"
            )

            plt.plot(
                Ns_all,
                total_type_all,
                marker="x",
                linestyle="--",
                label="upper: total N-gram types"
            )

            plt.plot(
                Ns_all,
                positive_type_all,
                marker=".",
                linestyle="-.",
                label="(P(next|prefix) < 1.0) N-gram types "
            )

            plt.xlabel("N (n-gram length)")
            plt.ylabel("cumulative total N-gram types")
            plt.title(" Group=All / Type Counts (2025)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_png_all, dpi=200)
            plt.close()
            print(f"# [Group=All] Saved cumulative plot to: {output_png_all}")

    # ignore-ids の確認（デバッグ用）
    all_ignore = data_loader_all.load_all_ignore_ids(settings_root)
    print(all_ignore.get("GCU", set()))

    # 7) 全グループ重ね描き（All/Unknown 除外、縦軸は positive_type）
    combined_groups = [g for g in sorted(per_group_data.keys(), key=group_sort_key)
                       if g not in ("All", "Unknown")]

    if combined_groups:
        # 7-1) Night を含むバージョン
        plt.figure(figsize=(8, 4))

        for g in combined_groups:
            data = per_group_data[g]
            Ns            = data["Ns"]
            positive_type = data["positive_type"]
            if not Ns:
                continue
            plt.plot(Ns, positive_type, marker=".", linestyle="-", label=g)

        plt.xlabel("N (n-gram length)")
        plt.ylabel("Number of N-gram types with P(next|prefix) < 1.0")
        plt.title("Group=ALL / Type Counts (2025)(except All/Unknown)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        all_png = os.path.join(output_dir, "ALL_GROUPS_2025_typecount.png")
        plt.savefig(all_png, dpi=200)
        plt.close()
        print(f"# [All groups] Saved plot to: {all_png}")

        # 7-2) Night を除外したバージョン
        combined_groups_no_night = [g for g in combined_groups if g != "Night"]

        if combined_groups_no_night:
            plt.figure(figsize=(8, 4))

            for g in combined_groups_no_night:
                data = per_group_data[g]
                Ns            = data["Ns"]
                positive_type = data["positive_type"]
                if not Ns:
                    continue
                plt.plot(Ns, positive_type, marker=".", linestyle="-", label=g)

            plt.xlabel("N (n-gram length)")
            plt.ylabel("Number of N-gram types with P(next|prefix) < 1.0")
            plt.title("Group=ALL / Type Counts (2025)(except All/Unknown/Night)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            all_png_nonight = os.path.join(output_dir, "ALL_GROUPS_2025_typecount_noNight.png")
            plt.savefig(all_png_nonight, dpi=200)
            plt.close()
            print(f"# [All groups except Night] Saved plot to: {all_png_nonight}")


if __name__ == "__main__":
    main()
