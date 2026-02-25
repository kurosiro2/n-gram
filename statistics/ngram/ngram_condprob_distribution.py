#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年だけに絞って、病棟を無視して
「同じ看護師グループごとに」weighted_positive_freq の曲線を出すスクリプト。

・全病棟の past-shifts を読む
・各病棟ごとに ngram_counts_by_group で N-gram カウント
・グループごとにカウンタを合算 → 全病棟をまとめたのと同じ結果になる
・グループごとに PNG を出力（All/Unknown は PNG からは除外）
・グローバル CSV: group,N,total,positive,ratio

デバッグ機能:
・N が大きいところで P(next|prefix) < 1.0 の N-gram が残っている場合、
  その上位パターンと、どの病棟のどの看護師が出しているかを表示する。

依存:
    - data_loader.py
    - data_loader_all.py
    - ngram_past_shifts_group.py  (ngram_counts_by_group を利用)
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


def weighted_positive_freq(n_counter: Counter, nm1_counter: Counter, N: int):
    """
    N-gram の出現回数 freq を集計し、
    そのうち P(next|prefix) < 1.0 の freq の合計を返す。
    N=1 の場合は nm1_counter が無いので total を positive とする。
    """
    total = sum(n_counter.values())

    # N=1 → 条件付き確率なし → 全て「positive」とみなす
    if N == 1:
        return total, total

    positive = 0

    for gram, c in n_counter.items():
        if len(gram) < 2:
            continue
        prefix = gram[:-1]
        base = nm1_counter.get(prefix, 0)
        if base <= 0:
            continue

        p = c / base
        # ほぼ 1.0 は除外
        if p < 1.0 - 1e-12:
            positive += c

    return total, positive


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
            dates = [d for (d, _) in seq]
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

    ward_seqs_2025 は全病棟の 2025 年シフト:
      { ward_name: {(nurse_id, name): [(date, shift), ...]} }
    """
    rows = []

    for gram, c in n_counter.items():
        if len(gram) < 2:
            continue

        prefix = gram[:-1]
        base = nm1_counter.get(prefix, 0)
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
            "python ngram_condprob_by_group_allwards_2025.py "
            "[past_shifts_dir] [settings_root] [N_max] [output_dir] [global_csv]"
        )
        sys.exit(1)

    past_shifts_dir = sys.argv[1]  # 例: exp/2019-2025-data/real-name/past-shifts
    settings_root   = sys.argv[2]  # 例: exp/2019-2025-data/real-name/group-settings
    N_max           = int(sys.argv[3])
    output_dir      = sys.argv[4]
    global_csv_path = sys.argv[5]

    os.makedirs(output_dir, exist_ok=True)

    # 1) 全病棟のタイムライン情報を読み込み
    #    { ward_name: { name: [(start_date, set(groups)), ...] } }
    all_timelines = data_loader_all.load_all_staff_group_timelines(settings_root)
    print("# Wards found in settings_root:", ", ".join(sorted(all_timelines.keys())))

    # 2) past-shifts（ディレクトリ or 単一ファイル）から ward ごとの .lp を探す
    ward_shift_files = {}

    if os.path.isdir(past_shifts_dir):
        # これまで通り: ディレクトリ配下の .lp を全部読む
        for fname in sorted(os.listdir(past_shifts_dir)):
            if not fname.endswith(".lp"):
                continue
            ward_name = os.path.splitext(fname)[0]
            shift_path = os.path.join(past_shifts_dir, fname)
            ward_shift_files[ward_name] = shift_path
    else:
        # 単一ファイルが渡された場合: その病棟だけ対象にする
        if not past_shifts_dir.endswith(".lp"):
            print(f"[ERROR] past_shifts_dir is not a directory and not a .lp file: {past_shifts_dir}")
            sys.exit(1)
        ward_name = os.path.splitext(os.path.basename(past_shifts_dir))[0]
        ward_shift_files[ward_name] = past_shifts_dir

    print("# Wards found in past_shifts_dir:", ", ".join(sorted(ward_shift_files.keys())))

    # 3) 各病棟のシフト（2025年だけ）を先に読み込んでおく
    #    ward_seqs_2025: { ward_name: {(nid, name): [(date, shift), ...]} }
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
        print("group,N,total,positive,ratio", file=csv_fp)

        # グループごとの曲線を保持:
        # { group: {"Ns": [...], "total": [...], "positive": [...], "ratio": [...]} }
        per_group_data = defaultdict(lambda: {"Ns": [], "total": [], "positive": [], "ratio": []})

        # 5) N = 1..N_max について、全病棟ぶんのカウンタを合算
        for N in range(1, N_max + 1):
            print(f"# ==== N={N} ====")

            # group -> Counter (aggregated across wards)
            aggregated_N   = defaultdict(Counter)
            aggregated_Nm1 = defaultdict(Counter)  # N>=2 のときだけ有効

            # (1) 各病棟で ngram_counts_by_group を計算し、グループごとに加算
            for ward_name, seqs in sorted(ward_seqs_2025.items()):
                group_timeline = all_timelines[ward_name]

                # N-gram
                counters_N_ward = ngram_counts_by_group(seqs, group_timeline, N)
                for g, ctr in counters_N_ward.items():
                    aggregated_N[g].update(ctr)

                # (N-1)-gram（条件付き確率の分母用）
                if N > 1:
                    counters_Nm1_ward = ngram_counts_by_group(seqs, group_timeline, N - 1)
                    for g, ctr in counters_Nm1_ward.items():
                        aggregated_Nm1[g].update(ctr)

            # (2) グループごとに weighted_positive_freq を計算
            for g in sorted(aggregated_N.keys(), key=group_sort_key):
                n_counter   = aggregated_N[g]
                nm1_counter = aggregated_Nm1[g] if N > 1 else Counter()

                total, positive = weighted_positive_freq(n_counter, nm1_counter, N)
                ratio = positive / total if total > 0 else 0.0

                per_group_data[g]["Ns"].append(N)
                per_group_data[g]["total"].append(total)
                per_group_data[g]["positive"].append(positive)
                per_group_data[g]["ratio"].append(ratio)

                # CSV: ward はまとめているので "ALL_WARDS" などに固定してもいいが、
                # ここでは病棟を無視したいので列には group だけ出す仕様にしている。
                print(f"{g},{N},{total},{positive},{ratio:.6f}", file=csv_fp)

                # ★ デバッグ出力:
                #    N が大きくても positive > 0 のグループについて、
                #    P(next|prefix) < 1.0 の N-gram を詳しく見てみる。
                #    （閾値は N>=15 にしている）
                if N >= 20 and positive > 0:
                    dump_positive_ngrams_for_group(
                        ward_seqs_2025=ward_seqs_2025,
                        group_name=g,
                        N=N,
                        n_counter=n_counter,
                        nm1_counter=nm1_counter,
                        max_rows=20,
                    )

    # 6) グループごとの PNG を出力（All / Unknown は PNG スキップ）
    for g in sorted(per_group_data.keys(), key=group_sort_key):
        if g in ("All", "Unknown"):
            # 集計自体は CSV に残っているので、PNG だけスキップ
            continue

        data = per_group_data[g]
        Ns            = data["Ns"]
        total_list    = data["total"]     # 理論上の上限 (全 N-gram 数)
        positive_list = data["positive"]  # P<1.0 の weighted count

        if not Ns:
            continue

        # 基準値として avg_total は計算するが、
        # 補助線は「各 N ごとの upper（total_list）に対する割合カーブ」に変更
        if total_list:
            base_value = sum(total_list) / len(total_list)
        else:
            base_value = 0.0

        # 傾きが最初にプラスになる N
        slope_up_N = None
        if len(positive_list) >= 2:
            for i in range(1, len(positive_list)):
                if positive_list[i] > positive_list[i - 1]:
                    slope_up_N = Ns[i]
                    break

        png_name   = f"{g}_2025.png"
        output_png = os.path.join(output_dir, png_name)

        plt.figure(figsize=(8, 4))

        # メイン：P<1.0 の weighted count の曲線
        line_label = "frequency count of P(next|prefix) < 1.0"
        plt.plot(Ns, positive_list, marker=".", linestyle="-", label=line_label)

        # ★ 上限値 (total_list) の曲線もプロット
        plt.plot(Ns, total_list, marker="x", linestyle="--",
                 label="upper(frequecy count of N-gram)")
        plt.xlabel("N (n-gram length)")
        plt.ylabel("frequency count of N-gram")
        plt.title(f"Group={g} (2025)")
        plt.grid(True)

        # ★ 補助線: 「各 N ごとの upper に対する 1〜5 割」のカーブ
        if total_list:
            for frac in [0.1,0.5]:
                curve = [frac * t for t in total_list]
                plt.plot(
                    Ns,
                    curve,
                    linestyle=":",
                    linewidth=1,
                    label=f"{int(frac * 100)}% of upper"
                )

        if slope_up_N is not None:
            plt.axvline(
                slope_up_N,
                linestyle=":",
                linewidth=1,
                label=f"slope>0 at N={slope_up_N}"
            )

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f"# [Group={g}] Saved plot to: {output_png}")

    # ★ 6.5) All グループだけをプロットする PNG も作る
    if "All" in per_group_data:
        data_all = per_group_data["All"]
        Ns_all            = data_all["Ns"]
        total_list_all    = data_all["total"]
        positive_list_all = data_all["positive"]

        if Ns_all:
            if total_list_all:
                base_value_all = sum(total_list_all) / len(total_list_all)
            else:
                base_value_all = 0.0

            slope_up_N_all = None
            if len(positive_list_all) >= 2:
                for i in range(1, len(positive_list_all)):
                    if positive_list_all[i] > positive_list_all[i - 1]:
                        slope_up_N_all = Ns_all[i]
                        break

            png_name_all   = "All_2025.png"
            output_png_all = os.path.join(output_dir, png_name_all)

            plt.figure(figsize=(8, 4))

            # P<1.0 の weighted count
            line_label_all = "frequency count of P(next|prefix) < 1.0"
            plt.plot(Ns_all, positive_list_all, marker=".", linestyle="-",
                     label=line_label_all)

            # 上限値（total_list_all）の曲線
            plt.plot(Ns_all, total_list_all, marker="x", linestyle="--",
                     label="upper(frequecy count of N-gram)")

            plt.xlabel("N (n-gram length)")
            plt.ylabel("frequency count of N-gram")
            plt.title("frequency count 2025/ Group=All (2025)")
            plt.grid(True)

            # ★ All についても「各 N ごとの upper に対する割合カーブ」
            if total_list_all:
                for frac in [0.1, 0.5]:
                    curve_all = [frac * t for t in total_list_all]
                    plt.plot(
                        Ns_all,
                        curve_all,
                        linestyle=":",
                        linewidth=1,
                        label=f"{int(frac * 100)}% of upper"
                    )

            if slope_up_N_all is not None:
                plt.axvline(
                    slope_up_N_all,
                    linestyle=":",
                    linewidth=1,
                    label=f"slope>0 at N={slope_up_N_all}"
                )

            plt.legend()
            plt.tight_layout()
            plt.savefig(output_png_all, dpi=200)
            plt.close()
            print(f"# [Group=All] Saved plot to: {output_png_all}")

    # ignore-ids の確認（デバッグ用）
    all_ignore = data_loader_all.load_all_ignore_ids(settings_root)
    print(all_ignore.get("GCU", set()))

    # 7) 全グループ重ね描き（All/Unknown 除外）
    combined_groups = [g for g in sorted(per_group_data.keys(), key=group_sort_key)
                       if g not in ("All", "Unknown")]

    if combined_groups:
        # 7-1) Night を含むバージョン
        plt.figure(figsize=(8, 4))

        for g in combined_groups:
            data = per_group_data[g]
            Ns            = data["Ns"]
            positive_list = data["positive"]
            if not Ns:
                continue
            plt.plot(Ns, positive_list, marker=".", linestyle="-", label=g)

        plt.xlabel("N (n-gram length)")
        plt.ylabel("frequency count of N-gram")
        plt.title("frequency count 2025: all groups (except All/Unknown)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        all_png = os.path.join(output_dir, "ALL_GROUPS_2025.png")
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
                positive_list = data["positive"]
                if not Ns:
                    continue
                plt.plot(Ns, positive_list, marker=".", linestyle="-", label=g)

            plt.xlabel("N (n-gram length)")
            plt.ylabel("frequency count of N-gram")
            plt.title("frequency count 2025:(except All/Unknown/Night)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            all_png_nonight = os.path.join(output_dir, "ALL_GROUPS_2025_noNight.png")
            plt.savefig(all_png_nonight, dpi=200)
            plt.close()
            print(f"# [All groups except Night] Saved plot to: {all_png_nonight}")


if __name__ == "__main__":
    main()
