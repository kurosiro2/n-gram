#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
past-shifts から determinism を計算

determinism = 1 - positive_type / total_type

--boxplot-det1 を付けると
det=1 の N-gram frequency 分布を
Nごとの箱ひげ図で出力
"""

import sys
import os
from collections import Counter, defaultdict

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader
from found_shifts_group import ngram_counts_by_group

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEAR_START = 20190101
YEAR_END = 20251231


def positive_stats(n_counter: Counter, nm1_counter: Counter):

    total_type = 0
    positive_type = 0

    for gram, c in n_counter.items():

        if len(gram) < 2:
            continue

        prefix = gram[:-1]
        base = nm1_counter.get(prefix, 0)

        if base <= 0:
            continue

        total_type += 1

        p = c / base

        if p < 1.0 - 1e-12:
            positive_type += 1

    return total_type, positive_type


def det1_frequency_list(n_counter: Counter, nm1_counter: Counter):

    freqs = []

    for gram, c in n_counter.items():

        if len(gram) < 2:
            continue

        prefix = gram[:-1]
        base = nm1_counter.get(prefix, 0)

        if base <= 0:
            continue

        p = c / base

        if abs(p - 1.0) < 1e-12:
            freqs.append(c)

    return freqs


def filter_seqs_by_date(seqs, start_date, end_date):

    filtered = {}

    for key, seq in seqs.items():

        sub = [(d, s) for (d, s) in seq if start_date <= d <= end_date]

        if sub:
            sub.sort(key=lambda t: t[0])
            filtered[key] = sub

    return filtered


def group_sort_key(g):

    if g == "All":
        return (0, "")
    if g == "Unknown":
        return (2, "")

    return (1, g.lower())


def main():

    if len(sys.argv) < 5:
        print(
            "python ngram_determinism_by_group.py "
            "[past_shift_file] [setting_dir] [N_max] [output_dir]"
        )
        sys.exit(1)

    past_shift_file = sys.argv[1]
    setting_dir = sys.argv[2]
    N_max = int(sys.argv[3])
    output_dir = sys.argv[4]

    boxplot_mode = "--boxplot-det1" in sys.argv

    os.makedirs(output_dir, exist_ok=True)

    print("# loading shifts")

    seqs = data_loader.load_past_shifts(past_shift_file)
    group_timeline = data_loader.load_staff_group_timeline(setting_dir)

    seqs_filtered = filter_seqs_by_date(seqs, YEAR_START, YEAR_END)

    print("# nurses with shifts:", len(seqs_filtered))

    shifts_by_staff = seqs_filtered
    groups_by_staff = group_timeline

    per_group_data = defaultdict(lambda: {"Ns": [], "determinism": []})

    # boxplot用
    per_group_det1_by_N = defaultdict(lambda: defaultdict(list))

    for N in range(2, N_max + 1):

        print("# N =", N)

        counters_N = ngram_counts_by_group(
            shifts_by_staff,
            groups_by_staff,
            N,
            date_from=YEAR_START,
            date_to=YEAR_END,
        )

        counters_Nm1 = ngram_counts_by_group(
            shifts_by_staff,
            groups_by_staff,
            N - 1,
            date_from=YEAR_START,
            date_to=YEAR_END,
        )

        for g in sorted(counters_N.keys(), key=group_sort_key):

            n_counter = counters_N[g]
            nm1_counter = counters_Nm1.get(g, Counter())

            total_t, pos_t = positive_stats(n_counter, nm1_counter)

            ratio = pos_t / total_t if total_t else 0
            determinism = 1 - ratio

            per_group_data[g]["Ns"].append(N)
            per_group_data[g]["determinism"].append(determinism)

            if boxplot_mode:

                freqs = det1_frequency_list(n_counter, nm1_counter)

                if freqs:
                    per_group_det1_by_N[g][N].extend(freqs)

    # determinism plot
    for g in sorted(per_group_data.keys(), key=group_sort_key):

        if g in ("All", "Unknown"):
            continue

        data = per_group_data[g]

        Ns = data["Ns"]
        determinism = data["determinism"]

        if not Ns:
            continue

        output_png = os.path.join(output_dir, f"{g}_determinism.png")

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 5))

        plt.plot(
            Ns,
            determinism,
            marker="o",
            linestyle="-",
        )

        plt.xlabel("N")
        plt.ylabel("Determinism (1 = fully determined)")
        plt.title(f"Group={g}")

        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()

        print("# saved:", output_png)

    # boxplot
    if boxplot_mode:

        print("# creating boxplots")

        for g in sorted(per_group_det1_by_N.keys(), key=group_sort_key):

            if g in ("All", "Unknown"):
                continue

            data_by_N = per_group_det1_by_N[g]

            if not data_by_N:
                continue

            Ns = sorted(data_by_N.keys())
            box_data = [data_by_N[N] for N in Ns]

            output_png = os.path.join(
                output_dir,
                f"{g}_det1_frequency_boxplot.png"
            )

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(10, 5))

            plt.boxplot(
                box_data,
                positions=Ns,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor="#dddddd", color="black"),
                medianprops=dict(color="#ff7f0e", linewidth=2),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
            )

            plt.xlabel("N (n-gram length)")
            plt.ylabel("Frequency of deterministic next-shift patterns")
            plt.title("deterministic next-shift pattern frequency")

            plt.grid(axis="y")

            plt.tight_layout()
            plt.savefig(output_png, dpi=200)
            plt.close()

            print("# saved:", output_png)


if __name__ == "__main__":
    main()
