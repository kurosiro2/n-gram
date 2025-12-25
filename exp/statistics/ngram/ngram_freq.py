#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALL グループ限定 n-gram 解析スクリプト
  ↓
グループ固定をやめて、期間（タイムライン）に合わせたグループで
任意/全グループを同じ処理で出せるようにした版。

従来の使い方（互換）:
  python ngram_freq.py [past_shifts] [settings] [output_png]

追加:
  --group を指定するとそのグループだけ（複数可）
  指定しなければ自動検出した全グループ（All/Unknown は除外。必要なら include オプションあり）

例:
  python ngram_freq.py exp/.../past-shifts/GCU.lp exp/.../group-settings/GCU/ out/freq.png
  python ngram_freq.py exp/.../past-shifts/GCU.lp exp/.../group-settings/GCU/ out/freq.png --group Leader --group Newcomer
"""

import sys
import os
import argparse
from collections import Counter, defaultdict

# -------------------------------------------------
# import path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader
from ngram_past_shifts_group import ngram_counts_by_group

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= 設定（必要ならここだけ編集） =================
YEAR_START = 20240101
YEAR_END   = 20251231

N_MIN = 1
N_MAX = 5

TOP_K_MAX = 300
BOXPLOT_SHOW_FLIERS = False
TARGET_CUM_SHARE = 0.90   # ★ 90%（0.95 などもOK）


# ================= ユーティリティ =================
def filter_seqs_by_date(seqs, start, end):
    out = {}
    for k, seq in seqs.items():
        sub = [(d, s) for d, s in seq if start <= d <= end]
        if sub:
            out[k] = sorted(sub, key=lambda x: x[0])
    return out


def gram_contains_subseq(longer, shorter):
    n, m = len(shorter), len(longer)
    for i in range(m - n + 1):
        if longer[i:i+n] == shorter:
            return True
    return False


def normalize_png(path):
    return path if path.lower().endswith(".png") else path + ".png"


def out_with_suffix(base, suf):
    b, e = os.path.splitext(base)
    return b + suf + e


def collect_past_files(past_input):
    """dir / file 両対応"""
    if os.path.isdir(past_input):
        return {
            os.path.splitext(f)[0]: os.path.join(past_input, f)
            for f in os.listdir(past_input)
            if f.endswith(".lp")
        }
    if os.path.isfile(past_input):
        ward = os.path.splitext(os.path.basename(past_input))[0]
        return {ward: past_input}
    raise FileNotFoundError(past_input)


def resolve_setting(settings_root, ward, single_mode):
    """
    settings_root の渡し方が複数あるので吸収する。

    - settings_root が setting.lp ならそのまま
    - 単一病棟で settings_root が ward ディレクトリならそのまま
    - 全病棟ルートなら settings_root/<ward> を探す
    - 単一病棟で settings_root が YYYY-MM-DD/ などでも data_loader が対応する想定で返す
    """
    if os.path.isfile(settings_root):
        return settings_root

    if os.path.isdir(settings_root):
        base = os.path.basename(os.path.normpath(settings_root))
        if single_mode and base == ward:
            return settings_root

        cand = os.path.join(settings_root, ward)
        if os.path.exists(cand):
            return cand

        return settings_root

    return None


def min_k_reaching_share(ratios, ks, target_share):
    """ratios(0..1) が target_share 以上になる最小Kを返す。なければ None"""
    for k, r in zip(ks, ratios):
        if r >= target_share:
            return k
    return None


def safe_tag(s: str) -> str:
    """ファイル名用に安全化"""
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts")
    ap.add_argument("settings")
    ap.add_argument("output_png")
    ap.add_argument("--group", action="append", default=[],
                    help="対象グループ（複数可）。省略すると自動検出した全グループ。")
    ap.add_argument("--include-all", action="store_true",
                    help="自動検出時に 'All' も含める（デフォルト除外）")
    ap.add_argument("--include-unknown", action="store_true",
                    help="自動検出時に 'Unknown' も含める（デフォルト除外）")
    args = ap.parse_args()

    past_input    = args.past_shifts
    settings_root = args.settings
    out_png_base  = normalize_png(args.output_png)

    os.makedirs(os.path.dirname(out_png_base) or ".", exist_ok=True)

    # ---- 入力収集 ----
    ward_files = collect_past_files(past_input)
    single_mode = os.path.isfile(past_input)

    ward_data = {}
    for ward, fpath in ward_files.items():
        setting_path = resolve_setting(settings_root, ward, single_mode)
        if setting_path is None:
            print(f"[skip] setting not found: ward={ward}")
            continue

        try:
            timeline = data_loader.load_staff_group_timeline(setting_path)
        except Exception as e:
            print(f"[skip] setting load failed: {ward} ({e})")
            continue

        seqs = data_loader.load_past_shifts(fpath)
        seqs = filter_seqs_by_date(seqs, YEAR_START, YEAR_END)
        if not seqs:
            print(f"[skip] no shifts in range: ward={ward}")
            continue

        ward_data[ward] = (seqs, timeline)

    if not ward_data:
        print("No data loaded.")
        return

    print("Loaded wards:", list(ward_data.keys()))

    # ============================================================
    # 1) まず「対象グループ一覧」を決める
    # ============================================================
    discovered_groups = set()
    for n in range(N_MIN, N_MAX + 1):
        for seqs, tl in ward_data.values():
            cnt_by_group = ngram_counts_by_group(seqs, tl, n)
            discovered_groups.update(cnt_by_group.keys())

    if not args.include_all and "All" in discovered_groups:
        discovered_groups.remove("All")
    if not args.include_unknown and "Unknown" in discovered_groups:
        discovered_groups.remove("Unknown")

    if args.group:
        groups_to_run = args.group
        print("[groups] manual:", groups_to_run)
    else:
        groups_to_run = sorted(discovered_groups, key=lambda x: str(x).lower())
        print("[groups] auto:", groups_to_run)

    if not groups_to_run:
        print("No target groups.")
        return

    # ============================================================
    # 2) グループ別に n-gram 集計（All固定を撤廃）
    #    aggregated_by_group[g][n] = Counter
    # ============================================================
    aggregated_by_group = {g: {} for g in groups_to_run}
    sorted_grams_by_group = {g: {} for g in groups_to_run}

    for n in range(N_MIN, N_MAX + 1):
        # wards を回して、各 ward の groupカウンタを足し合わせる
        tmp_by_group = {g: Counter() for g in groups_to_run}

        for seqs, tl in ward_data.values():
            cnt_by_group = ngram_counts_by_group(seqs, tl, n)
            for g in groups_to_run:
                if g in cnt_by_group:
                    tmp_by_group[g].update(cnt_by_group[g])

        for g in groups_to_run:
            total = tmp_by_group[g]
            if total:
                aggregated_by_group[g][n] = total
                print(f"[group={g} n={n}] types={len(total)}, total={sum(total.values())}")
            else:
                print(f"[group={g} n={n}] no data")

    # ============================================================
    # 3) グループごとにプロット4種を出力
    # ============================================================
    for g in groups_to_run:
        if not aggregated_by_group[g]:
            print(f"[skip] group={g} (no n-gram data)")
            continue

        tag = safe_tag(g)
        out_png = out_with_suffix(out_png_base, f"_{tag}")

        aggregated_by_n = aggregated_by_group[g]
        sorted_grams_by_n = {}

        # ================= (1) Top-K + K@share =================
        topk_by_n = {}
        k_at_share = {}  # {n: K@90}

        for n, counter in aggregated_by_n.items():
            items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            sorted_grams_by_n[n] = [gram for gram, _ in items]

            total = sum(v for _, v in items)
            if total <= 0:
                continue

            Ks, ratios, cum = [], [], 0
            for i, (_, v) in enumerate(items[:TOP_K_MAX]):
                cum += v
                Ks.append(i + 1)
                ratios.append(cum / total)

            topk_by_n[n] = (Ks, ratios)

            k_val = min_k_reaching_share(ratios, Ks, TARGET_CUM_SHARE)
            if k_val is None:
                k_val = Ks[-1] if Ks else 0
            k_at_share[n] = k_val

        # ---- (1) 描画 ----
        plt.figure(figsize=(8, 5))
        for n in sorted(topk_by_n.keys()):
            Ks, ratios = topk_by_n[n]
            plt.plot(Ks, [r * 100 for r in ratios], label=f"{n}-gram")

        plt.axhline(y=TARGET_CUM_SHARE * 100, linestyle="--")
        plt.xlabel("K")
        plt.ylabel("Cumulative frequency share (%)")
        plt.title(f"Top-K cumulative frequency share (group={g})")
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        # ================= (1.6) n vs K@share =================
        out_kshare = out_with_suffix(out_png, f"_k{int(TARGET_CUM_SHARE*100)}")

        ns = sorted(k_at_share.keys())
        ys = [k_at_share[n] for n in ns]

        plt.figure(figsize=(max(8, 0.55 * len(ns)), 4.8))
        plt.plot(ns, ys, marker="o", linestyle="-")
        plt.xticks(ns)
        plt.xlabel("n (n-gram)")
        plt.ylabel(f"K@{int(TARGET_CUM_SHARE*100)}%")
        plt.title(f"K needed to reach {int(TARGET_CUM_SHARE*100)}% (group={g})")
        plt.grid(True)

        for x, y in zip(ns, ys):
            plt.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 6),
                         ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(out_kshare, dpi=200)
        plt.close()

        # ================= (2) 部分列含有 =================
        overlap_by_n = {}

        for n in range(N_MIN, N_MAX):
            base = sorted_grams_by_n.get(n, [])
            if not base:
                continue

            Ks, ratios = [], []
            for k in range(1, min(TOP_K_MAX, len(base)) + 1):
                longer = []
                for m in range(n + 1, N_MAX + 1):
                    longer.extend(sorted_grams_by_n.get(m, [])[:k])

                hit = 0
                for gram in base[:k]:
                    if any(gram_contains_subseq(lg, gram) for lg in longer):
                        hit += 1

                Ks.append(k)
                ratios.append(hit / k)

            overlap_by_n[n] = (Ks, ratios)

        out_sub = out_with_suffix(out_png, "_subseq")
        plt.figure(figsize=(8, 5))
        for n in sorted(overlap_by_n.keys()):
            Ks, ratios = overlap_by_n[n]
            plt.plot(Ks, [r * 100 for r in ratios], label=f"{n}-gram")
        plt.xlabel("K")
        plt.ylabel("Included in longer n-grams (%)")
        plt.title(f"Subsequence inclusion ratio (group={g})")
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(out_sub, dpi=200)
        plt.close()

        # ================= (3) 箱ひげ図（生頻度） =================
        freq_by_n = {n: list(c.values()) for n, c in aggregated_by_n.items() if c}

        out_box = out_with_suffix(out_png, "_boxplot_freq")
        ns_box = sorted(freq_by_n.keys())
        data = [freq_by_n[n] for n in ns_box]

        plt.figure(figsize=(max(8, 0.6 * len(ns_box)), 5))
        plt.boxplot(
            data,
            labels=[str(n) for n in ns_box],
            showfliers=BOXPLOT_SHOW_FLIERS
        )
        plt.xlabel("n")
        plt.ylabel("n-gram frequency (count)")
        plt.title(f"Distribution of n-gram frequencies (group={g})")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(out_box, dpi=200)
        plt.close()

        print("[saved group]", g)
        print(" ", out_png)
        print(" ", out_kshare)
        print(" ", out_sub)
        print(" ", out_box)


if __name__ == "__main__":
    main()
