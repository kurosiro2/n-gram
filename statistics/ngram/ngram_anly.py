#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
n-gram frequency distribution (freq_share) comparison:
Compare past-shifts vs found-model and plot a bar chart.

★対応
- 入力は英語勤務記号 (D, SE, SN, WR, ...) のまま読む
- 図のラベルは日本語 (日勤, 短準夜, 短深夜, 週休, ...) にして出す
- matplotlib に日本語フォントを自動設定（見つかるものを優先順で採用）
- 横軸ラベルはデフォで横向き（rotation=0）
- フォントサイズを全体でまとめて大きくできる: --base-fs

Usage:
  python ngram_freq_dist_compare_ja.py past-shifts.lp setting.lp found-model.lp N output.png
"""

import sys
import os
import re
import math
import argparse
from collections import defaultdict, Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader


# -------------------------------------------------------------
# Shift label mapping
# -------------------------------------------------------------
SHIFT_LABELS = {
    "短準夜" : {"ja": "短準夜", "en": "SE"},
    "短深夜" : {"ja": "短深夜", "en": "SN"},
    "週休"   : {"ja": "週休",   "en": "WR"},
    "日勤"   : {"ja": "日勤",   "en": "D" },

    "長日勤" : {"ja": "長日勤", "en": "LD"},
    "早出"   : {"ja": "早日勤",   "en": "EM"},
    "遅出"   : {"ja": "遅日勤",   "en": "LM"},
    "準夜"   : {"ja": "準夜",   "en": "E" },
    "深夜"   : {"ja": "深夜",   "en": "N" },
    "公休"   : {"ja": "祝日",   "en": "PH"},
}

EN2JA = {}
for _k, v in SHIFT_LABELS.items():
    ja = v.get("ja")
    en = v.get("en")
    if isinstance(ja, str) and isinstance(en, str) and en:
        EN2JA[en] = ja

VALID_SHIFTS = set(EN2JA.keys())


# -------------------------------------------------------------
# found-model regex
# -------------------------------------------------------------
PAT_EXT   = re.compile(r'^out_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\("([^"]+)"\s*,\s*(\d+)\)\.')
PAT_STAFF = re.compile(r'^staff\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*"[^"]*"\s*,\s*"([^"]+)"')


# -------------------------------------------------------------
# Japanese font setup
# -------------------------------------------------------------
def setup_japanese_font():
    candidates = [
        "IPAexGothic",
        "IPAPGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "TakaoGothic",
        "Hiragino Sans",
        "Yu Gothic",
        "MS Gothic",
        "Meiryo",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            print(f"# matplotlib font set to: {name}")
            return name
    print("[WARN] No Japanese font found. CJK glyphs may be missing.")
    return None


# -------------------------------------------------------------
# Global font scaling
# -------------------------------------------------------------
def apply_global_fontsizes(base_fs: int):
    """
    base_fs を基準に、全体のフォントサイズをまとめて上げる。
    さらに必要なら個別オプションで上書き可能。
    """
    mpl.rcParams["font.size"] = base_fs
    mpl.rcParams["axes.titlesize"] = int(base_fs * 1.25)
    mpl.rcParams["axes.labelsize"] = int(base_fs * 1.15)
    mpl.rcParams["xtick.labelsize"] = int(base_fs * 1.00)
    mpl.rcParams["ytick.labelsize"] = int(base_fs * 1.00)
    mpl.rcParams["legend.fontsize"] = int(base_fs * 1.00)


# -------------------------------------------------------------
# Label helpers
# -------------------------------------------------------------
def wrap_label(s: str, width: int) -> str:
    if width is None or width <= 0:
        return s
    if "\n" in s:
        return s
    return "\n".join(s[i:i+width] for i in range(0, len(s), width))


def stagger_labels(labels):
    out = []
    for i, lab in enumerate(labels):
        out.append(("\n" + lab) if (i % 2 == 1) else lab)
    return out


# -------------------------------------------------------------
# Core utilities
# -------------------------------------------------------------
def filter_seqs_by_date(seqs_dict, start_date, end_date):
    if start_date is None and end_date is None:
        return seqs_dict
    filtered = {}
    for key, seq in seqs_dict.items():
        sub = []
        for d, s in seq:
            if start_date is not None and d < start_date:
                continue
            if end_date is not None and d > end_date:
                continue
            sub.append((d, s))
        if sub:
            sub.sort(key=lambda t: t[0])
            filtered[key] = sub
    return filtered


def ngram_counts_by_group_past(seqs_dict, group_timeline, n):
    from data_loader import get_groups_for_date
    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    for (nid, name), seq in sorted(seqs_dict.items(), key=lambda kv: kv[0][1]):
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            window = seq[i:i+n]
            dates  = [d for d, _ in window]
            shifts = [s for _, s in window]
            if any(s not in VALID_SHIFTS for s in shifts):
                continue

            ref_date = dates[-1]
            gram = tuple(shifts)
            groups = get_groups_for_date(name, ref_date, group_timeline, nurse_id=nid)
            if not groups:
                groups = {"Unknown"}

            gset = set(groups)
            gset.add("All")
            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


def ngram_counts_by_group_found(series_by_staff, groups_by_staff, n):
    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    for sidx, series in series_by_staff.items():
        if len(series) < n:
            continue
        for i in range(len(series) - n + 1):
            gram = tuple(series[i:i+n])
            if any(s not in VALID_SHIFTS for s in gram):
                continue

            gset = set(groups_by_staff.get(sidx, {"Unknown"}))
            gset.add("All")
            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


def compute_freq_share(counter: Counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def js_distance_sqrt(p: dict, q: dict):
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0

    def kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0.0 and bk > 0.0:
                s += ak * math.log(ak / bk)
        return s

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return math.sqrt(max(jsd, 0.0))


def shift_code_to_ja(code: str) -> str:
    return EN2JA.get(code, code)


def gram_to_label_ja(gram) -> str:
    return "-".join(shift_code_to_ja(x) for x in gram)


def parse_found_model(found_model_file):
    ext_by_staff = defaultdict(list)
    groups_by_staff = defaultdict(set)
    staff_info = {}

    with open(found_model_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = PAT_EXT.match(line)
            if m:
                sidx, didx, shift = int(m[1]), int(m[2]), m[3]
                ext_by_staff[sidx].append((didx, shift))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname, sidx = m[1], int(m[2])
                groups_by_staff[sidx].add(gname)
                continue

            m = PAT_STAFF.match(line)
            if m:
                sidx, name, nurse_id = int(m[1]), m[2], m[3]
                staff_info[sidx] = (name, nurse_id)
                continue

    series_by_staff = {}
    for sidx, pairs in ext_by_staff.items():
        pairs.sort(key=lambda t: t[0])
        series_by_staff[sidx] = [s for _, s in pairs]

    return series_by_staff, groups_by_staff, staff_info


def pick_grams(counter_past, counter_found, basis="union", topk=30, min_count=1):
    keys = set(counter_past.keys()) | set(counter_found.keys())
    cand = []
    for g in keys:
        cp = counter_past.get(g, 0)
        cf = counter_found.get(g, 0)
        if (cp + cf) < min_count:
            continue

        if basis == "past":
            score = cp
        elif basis == "found":
            score = cf
        else:
            score = cp + cf

        cand.append((g, score, cp, cf))

    cand.sort(key=lambda t: (-t[1], -t[2], -t[3], gram_to_label_ja(t[0])))
    return [g for g, _, _, _ in cand[:topk]]


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts.lp")
    ap.add_argument("setting_lp", help="setting.lp")
    ap.add_argument("found_model", help="found-model.lp")
    ap.add_argument("N", type=int, help="n-gram size (>=1)")
    ap.add_argument("output_png", help="output png path")

    ap.add_argument("--group", default="All")
    ap.add_argument("--date-start", type=int, default=None)
    ap.add_argument("--date-end",   type=int, default=None)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--basis", choices=["past", "found", "union"], default="union")
    ap.add_argument("--min-count", type=int, default=1)

    ap.add_argument("--fig-w", type=float, default=None)
    ap.add_argument("--fig-h", type=float, default=6.0)

    # ★全体フォントサイズ
    ap.add_argument("--base-fs", type=int, default=16, help="global base font size (default: 18)")

    # 個別（必要なら上書き）
    ap.add_argument("--fontsize-x", type=int, default=None)
    ap.add_argument("--fontsize-legend", type=int, default=None)

    # 横軸ラベル
    ap.add_argument("--x-rotate", type=float, default=0.0)
    ap.add_argument("--x-stagger", action="store_true")
    ap.add_argument("--wrap", type=int, default=0)

    args = ap.parse_args()

    # 日本語フォント + 全体フォント拡大
    setup_japanese_font()
    apply_global_fontsizes(args.base_fs)

    if args.N < 1:
        print("[ERROR] N must be >= 1", file=sys.stderr)
        sys.exit(1)

    for p in [args.past_shifts, args.setting_lp, args.found_model]:
        if not os.path.isfile(p):
            print(f"[ERROR] file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # 個別上書き（指定があるときだけ）
    if args.fontsize_x is not None:
        mpl.rcParams["xtick.labelsize"] = args.fontsize_x
    if args.fontsize_legend is not None:
        mpl.rcParams["legend.fontsize"] = args.fontsize_legend

    # 1) past
    seqs = data_loader.load_past_shifts(args.past_shifts)
    seqs = filter_seqs_by_date(seqs, args.date_start, args.date_end)
    if not seqs:
        print("[ERROR] No past-shifts after filtering.", file=sys.stderr)
        sys.exit(1)

    group_timeline = data_loader.load_staff_group_timeline(args.setting_lp)
    counters_past = ngram_counts_by_group_past(seqs, group_timeline, args.N)
    counter_past_g = counters_past.get(args.group, Counter())
    freq_past = compute_freq_share(counter_past_g)

    # 2) found
    series_by_staff, groups_by_staff, _staff_info = parse_found_model(args.found_model)
    counters_found = ngram_counts_by_group_found(series_by_staff, groups_by_staff, args.N)
    counter_found_g = counters_found.get(args.group, Counter())
    freq_found = compute_freq_share(counter_found_g)

    # 3) JS distance
    jsd = js_distance_sqrt(freq_past, freq_found)

    # 4) pick grams
    grams = pick_grams(counter_past_g, counter_found_g,
                       basis=args.basis, topk=args.topk, min_count=args.min_count)
    if not grams:
        print("[ERROR] No grams to plot.", file=sys.stderr)
        sys.exit(1)

    labels = [gram_to_label_ja(g) for g in grams]
    if args.wrap and args.wrap > 0:
        labels = [wrap_label(s, args.wrap) for s in labels]
    if args.x_stagger:
        labels = stagger_labels(labels)

    x = list(range(len(grams)))
    width = 0.42
    y_past = [freq_past.get(g, 0.0) for g in grams]
    y_found = [freq_found.get(g, 0.0) for g in grams]

    fig_w = max(8.0, len(grams) * 0.55) if args.fig_w is None else args.fig_w

    plt.figure(figsize=(fig_w, args.fig_h))
    plt.bar([xi - width / 2 for xi in x], y_past, width=width, label="MANUAL")
    plt.bar([xi + width / 2 for xi in x], y_found, width=width, label="AUTO")

    plt.xticks(x, labels, rotation=args.x_rotate, ha="center")
    plt.ylabel("freq_share")
    plt.title(f'{args.N}-gram')
    plt.legend()

    if args.x_rotate == 0:
        plt.gcf().subplots_adjust(bottom=0.25)

    plt.tight_layout()
    plt.savefig(args.output_png)
    plt.close()

    print(f"# saved: {args.output_png}")


if __name__ == "__main__":
    main()
