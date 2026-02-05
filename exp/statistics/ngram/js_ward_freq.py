#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
病棟ごと（ALL扱い）の n-gram 出現確率分布を作り、
病棟×病棟の JS distance (sqrt(JSD), ln) をヒートマップ出力する。

使い方（理想形）:
  python exp/statistics/ngram/js_ward_freq.py \
      exp/2019-2025-data/past-shifts/ \
      exp/2019-2025-data/group-settings/

前提:
  - past-shifts/ 配下に *.lp が複数あり、ファイル名（拡張子除く）が病棟名
      例: past-shifts/GCU.lp, past-shifts/NICU.lp ...
    （中身は shift_data("nurse_id","name",YYYYMMDD,"S").）
  - group-settings/ 配下に病棟ディレクトリがあり、その下に YYYY-MM-DD/setting.lp
      例: group-settings/GCU/2024-10-13/setting.lp ...

ポイント:
  - 病棟の「所属判定」は group_settings（timeline）側で行うのではなく、
    「その病棟の past-shifts から作る分布」として扱う（=病棟単位の実績分布比較）
  - 病棟名は表示用に短い英語ラベルに変換する
      例: 4階南病棟 -> 4S, 7階北病棟 -> 7N
  - プロット対象の病棟は CLI で指定できる
      --plot-wards ICU,NICU,GCU,4S,7N,...  (表示ラベルで指定)

描画:
  - vmax デフォルト 0.5
  - セル文字色は値域の半分以上なら黒、未満なら白
  - 日本語フォント警告を避けるため、表示は英語ラベルを基本にする
"""

import os
import sys
import math
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# FONT
# =============================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
STAT_DIR = os.path.dirname(CURRENT_DIR)  # .../exp/statistics
if STAT_DIR not in sys.path:
    sys.path.insert(0, STAT_DIR)

import data_loader


# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"}

# =============================================================
# Ward label mapping (basename without .lp -> label)
# =============================================================
WARD_LABEL_MAP: Dict[str, str] = {
    # 一般病棟（階＋方角）
    "1階西病棟": "1W",
    "2階西病棟": "2W",
    "3階西病棟": "3W",
    "4階北病棟": "4N",
    "4階南病棟": "4S",
    "4階西病棟": "4W",
    "5階北病棟": "5N",
    "5階南病棟": "5S",
    "5階西病棟": "5W",
    "6階北病棟": "6N",
    "6階南病棟": "6S",
    "6階西病棟": "6W",
    "7階北病棟": "7N",
    "7階南病棟": "7S",
    "7階西病棟": "7W",

    # クリティカル系
    "集中治療室": "ICU",
    "救急外来": "ER",
    "手術部": "OR",
    "GCU": "GCU",
    "NICU": "NICU",

    # 外来・中央部門
    "外来": "OPD",
    "外来受付": "OPD-Front",
    "材料部": "CSSD",
    "感染制御部": "ICT",
    "医療の質・安全管理部": "QPS",
    "医療チームセンター": "MTC",
    "総合がん診療部": "CCU-Onc",
    "治験センター": "CRC",

    # 教育・管理
    "臨床教育部": "Clinical-Edu",
    "特定行為研修センター": "Spec-Train",
    "看護部管理室": "Nursing-Admin",
    "看護部管理室A": "Nursing-Admin-A",
    "看護部管理室B": "Nursing-Admin-B",
    "看護部管理室C": "Nursing-Admin-C",
    "師長勤務表": "Head-Nurse",
    "総合支援部A": "Support-A",
    "総合支援部B": "Support-B",
    "総合支援部C": "Support-C",

    # テスト・その他
    "富士通テスト病棟": "Test-Ward",
}

# =============================================================
# utilities
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def parse_csv_set(s: Optional[str]) -> Optional[Set[str]]:
    """
    "A,B,C" -> {"A","B","C"} / None
    空文字や None のときは None を返す（=フィルタしない）
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out = set()
    for tok in s.split(","):
        t = tok.strip()
        if t:
            out.add(t)
    return out or None


# =============================================================
# past-shifts dir loader (ward(label) -> seqs)
# =============================================================
def list_past_shift_files(past_dir: str) -> List[str]:
    """
    past_dir 配下から *.lp を集める（再帰はしない）
    """
    if os.path.isfile(past_dir):
        return [past_dir]

    if not os.path.isdir(past_dir):
        raise FileNotFoundError(f"past-shifts path not found: {past_dir}")

    files = []
    for name in sorted(os.listdir(past_dir)):
        p = os.path.join(past_dir, name)
        if os.path.isfile(p) and name.lower().endswith(".lp"):
            files.append(p)

    if not files:
        raise FileNotFoundError(f"No *.lp found under: {past_dir}")

    return files


def load_past_shifts_by_ward(
    past_dir: str,
    plot_wards: Optional[Set[str]],
    strict_label: bool = False,
) -> Tuple[Dict[str, Dict[Tuple[str, str], List[Tuple[int, str]]]], Dict[str, str]]:
    """
    past_dir 配下の *.lp を全て読み、 ward_label -> seqs を返す。
      - ward_label は WARD_LABEL_MAP により日本語名から変換
      - 未定義名は:
          strict_label=False: "RAW:<元名>" で残す（ただし plot_wards があるなら弾かれがち）
          strict_label=True : スキップ
    戻り値:
      ward_to_seqs: {label: seqs}
      label_to_raw: {label: raw_name}  (デバッグ/ログ用)
    """
    ward_to_seqs = {}
    label_to_raw = {}

    for f in list_past_shift_files(past_dir):
        raw = os.path.splitext(os.path.basename(f))[0]

        if raw in WARD_LABEL_MAP:
            label = WARD_LABEL_MAP[raw]
        else:
            if strict_label:
                print(f"# skip (no label mapping): {raw}")
                continue
            label = f"RAW:{raw}"

        # プロット対象フィルタ（指定がある場合）
        if plot_wards is not None and label not in plot_wards:
            continue

        seqs = data_loader.load_past_shifts(f)
        if not seqs:
            continue

        # 同一ラベルが重複しない想定だが念のため
        if label in ward_to_seqs:
            print(f"# WARNING: duplicate label '{label}' from file: {f} (already exists). Overwrite.")
        ward_to_seqs[label] = seqs
        label_to_raw[label] = raw

    if not ward_to_seqs:
        if plot_wards is None:
            raise SystemExit("No wards loaded. Check mapping or past-shifts directory.")
        else:
            raise SystemExit(f"No wards loaded. Check --plot-wards={sorted(plot_wards)} and mapping.")

    return ward_to_seqs, label_to_raw


# =============================================================
# n-gram count for a ward (ALL within ward)
# =============================================================
def count_ngrams_in_ward(ward_seqs, n: int, d1: int, d2: int) -> Counter:
    """
    その病棟の全看護師のシフトから n-gram を数える（ALL扱い）
    ※病棟境界は「ファイル分割」で既に守られている前提
    """
    c = Counter()
    for (_nid, _name), seq in ward_seqs.items():
        seq = normalize_seq(seq)
        if not seq:
            continue
        seq.sort(key=lambda x: x[0])

        sseq = [(d, s) for d, s in seq if within_range(d, d1, d2)]
        if len(sseq) < n:
            continue

        for i in range(len(sseq) - n + 1):
            gram = tuple(sseq[i + k][1] for k in range(n))
            c[gram] += 1
    return c


# =============================================================
# JS distance (sqrt, ln)
# =============================================================
def js_distance(c1: Counter, c2: Counter, alpha: float, support: str, n: int) -> float:
    """
    JS distance = sqrt(JSD), ln
    Laplace smoothing with alpha, support:
      - observed_ab: V = |keys union|
      - all        : V = |VALID_SHIFTS|^n
    """
    if support == "observed_ab":
        vocab = set(c1) | set(c2)
        V = len(vocab)
    else:
        V = len(VALID_SHIFTS) ** n

    if V == 0:
        return 0.0

    t1, t2 = sum(c1.values()), sum(c2.values())
    d1 = t1 + alpha * V
    d2 = t2 + alpha * V

    kl1 = 0.0
    kl2 = 0.0
    keys = set(c1) | set(c2)

    for k in keys:
        p = (c1.get(k, 0) + alpha) / d1
        q = (c2.get(k, 0) + alpha) / d2
        m = 0.5 * (p + q)
        kl1 += p * math.log(p / m)
        kl2 += q * math.log(q / m)

    if support == "all":
        rest = V - len(keys)
        if rest > 0:
            p0 = alpha / d1
            q0 = alpha / d2
            m0 = 0.5 * (p0 + q0)
            kl1 += rest * p0 * math.log(p0 / m0)
            kl2 += rest * q0 * math.log(q0 / m0)

    return math.sqrt(0.5 * (kl1 + kl2))


# =============================================================
# plot
# =============================================================
def plot_heatmap(out_png: str, title: str, labels: List[str], mat: List[List[float]], vmin: float, vmax: float):
    R = len(labels)
    C = len(labels)

    fig_w = max(10.0, C * 1.0)
    fig_h = max(8.0, R * 1.0)

    threshold = vmin + 0.5 * (vmax - vmin)  # 値域の半分

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im)

    plt.xticks(range(C), labels, rotation=45, ha="right")
    plt.yticks(range(R), labels)

    for i in range(R):
        for j in range(C):
            v = float(mat[i][j])
            txt_color = "black" if v >= threshold else "white"
            plt.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts_dir", help="exp/2019-2025-data/past-shifts/ (or a single .lp)")
    ap.add_argument("group_settings_dir", help="exp/2019-2025-data/group-settings/ (not used; kept for compatibility)")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--laplace-support", choices=["observed_ab", "all"], default="observed_ab")
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--outdir", default="out/ward_vs_ward_total")

    # ★追加: プロット対象をラベルで指定
    ap.add_argument(
        "--plot-wards",
        default="",
        help="Comma-separated ward labels to include (e.g. 'ICU,NICU,GCU,4S,7N'). Empty means include all mapped wards.",
    )

    # ★追加: マッピングがない病棟をどうするか
    ap.add_argument(
        "--strict-label",
        action="store_true",
        help="If set, wards without label mapping are skipped instead of using 'RAW:<name>'.",
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 期間
    d1 = args.start_year * 10000 + 101
    d2 = args.end_year * 10000 + 1231

    plot_wards = parse_csv_set(args.plot_wards)

    # 病棟ごとに past-shifts を読む（表示ラベルに変換）
    ward_to_seqs, label_to_raw = load_past_shifts_by_ward(
        args.past_shifts_dir,
        plot_wards=plot_wards,
        strict_label=args.strict_label,
    )
    wards = sorted(list(ward_to_seqs.keys()), key=str.lower)

    # 1-gram 総数（ラベルに付ける）
    totals = {}
    for w in wards:
        totals[w] = sum(count_ngrams_in_ward(ward_to_seqs[w], 1, d1, d2).values())

    wards = [w for w in wards if totals[w] > 0]
    if not wards:
        raise SystemExit("All wards have 0 total counts in this date range.")

    # 表示ラベル（英語短縮）で統一
    labels = [f"{w}({totals[w]})" for w in wards]

    # デバッグログ: 何を読んだか
    print("# loaded wards (label -> raw):")
    for w in wards:
        print(f"#   {w:12s} <- {label_to_raw.get(w, '?')}  total_1gram={totals[w]}")

    # n ごとに病棟×病棟
    for n in range(args.nmin, args.nmax + 1):
        counters = {w: count_ngrams_in_ward(ward_to_seqs[w], n, d1, d2) for w in wards}

        mat = []
        for wi in wards:
            row = []
            for wj in wards:
                row.append(js_distance(counters[wi], counters[wj], args.alpha, args.laplace_support, n))
            mat.append(row)

        out = os.path.join(args.outdir, f"heatmap_ward_x_ward_{n}gram.png")
        title = f"P(gram), n={n}"
        plot_heatmap(out, title, labels, mat, vmin=0.0, vmax=args.vmax)
        print(f"# wrote: {out}")


if __name__ == "__main__":
    main()
