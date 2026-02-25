#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(半年) × (n=1..N の「2019~2025(past) を基準にした JS距離」) のヒートマップを 1枚で出す。
さらに、found-model の分布も「基準（2019~2025 past）」に対する JS距離として 行を追加する。

追加仕様:
  - 縦軸ラベルに「n=1 のデータ総数 T」を付ける（例: 2019H1 (T=12345)）
    ※Heads/NonHeads で T は別々に計算される（その行の 1-gram カウンタ総和）
  - found 行ラベルは「入力 found_path を厳密に反映」する:
      FOUND_MODEL: <basename(found_dir_or_file)>
  - found_path に「病棟直下ディレクトリ（例: found-model/GCU/）」を渡したとき、
      直下のサブディレクトリごとに “別々の found 行” を作る。
      (2024-10-13/, 2024-12-13/ ... を合算せず、各1行)

★summaryモード:
  - past の 2019H1..2025H2 行を “1行に集約” して表示（found行は残す）
  - 集約past行のセル:
      色(値) = 2019H1..2025H2 の中央値（median）
      表示文字 = 第1四分位数〜第3四分位数 (Q1–Q3)
  - found行は従来通り 色=値、表示文字=値（小数3桁）

切り替えオプション:
  --past-mode {full,summary}
    full    : 従来（半期ごとのpast行を全部出す） [デフォルト]
    summary : pastを1行に集約（色=中央値、表示=Q1–Q3）+ found行

★今回の追加変更（あなたの要望）:
  - summaryモードの past 行ラベルの (T) を
      「全past半期の n=1 出現数の平均」ではなく
      「全past半期の n=1 出現数の第一四分位数～第三四分位数」にする

★追加（今回）:
  - found 側の集計期間を年月日(YYYYMMDD)で指定できる（--date-from / --date-to）
      ※期間でフィルタした結果、日付が連続しない部分は別セグメントとして扱い、
        n-gram がギャップを跨がないようにする。

仕様:
  - past: グループ集合変化でセグメント分割（境界は跨がない）
  - Unknown は NonHeads 側に含める
  - found: staff_group / group で Heads/NonHeads 判定（タイムライン無し）
  - found: ext_assigned / out_assigned を読む（2引数 out_assigned は無視）
  - JS distance = sqrt(JSD), ln (natural log)
  - カラースケール統一: vmin=0, vmax=sqrt(ln 2)
"""

import os
import sys
import math
import argparse
import glob
import re
from datetime import date as _date
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import io
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# matplotlib: global font / layout (見やすさ調整)
# -------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,          # 全体の基本フォント
    "axes.titlesize": 16,     # タイトル
    "axes.labelsize": 14,     # 軸ラベル
    "xtick.labelsize": 12,    # x目盛
    "ytick.labelsize": 12,    # y目盛
    "legend.fontsize": 12,
    "figure.titlesize": 16,
})

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date
import foundmodel_data_loader as found_loader  # ★found 側ローダ（ignored-ids + staff(...)）


# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "SE", "SN", "WR", "PH", "E", "N",}
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


# =============================================================
# helpers
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target_group: str) -> bool:
    """past側: heads_name と完全一致（case-insensitive）で Heads 扱い"""
    tg = (target_group or "").strip().lower()
    if not tg:
        return False
    return any((g or "").strip().lower() == tg for g in group_set)


# =============================================================
# past: segment builder (group-timeline aware)
# =============================================================
class Segment:
    __slots__ = ("groups", "seq")

    def __init__(self, groups: FrozenSet[str]):
        self.groups = groups
        self.seq: List[Tuple[int, str]] = []


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
) -> List["Segment"]:
    """
    所属グループ集合が変わるたびにセグメント分割。
    Unknown は UNKNOWN_GROUP として保持（→ NonHeads 側に入る）
    """
    seq = normalize_seq(seq)
    if not seq:
        return []
    seq.sort(key=lambda t: t[0])

    segs: List[Segment] = []
    cur: Optional[Segment] = None

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)
        if not gs:
            gset = frozenset([UNKNOWN_GROUP])
        else:
            gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            segs.append(cur)

        cur.seq.append((d, s))

    return segs


def prebuild_all_segments(seqs: SeqDict, timeline: dict) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline)
    return out


def count_ngrams_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter]:
    """
    指定範囲の Heads / NonHeads(+Unknown) を集約カウント。
    セグメント境界は跨がない。
    """
    heads = Counter()
    nonheads = Counter()

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for _, s in sseq]
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i:i + n])
                if any(x not in VALID_SHIFTS for x in gram):
                    continue
                if is_heads:
                    heads[gram] += 1
                else:
                    nonheads[gram] += 1

    return heads, nonheads


# =============================================================
# found: load + ngram counts (no timeline)
#   ★ここを foundmodel_data_loader に差し替え
#   ★ログは「最初の1回だけ」出す（以降は捨てる）
#   ★ファイル読み込みはキャッシュ
# =============================================================
_FOUND_MODEL_CACHE: Dict[str, Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]] = {}
_FOUND_LOADER_LOGGED = False


def is_head_group_found(g: str) -> bool:
    if not g:
        return False
    gl = g.lower()
    return ("head" in gl) or ("師長" in g) or ("主任" in g)


def bucket_found(groups: Set[str], heads_name: str) -> str:
    """found側: heads_name 完全一致 or それっぽい語を含むなら Heads"""
    hn = (heads_name or "").strip().lower()
    for g in groups:
        if (g or "").strip().lower() == hn:
            return "Heads"
        if is_head_group_found(g):
            return "Heads"
    return "NonHeads"


def _pick_lp_files_in_dir(d: str) -> List[str]:
    """
    1ディレクトリ内の found-model を探す。
      1) found-model*.lp
      2) 無ければ *.lp
    """
    fs = sorted(glob.glob(os.path.join(d, "found-model*.lp")))
    if fs:
        return fs
    return sorted(glob.glob(os.path.join(d, "*.lp")))


def collect_found_entries(found_path: str) -> List[Tuple[str, List[str]]]:
    """
    found_path から “found行” の単位を作って返す。

    戻り値: [(label, [lp_files...]), ...]

    ルール:
      - found_path がファイル -> 1件
      - found_path がディレクトリで、その直下に lp がある -> 1件（従来通り）
      - found_path がディレクトリで、直下に lp が無い -> 直下サブdirごとに lp を探して “複数件”
      - それでも見つからない場合は再帰探索して、lp のあるディレクトリごとに1件
    """
    if os.path.isfile(found_path):
        label = os.path.basename(os.path.normpath(found_path))
        return [(label, [found_path])]

    if not os.path.isdir(found_path):
        return []

    found_path = os.path.normpath(found_path)

    # A) 直下に lp があるなら “1件” （従来互換）
    direct = _pick_lp_files_in_dir(found_path)
    if direct:
        label = os.path.basename(found_path)
        return [(label, direct)]

    # B) 直下サブdirごとに “複数件”
    entries: List[Tuple[str, List[str]]] = []
    try:
        for name in sorted(os.listdir(found_path)):
            sub = os.path.join(found_path, name)
            if not os.path.isdir(sub):
                continue
            fs = _pick_lp_files_in_dir(sub)
            if fs:
                entries.append((os.path.basename(sub), fs))
    except OSError:
        pass

    if entries:
        return entries

    # C) それでも無ければ再帰（lpファイルの “親dir” ごとにまとめる）
    cand = sorted(glob.glob(os.path.join(found_path, "**", "found-model*.lp"), recursive=True))
    if not cand:
        cand = sorted(glob.glob(os.path.join(found_path, "**", "*.lp"), recursive=True))
    if not cand:
        return []

    by_dir: Dict[str, List[str]] = defaultdict(list)
    for fp in cand:
        by_dir[os.path.dirname(fp)].append(fp)

    out: List[Tuple[str, List[str]]] = []
    for d in sorted(by_dir.keys()):
        files = sorted(by_dir[d])
        out.append((os.path.basename(d), files))
    return out


def load_found_model(path: str) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]:
    """
    ★差し替え版:
      foundmodel_data_loader.load_found_model を使う（ignored-ids対応）
    ローダの print は捕まえて、最初の1回だけ表示。
    さらにファイル単位でキャッシュ。
    """
    global _FOUND_LOADER_LOGGED

    path = os.path.abspath(path)
    if path in _FOUND_MODEL_CACHE:
        return _FOUND_MODEL_CACHE[path]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seqs_by_staff, groups_by_staff = found_loader.load_found_model(path, apply_ignore_ids=True)

    log = buf.getvalue().strip()
    if (not _FOUND_LOADER_LOGGED) and log:
        print(log)
        _FOUND_LOADER_LOGGED = True

    _FOUND_MODEL_CACHE[path] = (seqs_by_staff, groups_by_staff)
    return seqs_by_staff, groups_by_staff


# =============================================================
# ★追加: found 側の期間指定（YYYYMMDD）+ 連続日付分割
# =============================================================
DATE8_RE = re.compile(r"^\d{8}$")  # YYYYMMDD


def parse_yyyymmdd(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if not DATE8_RE.fullmatch(s):
        raise ValueError(f"Invalid date (YYYYMMDD): {s}")
    return int(s)


def _int_to_date(d: int) -> _date:
    y = d // 10000
    m = (d // 100) % 100
    dd = d % 100
    return _date(y, m, dd)


def is_next_day(d1: int, d2: int) -> bool:
    return (_int_to_date(d2) - _int_to_date(d1)).days == 1


def filter_and_split_by_consecutive_days(
    seq_sorted: List[Tuple[int, str]],
    date_from: Optional[int],
    date_to: Optional[int],
) -> List[List[str]]:
    """
    seq_sorted: [(YYYYMMDD, shift), ...] 日付昇順前提
    1) [date_from, date_to] でフィルタ（Noneは無制限）
    2) 連続日付ごとに分割（ギャップがあれば別セグメント）
    返り値: [ [shift, shift, ...], [shift, ...], ... ]
    """
    filtered: List[Tuple[int, str]] = []
    for d, sh in seq_sorted:
        if date_from is not None and d < date_from:
            continue
        if date_to is not None and d > date_to:
            continue
        filtered.append((d, sh))

    if not filtered:
        return []

    segments: List[List[str]] = []
    cur: List[str] = [filtered[0][1]]
    prev_d = filtered[0][0]

    for d, sh in filtered[1:]:
        if is_next_day(prev_d, d):
            cur.append(sh)
        else:
            segments.append(cur)
            cur = [sh]
        prev_d = d

    segments.append(cur)
    return segments


def count_ngrams_found_heads_nonheads(
    found_files: List[str],
    n: int,
    heads_name: str,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
) -> Tuple[Counter, Counter]:
    """
    found側: この found_files（= 1行ぶん）を合算して Heads / NonHeads を数える（タイムライン無し）

    ★追加:
      - date_from/date_to があれば、その期間内だけ集計
      - フィルタ後、日付ギャップを跨いで n-gram を数えない（連続日付セグメントごとに数える）
    """
    heads = Counter()
    nonheads = Counter()

    for fp in found_files:
        seqs_by_staff, groups_by_staff = load_found_model(fp)

        for sid, seq in seqs_by_staff.items():
            if len(seq) < n:
                continue
            seq_sorted = sorted(seq, key=lambda t: t[0])
            segments = filter_and_split_by_consecutive_days(seq_sorted, date_from, date_to)

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)

            for shifts in segments:
                if len(shifts) < n:
                    continue
                for i in range(len(shifts) - n + 1):
                    gram = tuple(shifts[i:i + n])
                    if any(x not in VALID_SHIFTS for x in gram):
                        continue
                    if bucket == "Heads":
                        heads[gram] += 1
                    else:
                        nonheads[gram] += 1

    return heads, nonheads


# =============================================================
# JS distance (sqrt(JSD)) [ln]
# =============================================================
def build_vocab(c1: Counter, c2: Counter) -> List[Tuple[str, ...]]:
    total = Counter()
    total.update(c1)
    total.update(c2)
    vocab = list(total.keys())
    vocab.sort()
    return vocab


def to_prob_vector(counter: Counter, vocab: List[Tuple[str, ...]], alpha: float) -> List[float]:
    total = float(sum(counter.values()))
    denom = total + alpha * len(vocab)
    if denom <= 0:
        return ([1.0 / len(vocab)] * len(vocab)) if vocab else []
    return [(counter.get(g, 0) + alpha) / denom for g in vocab]


def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi)  # ln
    return s


def js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_distance_from_counters(c1: Counter, c2: Counter, alpha: float) -> float:
    vocab = build_vocab(c1, c2)
    if not vocab:
        return 0.0
    p = to_prob_vector(c1, vocab, alpha)
    q = to_prob_vector(c2, vocab, alpha)
    d = js_divergence(p, q)
    if d < 0:
        d = 0.0
    return math.sqrt(d)


# =============================================================
# periods: half-year
# =============================================================
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
    """
    2019H1: 20190101..20190630
    2019H2: 20190701..20191231
    """
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101,  y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701,  y * 10000 + 1231))
    return periods


# =============================================================
# plotting
# =============================================================
def plot_heatmap_period_x_n(
    out_png: str,
    title: str,
    period_labels: List[str],
    n_labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
    ann_texts: Optional[List[List[str]]] = None,
) -> None:
    """
    mat: R x C の数値（色に使う）
    ann_texts: R x C のセル表示テキスト（Noneなら数値を小数3桁表示）
    """
    R = len(period_labels)
    C = len(n_labels)

    # ★見やすさ: figsize を少し広げる（特に縦）
    fig_w = max(10.5, C * 2.4)
    fig_h = max(8.5, R * 0.65)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")

    # ★カラーバーの文字サイズ
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)

    # ★目盛り文字サイズ
    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(list(range(R)), period_labels, fontsize=12)

    # ★セル内注釈の文字サイズ
    mid = (vmin + vmax) / 2.0
    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"
            if ann_texts is None:
                s = f"{val:.3f}"
            else:
                s = ann_texts[i][j]
            plt.text(j, i, s, ha="center", va="center", fontsize=12, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _percentile_linear(xs_sorted: List[float], p: float) -> float:
    """
    線形補間の percentile。
    xs_sorted は昇順ソート済みを想定。
    """
    if not xs_sorted:
        return 0.0
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    pos = (len(xs_sorted) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    w = pos - lo
    return xs_sorted[lo] * (1.0 - w) + xs_sorted[hi] * w


def summarize_past_rows_to_one(
    past_mat: List[List[float]],
    past_totals: List[int],
    past_row_label: str,
    qfmt: str = "{:.3f}",
) -> Tuple[List[List[float]], List[List[str]], List[str], str]:
    """
    past_mat: K x C (K=半期の数)
    色用: median（列ごと）
    表示文字: Q1–Q3（列ごと）

    ★今回の変更:
      past行ラベルの (T) は「平均」ではなく「TのQ1–Q3」を表示したいので、
      Tラベル用の文字列を返す（例: "1234–5678"）。

    返り値:
      mat1: 1 x C
      ann1: 1 x C
      labels1: [past_row_label]
      t_label: "T_Q1–T_Q3"（整数表示）
    """
    if not past_mat:
        return [[]], [[]], [past_row_label], "0–0"

    K = len(past_mat)
    C = len(past_mat[0])

    cols: List[List[float]] = [[] for _ in range(C)]
    for i in range(K):
        for j in range(C):
            cols[j].append(past_mat[i][j])

    mat1: List[List[float]] = [[]]
    ann1: List[List[str]] = [[]]

    for j in range(C):
        xs_sorted = sorted(cols[j])
        q1 = _percentile_linear(xs_sorted, 0.25)
        med = _percentile_linear(xs_sorted, 0.50)
        q3 = _percentile_linear(xs_sorted, 0.75)

        mat1[0].append(med)
        ann1[0].append(f"{qfmt.format(q1)}–{qfmt.format(q3)}")

    if past_totals:
        ts_sorted = sorted(float(x) for x in past_totals)
        t_q1 = _percentile_linear(ts_sorted, 0.25)
        t_q3 = _percentile_linear(ts_sorted, 0.75)
        t_label = f"{int(round(t_q1))}–{int(round(t_q3))}"
    else:
        t_label = "0–0"

    labels1 = [past_row_label]
    return mat1, ann1, labels1, t_label


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts *.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("found_path", help="found dir OR found-model.lp OR ward root dir (contains multiple found dirs)")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/halfyear+found-model_vs_total")
    ap.add_argument("--only-nonheads", action="store_true", help="NonHeads(+Unknown) だけ出力（Headsを出さない）")

    # past表示モード切替
    ap.add_argument(
        "--past-mode",
        choices=["full", "summary"],
        default="summary",
        help="full=従来の半期行すべて, summary=pastを1行に集約(色=中央値,表示=Q1–Q3)+found行",
    )

    # ★追加: found側 期間指定
    ap.add_argument("--date-from", default=None, help="found集計期間の開始日（YYYYMMDD, 例: 20241101）")
    ap.add_argument("--date-to", default=None, help="found集計期間の終了日（YYYYMMDD, 例: 20241130）")

    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")

    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    found_entries = collect_found_entries(args.found_path)
    if not found_entries:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

    # ★追加: found期間のパース
    date_from = parse_yyyymmdd(args.date_from)
    date_to = parse_yyyymmdd(args.date_to)
    if date_from is not None and date_to is not None and date_to < date_from:
        raise ValueError(f"--date-to must be >= --date-from (from={date_from}, to={date_to})")

    # load past + timeline
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # periods (half-year) + base(total) range
    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram (vs {base_key})" for n in ns]

    # base counters cache (nごとに1回だけ数える)
    base_heads_by_n: Dict[int, Counter] = {}
    base_non_by_n: Dict[int, Counter] = {}
    for n in ns:
        h_b, non_b = count_ngrams_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_heads_by_n[n] = h_b
        base_non_by_n[n] = non_b

    # found counters cache: (found_entryごと, nごとに1回だけ数える)
    found_heads_by_entry_n: Dict[str, Dict[int, Counter]] = {}
    found_non_by_entry_n: Dict[str, Dict[int, Counter]] = {}
    for label, files in found_entries:
        found_heads_by_entry_n[label] = {}
        found_non_by_entry_n[label] = {}
        for n in ns:
            # ★ここだけ変更: found側は期間指定 + 連続日付分割で数える
            h_f, non_f = count_ngrams_found_heads_nonheads(
                files, n=n, heads_name=args.heads_name, date_from=date_from, date_to=date_to
            )
            found_heads_by_entry_n[label][n] = h_f
            found_non_by_entry_n[label][n] = non_f

    # ---- 行定義: past半年 + found(サブdirごとに1行)
    # periods: (row_label, d1, d2, kind) kind in {"past","found"}
    periods: List[Tuple[str, Optional[int], Optional[int], str]] = []
    for k, d1, d2 in half_periods:
        periods.append((k, d1, d2, "past"))
    for label, _files in found_entries:
        periods.append((f"AUTO: {label}", None, None, "found"))

    # global scale (ln): vmax = sqrt(ln2) ~ 0.8326
    vmin = 0.0
    vmax = 0.5

    def build_matrix(is_heads: bool) -> Tuple[List[List[float]], List[int]]:
        """
        periods の順で:
          - 行の 1-gram 総数 totals
          - 行の jsd row（列=ns）
        """
        mat: List[List[float]] = []
        totals: List[int] = []

        for (pkey, d1, d2, kind) in periods:
            row: List[float] = []

            # ---- n=1 の総数（表示用）
            if kind == "found":
                label = pkey.replace("AUTO: ", "", 1)
                c_p_1 = found_heads_by_entry_n[label][1] if is_heads else found_non_by_entry_n[label][1]
            else:
                assert d1 is not None and d2 is not None
                h_p1, non_p1 = count_ngrams_heads_nonheads_in_range(
                    segs_by_person, n=1, heads_name=args.heads_name, date_start=d1, date_end=d2
                )
                c_p_1 = h_p1 if is_heads else non_p1
            totals.append(int(sum(c_p_1.values())))

            # ---- JS距離（n=ns）
            for n in ns:
                c_b = base_heads_by_n[n] if is_heads else base_non_by_n[n]

                if kind == "found":
                    label = pkey.replace("AUTO: ", "", 1)
                    c_p = found_heads_by_entry_n[label][n] if is_heads else found_non_by_entry_n[label][n]
                else:
                    assert d1 is not None and d2 is not None
                    h_p, non_p = count_ngrams_heads_nonheads_in_range(
                        segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                    )
                    c_p = h_p if is_heads else non_p

                row.append(js_distance_from_counters(c_p, c_b, args.alpha))

            mat.append(row)

        return mat, totals

    def apply_past_mode(
        mat: List[List[float]],
        totals: List[int],
        is_heads: bool,
    ) -> Tuple[List[List[float]], List[List[str]], List[str], List[int]]:
        """
        full:
          - 色=値、表示=値(3桁)
        summary:
          - past部分を1行に集約（色=median, 表示=Q1–Q3）
          - found行はそのまま（色=値, 表示=値）
          - ★past行ラベルの (T) は「TのQ1–Q3」
        """
        K = len(half_periods)
        past_mat = mat[:K]
        past_totals = totals[:K]
        found_mat = mat[K:]
        found_totals = totals[K:]

        period_labels_base = [k for (k, _, _, _) in periods]

        if args.past_mode == "full":
            labels = [f"{lbl} ({t})" for lbl, t in zip(period_labels_base, totals)]
            ann = [[f"{v:.3f}" for v in row] for row in mat]
            return mat, ann, labels, totals

        past_label = f"MANUAL:{args.start_year}H1–{args.end_year}H2"
        sum_mat1, sum_ann1, sum_labels1, t_iqr_label = summarize_past_rows_to_one(
            past_mat=past_mat,
            past_totals=past_totals,
            past_row_label=past_label,
            qfmt="{:.3f}",
        )

        found_ann = [[f"{v:.3f}" for v in row] for row in found_mat]
        found_labels = period_labels_base[K:]
        found_labels = [f"{lbl} ({t})" for lbl, t in zip(found_labels, found_totals)]

        mat2 = sum_mat1 + found_mat
        ann2 = sum_ann1 + found_ann
        labels2 = [f"{sum_labels1[0]} ({t_iqr_label})"] + found_labels

        if past_totals:
            ts_sorted = sorted(past_totals)
            med_idx = (len(ts_sorted) - 1) // 2
            rep = int(ts_sorted[med_idx])
        else:
            rep = 0
        totals2 = [rep] + found_totals

        return mat2, ann2, labels2, totals2

    # Heads heatmap
    if not args.only_nonheads:
        mat_h, totals_h = build_matrix(is_heads=True)
        mat_h2, ann_h2, labels_h2, _totals_h2 = apply_past_mode(mat_h, totals_h, is_heads=True)

        suffix = f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{args.past_mode}"
        out_h = os.path.join(
            args.outdir,
            f"heatmap_halfyear_plus_foundEntries_x_ngram_heads_{suffix}.png"
        )
        plot_heatmap_period_x_n(
            out_h,
            title=f"Heads: JSdist(past / each-found-entry vs {base_key}(past)) for n={args.nmin}..{args.nmax} [ln] (past-mode={args.past_mode})",
            period_labels=labels_h2,
            n_labels=n_labels,
            mat=mat_h2,
            vmin=vmin,
            vmax=vmax,
            ann_texts=ann_h2,
        )
        print(f"# wrote: {out_h}")

    # NonHeads heatmap
    mat_n, totals_n = build_matrix(is_heads=False)
    mat_n2, ann_n2, labels_n2, _totals_n2 = apply_past_mode(mat_n, totals_n, is_heads=False)

    suffix = f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{args.past_mode}"
    out_n = os.path.join(
        args.outdir,
        f"heatmap_halfyear_plus_foundEntries_x_ngram_nonheads_{suffix}.png"
    )
    plot_heatmap_period_x_n(
        out_n,
        title=f"NonHeads: JSD(AUTO vs {base_key} MANUAL)",
        period_labels=labels_n2,
        n_labels=n_labels,
        mat=mat_n2,
        vmin=vmin,
        vmax=vmax,
        ann_texts=ann_n2,
    )
    print(f"# wrote: {out_n}")


if __name__ == "__main__":
    main()
