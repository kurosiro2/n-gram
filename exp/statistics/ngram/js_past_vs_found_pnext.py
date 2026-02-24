#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(半年) × (n=1..N の「2019~2025(past) を基準にした 条件付き確率 P(next|prefix) の JS距離」) のヒートマップを 1枚で出す。
さらに、found-model の分布も「基準（2019~2025 past）」に対する JS距離として 行を追加する。

仕様（freq版の方針を pnext に移植）:
  - 縦軸ラベルに「n=1 のデータ総数 T」を付ける（例: 2019H1 (T=12345)）
    ※Heads/NonHeads で T は別々（その行の prefixイベント総数 = sum(prefN)）
  - found 行ラベルは「入力 found_path を厳密に反映」する:
      FOUND_ENTRY: <basename(dir_or_file_or_subdir)>
  - found_path に「病棟直下ディレクトリ（例: found-model/GCU/）」を渡したとき、
      直下のサブディレクトリごとに “別々の found 行” を作る（合算しない）

★past集約（summaryモード）:
  - past の 2019H1..2025H2 行を “1行に集約” して表示できる（found行は残す）
  - 集約past行のセル:
      色(値) = 2019H1..2025H2 の中央値（median）   ※「そのセルの色用スカラー」を半期行から集約
      表示文字 = 第1四分位数〜第3四分位数 (Q1–Q3)
  - found行は従来通り（色=値、表示文字=値 or avg-mode=iqrならその表記）

切り替えオプション:
  --past-mode {full,summary}
    full    : 従来（半期ごとのpast行を全部出す） [デフォルト: summary]
    summary : pastを1行に集約（色=中央値、表示=Q1–Q3）+ found行

★追加変更（あなたの要望）:
  - summaryモードの past 行ラベルの (T) は
      「平均」ではなく
      「全past半期の n=1 出現数T の第一四分位数～第三四分位数」にする（四捨五入 int）

pnext集計:
  - past: グループ集合変化でセグメント分割（境界は跨がない）
  - Unknown は NonHeads 側に含める
  - found: staff_group / group で Heads/NonHeads 判定（タイムライン無し）
  - found: ext_assigned / out_assigned を読む（2引数 out_assigned は無視）
  - JS distance = sqrt(JSD), ln（自然対数）

avg-mode:
  --avg-mode weighted:
      各prefix y について JSD(P(.|y), Q(.|y)) を計算し、重み w(y)=c_p(y)+c_b(y) で平均 → sqrt
  --avg-mode uniform:
      prefixごとの JSD を等重みで平均 → sqrt
  --avg-mode iqr:
      prefixごとの JS距離（sqrt(JSD)）の分布を作り、
        色 = median
        表示 = Q1–Q3

Laplace:
  --laplace-k > 0
  --laplace-support all:
      X = VALID_SHIFTS（10固定）に add-k
  --laplace-support observed_ab:
      prefixごとに A/B で観測された next の union のみに add-k（|X|可変）
      ただし表示ベクトルは常に10次元（未支持は加算なし）
"""

import os
import sys
import math
import argparse
import glob
import re
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
X_SIZE = 10

UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]


# =============================================================
# helpers
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


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


def build_segments_for_person(seq, name, nid, timeline) -> List["Segment"]:
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


# =============================================================
# past: conditional counts for P(next|prefix)
# =============================================================
def count_conditional_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Dict[Prefix, Counter], Counter, Dict[Prefix, Counter], Counter]:
    """
    戻り:
      heads_cond[prefix][next] = c(y,x)
      heads_prefN[prefix]      = c(y)    (prefixイベント回数)
      non_cond, non_prefN も同様
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    heads_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    non_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    heads_prefN: Counter = Counter()
    non_prefN: Counter = Counter()

    pref_len = n - 1

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for (_, s) in sseq]

            if n == 1:
                pfx: Prefix = tuple()
                for x in shifts:
                    if is_heads:
                        heads_cond[pfx][x] += 1
                        heads_prefN[pfx] += 1
                    else:
                        non_cond[pfx][x] += 1
                        non_prefN[pfx] += 1
                continue

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i: i + pref_len])
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# =============================================================
# found: load + conditional counts (no timeline)
# =============================================================
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_OUT = re.compile(r'^out_assigned\(\s*(\d+)\s*,\s*(\d{8})\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')
PAT_GROUP2 = re.compile(r'^group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')


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

    # A) 直下に lp があるなら “1件”
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

    # C) 再帰（lpファイルの “親dir” ごとにまとめる）
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


def load_found_model(path: str) -> Tuple[Dict[int, List[str]], Dict[int, Set[str]]]:
    """
    found-model.lp を読んで:
      - shifts_by_staff: {staff_id: [shift, shift, ...]}
      - groups_by_staff: {staff_id: set(groupname)}
    """
    seqs_by_staff: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    groups_by_staff: Dict[int, Set[str]] = defaultdict(set)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue

            m = PAT_EXT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))
                sh = m.group(3)
                if sh in VALID_SHIFTS_SET:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_OUT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))  # yyyymmdd
                sh = m.group(3)
                if sh in VALID_SHIFTS_SET:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

            m = PAT_GROUP2.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

    shifts_by_staff: Dict[int, List[str]] = {}
    for sid, pairs in seqs_by_staff.items():
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        shifts_by_staff[sid] = [s for _, s in pairs_sorted]

    return shifts_by_staff, groups_by_staff


def count_conditional_found_heads_nonheads(
    found_files: List[str],
    n: int,
    heads_name: str,
) -> Tuple[Dict[Prefix, Counter], Counter, Dict[Prefix, Counter], Counter]:
    """
    found側: found_files（= 1行ぶん）を合算して Heads / NonHeads の P(next|prefix) カウントを作る（タイムライン無し）
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    heads_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    non_cond: Dict[Prefix, Counter] = defaultdict(Counter)
    heads_prefN: Counter = Counter()
    non_prefN: Counter = Counter()

    pref_len = n - 1

    for fp in found_files:
        shifts_by_staff, groups_by_staff = load_found_model(fp)

        for sid, shifts in shifts_by_staff.items():
            if len(shifts) < n:
                continue

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)

            if n == 1:
                pfx: Prefix = tuple()
                for x in shifts:
                    if bucket == "Heads":
                        heads_cond[pfx][x] += 1
                        heads_prefN[pfx] += 1
                    else:
                        non_cond[pfx][x] += 1
                        non_prefN[pfx] += 1
                continue

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i: i + pref_len])
                nxt = shifts[i + pref_len]
                if bucket == "Heads":
                    heads_cond[pfx][nxt] += 1
                    heads_prefN[pfx] += 1
                else:
                    non_cond[pfx][nxt] += 1
                    non_prefN[pfx] += 1

    return heads_cond, heads_prefN, non_cond, non_prefN


# =============================================================
# JS distance for vectors (sqrt(JSD)) [ln]
# =============================================================
def kl_div(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi)
    return s


def js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_distance_vec(p: List[float], q: List[float]) -> float:
    d = js_divergence(p, q)
    if d < 0.0:
        d = 0.0
    return math.sqrt(d)


def laplace_pnext_vector(
    cond: Dict[Prefix, Counter],
    prefN: Counter,
    prefix: Prefix,
    k: float,
    support: Optional[Set[str]] = None,
) -> List[float]:
    Ny = float(prefN.get(prefix, 0))
    cxy = cond.get(prefix, Counter())

    if support is None:
        support = VALID_SHIFTS_SET
    if not support:
        support = VALID_SHIFTS_SET

    denom = Ny + k * len(support)
    if denom <= 0.0:
        return [1.0 / X_SIZE] * X_SIZE

    out: List[float] = []
    for x in VALID_SHIFTS:
        if x in support:
            out.append((cxy.get(x, 0) + k) / denom)
        else:
            out.append(cxy.get(x, 0) / denom)
    return out


# =============================================================
# quantiles (no numpy)
# =============================================================
def _quantile_sorted(xs_sorted: List[float], q: float) -> float:
    n = len(xs_sorted)
    if n <= 0:
        return 0.0
    if n == 1:
        return xs_sorted[0]
    if q <= 0.0:
        return xs_sorted[0]
    if q >= 1.0:
        return xs_sorted[-1]

    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    frac = pos - lo
    return xs_sorted[lo] * (1.0 - frac) + xs_sorted[hi] * frac


def _q1_med_q3(xs: List[float]) -> Tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    s = sorted(xs)
    q1 = _quantile_sorted(s, 0.25)
    med = _quantile_sorted(s, 0.50)
    q3 = _quantile_sorted(s, 0.75)
    return q1, med, q3


# =============================================================
# aggregate per-prefix distances into one cell
# =============================================================
def js_distance_pnext_aggregate(
    cond_p: Dict[Prefix, Counter],
    prefN_p: Counter,
    cond_b: Dict[Prefix, Counter],
    prefN_b: Counter,
    laplace_k: float,
    avg_mode: str,
    laplace_support: str,
) -> Tuple[float, str]:
    """
    戻り値:
      color_value: ヒートマップの色（float）
      text_value : セル内の表示文字列

    avg_mode:
      weighted/uniform:
        セル値 = sqrt( avg JSD ) の1値（色=値, 表示=値）
      iqr:
        prefixごとの JS距離（sqrt(JSD)）の分布から
          色=median, 表示=Q1–Q3
    """
    prefixes = set(prefN_p.keys()) | set(prefN_b.keys())
    if not prefixes:
        return 0.0, "0.000"

    if avg_mode not in ("weighted", "uniform", "iqr"):
        raise ValueError(f"avg_mode must be weighted|uniform|iqr, got {avg_mode}")
    if laplace_support not in ("all", "observed_ab"):
        raise ValueError(f"laplace_support must be all|observed_ab, got {laplace_support}")

    def _support_for_prefix(y: Prefix) -> Optional[Set[str]]:
        if laplace_support == "all":
            return None
        c_p = cond_p.get(y, Counter())
        c_b = cond_b.get(y, Counter())
        return set(c_p.keys()) | set(c_b.keys())

    if avg_mode == "iqr":
        dists: List[float] = []
        for y in prefixes:
            sup = _support_for_prefix(y)
            pvec = laplace_pnext_vector(cond_p, prefN_p, y, laplace_k, support=sup)
            qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
            dists.append(js_distance_vec(pvec, qvec))

        q1, med, q3 = _q1_med_q3(dists)
        return med, f"{q1:.3f}–{q3:.3f}"

    if avg_mode == "uniform":
        jsd_sum = 0.0
        mcnt = 0
        for y in prefixes:
            sup = _support_for_prefix(y)
            pvec = laplace_pnext_vector(cond_p, prefN_p, y, laplace_k, support=sup)
            qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
            jsd = js_divergence(pvec, qvec)
            if jsd < 0.0:
                jsd = 0.0
            jsd_sum += jsd
            mcnt += 1
        if mcnt <= 0:
            return 0.0, "0.000"
        val = math.sqrt(jsd_sum / float(mcnt))
        return val, f"{val:.3f}"

    # weighted
    wsum = 0.0
    weights: Dict[Prefix, float] = {}
    for y in prefixes:
        w = float(prefN_p.get(y, 0) + prefN_b.get(y, 0))
        weights[y] = w
        wsum += w
    if wsum <= 0.0:
        wsum = float(len(prefixes))
        for y in prefixes:
            weights[y] = 1.0

    jsd_agg = 0.0
    for y in prefixes:
        sup = _support_for_prefix(y)
        pvec = laplace_pnext_vector(cond_p, prefN_p, y, laplace_k, support=sup)
        qvec = laplace_pnext_vector(cond_b, prefN_b, y, laplace_k, support=sup)
        jsd = js_divergence(pvec, qvec)
        if jsd < 0.0:
            jsd = 0.0
        jsd_agg += (weights[y] / wsum) * jsd

    val = math.sqrt(jsd_agg)
    return val, f"{val:.3f}"


# =============================================================
# periods: half-year
# =============================================================
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
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
    color_mat: List[List[float]],
    vmin: float,
    vmax: float,
    text_mat: List[List[str]],
) -> None:
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(7.0, R * 0.55)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(color_mat, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im)

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), period_labels)

    mid = (vmin + vmax) / 2.0
    for i in range(R):
        for j in range(C):
            val = color_mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(j, i, text_mat[i][j], ha="center", va="center", fontsize=9, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================================================
# summary (past rows -> one row) for pnext heatmap
# =============================================================
def summarize_past_rows_to_one(
    past_color_mat: List[List[float]],   # K x C (色用スカラー)
    past_Ts: List[int],                  # K (n=1 prefixイベント数)
    past_row_label: str,
) -> Tuple[List[List[float]], List[List[str]], List[str], str]:
    """
    pnext版: past半期行の「セル色用スカラー」を列ごとに集約
      色用: median（列ごと）
      表示文字: Q1–Q3（列ごと）

    ★変更点:
      past行ラベルのTは「平均」ではなく「TのQ1–Q3」にしたいので、
      文字列 t_iqr_label ("1234–5678") を返す。
    """
    if not past_color_mat:
        return [[]], [[]], [past_row_label], "0–0"

    K = len(past_color_mat)
    C = len(past_color_mat[0])

    cols: List[List[float]] = [[] for _ in range(C)]
    for i in range(K):
        for j in range(C):
            cols[j].append(past_color_mat[i][j])

    out_color: List[List[float]] = [[]]
    out_text: List[List[str]] = [[]]

    for j in range(C):
        xs = cols[j]
        q1, med, q3 = _q1_med_q3(xs)
        out_color[0].append(med)
        out_text[0].append(f"{q1:.3f}–{q3:.3f}")

    # ---- ★TのIQR（Q1–Q3）
    if past_Ts:
        ts = [float(x) for x in past_Ts]
        t_q1, _t_med, t_q3 = _q1_med_q3(ts)
        t_iqr_label = f"{int(round(t_q1))}–{int(round(t_q3))}"
    else:
        t_iqr_label = "0–0"

    return out_color, out_text, [past_row_label], t_iqr_label


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
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--laplace-k", type=float, default=1.0)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/halfyear+foundEntries_vs_total_pnext")
    ap.add_argument("--only-nonheads", action="store_true")

    ap.add_argument(
        "--avg-mode",
        choices=["weighted", "uniform", "iqr"],
        default="weighted",
        help=(
            "weighted/uniform: avg JSD then sqrt (single value). "
            "iqr: per-prefix sqrt(JSD) distribution -> color=median, text=Q1–Q3."
        ),
    )
    ap.add_argument(
        "--laplace-support",
        choices=["all", "observed_ab"],
        default="all",
        help='Laplace support X: all=10 fixed, observed_ab=union of observed next in compared A/B (per prefix).',
    )

    # ★past表示モード
    ap.add_argument(
        "--past-mode",
        choices=["full", "summary"],
        default="summary",
        help="full=half-year past rows, summary=compress past rows into 1 (color=median, text=Q1–Q3) + found rows",
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")
    if args.laplace_k <= 0:
        raise ValueError("--laplace-k must be > 0")

    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    found_entries = collect_found_entries(args.found_path)
    if not found_entries:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

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

    # base cache (past total): per n
    base_heads_cond_by_n: Dict[int, Dict[Prefix, Counter]] = {}
    base_heads_prefN_by_n: Dict[int, Counter] = {}
    base_non_cond_by_n: Dict[int, Dict[Prefix, Counter]] = {}
    base_non_prefN_by_n: Dict[int, Counter] = {}

    for n in ns:
        h_cond, h_prefN, non_cond, non_prefN = count_conditional_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_heads_cond_by_n[n] = h_cond
        base_heads_prefN_by_n[n] = h_prefN
        base_non_cond_by_n[n] = non_cond
        base_non_prefN_by_n[n] = non_prefN

    # found cache: per entry label, per n
    found_heads_cond_by_entry_n: Dict[str, Dict[int, Dict[Prefix, Counter]]] = {}
    found_heads_prefN_by_entry_n: Dict[str, Dict[int, Counter]] = {}
    found_non_cond_by_entry_n: Dict[str, Dict[int, Dict[Prefix, Counter]]] = {}
    found_non_prefN_by_entry_n: Dict[str, Dict[int, Counter]] = {}

    for label, files in found_entries:
        found_heads_cond_by_entry_n[label] = {}
        found_heads_prefN_by_entry_n[label] = {}
        found_non_cond_by_entry_n[label] = {}
        found_non_prefN_by_entry_n[label] = {}
        for n in ns + [1]:  # 1はT計算にも使う
            h_cond, h_prefN, non_cond, non_prefN = count_conditional_found_heads_nonheads(
                files, n=n, heads_name=args.heads_name
            )
            found_heads_cond_by_entry_n[label][n] = h_cond
            found_heads_prefN_by_entry_n[label][n] = h_prefN
            found_non_cond_by_entry_n[label][n] = non_cond
            found_non_prefN_by_entry_n[label][n] = non_prefN

    # ---- 行定義: past半年 + found
    rows: List[Tuple[str, Optional[int], Optional[int], str]] = []
    for k, d1, d2 in half_periods:
        rows.append((k, d1, d2, "past"))
    for label, _files in found_entries:
        rows.append((f"AUTO: {label}", None, None, "found"))

    # color scale (ln): vmax = sqrt(ln2)
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

    def build_color_text_and_T(is_heads: bool) -> Tuple[List[List[float]], List[List[str]], List[int]]:
        """
        rows順で:
          - T: n=1 の prefixイベント総数 = sum(prefN.values())
          - 各セル: js_distance_pnext_aggregate(...) -> (color_scalar, text)
        """
        color_mat: List[List[float]] = []
        text_mat: List[List[str]] = []
        Ts: List[int] = []

        for (rkey, d1, d2, kind) in rows:
            # ---- T（n=1）
            if kind == "found":
                label = rkey.replace("AUTO: ", "", 1)
                prefN_1 = found_heads_prefN_by_entry_n[label][1] if is_heads else found_non_prefN_by_entry_n[label][1]
            else:
                assert d1 is not None and d2 is not None
                _h1, hN1, _n1, nN1 = count_conditional_heads_nonheads_in_range(
                    segs_by_person, n=1, heads_name=args.heads_name, date_start=d1, date_end=d2
                )
                prefN_1 = hN1 if is_heads else nN1
            Ts.append(int(sum(prefN_1.values())))

            # ---- cells
            c_row: List[float] = []
            t_row: List[str] = []
            for n in ns:
                if kind == "found":
                    label = rkey.replace("AUTO: ", "", 1)
                    cond_p = found_heads_cond_by_entry_n[label][n] if is_heads else found_non_cond_by_entry_n[label][n]
                    prefN_p = found_heads_prefN_by_entry_n[label][n] if is_heads else found_non_prefN_by_entry_n[label][n]
                else:
                    assert d1 is not None and d2 is not None
                    h_cond_p, h_prefN_p, non_cond_p, non_prefN_p = count_conditional_heads_nonheads_in_range(
                        segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                    )
                    cond_p = h_cond_p if is_heads else non_cond_p
                    prefN_p = h_prefN_p if is_heads else non_prefN_p

                cond_b = base_heads_cond_by_n[n] if is_heads else base_non_cond_by_n[n]
                prefN_b = base_heads_prefN_by_n[n] if is_heads else base_non_prefN_by_n[n]

                color_val, text = js_distance_pnext_aggregate(
                    cond_p, prefN_p, cond_b, prefN_b,
                    laplace_k=args.laplace_k,
                    avg_mode=args.avg_mode,
                    laplace_support=args.laplace_support,
                )
                c_row.append(color_val)
                t_row.append(text)

            color_mat.append(c_row)
            text_mat.append(t_row)

        return color_mat, text_mat, Ts

    def apply_past_mode(
        color_mat: List[List[float]],
        text_mat: List[List[str]],
        Ts: List[int],
    ) -> Tuple[List[List[float]], List[List[str]], List[str], List[int]]:
        """
        full:
          - 行は全部（past半期 + found）
          - labels: "<row>(T=xxx)"
        summary:
          - past(K行) を 1行に集約（色=median, 表示=Q1–Q3）
          - found行はそのまま
          - ★pastの(T)は「TのQ1–Q3」
        """
        K = len(half_periods)
        base_labels = [r[0] for r in rows]

        if args.past_mode == "full":
            labels = [f"{lbl} (T={t})" for lbl, t in zip(base_labels, Ts)]
            return color_mat, text_mat, labels, Ts

        # summary
        past_color = color_mat[:K]
        past_Ts = Ts[:K]
        found_color = color_mat[K:]
        found_text = text_mat[K:]
        found_Ts = Ts[K:]
        found_labels = base_labels[K:]

        past_label = f"MANUAL: {args.start_year}H1–{args.end_year}H2"

        sum_color1, sum_text1, sum_labels1, t_iqr_label = summarize_past_rows_to_one(
            past_color_mat=past_color,
            past_Ts=past_Ts,
            past_row_label=past_label,
        )

        # found labels + T（従来通り「合計T」）
        found_labels2 = [f"{lbl} (T={t})" for lbl, t in zip(found_labels, found_Ts)]

        # 合成
        color2 = sum_color1 + found_color
        text2 = sum_text1 + found_text
        labels2 = [f"{sum_labels1[0]} (T={t_iqr_label})"] + found_labels2

        # Ts2 は内部用に返すだけ（ラベル表示は labels2 が担当）
        Ts2 = [int(round(_q1_med_q3([float(x) for x in past_Ts])[1]))] + found_Ts
        return color2, text2, labels2, Ts2

    suffix = f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_k{args.laplace_k}_{args.avg_mode}_{args.laplace_support}_{args.past_mode}"

    # Heads
    if not args.only_nonheads:
        color_h, text_h, Ts_h = build_color_text_and_T(is_heads=True)
        color_h2, text_h2, labels_h2, _ = apply_past_mode(color_h, text_h, Ts_h)

        out_h = os.path.join(
            args.outdir,
            f"heatmap_pastHalfYear_plus_foundEntries_x_pnext_heads_{suffix}.png",
        )
        plot_heatmap_period_x_n(
            out_h,
            title=f"Heads: JSdist(pastHalfYear + foundEntries vs {base_key}(past)) on P(next|prefix) n={args.nmin}..{args.nmax} [ln] (past-mode={args.past_mode})",
            period_labels=labels_h2,
            n_labels=n_labels,
            color_mat=color_h2,
            vmin=vmin,
            vmax=vmax,
            text_mat=text_h2,
        )
        print(f"# wrote: {out_h}")

    # NonHeads
    color_n, text_n, Ts_n = build_color_text_and_T(is_heads=False)
    color_n2, text_n2, labels_n2, _ = apply_past_mode(color_n, text_n, Ts_n)

    out_n = os.path.join(
        args.outdir,
        f"heatmap_pastHalfYear_plus_foundEntries_x_pnext_nonheads_{suffix}.png",
    )
    plot_heatmap_period_x_n(
        out_n,
        title=f"NonHeads: JSD(AUTO vs {base_key} MANUAL) on pnext n={args.nmin}..{args.nmax}",
        period_labels=labels_n2,
        n_labels=n_labels,
        color_mat=color_n2,
        vmin=vmin,
        vmax=vmax,
        text_mat=text_n2,
    )
    print(f"# wrote: {out_n}")


if __name__ == "__main__":
    main()
