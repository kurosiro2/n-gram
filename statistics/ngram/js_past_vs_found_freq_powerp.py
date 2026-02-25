#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(半年) × (n=1..N の「2019~2025(past) を基準にした JS距離」) のヒートマップを 1枚で出す。
さらに、found-model の分布も「基準（2019~2025 past）」に対する JS距離として 行を追加する。

★変更（今回）: past半期行の比較基準を「全期間から当該半期を除外した base」にする（leave-one-halfyear-out）
  - past半期 P のセル: JSdist( P , base_excl(P) )
  - found行のセル: JSdist( found , base_full )  ※従来通り

注意:
  - base_excl(P) を作るとき、単純に base_full - P はしない
    （半期境界を跨ぐn-gramが残るため）
  - 代わりに [base_start..P_start-1] と [P_end+1..base_end] を別々に数えて足す。

summaryモード / boxplot などは元仕様を維持。
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
# matplotlib font sizing (global)
# -------------------------------------------------------------
def apply_font_scale(scale: float) -> None:
    if scale <= 0:
        scale = 1.0
    base = {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
    for k, v in base.items():
        plt.rcParams[k] = v * scale


# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader
import foundmodel_data_loader as found_loader

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "SE", "SN", "WR", "PH", "E", "N"}
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


def count_ngrams_heads_nonheads_in_two_ranges(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    r1: Optional[Tuple[int, int]],
    r2: Optional[Tuple[int, int]],
) -> Tuple[Counter, Counter]:
    """
    2区間をそれぞれカウントして足す（区間間の“跨ぎn-gram”は自然に消える）。
    r1/r2 が None の場合は無視。
    """
    heads = Counter()
    non = Counter()

    if r1 is not None:
        a, b = r1
        if a <= b:
            h1, n1 = count_ngrams_heads_nonheads_in_range(segs_by_person, n, heads_name, a, b)
            heads += h1
            non += n1

    if r2 is not None:
        a, b = r2
        if a <= b:
            h2, n2 = count_ngrams_heads_nonheads_in_range(segs_by_person, n, heads_name, a, b)
            heads += h2
            non += n2

    return heads, non


# =============================================================
# found: load + ngram counts (no timeline)
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
      - found_path がディレクトリで、その直下に lp がある -> 1件
      - found_path がディレクトリで、直下に lp が無い -> 直下サブdirごとに複数件
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

    # B) 直下サブdirごと
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

    # C) 再帰（lpの親dirごと）
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
# found: period filter (YYYYMMDD) + split by consecutive days
# =============================================================
DATE8_RE = re.compile(r"^\d{8}$")


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
        s += pi * math.log(pi / qi)
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
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101,  y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701,  y * 10000 + 1231))
    return periods


# =============================================================
# plotting: heatmap
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
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(12.0, C * 2.6)
    fig_h = max(9.0, R * 0.75)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right", fontsize=16)
    plt.yticks(list(range(R)), period_labels, fontsize=16)

    VALUE_FS = 19
    MANUAL_FS = 19
    BAR_FS = 12
    DIFF_FS = 16

    mid = (vmin + vmax) / 2.0

    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"

            if ann_texts is None:
                plt.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=VALUE_FS, color=txt_color)
                continue

            s = (ann_texts[i][j] or "").strip()

            if "\n" in s:
                parts = [p.strip() for p in s.split("\n")]

                if len(parts) == 2 and parts[1].startswith("(") and parts[1].endswith(")"):
                    top, bottom = parts
                    plt.text(j, i - 0.18, top, ha="center", va="center",
                             fontsize=VALUE_FS, color=txt_color)
                    plt.text(j, i + 0.22, bottom, ha="center", va="center",
                             fontsize=DIFF_FS, color=txt_color)
                    continue

                if len(parts) >= 3:
                    top = parts[0]
                    midbar = parts[1] if len(parts) >= 2 else "|"
                    bottom = parts[2]

                    plt.text(j, i - 0.28, top, ha="center", va="center",
                             fontsize=MANUAL_FS, color=txt_color)
                    plt.text(j, i, midbar, ha="center", va="center",
                             fontsize=BAR_FS, color=txt_color)
                    plt.text(j, i + 0.28, bottom, ha="center", va="center",
                             fontsize=MANUAL_FS, color=txt_color)
                    continue

                if len(parts) == 2:
                    top, bottom = parts
                    plt.text(j, i - 0.18, top, ha="center", va="center",
                             fontsize=MANUAL_FS, color=txt_color)
                    plt.text(j, i + 0.22, bottom, ha="center", va="center",
                             fontsize=MANUAL_FS, color=txt_color)
                    continue

            plt.text(j, i, s, ha="center", va="center",
                     fontsize=VALUE_FS, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =============================================================
# stats: median/IQR (manual)
# =============================================================
def _percentile_linear(xs_sorted: List[float], p: float) -> float:
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


def manual_iqr_by_col(past_mat: List[List[float]]) -> Tuple[List[float], List[float], List[float]]:
    if not past_mat:
        return [], [], []
    K = len(past_mat)
    C = len(past_mat[0])
    cols: List[List[float]] = [[] for _ in range(C)]
    for i in range(K):
        for j in range(C):
            cols[j].append(past_mat[i][j])

    q1s: List[float] = []
    meds: List[float] = []
    q3s: List[float] = []
    for j in range(C):
        xs = sorted(cols[j])
        q1s.append(_percentile_linear(xs, 0.25))
        meds.append(_percentile_linear(xs, 0.50))
        q3s.append(_percentile_linear(xs, 0.75))
    return q1s, meds, q3s


def summarize_past_rows_to_one(
    past_mat: List[List[float]],
    past_row_label: str,
    qfmt: str = "{:.3f}",
) -> Tuple[List[List[float]], List[List[str]], List[str]]:
    if not past_mat:
        return [[]], [[]], [past_row_label]

    q1s, meds, q3s = manual_iqr_by_col(past_mat)

    mat1: List[List[float]] = [meds]
    ann1: List[List[str]] = [[f"{qfmt.format(q1)}~\n|\n{qfmt.format(q3)}" for q1, q3 in zip(q1s, q3s)]]
    labels1 = [past_row_label]
    return mat1, ann1, labels1


def diff_from_iqr(x: float, q1: float, q3: float) -> float:
    if x < q1:
        return x - q1
    if x > q3:
        return x - q3
    return 0.0


# =============================================================
# plotting: boxplot + found points (+ minmax markers)
# =============================================================
def _finite_only(vals: List[float]) -> List[float]:
    return [v for v in vals if (v is not None and math.isfinite(v))]


def plot_boxplot_by_n_with_found(
    out_png: str,
    title: str,
    n_labels: List[str],
    past_data_by_n: List[List[float]],
    found_points_by_entry: Dict[str, List[float]],
    ymin: float,
    ymax: float,
    show_minmax_markers: bool = False,
    minmax_marker_size: Optional[float] = None,
    found_marker_size: Optional[float] = None,
) -> None:
    C = len(n_labels)
    fig_w = max(10.0, C * 1.8)
    fig_h = 7.0

    clean_past: List[List[float]] = [_finite_only(vs) for vs in past_data_by_n]

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    ax.boxplot(
        clean_past,
        tick_labels=n_labels,
        showfliers=False,
    )

    xs = list(range(1, C + 1))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_xticklabels(n_labels, rotation=45, ha="right")
    ax.set_ylabel("JSDistance")
    ax.set_title(title)

    legend_handles = []
    legend_labels = []

    if show_minmax_markers:
        mins: List[Optional[float]] = []
        maxs: List[Optional[float]] = []
        for vals in clean_past:
            if not vals:
                mins.append(None)
                maxs.append(None)
            else:
                mins.append(min(vals))
                maxs.append(max(vals))

        ms = float(minmax_marker_size) if (minmax_marker_size is not None) else float(plt.rcParams["font.size"] * 1.2)
        s_area_open = ms * ms
        s_area_dot = (ms * 2.0) * (ms * 2.0)

        x_min: List[int] = []
        y_min: List[float] = []
        x_max: List[int] = []
        y_max: List[float] = []

        for x, mn, mx in zip(xs, mins, maxs):
            if mn is not None and math.isfinite(mn):
                x_min.append(x); y_min.append(mn)
            if mx is not None and math.isfinite(mx):
                x_max.append(x); y_max.append(mx)

        sc_min = ax.scatter(
            x_min, y_min,
            marker="o",
            s=s_area_open,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            zorder=50,
            clip_on=False,
        )
        sc_max = ax.scatter(
            x_max, y_max,
            marker=".",
            s=s_area_dot,
            color="black",
            zorder=50,
            clip_on=False,
        )

        legend_handles += [sc_max, sc_min]
        legend_labels += ["MANUAL: max", "MANUAL: min"]

    entry_names = list(found_points_by_entry.keys())
    if entry_names:
        msf = float(found_marker_size) if (found_marker_size is not None) else float(plt.rcParams["font.size"] * 0.9)
        s_found = msf * msf

        for name in entry_names:
            vals = found_points_by_entry[name]
            yj = [float(v) if (v is not None and math.isfinite(v)) else float("nan") for v in vals]
            sc = ax.scatter(
                xs, yj,
                marker="o",
                s=s_found,
                zorder=60,
                clip_on=False,
            )
            legend_handles.append(sc)
            legend_labels.append(f"AUTO: {name}")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


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

    ap.add_argument(
        "--past-mode",
        choices=["full", "summary"],
        default="summary",
        help="full=従来の半期行すべて, summary=pastを1行に集約(色=中央値,表示=Q1~/|/Q3)+found行",
    )

    # ★今回: 半期を抜いた基準で比較するか
    ap.add_argument(
        "--past-base",
        choices=["full", "loo"],
        default="loo",
        help="past半期行の比較基準: full=全期間そのまま, loo=当該半期を除外した全期間(2区間合算)",
    )

    ap.add_argument("--date-from", default=None, help="found集計期間の開始日（YYYYMMDD, 例: 20241101）")
    ap.add_argument("--date-to", default=None, help="found集計期間の終了日（YYYYMMDD, 例: 20241130）")

    ap.add_argument("--boxplot", action="store_true", help="Draw boxplot by n (past half-years) + overlay found points.")
    ap.add_argument("--boxplot-only", action="store_true", help="Only draw boxplot (skip heatmap).")
    ap.add_argument("--boxplot-minmax-markers", action="store_true",
                    help="Overlay min/max markers on boxplot (past min=open circle, past max=dot).")
    ap.add_argument("--minmax-marker-size", type=float, default=10.0,
                    help="Marker size base for min/max overlays. If omitted, uses rcParams font.size.")
    ap.add_argument("--found-marker-size", type=float, default=10.0,
                    help="Marker size base for found overlay points. If omitted, uses rcParams font.size.")
    ap.add_argument("--font-scale", type=float, default=1.0, help="Global font scale (e.g., 1.2, 1.35).")

    args = ap.parse_args()

    apply_font_scale(args.font_scale)
    ensure_dir(args.outdir)

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")

    if args.boxplot_only:
        args.boxplot = True

    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    found_entries = collect_found_entries(args.found_path)
    if not found_entries:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

    date_from = parse_yyyymmdd(args.date_from)
    date_to = parse_yyyymmdd(args.date_to)
    if date_from is not None and date_to is not None and date_to < date_from:
        raise ValueError(f"--date-to must be >= --date-from (from={date_from}, to={date_to})")

    # load past + timeline
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram" for n in ns]

    # -------------------------
    # 1) past半期ごとのカウント（後で再利用する）
    # -------------------------
    past_counts_by_halfyear_n: Dict[str, Dict[int, Tuple[Counter, Counter]]] = defaultdict(dict)
    for (pkey, d1, d2) in half_periods:
        for n in ns:
            h_p, non_p = count_ngrams_heads_nonheads_in_range(
                segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
            )
            past_counts_by_halfyear_n[pkey][n] = (h_p, non_p)

    # -------------------------
    # 2) base_full（全期間）カウント（found行の比較用に固定）
    # -------------------------
    base_full_heads_by_n: Dict[int, Counter] = {}
    base_full_non_by_n: Dict[int, Counter] = {}
    for n in ns:
        h_b, non_b = count_ngrams_heads_nonheads_in_range(
            segs_by_person, n=n, heads_name=args.heads_name, date_start=base_start, date_end=base_end
        )
        base_full_heads_by_n[n] = h_b
        base_full_non_by_n[n] = non_b

    # -------------------------
    # 3) base_excl(P)（半期を抜いた基準）を半期ごと×nで作る
    #    ※2区間合算方式（跨ぎn-gramが残らない）
    # -------------------------
    base_excl_heads_by_halfyear_n: Dict[str, Dict[int, Counter]] = defaultdict(dict)
    base_excl_non_by_halfyear_n: Dict[str, Dict[int, Counter]] = defaultdict(dict)

    if args.past_base == "loo":
        for (pkey, d1, d2) in half_periods:
            r1 = (base_start, d1 - 1) if base_start <= d1 - 1 else None
            r2 = (d2 + 1, base_end) if d2 + 1 <= base_end else None
            for n in ns:
                h_ex, non_ex = count_ngrams_heads_nonheads_in_two_ranges(
                    segs_by_person, n=n, heads_name=args.heads_name, r1=r1, r2=r2
                )
                base_excl_heads_by_halfyear_n[pkey][n] = h_ex
                base_excl_non_by_halfyear_n[pkey][n] = non_ex

    # -------------------------
    # 4) found counters cache
    # -------------------------
    found_heads_by_entry_n: Dict[str, Dict[int, Counter]] = {}
    found_non_by_entry_n: Dict[str, Dict[int, Counter]] = {}
    for label, files in found_entries:
        found_heads_by_entry_n[label] = {}
        found_non_by_entry_n[label] = {}
        for n in ns:
            h_f, non_f = count_ngrams_found_heads_nonheads(
                files, n=n, heads_name=args.heads_name, date_from=date_from, date_to=date_to
            )
            found_heads_by_entry_n[label][n] = h_f
            found_non_by_entry_n[label][n] = non_f

    # rows definition: (row_label, d1, d2, kind)
    periods: List[Tuple[str, Optional[int], Optional[int], str]] = []
    for k, d1, d2 in half_periods:
        periods.append((k, d1, d2, "past"))
    for label, _files in found_entries:
        periods.append((f"AUTO: {label}", None, None, "found"))

    vmin = 0.0
    vmax = 0.45  # sqrt(ln2)

    def pick_base_counter(is_heads: bool, kind: str, pkey: str, n: int) -> Counter:
        """
        基準分布（base）を返す。
          - past行:
              --past-base=full -> base_full
              --past-base=loo  -> base_excl(pkey)
          - found行:
              常に base_full（従来通り）
        """
        if kind == "past":
            if args.past_base == "loo":
                if is_heads:
                    return base_excl_heads_by_halfyear_n[pkey][n]
                return base_excl_non_by_halfyear_n[pkey][n]
            else:
                if is_heads:
                    return base_full_heads_by_n[n]
                return base_full_non_by_n[n]
        else:
            # found
            if is_heads:
                return base_full_heads_by_n[n]
            return base_full_non_by_n[n]

    def build_matrix(is_heads: bool) -> List[List[float]]:
        mat: List[List[float]] = []
        for (pkey, d1, d2, kind) in periods:
            row: List[float] = []
            for n in ns:
                c_b = pick_base_counter(is_heads, kind, pkey, n)

                if kind == "found":
                    label = pkey.replace("AUTO: ", "", 1)
                    c_p = found_heads_by_entry_n[label][n] if is_heads else found_non_by_entry_n[label][n]
                else:
                    # past 半期カウントは事前計算済みを使う
                    h_p, non_p = past_counts_by_halfyear_n[pkey][n]
                    c_p = h_p if is_heads else non_p

                row.append(js_distance_from_counters(c_p, c_b, args.alpha))
            mat.append(row)
        return mat

    def apply_past_mode_and_annotations(
        mat: List[List[float]],
    ) -> Tuple[List[List[float]], List[List[str]], List[str], List[List[float]], List[List[float]]]:
        """
        戻り値:
          mat2, ann2, labels2, past_mat(half-years only), found_mat(rows for each entry)
        """
        K = len(half_periods)
        past_mat = mat[:K]
        found_mat = mat[K:]

        # MANUAL IQR（past半期行の値から列ごとに計算） -> found注釈diff用
        q1s, meds, q3s = manual_iqr_by_col(past_mat)

        period_labels_base = [k for (k, _, _, _) in periods]

        found_ann: List[List[str]] = []
        for row in found_mat:
            arow: List[str] = []
            for j, v in enumerate(row):
                d = diff_from_iqr(v, q1s[j], q3s[j]) if j < len(q1s) else 0.0
                arow.append(f"{v:.3f}\n({d:+.3f})")
            found_ann.append(arow)

        if args.past_mode == "full":
            past_ann = [[f"{v:.3f}" for v in row] for row in past_mat]
            ann = past_ann + found_ann
            labels = period_labels_base
            return mat, ann, labels, past_mat, found_mat

        past_label = "MANUAL"
        sum_mat1, sum_ann1, sum_labels1 = summarize_past_rows_to_one(
            past_mat=past_mat,
            past_row_label=past_label,
            qfmt="{:.3f}",
        )

        labels2 = sum_labels1 + period_labels_base[K:]
        mat2 = sum_mat1 + found_mat
        ann2 = sum_ann1 + found_ann
        return mat2, ann2, labels2, past_mat, found_mat

    def past_mat_to_boxdata(past_mat: List[List[float]]) -> List[List[float]]:
        cols: List[List[float]] = [[] for _ in ns]
        for i in range(len(past_mat)):
            for j in range(len(ns)):
                cols[j].append(past_mat[i][j])
        return cols

    def found_mat_to_points(found_mat: List[List[float]]) -> Dict[str, List[float]]:
        points: Dict[str, List[float]] = {}
        for (pkey, _d1, _d2, kind), row in zip(periods[len(half_periods):], found_mat):
            if kind != "found":
                continue
            label = pkey.replace("AUTO: ", "", 1)
            points[label] = list(row)
        return points

    suffix = f"{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}_{args.past_mode}_pastbase-{args.past_base}"

    # -------------------------
    # Heads
    # -------------------------
    if not args.only_nonheads:
        mat_h = build_matrix(is_heads=True)
        mat_h2, ann_h2, labels_h2, past_h, found_h = apply_past_mode_and_annotations(mat_h)

        if not args.boxplot_only:
            out_h = os.path.join(args.outdir, f"heatmap_halfyear_plus_foundEntries_x_ngram_heads_{suffix}.png")
            plot_heatmap_period_x_n(
                out_h,
                title=f"Heads: JSdist(past(half-year) vs base[{args.past_base}], found vs base[full]) for n={args.nmin}..{args.nmax} [ln] (past-mode={args.past_mode})",
                period_labels=labels_h2,
                n_labels=n_labels,
                mat=mat_h2,
                vmin=vmin,
                vmax=vmax,
                ann_texts=ann_h2,
            )
            print(f"# wrote: {out_h}")

        if args.boxplot:
            box_h = past_mat_to_boxdata(past_h)
            found_pts_h = found_mat_to_points(found_h)
            out_bh = os.path.join(args.outdir, f"boxplot_halfyear_jsdist_heads_{suffix}.png")
            plot_boxplot_by_n_with_found(
                out_bh,
                title=f"Heads: past half-year JSdist (base={args.past_base}) + found (base=full)",
                n_labels=n_labels,
                past_data_by_n=box_h,
                found_points_by_entry=found_pts_h,
                ymin=vmin,
                ymax=vmax,
                show_minmax_markers=args.boxplot_minmax_markers,
                minmax_marker_size=args.minmax_marker_size,
                found_marker_size=args.found_marker_size,
            )
            print(f"# wrote: {out_bh}")

    # -------------------------
    # NonHeads
    # -------------------------
    mat_n = build_matrix(is_heads=False)
    mat_n2, ann_n2, labels_n2, past_n, found_n = apply_past_mode_and_annotations(mat_n)

    if not args.boxplot_only:
        out_n = os.path.join(args.outdir, f"heatmap_halfyear_plus_foundEntries_x_ngram_nonheads_{suffix}.png")
        plot_heatmap_period_x_n(
            out_n,
            title=f"MANUAL vs AUTO",
            period_labels=labels_n2,
            n_labels=n_labels,
            mat=mat_n2,
            vmin=vmin,
            vmax=vmax,
            ann_texts=ann_n2,
        )
        print(f"# wrote: {out_n}")

    if args.boxplot:
        box_n = past_mat_to_boxdata(past_n)
        found_pts_n = found_mat_to_points(found_n)
        out_bn = os.path.join(args.outdir, f"boxplot_halfyear_jsdist_nonheads_{suffix}.png")
        plot_boxplot_by_n_with_found(
            out_bn,
            title=f"MANUAL vs AUTO",
            n_labels=n_labels,
            past_data_by_n=box_n,
            found_points_by_entry=found_pts_n,
            ymin=vmin,
            ymax=vmax,
            show_minmax_markers=args.boxplot_minmax_markers,
            minmax_marker_size=args.minmax_marker_size,
            found_marker_size=args.found_marker_size,
        )
        print(f"# wrote: {out_bn}")


if __name__ == "__main__":
    main()
