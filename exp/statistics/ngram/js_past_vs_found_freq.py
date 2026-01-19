#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
period(半年) × (n=1..N の「2019~2025(past) を基準にした JS距離」) のヒートマップを 1枚で出す。
さらに、found-model の分布も「基準（2019~2025 past）」に対する JS距離として 1行追加する。

追加仕様（今回）:
  - 縦軸ラベルに「n=1 のデータ総数 T」を付ける（例: 2019H1 (T=12345)）
    ※Heads/NonHeads で T は別々に計算される（その行の 1-gram カウンタ総和）
  - found 行ラベルは「入力 found_path を厳密に反映」する
      FOUND_MODEL: <basename(found_path)>
    例: found-model/2024-10-13-GCU/ -> "FOUND_MODEL: 2024-10-13-GCU"

仕様:
  - past: グループ集合変化でセグメント分割（境界は跨がない）
  - Unknown は NonHeads 側に含める
  - found: staff_group / group で Heads/NonHeads 判定（タイムライン無し）
  - found: ext_assigned / out_assigned を読む（2引数 out_assigned は無視）
  - JS distance = sqrt(JSD), ln (natural log)
  - カラースケール統一: vmin=0, vmax=sqrt(ln 2)
  - 各セルに値（小数3桁）を表示
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
VALID_SHIFTS = {"D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"}
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
# =============================================================
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')

# out_assigned(id, yyyymmdd, "SHIFT").（2引数 out_assigned はマッチしないので無視）
PAT_OUT = re.compile(r'^out_assigned\(\s*(\d+)\s*,\s*(\d{8})\s*,\s*"([^"]+)"\s*\)\.')

# group("GROUPNAME", staff_id).
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


def list_found_files(found_path: str) -> List[str]:
    """
    found_path が:
      - ファイルならそれ1本
      - ディレクトリなら:
          1) found-model*.lp
          2) それが無ければ *.lp
    を読む
    """
    if os.path.isfile(found_path):
        return [found_path]
    if not os.path.isdir(found_path):
        return []
    fs = sorted(glob.glob(os.path.join(found_path, "found-model*.lp")))
    if fs:
        return fs
    fs = sorted(glob.glob(os.path.join(found_path, "*.lp")))
    return fs


def load_found_model(path: str) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]:
    """
    found-model.lp を読んで:
      - seqs_by_staff: {staff_id: [(day, shift), ...]}
      - groups_by_staff: {staff_id: set(groupname)}

    対応:
      - ext_assigned(staff_id, day, "SHIFT").
      - out_assigned(staff_id, yyyymmdd, "SHIFT").
      - staff_group("GROUPNAME", staff_id).
      - group("GROUPNAME", staff_id).
      - out_assigned(staff_id, yyyymmdd). は無視（パターン不一致）
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
                if sh in VALID_SHIFTS:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_OUT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))  # yyyymmdd
                sh = m.group(3)
                if sh in VALID_SHIFTS:
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

    return seqs_by_staff, groups_by_staff


def count_ngrams_found_heads_nonheads(found_files: List[str], n: int, heads_name: str) -> Tuple[Counter, Counter]:
    """
    found側: 複数 found-model を合算して Heads / NonHeads を数える（タイムライン無し）
    """
    heads = Counter()
    nonheads = Counter()

    for fp in found_files:
        seqs_by_staff, groups_by_staff = load_found_model(fp)

        for sid, seq in seqs_by_staff.items():
            if len(seq) < n:
                continue
            seq_sorted = sorted(seq, key=lambda t: t[0])
            shifts = [s for _, s in seq_sorted]

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)
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


def plot_heatmap_period_x_n(
    out_png: str,
    title: str,
    period_labels: List[str],
    n_labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
) -> None:
    R = len(period_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(7.0, R * 0.55)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im)

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), period_labels)

    mid = (vmin + vmax) / 2.0
    for i in range(R):
        for j in range(C):
            val = mat[i][j]
            txt_color = "black" if val > mid else "white"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts *.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("found_path", help="found dir OR found-model.lp")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--outdir", default="out/halfyear+found-model_vs_total")
    ap.add_argument("--only-nonheads", action="store_true", help="NonHeads(+Unknown) だけ出力（Headsを出さない）")
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

    found_files = list_found_files(args.found_path)
    if not found_files:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

    # ---- found 行ラベルを入力から厳密に生成（タイトル/ファイル名には使わない）
    found_label = f" {os.path.basename(os.path.normpath(args.found_path))}"

    # load past + timeline
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # periods (half-year) + base(total) range
    half_periods = make_halfyear_periods(args.start_year, args.end_year)
    base_key = f"{args.start_year}~{args.end_year}"
    base_start = args.start_year * 10000 + 101
    base_end = args.end_year * 10000 + 1231

    # 表示する行：半年 + found_label（基準行は表示しない）
    periods = list(half_periods) + [(found_label, None, None)]  # type: ignore

    period_labels_base = [k for (k, _, _) in periods]
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

    # found counters cache (nごとに1回だけ数える)
    found_heads_by_n: Dict[int, Counter] = {}
    found_non_by_n: Dict[int, Counter] = {}
    for n in ns:
        h_f, non_f = count_ngrams_found_heads_nonheads(found_files, n=n, heads_name=args.heads_name)
        found_heads_by_n[n] = h_f
        found_non_by_n[n] = non_f

    # global scale (ln): vmax = sqrt(ln2) ~ 0.8326
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

    # n=1 の総数 T を各行で計算して返す
    def build_matrix(is_heads: bool) -> Tuple[List[List[float]], List[int]]:
        mat: List[List[float]] = []
        totals: List[int] = []

        for (pkey, d1, d2) in periods:
            row: List[float] = []

            # ---- n=1 の総数（表示用）
            if pkey == found_label:
                c_p_1 = found_heads_by_n[1] if is_heads else found_non_by_n[1]
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

                if pkey == found_label:
                    c_p = found_heads_by_n[n] if is_heads else found_non_by_n[n]
                else:
                    assert d1 is not None and d2 is not None
                    h_p, non_p = count_ngrams_heads_nonheads_in_range(
                        segs_by_person, n=n, heads_name=args.heads_name, date_start=d1, date_end=d2
                    )
                    c_p = h_p if is_heads else non_p

                row.append(js_distance_from_counters(c_p, c_b, args.alpha))

            mat.append(row)

        return mat, totals

    # Heads heatmap
    if not args.only_nonheads:
        mat_h, totals_h = build_matrix(is_heads=True)
        period_labels_h = [f"{lbl} ({t})" for lbl, t in zip(period_labels_base, totals_h)]

        out_h = os.path.join(
            args.outdir,
            f"heatmap_halfyear_plus_found_x_ngram_heads_{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}.png"
        )
        plot_heatmap_period_x_n(
            out_h,
            title=f"Heads: JSdist(period/{found_label} vs {base_key}(past)) for n={args.nmin}..{args.nmax} [ln]",
            period_labels=period_labels_h,
            n_labels=n_labels,
            mat=mat_h,
            vmin=vmin,
            vmax=vmax,
        )
        print(f"# wrote: {out_h}")

    # NonHeads heatmap
    mat_n, totals_n = build_matrix(is_heads=False)
    period_labels_n = [f"{lbl} ({t})" for lbl, t in zip(period_labels_base, totals_n)]

    out_n = os.path.join(
        args.outdir,
        f"heatmap_halfyear_plus_found_x_ngram_nonheads_{args.start_year}-{args.end_year}_n{args.nmin}-{args.nmax}.png"
    )
    plot_heatmap_period_x_n(
        out_n,
        title=f"NonHeads: JSdist(period/{found_label} vs {base_key}(past)) for n={args.nmin}..{args.nmax} [ln]",
        period_labels=period_labels_n,
        n_labels=n_labels,
        mat=mat_n,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"# wrote: {out_n}")


if __name__ == "__main__":
    main()
