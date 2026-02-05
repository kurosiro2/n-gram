#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【事前評価】prefix固定で、2期間の P(next|prefix) を同じ棒グラフで比較
  - MLE と Laplace(add-k) を同一図に重ねて比較
  - 2人の看護師IDを直接指定して、その2人同士の分布比較も可能（--nurse-a-id/--nurse-b-id）
  - 円滑化前（MLE-only）のグラフも別PNGで出力
  - NEW: 横軸displayは固定10種ではなく「A/B両分布のどちらかで観測された next 記号のみ（union）」を表示
        ※表示順は VALID_SHIFTS の順を維持

要点:
  - past-shifts(.lp) を読み、group-settings（timeline）で「グループ集合が変わるたびに」セグメント分割
    → セグメント境界を跨ぐ n-gram は数えない
  - デフォルト: Heads / NonHeads の2値（Heads-name で判定）
  - --nurse-a-id と --nurse-b-id を両方指定すると「看護師A vs 看護師B」モードになり、
    Heads/NonHeads 集計は行わず、指定2名のみを比較する
  - 指定prefix（長さ = N-1）に対する next 分布（10種から観測されたものだけ表示）を出す

Laplace 支持集合:
  --laplace-support all          : VALID_SHIFTS(10種) に add-k（|X|=10固定）
  --laplace-support observed_ab  : 比較対象のA/Bで観測された next の union のみに add-k（|X|可変）

使い方（単一病棟: Heads/NonHeads 既定）:
  python pnext_prefix_compare_2periods_laplace.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --laplace-k 1.0 \
    --laplace-support observed_ab \
    --heads-name Heads \
    --outdir out/prefix_compare

使い方（単一病棟: 看護師A vs 看護師B）:
  python pnext_prefix_compare_2periods_laplace.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --nurse-a-id 101 --nurse-b-id 202 \
    --laplace-k 1.0 \
    --laplace-support observed_ab \
    --outdir out/prefix_compare_nurses

使い方（全病棟: past-shifts-dir × group-settings-root）:
  python pnext_prefix_compare_2periods_laplace.py \
    /workspace/2025/past-shifts \
    /workspace/2025/group-settings \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20240101 --a-end 20240630 \
    --b-start 20240701 --b-end 20241231 \
    --laplace-k 1.0 \
    --laplace-support observed_ab \
    --outdir out/prefix_compare_all
"""

import os
import sys
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# ★追加：グラフの文字を全体的に大きくする（全図に適用）
# =============================================================
plt.rcParams.update({
    "font.size": 16,          # 基本
    "axes.titlesize": 18,     # タイトル
    "axes.labelsize": 16,     # 軸ラベル
    "xtick.labelsize": 15,    # x目盛り
    "ytick.labelsize": 15,    # y目盛り
    "legend.fontsize": 15,    # 凡例
})

# -------------------------------------------------------------
# import path (exp/statistics 配下で使う想定)
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = ["D", "LD", "EM", "LM", "E", "SE", "N", "SN", "WR", "PH"]
VALID_SHIFTS_SET = set(VALID_SHIFTS)
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


def within_range(d: int, start: Optional[int], end: Optional[int]) -> bool:
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target_group: str) -> bool:
    tg = target_group.lower()
    return any(g.lower() == tg for g in group_set)


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
    境界を跨ぐ n-gram を数えないための前処理。
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


def parse_prefix(prefix_str: str, expected_len: int) -> Prefix:
    s = prefix_str.strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f'--prefix must have length {expected_len} (got {len(parts)}): "{prefix_str}"')
    for p in parts:
        if p not in VALID_SHIFTS_SET:
            raise ValueError(f"--prefix contains invalid shift: {p}")
    return tuple(parts)


# -------------------------------------------------------------
# NEW: display shifts (x-axis) selector
# -------------------------------------------------------------
def build_display_shifts(nextA: Counter, nextB: Counter) -> List[str]:
    """
    横軸に表示する勤務記号:
      A/Bのどちらかで観測された next の union を VALID_SHIFTS順で返す。
    もし何も観測されてない場合（Ny=0など）は保険で全10種を返す。
    """
    observed = (set(nextA.keys()) | set(nextB.keys())) & VALID_SHIFTS_SET
    display = [s for s in VALID_SHIFTS if s in observed]
    return display if display else list(VALID_SHIFTS)


# -------------------------------------------------------------
# NEW: nurse id resolver for past-shifts key
# -------------------------------------------------------------
def find_person_key_by_id(seqs: SeqDict, nurse_id: int) -> PersonKey:
    """
    past-shifts のキーは (nid, name) だが nid が str/ゼロ埋め等の可能性に備えて頑健に探す。
    """
    keys = list(seqs.keys())
    nid_int = int(nurse_id)
    nid_str = str(nurse_id).strip()

    # 1) int 完全一致
    cands = [k for k in keys if isinstance(k[0], int) and k[0] == nid_int]
    if cands:
        return cands[0]

    # 2) 文字列一致（ゼロ埋めも許容）
    def norm(x) -> str:
        s = str(x).strip()
        s2 = s.lstrip("0")
        return s2 if s2 else s

    cands = [k for k in keys if norm(k[0]) == norm(nid_str)]
    if cands:
        return cands[0]

    raise ValueError(f"nurse_id={nurse_id} が past-shifts に見つからない")


# -------------------------------------------------------------
# counting functions
# -------------------------------------------------------------
def count_next_for_prefix_for_one_person_in_period(
    segs: List[Segment],
    n: int,
    target_prefix: Prefix,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, int]:
    """
    1人分：期間内で target_prefix に一致する (prefix->next) だけ数える
    戻り値:
      next_counter, Ny
    """
    next_counter = Counter()
    Ny = 0
    pref_len = n - 1

    for seg in segs:
        sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
        if len(sseq) < n:
            continue

        shifts = [s for (_, s) in sseq]
        for i in range(len(shifts) - n + 1):
            pfx = tuple(shifts[i : i + pref_len])
            if pfx != target_prefix:
                continue
            nxt = shifts[i + pref_len]
            next_counter[nxt] += 1
            Ny += 1

    return next_counter, Ny


def count_next_for_prefix_in_period(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter, int, int]:
    """
    期間内で、target_prefix に一致する (prefix->next) だけ数える（集計: Heads vs NonHeads）
    戻り値:
      heads_next_counter, nonheads_next_counter, heads_Ny, nonheads_Ny
    """
    heads_next = Counter()
    non_next = Counter()
    Ny_h = 0
    Ny_n = 0

    pref_len = n - 1

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for (_, s) in sseq]
            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i : i + pref_len])
                if pfx != target_prefix:
                    continue
                nxt = shifts[i + pref_len]
                if is_heads:
                    heads_next[nxt] += 1
                    Ny_h += 1
                else:
                    non_next[nxt] += 1
                    Ny_n += 1

    return heads_next, non_next, Ny_h, Ny_n


# -----------------------------
# probability vectors (display_shifts-driven)
# -----------------------------
def mle_probs(next_counter: Counter, Ny: int, display_shifts: List[str]) -> List[float]:
    """
    MLE確率ベクトル（display_shifts の順で返す）
    Ny=0 は本来「未定義」だが、図を出すため 0ベクトルで返す（=未観測を可視化）
    """
    if Ny <= 0:
        return [0.0] * len(display_shifts)
    return [next_counter.get(x, 0) / Ny for x in display_shifts]


def laplace_probs(
    next_counter: Counter,
    Ny: int,
    k: float,
    display_shifts: List[str],
    support: Optional[Set[str]] = None,
) -> List[float]:
    """
    Laplace(add-k) 確率ベクトル（display_shifts の順で返す）
      support=None -> VALID_SHIFTS 全部に add-k（|X|=10）
      support=set -> support に含まれる next のみに add-k（分母は Ny + k*|support|）
    """
    if support is None:
        support = VALID_SHIFTS_SET
    if not support:
        support = VALID_SHIFTS_SET

    denom = Ny + k * len(support)
    if denom <= 0:
        uni = 1.0 / max(1, len(display_shifts))
        return [uni] * len(display_shifts)

    out: List[float] = []
    for x in display_shifts:
        c = next_counter.get(x, 0)
        if x in support:
            out.append((c + k) / denom)
        else:
            out.append(c / denom)
    return out


def print_debug(
    tag: str,
    prefix: Prefix,
    next_counter: Counter,
    Ny: int,
    k: float,
    display_shifts: List[str],
    support: Optional[Set[str]] = None,
) -> None:
    if support is None:
        support = VALID_SHIFTS_SET
    if not support:
        support = VALID_SHIFTS_SET

    print("")
    print("=" * 110)
    print(tag)
    print("=" * 110)
    print(f"prefix={','.join(prefix)}  Ny=c(prefix)={Ny}   |support|={len(support)}  laplace_k={k}")
    print("display_shifts:", ",".join(display_shifts))
    print("laplace_support_set:", ",".join(sorted(support)))
    top = next_counter.most_common(10)
    if top:
        print("top next counts:", ", ".join([f"{x}:{c}" for x, c in top]))
    else:
        print("top next counts: (none)")

    pmle = mle_probs(next_counter, Ny, display_shifts)
    plap = laplace_probs(next_counter, Ny, k, display_shifts, support=support)

    print(f"{'next':>4}  {'count':>6}  {'P_MLE':>10}  {'P_LAP':>10}")
    for x, m, l in zip(display_shifts, pmle, plap):
        print(f"{x:>4}  {next_counter.get(x,0):>6}  {m:>10.6f}  {l:>10.6f}")


# -----------------------------
# plotting
# -----------------------------
def plot_compare_2bars(
    out_png: str,
    title: str,
    display_shifts: List[str],
    probs_A: List[float],
    probs_B: List[float],
    labelA: str,
    labelB: str,
    ylabel: str,
) -> None:
    """
    MLE-only（円滑化前）: 2系列（A/B）だけの棒グラフ
    """
    xs = list(range(len(display_shifts)))
    width = 0.32

    plt.figure(figsize=(13.0, 5.6))

    xsA = [x - 0.5 * width for x in xs]
    xsB = [x + 0.5 * width for x in xs]

    plt.bar(xsA, probs_A, width=width, label=labelA)
    plt.bar(xsB, probs_B, width=width, label=labelB)

    plt.xticks(xs, display_shifts)
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.xlabel("next shift")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


def plot_compare_4bars(
    out_png: str,
    title: str,
    display_shifts: List[str],
    probs_A_mle: List[float],
    probs_B_mle: List[float],
    probs_A_lap: List[float],
    probs_B_lap: List[float],
    labelA: str,
    labelB: str,
    ylabel: str,
) -> None:
    """
    1つの図に4系列:
      A-MLE, B-MLE, A-LAP, B-LAP
    """
    xs = list(range(len(display_shifts)))
    width = 0.18

    plt.figure(figsize=(13.0, 5.8))

    xsA_mle = [x - 1.5 * width for x in xs]
    xsB_mle = [x - 0.5 * width for x in xs]
    xsA_lap = [x + 0.5 * width for x in xs]
    xsB_lap = [x + 1.5 * width for x in xs]

    plt.bar(xsA_mle, probs_A_mle, width=width, label=f"{labelA} ")
    plt.bar(xsB_mle, probs_B_mle, width=width, label=f"{labelB} ")
    plt.bar(xsA_lap, probs_A_lap, width=width, label=f"{labelA} (Laplace)")
    plt.bar(xsB_lap, probs_B_lap, width=width, label=f"{labelB} (Laplace)")

    plt.xticks(xs, display_shifts)
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.xlabel("next shift")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


# -------------------------------------------------------------
# per-ward runners
# -------------------------------------------------------------
def run_one_ward_heads_nonheads(
    ward_name: str,
    past_shifts_file: str,
    group_settings_dir: str,
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
    laplace_k: float,
    laplace_support: str,
    outdir: str,
):
    """
    従来モード: Heads vs NonHeads を、それぞれ Period A vs Period B で比較
    追加: MLE-only も別PNGで出す
    追加: 横軸は観測された next のみ（A/B union）
    """
    seqs = data_loader.load_past_shifts(past_shifts_file)
    timeline = data_loader.load_staff_group_timeline(group_settings_dir)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # ---- period A/B
    hA, nA, Ny_hA, Ny_nA = count_next_for_prefix_in_period(
        segs_by_person, n, target_prefix, heads_name, a_start, a_end
    )
    hB, nB, Ny_hB, Ny_nB = count_next_for_prefix_in_period(
        segs_by_person, n, target_prefix, heads_name, b_start, b_end
    )

    # ---- display shifts (observed only)
    display_h = build_display_shifts(hA, hB)
    display_n = build_display_shifts(nA, nB)

    # ---- Laplace support set
    if laplace_support == "observed_ab":
        support_h = (set(hA.keys()) | set(hB.keys())) & VALID_SHIFTS_SET
        support_n = (set(nA.keys()) | set(nB.keys())) & VALID_SHIFTS_SET
        if not support_h:
            support_h = set(VALID_SHIFTS)
        if not support_n:
            support_n = set(VALID_SHIFTS)
    else:
        support_h = VALID_SHIFTS_SET
        support_n = VALID_SHIFTS_SET

    prefix_str = ",".join(target_prefix)
    labelA = f"A:{a_start}-{a_end}"
    labelB = f"B:{b_start}-{b_end}"

    # debug print
    print_debug(f'Ward="{ward_name}" Heads  {labelA}', target_prefix, hA, Ny_hA, laplace_k, display_h, support=support_h)
    print_debug(f'Ward="{ward_name}" Heads  {labelB}', target_prefix, hB, Ny_hB, laplace_k, display_h, support=support_h)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelA}', target_prefix, nA, Ny_nA, laplace_k, display_n, support=support_n)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelB}', target_prefix, nB, Ny_nB, laplace_k, display_n, support=support_n)

    # probs
    phA_mle = mle_probs(hA, Ny_hA, display_h)
    phB_mle = mle_probs(hB, Ny_hB, display_h)
    pnA_mle = mle_probs(nA, Ny_nA, display_n)
    pnB_mle = mle_probs(nB, Ny_nB, display_n)

    phA_lap = laplace_probs(hA, Ny_hA, laplace_k, display_h, support=support_h)
    phB_lap = laplace_probs(hB, Ny_hB, laplace_k, display_h, support=support_h)
    pnA_lap = laplace_probs(nA, Ny_nA, laplace_k, display_n, support=support_n)
    pnB_lap = laplace_probs(nB, Ny_nB, laplace_k, display_n, support=support_n)

    note_h = f"(Ny A={Ny_hA}, Ny B={Ny_hB}, k={laplace_k}, support={laplace_support}, |support|={len(support_h)})"
    note_n = f"(Ny A={Ny_nA}, Ny B={Ny_nB}, k={laplace_k}, support={laplace_support}, |support|={len(support_n)})"

    safe_ward = ward_name.replace("/", "_")
    safe_prefix = prefix_str.replace(",", "-")

    out_h_mle = os.path.join(
        outdir,
        f"compare_prefix-{safe_prefix}_heads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}_MLEonly.png",
    )
    out_h_lap = os.path.join(
        outdir,
        f"compare_prefix-{safe_prefix}_heads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}_withLaplace.png",
    )
    out_n_mle = os.path.join(
        outdir,
        f"compare_prefix-{safe_prefix}_nonheads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}_MLEonly.png",
    )
    out_n_lap = os.path.join(
        outdir,
        f"compare_prefix-{safe_prefix}_nonheads_ward-{safe_ward}_N{n}_{labelA}_vs_{labelB}_withLaplace.png",
    )

    # MLE-only（円滑化前）
    plot_compare_2bars(
        out_png=out_h_mle,
        title=f"Ward={ward_name} Heads | prefix={prefix_str} | N={n} | {note_h} | MLE-only",
        display_shifts=display_h,
        probs_A=phA_mle,
        probs_B=phB_mle,
        labelA=labelA,
        labelB=labelB,
        ylabel="P(next|prefix) (MLE)",
    )
    plot_compare_2bars(
        out_png=out_n_mle,
        title=f"Ward={ward_name} NonHeads | prefix={prefix_str} | N={n} | {note_n} | MLE-only",
        display_shifts=display_n,
        probs_A=pnA_mle,
        probs_B=pnB_mle,
        labelA=labelA,
        labelB=labelB,
        ylabel="P(next|prefix) (MLE)",
    )

    # MLE + Laplace（4本）
    plot_compare_4bars(
        out_png=out_h_lap,
        title=f"Ward={ward_name} Heads | prefix={prefix_str} | N={n} | {note_h}",
        display_shifts=display_h,
        probs_A_mle=phA_mle,
        probs_B_mle=phB_mle,
        probs_A_lap=phA_lap,
        probs_B_lap=phB_lap,
        labelA=labelA,
        labelB=labelB,
        ylabel="P(next|prefix)",
    )
    plot_compare_4bars(
        out_png=out_n_lap,
        title=f"Ward={ward_name} NonHeads | prefix={prefix_str} | N={n} | {note_n}",
        display_shifts=display_n,
        probs_A_mle=pnA_mle,
        probs_B_mle=pnB_mle,
        probs_A_lap=pnA_lap,
        probs_B_lap=pnB_lap,
        labelA=labelA,
        labelB=labelB,
        ylabel="P(next|prefix)",
    )


def run_one_ward_two_nurses(
    ward_name: str,
    past_shifts_file: str,
    group_settings_dir: str,
    n: int,
    target_prefix: Prefix,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
    laplace_k: float,
    laplace_support: str,
    nurse_a_id: int,
    nurse_b_id: int,
    outdir: str,
):
    """
    看護師比較モード:
      - Period A：NurseA vs NurseB（MLE-only と withLaplace）
      - Period B：NurseA vs NurseB（MLE-only と withLaplace）
    横軸は「各Periodごとの A/B union（観測された next のみ）」
    """
    seqs = data_loader.load_past_shifts(past_shifts_file)
    timeline = data_loader.load_staff_group_timeline(group_settings_dir)

    keyA = find_person_key_by_id(seqs, nurse_a_id)
    keyB = find_person_key_by_id(seqs, nurse_b_id)

    segsA = build_segments_for_person(seqs[keyA], keyA[1], int(keyA[0]), timeline)
    segsB = build_segments_for_person(seqs[keyB], keyB[1], int(keyB[0]), timeline)

    # ---- Period A counts
    nextA_A, NyA_A = count_next_for_prefix_for_one_person_in_period(segsA, n, target_prefix, a_start, a_end)
    nextB_A, NyB_A = count_next_for_prefix_for_one_person_in_period(segsB, n, target_prefix, a_start, a_end)

    # ---- Period B counts
    nextA_B, NyA_B = count_next_for_prefix_for_one_person_in_period(segsA, n, target_prefix, b_start, b_end)
    nextB_B, NyB_B = count_next_for_prefix_for_one_person_in_period(segsB, n, target_prefix, b_start, b_end)

    # ---- display shifts (observed only, per period)
    display_A = build_display_shifts(nextA_A, nextB_A)
    display_B = build_display_shifts(nextA_B, nextB_B)

    # ---- Laplace support (per period)
    if laplace_support == "observed_ab":
        support_A = (set(nextA_A.keys()) | set(nextB_A.keys())) & VALID_SHIFTS_SET
        support_B = (set(nextA_B.keys()) | set(nextB_B.keys())) & VALID_SHIFTS_SET
        if not support_A:
            support_A = set(VALID_SHIFTS)
        if not support_B:
            support_B = set(VALID_SHIFTS)
    else:
        support_A = VALID_SHIFTS_SET
        support_B = VALID_SHIFTS_SET

    prefix_str = ",".join(target_prefix)
    safe_ward = ward_name.replace("/", "_")
    safe_prefix = prefix_str.replace(",", "-")

    labelNA = f"A({keyA[1]})"
    labelNB = f"B({keyB[1]})"

    # debug
    print_debug(f'Ward="{ward_name}" {labelNA}  PeriodA', target_prefix, nextA_A, NyA_A, laplace_k, display_A, support=support_A)
    print_debug(f'Ward="{ward_name}" {labelNB}  PeriodA', target_prefix, nextB_A, NyB_A, laplace_k, display_A, support=support_A)
    print_debug(f'Ward="{ward_name}" {labelNA}  PeriodB', target_prefix, nextA_B, NyA_B, laplace_k, display_B, support=support_B)
    print_debug(f'Ward="{ward_name}" {labelNB}  PeriodB', target_prefix, nextB_B, NyB_B, laplace_k, display_B, support=support_B)

    # probs: Period A
    pA_A_mle = mle_probs(nextA_A, NyA_A, display_A)
    pB_A_mle = mle_probs(nextB_A, NyB_A, display_A)
    pA_A_lap = laplace_probs(nextA_A, NyA_A, laplace_k, display_A, support=support_A)
    pB_A_lap = laplace_probs(nextB_A, NyB_A, laplace_k, display_A, support=support_A)

    # probs: Period B
    pA_B_mle = mle_probs(nextA_B, NyA_B, display_B)
    pB_B_mle = mle_probs(nextB_B, NyB_B, display_B)
    pA_B_lap = laplace_probs(nextA_B, NyA_B, laplace_k, display_B, support=support_B)
    pB_B_lap = laplace_probs(nextB_B, NyB_B, laplace_k, display_B, support=support_B)

    # notes
    noteA = (
        f"(PeriodA {a_start}-{a_end}, Ny A={NyA_A}, Ny B={NyB_A}, k={laplace_k}, "
        f"support={laplace_support}, |support|={len(support_A)})"
    )
    noteB = (
        f"(PeriodB {b_start}-{b_end}, Ny A={NyA_B}, Ny B={NyB_B}, k={laplace_k}, "
        f"support={laplace_support}, |support|={len(support_B)})"
    )

    baseA = f"compare_prefix-{safe_prefix}_nurses_ward-{safe_ward}_N{n}_A{keyA[0]}_B{keyB[0]}_PeriodA"
    baseB = f"compare_prefix-{safe_prefix}_nurses_ward-{safe_ward}_N{n}_A{keyA[0]}_B{keyB[0]}_PeriodB"

    outA_mle = os.path.join(outdir, baseA + "_MLEonly.png")
    outA_lap = os.path.join(outdir, baseA + "_withLaplace.png")
    outB_mle = os.path.join(outdir, baseB + "_MLEonly.png")
    outB_lap = os.path.join(outdir, baseB + "_withLaplace.png")

    # Period A plots
    plot_compare_2bars(
        out_png=outA_mle,
        title=f"P(next|(n-1)gram) : (n-1)gram={prefix_str}",
        display_shifts=display_A,
        probs_A=pA_A_mle,
        probs_B=pB_A_mle,
        labelA=labelNA,
        labelB=labelNB,
        ylabel="P(next|(n-1)gram)",
    )
    plot_compare_4bars(
        out_png=outA_lap,
        title=f"Ward={ward_name} Nurses | (n-1)gram={prefix_str} | N={n} | {noteA}",
        display_shifts=display_A,
        probs_A_mle=pA_A_mle,
        probs_B_mle=pB_A_mle,
        probs_A_lap=pA_A_lap,
        probs_B_lap=pB_A_lap,
        labelA=labelNA,
        labelB=labelNB,
        ylabel="P(next|prefix)",
    )

    # Period B plots
    plot_compare_2bars(
        out_png=outB_mle,
        title=f"Ward={ward_name} Nurses | prefix={prefix_str} | N={n} | {noteB} | MLE-only",
        display_shifts=display_B,
        probs_A=pA_B_mle,
        probs_B=pB_B_mle,
        labelA=labelNA,
        labelB=labelNB,
        ylabel="P(next|prefix) (MLE)",
    )
    plot_compare_4bars(
        out_png=outB_lap,
        title=f"Ward={ward_name} Nurses | prefix={prefix_str} | N={n} | {noteB}",
        display_shifts=display_B,
        probs_A_mle=pA_B_mle,
        probs_B_mle=pB_B_mle,
        probs_A_lap=pA_B_lap,
        probs_B_lap=pB_B_lap,
        labelA=labelNA,
        labelB=labelNB,
        ylabel="P(next|prefix)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts の .lp か、.lpが並ぶディレクトリ")
    ap.add_argument("group_settings", help="病棟の group-settings ディレクトリ、または group-settings-root")
    ap.add_argument("--n", type=int, required=True, help="n-gram の N（>=2）")
    ap.add_argument("--prefix", required=True, help='例: "SE,SN"（N=3なら2個）')

    ap.add_argument("--a-start", type=int, required=True, help="期間A start YYYYMMDD")
    ap.add_argument("--a-end", type=int, required=True, help="期間A end YYYYMMDD")
    ap.add_argument("--b-start", type=int, required=True, help="期間B start YYYYMMDD")
    ap.add_argument("--b-end", type=int, required=True, help="期間B end YYYYMMDD")

    ap.add_argument("--laplace-k", type=float, default=1.0, help="Laplace add-k の k（add-oneなら1.0）")
    ap.add_argument(
        "--laplace-support",
        choices=["all", "observed_ab"],
        default="all",
        help='Laplace の支持集合X: "all"=10種固定, "observed_ab"=比較対象A/Bで観測されたnextのunionのみ',
    )
    ap.add_argument("--heads-name", default="Heads", help='Heads 判定に使うグループ名（default "Heads"）')
    ap.add_argument("--outdir", default="out/prefix_compare", help="出力先ディレクトリ")

    # nurse compare
    ap.add_argument("--nurse-a-id", type=int, default=None,
                    help="看護師A nurse_id（この2つが指定されると看護師比較モード）")
    ap.add_argument("--nurse-b-id", type=int, default=None,
                    help="看護師B nurse_id（この2つが指定されると看護師比較モード）")

    args = ap.parse_args()

    if args.n < 2:
        print("[ERROR] --n は 2 以上（prefixが必要）", file=sys.stderr)
        sys.exit(1)
    if args.laplace_k <= 0:
        print("[ERROR] --laplace-k は > 0 にして", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.outdir)
    target_prefix = parse_prefix(args.prefix, expected_len=args.n - 1)

    nurse_mode = (args.nurse_a_id is not None) or (args.nurse_b_id is not None)
    if nurse_mode and (args.nurse_a_id is None or args.nurse_b_id is None):
        print("[ERROR] 看護師比較モードは --nurse-a-id と --nurse-b-id を両方指定して", file=sys.stderr)
        sys.exit(1)

    # 全病棟モード
    if os.path.isdir(args.past_shifts):
        past_dir = args.past_shifts
        settings_root = args.group_settings
        if not os.path.isdir(settings_root):
            print(f"[ERROR] group_settings-root がディレクトリではない: {settings_root}", file=sys.stderr)
            sys.exit(1)

        lp_files = [f for f in sorted(os.listdir(past_dir)) if f.endswith(".lp")]
        if not lp_files:
            print(f"[ERROR] .lp が見つからない: {past_dir}", file=sys.stderr)
            sys.exit(1)

        for fname in lp_files:
            ward = os.path.splitext(fname)[0]
            past_file = os.path.join(past_dir, fname)
            ward_settings_dir = os.path.join(settings_root, ward)
            if not os.path.isdir(ward_settings_dir):
                print(f'# [SKIP] ward="{ward}" settings not found: {ward_settings_dir}')
                continue

            print("\n" + "#" * 120)
            if nurse_mode:
                print(
                    f'# RUN ward="{ward}" prefix="{args.prefix}" N={args.n}  '
                    f'PeriodA={args.a_start}-{args.a_end}  PeriodB={args.b_start}-{args.b_end}  '
                    f'nurseA={args.nurse_a_id} nurseB={args.nurse_b_id}  '
                    f'k={args.laplace_k} support={args.laplace_support}'
                )
            else:
                print(
                    f'# RUN ward="{ward}" prefix="{args.prefix}" N={args.n}  '
                    f'A={args.a_start}-{args.a_end}  B={args.b_start}-{args.b_end}  '
                    f'k={args.laplace_k} support={args.laplace_support}'
                )
            print("#" * 120)

            try:
                if nurse_mode:
                    run_one_ward_two_nurses(
                        ward_name=ward,
                        past_shifts_file=past_file,
                        group_settings_dir=ward_settings_dir,
                        n=args.n,
                        target_prefix=target_prefix,
                        a_start=args.a_start,
                        a_end=args.a_end,
                        b_start=args.b_start,
                        b_end=args.b_end,
                        laplace_k=args.laplace_k,
                        laplace_support=args.laplace_support,
                        nurse_a_id=args.nurse_a_id,
                        nurse_b_id=args.nurse_b_id,
                        outdir=args.outdir,
                    )
                else:
                    run_one_ward_heads_nonheads(
                        ward_name=ward,
                        past_shifts_file=past_file,
                        group_settings_dir=ward_settings_dir,
                        n=args.n,
                        target_prefix=target_prefix,
                        heads_name=args.heads_name,
                        a_start=args.a_start,
                        a_end=args.a_end,
                        b_start=args.b_start,
                        b_end=args.b_end,
                        laplace_k=args.laplace_k,
                        laplace_support=args.laplace_support,
                        outdir=args.outdir,
                    )
            except Exception as e:
                print(f'# [SKIP/ERROR] ward="{ward}" reason: {e}')
                continue
        return

    # 単一病棟モード
    past_file = args.past_shifts
    settings_dir = args.group_settings

    if not os.path.isfile(past_file):
        print(f"[ERROR] past_shifts がファイルじゃない: {past_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(settings_dir):
        print(f"[ERROR] group_settings は病棟のディレクトリを指定して: {settings_dir}", file=sys.stderr)
        sys.exit(1)

    ward = os.path.splitext(os.path.basename(past_file))[0]
    if nurse_mode:
        run_one_ward_two_nurses(
            ward_name=ward,
            past_shifts_file=past_file,
            group_settings_dir=settings_dir,
            n=args.n,
            target_prefix=target_prefix,
            a_start=args.a_start,
            a_end=args.a_end,
            b_start=args.b_start,
            b_end=args.b_end,
            laplace_k=args.laplace_k,
            laplace_support=args.laplace_support,
            nurse_a_id=args.nurse_a_id,
            nurse_b_id=args.nurse_b_id,
            outdir=args.outdir,
        )
    else:
        run_one_ward_heads_nonheads(
            ward_name=ward,
            past_shifts_file=past_file,
            group_settings_dir=settings_dir,
            n=args.n,
            target_prefix=target_prefix,
            heads_name=args.heads_name,
            a_start=args.a_start,
            a_end=args.a_end,
            b_start=args.b_start,
            b_end=args.b_end,
            laplace_k=args.laplace_k,
            laplace_support=args.laplace_support,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
