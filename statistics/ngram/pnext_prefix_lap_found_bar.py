#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【比較】prefix固定で、MANUAL(期間指定) と FOUND(全データ合算) の P(next|prefix) を棒グラフで比較
  - 棒グラフは「円滑化後（Laplace add-k）」だけを表示（= 2系列: MANUAL-LAP vs FOUND-LAP）

あなたの修正点:
  1) 棒グラフは円滑化後だけでOK → MLE系列を削除
  2) found-model の読み込み/集計が怪しくて Ny_F が小さい → 参照スクリプトの方式に合わせる
     ★重要: 複数 found-model を “連結して1本の時系列にする” と、同じ月の勤務表6回などで
       日付が重複して混ざり、n-gramが壊れて Ny が不自然に小さくなることがある。
     → 参照コード同様、「ファイル単位で staff の列を作り、その中で n-gram を数え、最後に合算」する。

要点:
  - past: group-settings（timeline）で所属グループ集合が変わるたびにセグメント分割（境界は跨がない）
  - past: Heads 判定は heads-name 完全一致（case-insensitive）。Unknown は NonHeads 扱い
  - found: staff_group/group で Heads/NonHeads 判定（タイムライン無し）
  - found: ext_assigned / out_assigned を読む（2引数 out_assigned は無視）
  - found: “全データ合算” は「found_path配下のlpを全部拾って、ファイルごとに数えて合算」する（連結しない）
  - Laplace support:
      --laplace-support all         : VALID_SHIFTS(10) 全部に add-k（|X|=10固定）
      --laplace-support observed_mf : MANUAL/FOUND のどちらかで観測された next の union のみに add-k（|X|可変）

使い方（単一病棟）:
  python pnext_prefix_compare_manual_vs_found_laplace_only.py \
    exp/2019-2025-data/past-shifts/GCU.lp \
    exp/2019-2025-data/group-settings/GCU/ \
    exp/found-model/GCU/2024-10/ \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20241001 --a-end 20241031 \
    --laplace-k 1.0 \
    --laplace-support observed_mf \
    --heads-name Heads \
    --outdir out/prefix_compare_m_vs_f

使い方（全病棟: past-shifts-dir × group-settings-root × found-root）:
  python pnext_prefix_compare_manual_vs_found_laplace_only.py \
    /workspace/2025/past-shifts \
    /workspace/2025/group-settings \
    /workspace/2025/found-model \
    --n 3 \
    --prefix "SE,SN" \
    --a-start 20241001 --a-end 20241031 \
    --laplace-k 1.0 \
    --laplace-support observed_mf \
    --outdir out/prefix_compare_all_m_vs_f

found_root の解釈（全病棟モード）:
  - found_root/<ward>/ が存在すればそこを使う
  - 無ければ found_root 直下を使う（共通foundを渡したい場合）
"""

import os
import sys
import re
import glob
import argparse
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
X_SIZE = 10
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Prefix = Tuple[str, ...]


# =============================================================
# misc helpers
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS_SET]


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def parse_prefix(prefix_str: str, expected_len: int) -> Prefix:
    s = prefix_str.strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f'--prefix must have length {expected_len} (got {len(parts)}): "{prefix_str}"')
    for p in parts:
        if p not in VALID_SHIFTS_SET:
            raise ValueError(f"--prefix contains invalid shift: {p}")
    return tuple(parts)


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target_group: str) -> bool:
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
    seq = normalize_seq(seq)
    if not seq:
        return []
    seq.sort(key=lambda t: t[0])

    segs: List[Segment] = []
    cur: Optional[Segment] = None

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)
        if not gs:
            gset = frozenset([UNKNOWN_GROUP])  # Unknown → NonHeads 扱い
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
# past: count next for fixed prefix in period
# =============================================================
def count_next_for_prefix_in_period_past(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter, int, int]:
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

            shifts = [s for (_d, s) in sseq]
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


# =============================================================
# found: load + group bucket + count (NO period filter)
#   ★参照スクリプト方式: ファイルごとに shifts_by_staff を作り、n-gram を数えて最後に合算
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
    hn = (heads_name or "").strip().lower()
    for g in groups:
        if (g or "").strip().lower() == hn:
            return "Heads"
        if is_head_group_found(g):
            return "Heads"
    return "NonHeads"


def _pick_lp_files_in_dir(d: str) -> List[str]:
    fs = sorted(glob.glob(os.path.join(d, "found-model*.lp")))
    if fs:
        return fs
    return sorted(glob.glob(os.path.join(d, "*.lp")))


def collect_all_found_lp_files(found_path: str) -> Tuple[str, List[str]]:
    """
    found_path 配下の lp を “全部合算” したいので全部拾う。
    ただし「直下にある found-model*.lp があればそれだけ」を優先（参照の _pick_lp_files_in_dir に合わせる）
    """
    if os.path.isfile(found_path):
        label = os.path.basename(os.path.normpath(found_path))
        return label, [found_path]

    if not os.path.isdir(found_path):
        return os.path.basename(os.path.normpath(found_path)), []

    found_path = os.path.normpath(found_path)
    label = os.path.basename(found_path)

    direct = _pick_lp_files_in_dir(found_path)
    if direct:
        return label, direct

    cand = sorted(glob.glob(os.path.join(found_path, "**", "found-model*.lp"), recursive=True))
    if not cand:
        cand = sorted(glob.glob(os.path.join(found_path, "**", "*.lp"), recursive=True))
    return label, cand


def load_found_model_one_file(path: str) -> Tuple[Dict[int, List[str]], Dict[int, Set[str]]]:
    """
    参照スクリプトの load_found_model と同等:
      - shifts_by_staff: {staff_id: [shift, shift, ...]}  ※dayでソートして shift列にする
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
        shifts_by_staff[sid] = [s for (_d, s) in pairs_sorted]

    return shifts_by_staff, groups_by_staff


def count_next_for_prefix_found_all_files(
    found_files: List[str],
    n: int,
    target_prefix: Prefix,
    heads_name: str,
) -> Tuple[Counter, Counter, int, int]:
    """
    found_files（複数）を “合算” して target_prefix の next を数える。
    ★ただしファイル間で staff列を連結しない（参照スクリプト方式）。
    """
    heads_next = Counter()
    non_next = Counter()
    Ny_h = 0
    Ny_n = 0

    pref_len = n - 1

    for fp in found_files:
        shifts_by_staff, groups_by_staff = load_found_model_one_file(fp)

        for sid, shifts in shifts_by_staff.items():
            if len(shifts) < n:
                continue

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)

            for i in range(len(shifts) - n + 1):
                pfx = tuple(shifts[i : i + pref_len])
                if pfx != target_prefix:
                    continue
                nxt = shifts[i + pref_len]
                if bucket == "Heads":
                    heads_next[nxt] += 1
                    Ny_h += 1
                else:
                    non_next[nxt] += 1
                    Ny_n += 1

    return heads_next, non_next, Ny_h, Ny_n


# =============================================================
# Laplace probabilities (only)
# =============================================================
def laplace_probs(next_counter: Counter, Ny: int, k: float, support: Optional[Set[str]] = None) -> List[float]:
    """
    Laplace(add-k) 確率ベクトル（支持集合 support 対応）:
      support=None -> VALID_SHIFTS 全部に add-k（|X|=10）
      support=set  -> support のみに add-k（分母 Ny + k*|support|）
    返すベクトルは常に VALID_SHIFTS の順で10本。
    """
    if support is None:
        support = VALID_SHIFTS_SET
    if not support:
        support = VALID_SHIFTS_SET

    denom = Ny + k * len(support)
    if denom <= 0:
        # Ny=0 でも k>0 なら denom>0 のはずだが、念のため
        return [1.0 / X_SIZE] * X_SIZE

    out: List[float] = []
    for x in VALID_SHIFTS:
        c = next_counter.get(x, 0)
        if x in support:
            out.append((c + k) / denom)
        else:
            out.append(c / denom)
    return out


def print_debug(tag: str, prefix: Prefix, next_counter: Counter, Ny: int, k: float, support: Set[str]) -> None:
    print("")
    print("=" * 110)
    print(tag)
    print("=" * 110)
    print(f"prefix={','.join(prefix)}  Ny=c(prefix)={Ny}   |X|={len(support)}  laplace_k={k}")
    print("laplace_support_set:", ",".join(sorted(support)))
    top = next_counter.most_common(10)
    if top:
        print("top next counts:", ", ".join([f"{x}:{c}" for x, c in top]))
    else:
        print("top next counts: (none)")

    plap = laplace_probs(next_counter, Ny, k, support=support)
    print(f"{'next':>4}  {'count':>6}  {'P_LAP':>10}")
    for x, l in zip(VALID_SHIFTS, plap):
        print(f"{x:>4}  {next_counter.get(x,0):>6}  {l:>10.6f}")


# =============================================================
# plotting (Laplace only: 2 series)
# =============================================================
def plot_compare_2bars_laplace(
    out_png: str,
    title: str,
    probs_M_lap: List[float],
    probs_F_lap: List[float],
    labelM: str,
    labelF: str,
) -> None:
    xs = list(range(len(VALID_SHIFTS)))
    width = 0.35

    plt.figure(figsize=(13.0, 5.8))
    xsM = [x - 0.5 * width for x in xs]
    xsF = [x + 0.5 * width for x in xs]

    plt.bar(xsM, probs_M_lap, width=width, label=f"{labelM} (Laplace)")
    plt.bar(xsF, probs_F_lap, width=width, label=f"{labelF} (Laplace)")

    plt.xticks(xs, VALID_SHIFTS)
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(next|prefix) [Laplace]")
    plt.xlabel("next shift (display=10)")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"# wrote: {out_png}")


# =============================================================
# main worker
# =============================================================
def run_one_ward(
    ward_name: str,
    past_shifts_file: str,
    group_settings_dir: str,
    found_path_for_ward: str,
    n: int,
    target_prefix: Prefix,
    heads_name: str,
    a_start: int,
    a_end: int,
    laplace_k: float,
    laplace_support: str,  # all / observed_mf
    outdir: str,
) -> None:
    # ---- load past + timeline
    seqs = data_loader.load_past_shifts(past_shifts_file)
    timeline = data_loader.load_staff_group_timeline(group_settings_dir)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # ---- MANUAL (period specified)
    hM, nM, Ny_hM, Ny_nM = count_next_for_prefix_in_period_past(
        segs_by_person, n, target_prefix, heads_name, a_start, a_end
    )

    # ---- FOUND (all data under found_path_for_ward)
    found_label, found_files = collect_all_found_lp_files(found_path_for_ward)
    if not found_files:
        print(f'# [WARN] ward="{ward_name}" no found lp files under: {found_path_for_ward}')
        hF = Counter()
        nF = Counter()
        Ny_hF = 0
        Ny_nF = 0
    else:
        # ★参照方式: ファイルごとに数えて合算
        hF, nF, Ny_hF, Ny_nF = count_next_for_prefix_found_all_files(
            found_files, n, target_prefix, heads_name
        )

    # ---- Laplace support set
    if laplace_support == "observed_mf":
        support_h = set(hM.keys()) | set(hF.keys())
        support_n = set(nM.keys()) | set(nF.keys())
        if not support_h:
            support_h = set(VALID_SHIFTS_SET)
        if not support_n:
            support_n = set(VALID_SHIFTS_SET)
    else:
        support_h = set(VALID_SHIFTS_SET)
        support_n = set(VALID_SHIFTS_SET)

    prefix_str = ",".join(target_prefix)
    labelM = f"MANUAL:{a_start}-{a_end}"
    labelF = f"FOUND(all):{found_label}"

    # debug print
    print_debug(f'Ward="{ward_name}" Heads  {labelM}', target_prefix, hM, Ny_hM, laplace_k, support=support_h)
    print_debug(f'Ward="{ward_name}" Heads  {labelF}', target_prefix, hF, Ny_hF, laplace_k, support=support_h)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelM}', target_prefix, nM, Ny_nM, laplace_k, support=support_n)
    print_debug(f'Ward="{ward_name}" NonHeads  {labelF}', target_prefix, nF, Ny_nF, laplace_k, support=support_n)

    # probs (Laplace only)
    phM_lap = laplace_probs(hM, Ny_hM, laplace_k, support=support_h)
    phF_lap = laplace_probs(hF, Ny_hF, laplace_k, support=support_h)
    pnM_lap = laplace_probs(nM, Ny_nM, laplace_k, support=support_n)
    pnF_lap = laplace_probs(nF, Ny_nF, laplace_k, support=support_n)

    note_h = f"(M={Ny_hM}, A={Ny_hF})"
    note_n = f"(M={Ny_nM}, A={Ny_nF})"

    safe_ward = ward_name.replace("/", "_")
    safe_prefix = prefix_str.replace(",", "-")

    out_h = os.path.join(
        outdir,
        f"compare_manual_vs_found_LAPLACEONLY_prefix-{safe_prefix}_heads_ward-{safe_ward}_N{n}_{labelM}_vs_{labelF}.png",
    )
    out_n = os.path.join(
        outdir,
        f"compare_manual_vs_found_LAPLACEONLY_prefix-{safe_prefix}_nonheads_ward-{safe_ward}_N{n}_{labelM}_vs_{labelF}.png",
    )

    plot_compare_2bars_laplace(
        out_png=out_h,
        title=f"Ward={ward_name} Heads | prefix={prefix_str} | N={n} | {note_h}",
        probs_M_lap=phM_lap,
        probs_F_lap=phF_lap,
        labelM=labelM,
        labelF=labelF,
    )
    plot_compare_2bars_laplace(
        out_png=out_n,
        title=f"Ward={ward_name} NonHeads | prefix={prefix_str} | N={n} | {note_n}",
        probs_M_lap=pnM_lap,
        probs_F_lap=pnF_lap,
        labelM=labelM,
        labelF=labelF,
    )


# =============================================================
# entry
# =============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts の .lp か、.lpが並ぶディレクトリ")
    ap.add_argument("group_settings", help="病棟の group-settings ディレクトリ、または group-settings-root")
    ap.add_argument("found_path", help="found-model の .lp / ディレクトリ / found-root（全病棟ならroot推奨）")

    ap.add_argument("--n", type=int, required=True, help="n-gram の N（>=2）")
    ap.add_argument("--prefix", required=True, help='例: "SE,SN"（N=3なら2個）')

    ap.add_argument("--a-start", type=int, required=True, help="MANUAL 期間 start YYYYMMDD")
    ap.add_argument("--a-end", type=int, required=True, help="MANUAL 期間 end YYYYMMDD")

    ap.add_argument("--laplace-k", type=float, default=1.0, help="Laplace add-k の k（add-oneなら1.0）")
    ap.add_argument(
        "--laplace-support",
        choices=["all", "observed_mf"],
        default="all",
        help='Laplace の支持集合X: "all"=10種固定, "observed_mf"=MANUAL/FOUNDで観測されたnextのunion',
    )
    ap.add_argument("--heads-name", default="Heads", help='Heads 判定に使うグループ名（default "Heads"）')
    ap.add_argument("--outdir", default="out/prefix_compare_manual_vs_found_laplace_only", help="出力先ディレクトリ")
    args = ap.parse_args()

    if args.n < 2:
        print("[ERROR] --n は 2 以上（prefixが必要）", file=sys.stderr)
        sys.exit(1)
    if args.laplace_k <= 0:
        print("[ERROR] --laplace-k は > 0 にして", file=sys.stderr)
        sys.exit(1)
    if args.a_start > args.a_end:
        print("[ERROR] --a-start <= --a-end にして", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.outdir)
    target_prefix = parse_prefix(args.prefix, expected_len=args.n - 1)

    # 全病棟モード
    if os.path.isdir(args.past_shifts):
        past_dir = args.past_shifts
        settings_root = args.group_settings
        found_root = args.found_path

        if not os.path.isdir(settings_root):
            print(f"[ERROR] group_settings-root がディレクトリではない: {settings_root}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(found_root) and not os.path.isfile(found_root):
            print(f"[ERROR] found_path が存在しない: {found_root}", file=sys.stderr)
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

            # found: found_root/<ward>/ があればそれ。無ければ found_root をそのまま使う
            ward_found_path = os.path.join(found_root, ward) if os.path.isdir(found_root) else found_root
            if os.path.isdir(found_root) and not os.path.exists(ward_found_path):
                ward_found_path = found_root

            print("\n" + "#" * 120)
            print(
                f'# RUN ward="{ward}" prefix="{args.prefix}" N={args.n}  '
                f'MANUAL={args.a_start}-{args.a_end}  '
                f'FOUND(all)={ward_found_path}  '
                f'k={args.laplace_k} support={args.laplace_support}'
            )
            print("#" * 120)

            run_one_ward(
                ward_name=ward,
                past_shifts_file=past_file,
                group_settings_dir=ward_settings_dir,
                found_path_for_ward=ward_found_path,
                n=args.n,
                target_prefix=target_prefix,
                heads_name=args.heads_name,
                a_start=args.a_start,
                a_end=args.a_end,
                laplace_k=args.laplace_k,
                laplace_support=args.laplace_support,
                outdir=args.outdir,
            )
        return

    # 単一病棟モード
    past_file = args.past_shifts
    settings_dir = args.group_settings
    found_path = args.found_path

    if not os.path.isfile(past_file):
        print(f"[ERROR] past_shifts がファイルじゃない: {past_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(settings_dir):
        print(f"[ERROR] group_settings は病棟のディレクトリを指定して: {settings_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(found_path):
        print(f"[ERROR] found_path が存在しない: {found_path}", file=sys.stderr)
        sys.exit(1)

    ward = os.path.splitext(os.path.basename(past_file))[0]
    run_one_ward(
        ward_name=ward,
        past_shifts_file=past_file,
        group_settings_dir=settings_dir,
        found_path_for_ward=found_path,
        n=args.n,
        target_prefix=target_prefix,
        heads_name=args.heads_name,
        a_start=args.a_start,
        a_end=args.a_end,
        laplace_k=args.laplace_k,
        laplace_support=args.laplace_support,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
