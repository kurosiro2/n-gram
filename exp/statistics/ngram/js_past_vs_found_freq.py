#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
js_past_vs_found.py
過去勤務（手動） vs 自動生成（found）を、Head/Other（+All）別に n-gram 分布の JS距離で比較する。

✅ 入力（柔軟対応）
  python js_past_vs_found.py <past_path> <settings_path> <found_dir> [options]

- past_path:
    A) ディレクトリ（直下に *.lp が並ぶ：ファイル名=病棟名）
    B) 単体ファイル（例: .../GCU.lp）

- settings_path:
    A) root（<root>/<ward>/...）
    B) 単体 ward ディレクトリ（例: .../group-settings/GCU/）

- found_dir:
    A) <found_dir>/<ward>/found-model*.lp
    B) <found_dir> 直下に found-model*.lp（単一ward想定）→ --ward 推奨

出力:
  - CSV（任意）: ward, group, n, jsdist, past_total, found_total
  - heatmap: wards × n (1..Nmax) を group別にPNG（All/Head/Other）

仕様:
  - JS distance = sqrt(JSD), ln（自然対数）
  - smoothing: add-alpha（デフォルト alpha=1e-3）
  - past側: グループ集合変化でセグメント分割（境界は跨がない）
  - found側: staff_group で Head/Other 判定（タイムライン無し）
  - past側の Head 判定: group名が --heads-name と「完全一致（case-insensitive）」なら Head
  - found側の Head 判定: groupname に "head" / "師長" / "主任" を含めば Head
  - Unknown は Other 扱い
"""

import os
import sys
import math
import argparse
import glob
import csv
from collections import Counter, defaultdict
import re
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

PersonKey = Tuple[int, str]  # (nurse_id, name)
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]


# =============================================================
# small helpers
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def within_range(d: int, start: Optional[int], end: Optional[int]) -> bool:
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def collect_lp_files_dir(dir_path: str) -> List[str]:
    """dir直下の *.lp を列挙（安定ソート）"""
    out = []
    for fn in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p) and fn.endswith(".lp"):
            out.append(p)
    return out


def ward_name_from_lp(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def is_same_ward_dir(settings_path: str, ward: str) -> bool:
    """settings_path が .../<ward>/ を指しているか（末尾だけで判定）"""
    base = os.path.basename(os.path.normpath(settings_path))
    return base == ward


# =============================================================
# Head / Other 判定
# =============================================================
def is_head_groupname_found(g: str) -> bool:
    """found側の groupname から Head っぽさを判定（緩め）"""
    if not g:
        return False
    gl = g.lower()
    if "head" in gl:
        return True
    if "師長" in g:
        return True
    if "主任" in g:
        return True
    return False


def bucket_from_groupnames(groupnames: Set[str]) -> str:
    """headを含むなら Head、それ以外は Other（空/UnknownもOther）"""
    for g in groupnames:
        if is_head_groupname_found(g):
            return "Head"
    return "Other"


def group_set_contains_exact_ci(group_set: FrozenSet[str], target_group: str) -> bool:
    """past側: heads_name と完全一致 (case-insensitive) なら head 扱い"""
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


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
) -> List[Segment]:
    """
    所属グループ集合が変わるたびにセグメント分割。
    Unknown は UNKNOWN_GROUP として保持（→ Other 扱い）
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


def count_ngrams_past_heads_other(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: Optional[int],
    date_end: Optional[int],
) -> Dict[str, Counter]:
    """
    past側: セグメント境界を跨がずに n-gram を数える。
    出力: {"All":Counter, "Head":Counter, "Other":Counter}
    """
    out = {"All": Counter(), "Head": Counter(), "Other": Counter()}

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains_exact_ci(seg.groups, heads_name)

            sseq = [(d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)]
            if len(sseq) < n:
                continue

            shifts = [s for _, s in sseq]
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i:i + n])
                if any(x not in VALID_SHIFTS for x in gram):
                    continue
                out["All"][gram] += 1
                if is_heads:
                    out["Head"][gram] += 1
                else:
                    out["Other"][gram] += 1

    return out


# =============================================================
# found: load + ngram counts (no timeline)
# =============================================================
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')


def load_found_model(path: str) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]:
    """
    found-model.lp を読んで:
      - seqs_by_staff: {staff_id: [(day, shift), ...]}
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
                if sh in VALID_SHIFTS:
                    seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

    return seqs_by_staff, groups_by_staff


def count_ngrams_found_heads_other(
    found_files: List[str],
    n: int,
) -> Dict[str, Counter]:
    """
    found側: found-model*.lp を全部合算して、Head/Other(+All) の n-gram を数える
    Unknown は Other 扱い（= staff_group 空なら Other）
    """
    out = {"All": Counter(), "Head": Counter(), "Other": Counter()}

    for fp in found_files:
        seqs_by_staff, groups_by_staff = load_found_model(fp)

        for sid, seq in seqs_by_staff.items():
            if len(seq) < n:
                continue
            seq_sorted = sorted(seq, key=lambda t: t[0])
            shifts = [s for _, s in seq_sorted]

            bucket = bucket_from_groupnames(groups_by_staff.get(sid, set()))  # empty -> Other
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i:i + n])
                if any(x not in VALID_SHIFTS for x in gram):
                    continue
                out["All"][gram] += 1
                out[bucket][gram] += 1

    return out


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
# plotting
# =============================================================
def plot_heatmap_wards_x_n(
    out_png: str,
    title: str,
    ward_labels: List[str],
    n_labels: List[str],
    mat: List[List[float]],
    vmin: float,
    vmax: float,
) -> None:
    R = len(ward_labels)
    C = len(n_labels)

    fig_w = max(9.0, C * 2.2)
    fig_h = max(7.0, R * 0.55)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im)

    plt.xticks(list(range(C)), n_labels, rotation=45, ha="right")
    plt.yticks(list(range(R)), ward_labels)

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
# found dir mapping
# =============================================================
def find_found_files_for_ward(found_dir: str, ward: str) -> List[str]:
    """
    found_dir から ward に対応する found-model*.lp を探す。

    1) found_dir/ward/ があれば、その直下の found-model*.lp（なければ *.lp）
    2) found_dir 直下に found-model*.lp があれば、それ（単一ward用）
    """
    ward_sub = os.path.join(found_dir, ward)
    if os.path.isdir(ward_sub):
        fs = sorted(glob.glob(os.path.join(ward_sub, "found-model*.lp")))
        if fs:
            return fs
        return sorted(glob.glob(os.path.join(ward_sub, "*.lp")))

    # fallback: found_dir直下
    fs = sorted(glob.glob(os.path.join(found_dir, "found-model*.lp")))
    if fs:
        return fs
    return []


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_path", help="past-shifts path (directory OR single ward .lp)")
    ap.add_argument("settings_path", help="group-settings path (root OR single ward dir)")
    ap.add_argument("found_dir", help="found directory (contains <ward>/found-model*.lp OR direct found-model*.lp)")
    ap.add_argument("--ward", default=None,
                    help="単一ward構成っぽい場合に ward名を明示（past_pathが単体lpのときも推奨）")
    ap.add_argument("--date-start", type=int, default=None, help="YYYYMMDD (past側のみ) 例: 20240701")
    ap.add_argument("--date-end", type=int, default=None, help="YYYYMMDD (past側のみ) 例: 20241231")
    ap.add_argument("--heads-name", default="Heads", help="past側で Head とみなすグループ名（例: Heads）")
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--outdir", default="out/js_past_vs_found")
    ap.add_argument("--csv", default=None, help="CSV 出力パス（省略可）")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("nmin/nmax must satisfy 1 <= nmin <= nmax")

    if not os.path.exists(args.past_path):
        raise FileNotFoundError(f"past_path not found: {args.past_path}")
    if not os.path.exists(args.settings_path):
        raise FileNotFoundError(f"settings_path not found: {args.settings_path}")
    if not os.path.isdir(args.found_dir):
        raise FileNotFoundError(f"found_dir not found: {args.found_dir}")

    # --------------------------
    # past wards
    # --------------------------
    if os.path.isdir(args.past_path):
        past_files = collect_lp_files_dir(args.past_path)
    elif os.path.isfile(args.past_path) and args.past_path.endswith(".lp"):
        past_files = [args.past_path]
    else:
        raise ValueError(f"past_path must be a directory or a .lp file: {args.past_path}")

    if not past_files:
        raise FileNotFoundError(f"No *.lp found under past_path: {args.past_path}")

    # ward list
    wards: List[str] = []
    ward_to_past: Dict[str, str] = {}
    for fp in past_files:
        ward = ward_name_from_lp(fp)
        ward_to_past[ward] = fp
        wards.append(ward)
    wards.sort()

    # --------------------------
    # found mode detection
    # --------------------------
    found_dir_direct = sorted(glob.glob(os.path.join(args.found_dir, "found-model*.lp")))
    found_has_ward_subdirs = any(os.path.isdir(os.path.join(args.found_dir, w)) for w in wards)

    if found_dir_direct and not found_has_ward_subdirs:
        if args.ward:
            wards = [args.ward]
        else:
            # past が単体ならそこから ward 推測
            if len(past_files) == 1:
                wards = [ward_name_from_lp(past_files[0])]
                print(f"# [info] --ward not given; inferred ward={wards[0]} from past_path", file=sys.stderr)
            else:
                raise ValueError(
                    "found_dir直下に found-model*.lp がある単一ward構成っぽいです。"
                    " 複数ward処理はできないので --ward <病棟名> を指定してください。"
                )


    # past_path が単体 lp の場合も、settings_path も単体 ward dir の可能性が高いので ward は明示推奨
    if len(past_files) == 1 and not args.ward:
        # ただし wards は past名から取れるので、ここでは警告だけ（動作はさせる）
        print(f"# [warn] past_path is single file. Consider adding --ward {wards[0]} for found/settings alignment.", file=sys.stderr)

    ns = list(range(args.nmin, args.nmax + 1))
    n_labels = [f"{n}-gram" for n in ns]

    csv_rows: List[dict] = []

    groups = ["All", "Head", "Other"]
    mats: Dict[str, List[List[float]]] = {g: [] for g in groups}
    ward_labels_out: List[str] = []

    # global color scale (ln): vmax = sqrt(ln2)
    vmin = 0.0
    vmax = math.sqrt(math.log(2.0))

    processed = 0
    skipped = 0

    for ward in wards:
        # past lp
        past_lp = ward_to_past.get(ward)
        if not past_lp or not os.path.isfile(past_lp):
            skipped += 1
            continue

        # settings dir: root/<ward> or directly ward dir
        if os.path.isdir(args.settings_path) and is_same_ward_dir(args.settings_path, ward):
            setting_dir = args.settings_path
        else:
            setting_dir = os.path.join(args.settings_path, ward)

        if not os.path.isdir(setting_dir):
            print(f"# [skip] settings not found for ward={ward}: {setting_dir}", file=sys.stderr)
            skipped += 1
            continue

        # found files
        found_files = find_found_files_for_ward(args.found_dir, ward)
        if not found_files:
            print(f"# [skip] found-model not found for ward={ward} under: {args.found_dir}", file=sys.stderr)
            skipped += 1
            continue

        # load past
        seqs = data_loader.load_past_shifts(past_lp)
        timeline = data_loader.load_staff_group_timeline(setting_dir)
        segs_by_person = prebuild_all_segments(seqs, timeline)

        row_by_group: Dict[str, List[float]] = {g: [] for g in groups}

        for n in ns:
            past_c = count_ngrams_past_heads_other(
                segs_by_person,
                n=n,
                heads_name=args.heads_name,
                date_start=args.date_start,
                date_end=args.date_end,
            )
            found_c = count_ngrams_found_heads_other(found_files, n=n)

            for g in groups:
                d = js_distance_from_counters(past_c[g], found_c[g], args.alpha)
                row_by_group[g].append(d)

                csv_rows.append({
                    "ward": ward,
                    "group": g,
                    "n": n,
                    "jsdist": d,
                    "past_total": int(sum(past_c[g].values())),
                    "found_total": int(sum(found_c[g].values())),
                })

        for g in groups:
            mats[g].append(row_by_group[g])

        ward_labels_out.append(ward)
        processed += 1

    print(f"# processed={processed}, skipped={skipped}")
    if processed == 0:
        print("# No wards processed. (ward名/ディレクトリ構造が合ってるか確認して)", file=sys.stderr)
        sys.exit(1)

    # write CSV
    if args.csv:
        fieldnames = ["ward", "group", "n", "jsdist", "past_total", "found_total"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"# wrote csv: {args.csv}")

    # plot heatmaps
    for g in groups:
        out_png = os.path.join(args.outdir, f"heatmap_wards_x_ngram_{g.lower()}_n{args.nmin}-{args.nmax}.png")
        plot_heatmap_wards_x_n(
            out_png=out_png,
            title=f"{g}: JSdist(past vs found)  n={args.nmin}..{args.nmax}  [ln]",
            ward_labels=ward_labels_out,
            n_labels=n_labels,
            mat=mats[g],
            vmin=vmin,
            vmax=vmax,
        )
        print(f"# wrote: {out_png}")


if __name__ == "__main__":
    main()
