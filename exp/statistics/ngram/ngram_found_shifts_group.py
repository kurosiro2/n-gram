#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【統合版】found-model(lp) を読み、n-gram を「Head/Other（+All）」で集計して表示する。

✅ 対応モード
  1) ファイルモード:
       python ngram_found_shifts_group.py <found-model.lp> <N> [csv_path]

  2) ディレクトリモード:
       python ngram_found_shifts_group.py <found-model-dir/> <N> [csv_path]

  - found-model-dir は直下の found-model*.lp を列挙（無ければ *.lp）
  - ディレクトリモードでは
      (a) 各found-modelごとの結果
      (b) 全found-model合算（TOTAL）
    を同じ出力形式で表示する

出力:
  - N=1: 1-gram（勤務記号の割合）
  - N=2: 2-gram の freq_share と P(next|prefix)
  - N>=3: N-gram の freq_share と P(next|prefix)
  - CSV（任意）: model, group, N, gram, count, freq_share, cond_prob

グループ:
  - Head / Other の2値（+ All）

注意:
  - found-model 内の述語（想定）
      ext_assigned(staff_id, day, "SHIFT").
      staff_group("GROUPNAME", staff_id).
"""

import sys
import os
import csv
import glob
import re
from collections import defaultdict, Counter

# -------------------------------------------------------------
# ★ 任意：勤務シフト（8種）+ 休暇（2種）のみカウント
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH"
}

# -------------------------------------------------------------
# 正規表現（既存の found-model 用スクリプト互換）
# -------------------------------------------------------------
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')


# =============================================================
# helpers
# =============================================================
def _stable_most_common(counter: Counter):
    """頻度降順・語順安定の並べ替え"""
    return sorted(counter.items(), key=lambda kv: (-kv[1], tuple(kv[0])))


def group_sort_key(g):
    if g == "All":
        return (0, "")
    if g == "Head":
        return (1, "")
    if g == "Other":
        return (2, "")
    return (9, g.lower())


def merge_group_counters(dst, src):
    """defaultdict(Counter) 同士を加算マージ"""
    for g, c in src.items():
        dst[g].update(c)


def list_model_files(dir_path: str):
    """
    dir直下の found-model*.lp を優先。
    無ければ *.lp を読む。
    """
    cand = sorted(glob.glob(os.path.join(dir_path, "found-model*.lp")))
    if cand:
        return cand
    return sorted(glob.glob(os.path.join(dir_path, "*.lp")))


# =============================================================
# Head / Other 判定（暫定）
# =============================================================
def is_head_groupname(g: str) -> bool:
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


def bucket_group(groupnames):
    """
    groupnames: iterable[str]
    return: "Head" or "Other"
    方針: head っぽいものが1つでもあれば Head、そうでなければ Other
    ※ groupnames が空でも Other（= head 以外）
    """
    for g in groupnames:
        if is_head_groupname(g):
            return "Head"
    return "Other"


# =============================================================
# found-model 読み込み
# =============================================================
def load_found_model(path):
    """
    found-model.lp を読んで
      - seqs_by_staff: {staff_id: [(day:int, shift:str), ...]}
      - groups_by_staff: {staff_id: set(groupname)}
    を返す
    """
    seqs_by_staff = defaultdict(list)
    groups_by_staff = defaultdict(set)

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
                seqs_by_staff[sid].append((day, sh))
                continue

            m = PAT_GROUP.match(line)
            if m:
                gname = m.group(1)
                sid = int(m.group(2))
                groups_by_staff[sid].add(gname)
                continue

    return seqs_by_staff, groups_by_staff


# =============================================================
# n-gram 集計（found-model版）
# =============================================================
def ngram_counts_by_group(seqs_by_staff, groups_by_staff, n):
    """
    n-gram 出現回数を Head/Other(+All) でカウントする。

    仕様:
      - staff ごとに day 昇順に並べ、連続列から n-gram を切る
      - VALID_SHIFTS 以外を含む n-gram は無視
      - グループは bucket_group(staff_group) で Head/Other の2値
      - すべて "All" にも加算
    """
    counters = defaultdict(Counter)
    if n <= 0:
        return counters

    for sid, seq in seqs_by_staff.items():
        if len(seq) < n:
            continue

        seq_sorted = sorted(seq, key=lambda t: t[0])
        shifts = [sh for _, sh in seq_sorted]

        g_bucket = bucket_group(groups_by_staff.get(sid, set()))
        gset = {g_bucket, "All"}

        for i in range(len(shifts) - n + 1):
            gram = tuple(shifts[i:i + n])
            if any(s not in VALID_SHIFTS for s in gram):
                continue
            for g in gset:
                counters[g][gram] += 1

    return counters


# =============================================================
# 出力（あなたの形式に合わせる）
# =============================================================
def print_unigram_share(model: str, group: str, uni_counter: Counter, csv_rows=None):
    total = sum(uni_counter.values())
    print(f'\n----- Model="{model}" | Group="{group}" | 1-gram（勤務記号の割合） -----')
    if total == 0:
        print("  (no data)")
        return

    for (gram, c) in _stable_most_common(uni_counter):
        s = gram[0]
        share = c / total
        print(f" {c:6d}  {s:<3}   {share*100:6.2f}%")
        if csv_rows is not None:
            csv_rows.append({
                "model": model,
                "group": group,
                "N": 1,
                "gram": s,
                "count": c,
                "freq_share": share,
                "cond_prob": "",
            })


def print_bigram_score(model: str, group: str, uni_counter: Counter, bi_counter: Counter, csv_rows=None):
    total_bi = sum(bi_counter.values())
    print(f'\n----- Model="{model}" | Group="{group}" | 2-gram（freq_share と P(next|prefix)） -----')
    print("  freq   prefix -> next         freq_share    P(next|prefix)")
    if total_bi == 0:
        print("  (no data)")
        return

    rows = []
    for gram, c in bi_counter.items():
        prefix = (gram[0],)
        base_prefix = uni_counter.get(prefix, 0)
        if base_prefix <= 0:
            continue
        freq_share = c / total_bi
        cond_prob = c / base_prefix
        rows.append((freq_share, c, gram, cond_prob))

    rows.sort(key=lambda x: (-x[0], -x[1], tuple(x[2])))

    for (freq_share, c, gram, cond_prob) in rows:
        arrow = f"{gram[0]}->{gram[1]}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")
        if csv_rows is not None:
            csv_rows.append({
                "model": model,
                "group": group,
                "N": 2,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def print_ngramN_score(model: str, group: str, n_counter: Counter, nm1_counter: Counter, N: int, csv_rows=None):
    total_N = sum(n_counter.values())
    print(f'\n----- Model="{model}" | Group="{group}" | {N}-gram（freq_share と P(next|prefix)） -----')
    print("  freq   prefix -> next         freq_share    P(next|prefix)")
    if total_N == 0:
        print("  (no data)")
        return

    rows = []
    for gram, c in n_counter.items():
        prefix = gram[:-1]
        base_prefix = nm1_counter.get(prefix, 0)
        if base_prefix <= 0:
            continue
        freq_share = c / total_N
        cond_prob = c / base_prefix
        rows.append((freq_share, c, gram, cond_prob))

    rows.sort(key=lambda x: (-x[0], -x[1], tuple(x[2])))

    for (freq_share, c, gram, cond_prob) in rows:
        prefix_str = "-".join(gram[:-1])
        nxt = gram[-1]
        arrow = f"{prefix_str}->{nxt}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")
        if csv_rows is not None:
            csv_rows.append({
                "model": model,
                "group": group,
                "N": N,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def print_model_block(model_label: str, N_eff: int,
                      counters_1, counters_2, counters_N, counters_Nm1,
                      csv_rows=None):
    """
    model_label: 表示名（例: found-model1.lp, TOTAL）
    counters_*: group->Counter
    """
    for g in sorted(counters_N.keys(), key=group_sort_key):
        if g not in ("All", "Head", "Other"):
            continue

        if N_eff == 1:
            print_unigram_share(model_label, g, counters_N.get(g, Counter()), csv_rows=csv_rows)
        elif N_eff == 2:
            uni = counters_1.get(g, Counter())
            bi = counters_2.get(g, Counter())
            print_bigram_score(model_label, g, uni, bi, csv_rows=csv_rows)
        else:
            n_counter = counters_N.get(g, Counter())
            nm1_counter = counters_Nm1.get(g, Counter())
            print_ngramN_score(model_label, g, n_counter, nm1_counter, N_eff, csv_rows=csv_rows)


# =============================================================
# main
# =============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python ngram_found_shifts_group.py <found-model.lp or dir> <N> [csv_path(optional)]")
        sys.exit(1)

    target = sys.argv[1]
    N = int(sys.argv[2])
    csv_path = sys.argv[3] if len(sys.argv) >= 4 else None
    csv_rows = [] if csv_path else None

    N_eff = max(1, N)

    # =========================================================
    # ファイルモード
    # =========================================================
    if os.path.isfile(target):
        seqs_by_staff, groups_by_staff = load_found_model(target)

        counters_1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 1)
        counters_2 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 2)
        counters_N = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff)
        counters_Nm1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff - 1) if N_eff >= 2 else defaultdict(Counter)

        # グループキーを Head/Other/All に揃える（空でも表示できるように）
        for g in ("All", "Head", "Other"):
            counters_1.setdefault(g, Counter())
            counters_2.setdefault(g, Counter())
            counters_N.setdefault(g, Counter())
            if N_eff >= 2:
                counters_Nm1.setdefault(g, Counter())

        print(f'# [File mode] {target}')
        print_model_block(os.path.basename(target), N_eff, counters_1, counters_2, counters_N, counters_Nm1, csv_rows=csv_rows)

        if csv_path:
            fieldnames = ["model", "group", "N", "gram", "count", "freq_share", "cond_prob"]
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"# CSV written to: {csv_path}")

        return

    # =========================================================
    # ディレクトリモード
    # =========================================================
    if not os.path.isdir(target):
        print(f"[ERROR] ファイルでもディレクトリでもありません: {target}", file=sys.stderr)
        sys.exit(1)

    model_files = list_model_files(target)
    if not model_files:
        print(f"[ERROR] ディレクトリ直下に .lp が見つかりません: {target}", file=sys.stderr)
        sys.exit(1)

    print(f'# [Directory mode] {target}')
    print("# Models:")
    for p in model_files:
        print(f"#   - {os.path.basename(p)}")

    # 合計（TOTAL）用
    counters_1_all = defaultdict(Counter)
    counters_2_all = defaultdict(Counter)
    counters_N_all = defaultdict(Counter)
    counters_Nm1_all = defaultdict(Counter) if N_eff >= 2 else defaultdict(Counter)

    # まず各モデルを出す（あなたの形式）
    for p in model_files:
        model_label = os.path.basename(p)

        seqs_by_staff, groups_by_staff = load_found_model(p)
        c1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 1)
        c2 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 2)
        cN = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff)
        cNm1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff - 1) if N_eff >= 2 else defaultdict(Counter)

        # グループキーを固定
        for g in ("All", "Head", "Other"):
            c1.setdefault(g, Counter())
            c2.setdefault(g, Counter())
            cN.setdefault(g, Counter())
            if N_eff >= 2:
                cNm1.setdefault(g, Counter())

        print_model_block(model_label, N_eff, c1, c2, cN, cNm1, csv_rows=csv_rows)

        # TOTAL に加算
        merge_group_counters(counters_1_all, c1)
        merge_group_counters(counters_2_all, c2)
        merge_group_counters(counters_N_all, cN)
        if N_eff >= 2:
            merge_group_counters(counters_Nm1_all, cNm1)

    # 最後に合算結果
    print("\n# =====================")
    print("# --- TOTAL (sum of all found-models) ---")
    print("# =====================")

    for g in ("All", "Head", "Other"):
        counters_1_all.setdefault(g, Counter())
        counters_2_all.setdefault(g, Counter())
        counters_N_all.setdefault(g, Counter())
        if N_eff >= 2:
            counters_Nm1_all.setdefault(g, Counter())

    print_model_block("TOTAL", N_eff,
                      counters_1_all, counters_2_all, counters_N_all, counters_Nm1_all,
                      csv_rows=csv_rows)

    if csv_path:
        fieldnames = ["model", "group", "N", "gram", "count", "freq_share", "cond_prob"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"# CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
