#!/usr/bin/env python3
import sys
import os
import re
import csv
from collections import defaultdict, Counter

# -------------------------------------------------------------
# 親ディレクトリを import パスに追加して data_loader を読めるようにする
#   このファイル: exp/statistics/ngram/ngram_past_shifts_group.py
#   data_loader:   exp/statistics/data_loader.py
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, load_staff_groups, get_groups_for_date
import data_loader_all  # 全病棟モード用

# -------------------------------------------------------------
# ★ 期間指定（ここを書き換えて使う）
#   - None にするとその側の制限なし
#   - 例: 2025年だけ → DATE_START = 20250101, DATE_END = 20251231
# -------------------------------------------------------------
DATE_START = 20250101  # 例: 20250101
DATE_END   = 20251231  # 例: 20251231


# -------------------------------------------------------------
# 勤務シフト（8種）+ 休暇シフト（2種）だけを有効とする
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "E", "SE", "N", "SN",
    "WR", "PH",
}


# -------------------------------------------------------------
# ★ 日付フィルタ（シーケンス全体に適用）
# -------------------------------------------------------------
def filter_seqs_by_date(seqs_dict, start_date, end_date):
    """
    seqs_dict: {(nurse_id, name): [(date, shift), ...]}
    start_date, end_date: int (YYYYMMDD) or None

    指定された日付範囲 [start_date, end_date] に入るシフトだけを残す。
    1人のシフトが全て範囲外なら、その人自体を削除する。
    """
    if start_date is None and end_date is None:
        # 期間指定なし → そのまま返す
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


# ------------------ n-gram 集計ロジック ------------------


def ngram_counts_by_group(seqs_dict, group_timeline, n):
    """
    n-gram 出現回数をグループ別にカウント（タイムライン対応版）。

    - seqs_dict: {(nurse_id, name): [(date, shift), ...]}
      * data_loader.load_past_shifts() の戻り値を想定
    - group_timeline: { name: [(start_date, set(groups)), ...] }
      * data_loader.load_staff_group_timeline() の戻り値を想定
    - n: n-gram の n

    仕様:
      - 各 n-gram について、その n-gram の「最後の日付」を基準に
        その時点でのグループを get_groups_for_date(name, date, group_timeline)
        で引く。
      - グループが見つからなければ { "Unknown" } として扱う。
      - すべての n-gram は "All" グループにもカウントする。
      - ★ 勤務シフト8種 + 休暇シフト2種以外のシフトを含む n-gram は無視する。
    """
    from data_loader import get_groups_for_date  # 明示 import（読みやすさ用）

    group_counters = defaultdict(Counter)
    if n <= 0:
        return group_counters

    # 名前順にソート（見やすさ用）
    for (nid, name), seq in sorted(seqs_dict.items(), key=lambda kv: kv[0][1]):
        if len(seq) < n:
            continue

        # seq: [(date, shift), ...] で、load_past_shifts() 側ですでに日付ソート済みのはず
        for i in range(len(seq) - n + 1):
            window = seq[i: i + n]
            dates = [d for (d, _) in window]
            shifts = [s for (_, s) in window]

            # ★ 無効なシフト記号を含む n-gram はスキップ
            if any(s not in VALID_SHIFTS for s in shifts):
                continue

            ref_date = dates[-1]  # 「最後の日」のグループを使う
            gram = tuple(shifts)

            groups = get_groups_for_date(name, ref_date, group_timeline)
            if not groups:
                groups = {"Unknown"}

            gset = set(groups)
            gset.add("All")

            for g in gset:
                group_counters[g][gram] += 1

    return group_counters


def _stable_most_common(counter: Counter):
    """頻度降順・語順安定の並べ替え"""
    return sorted(counter.items(), key=lambda kv: (-kv[1], tuple(kv[0])))


# ---------- 出力（＋CSV 用の収集） ----------


def print_unigram_share(group: str, uni_counter: Counter, csv_rows=None):
    """n=1: 勤務記号の割合（%）を表示 & CSV 行を追加"""
    total = sum(uni_counter.values()) or 1
    print(f'\n----- Group="{group}" | 1-gram（勤務記号の割合） -----')
    for (gram,), c in _stable_most_common(uni_counter):
        pct = c / total * 100.0
        print(f"{c:>6}  {gram:<4}  {pct:6.2f}%")
        if csv_rows is not None:
            csv_rows.append({
                "group": group,
                "N": 1,
                "gram": gram,
                "count": c,
                "freq_share": c / total,
                "mc_prob": "",
                "cond_prob": "",
                "prefix_prob": "",
                "chain_prob": "",
                "score_mc": "",
                "score_chain": "",
                "score_mix": "",
                "mix_src": "",
            })


def print_bigram_score(group: str, uni_counter: Counter, bi_counter: Counter, csv_rows=None):
    """
    n=2: score = P(next|prev) × freq_share
      - P(next|prev) = c(prev,next) / c(prev)
      - freq_share   = c(prev,next) / Σ c(·,·)
    """
    total_bi = sum(bi_counter.values()) or 1
    rows = []
    for (s1, s2), c in bi_counter.items():
        base = uni_counter.get((s1,), 0)
        if base <= 0:
            continue
        prob = c / base                   # P(s2 | s1)
        freq_share = c / total_bi         # bigram の相対度数
        score = prob * freq_share
        rows.append((score, c, s1, s2, prob, freq_share))
    rows.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    print(f'\n----- Group="{group}" | 2-gram（ 条件付き確率 × freq_share ） -----')
    print("  freq   pair        P(next|prev)   freq_share      score")
    for score, c, s1, s2, prob, freq_share in rows:
        print(
            f"{c:>6}  {s1}->{s2:<4}   "
            f"{prob:12.6f}   {freq_share:11.6f}   {score:11.8f}"
        )
        if csv_rows is not None:
            gram_str = f"{s1}-{s2}"
            csv_rows.append({
                "group": group,
                "N": 2,
                "gram": gram_str,
                "count": c,
                "freq_share": freq_share,
                # bigram なので mc_prob = cond_prob とみなしておく
                "mc_prob": prob,
                "cond_prob": prob,
                "prefix_prob": "",
                "chain_prob": prob,
                "score_mc": freq_share * prob,
                "score_chain": score,
                "score_mix": score,
                "mix_src": "bigram",
            })


def print_ngramN_score(
    group: str,
    n_counter: Counter,
    nm1_counter: Counter,
    N: int,
    bi_counter: Counter,
    csv_rows=None,
):
    """
    n>=3:
      - freq_share      = c(gram) / Σ cN(·)                    （その N-gram の相対頻度）
      - P(next|prefix)  = c(gram) / cNm1(prefix)               （prefix = 長さ N-1）  cond_prob
      - P(prefix)       = P(s_{N-1} | s_{N-2})                 （2-gram の条件付き確率） prefix_prob
      - chain_prob      = P(prefix) * P(next|prefix)
      - mc_prob         = Π_i P(s_{i+1} | s_i)                 （1階マルコフ連鎖確率）
      - score_mc        = freq_share * mc_prob
      - score_chain     = freq_share * chain_prob
      - score_mix       = max(score_mc, score_chain)
    """
    if not n_counter or not nm1_counter:
        return

    total_N = sum(n_counter.values()) or 1

    # bigram 行合計（P(s_{i+1}|s_i) 計算用）
    row_totals = defaultdict(int)
    for (a, b), cnt in bi_counter.items():
        row_totals[a] += cnt

    # 2-gram から P(b|a) を作る（prefix用）
    bigram_prob = {}
    for (a, b), cnt in bi_counter.items():
        base = row_totals[a]
        if base > 0:
            bigram_prob[(a, b)] = cnt / base

    rows = []
    for gram, c in n_counter.items():
        if len(gram) < 3:
            continue

        prefix = gram[:-1]
        last = gram[-1]

        base_prefix = nm1_counter.get(prefix, 0)
        if base_prefix <= 0:
            continue

        # N-gram 相対頻度
        freq_share = c / total_N

        # P(next | prefix)
        cond_prob = c / base_prefix

        # P(prefix) は prefix の最後の 2-gram から取る（なければ 0 扱い）
        if len(prefix) >= 2:
            prev2 = (prefix[-2], prefix[-1])
            prefix_prob = bigram_prob.get(prev2, 0.0)
        else:
            prefix_prob = 0.0

        # mc_prob = Π P(s_{i+1} | s_i)
        mc_prob = 1.0
        for i in range(len(gram) - 1):
            a = gram[i]
            b = gram[i + 1]
            pair_cnt = bi_counter.get((a, b), 0)
            base_row = row_totals.get(a, 0)
            if base_row <= 0 or pair_cnt <= 0:
                mc_prob = 0.0
                break
            mc_prob *= (pair_cnt / base_row)

        chain_prob = prefix_prob * cond_prob

        score_mc = freq_share * mc_prob
        score_chain = freq_share * chain_prob
        if score_mc >= score_chain:
            score_mix = score_mc
            mix_src = "mc"
        else:
            score_mix = score_chain
            mix_src = "chain"

        rows.append(
            (
                score_mix,   # ソートキー
                c,
                gram,
                freq_share,
                mc_prob,
                cond_prob,
                prefix_prob,
                chain_prob,
                score_mc,
                score_chain,
                score_mix,
                mix_src,
            )
        )

    rows.sort(key=lambda x: (-x[0], -x[1], tuple(x[2])))

    print(f'\n----- Group="{group}" | {N}-gram（ freq_share・mc_prob・P(prefix)・各種スコア ） -----')
    print(
        "  freq   prefix -> next         "
        "freq_share    mc_prob     P(next|prefix)   P(prefix)    chain_prob   score_mc    score_chain   score_mix   mix_src"
    )
    for (
        _,
        c,
        gram,
        freq_share,
        mc_prob,
        cond_prob,
        prefix_prob,
        chain_prob,
        score_mc,
        score_chain,
        score_mix,
        mix_src,
    ) in rows:
        prefix_str = "-".join(gram[:-1])
        nxt = gram[-1]
        arrow = f"{prefix_str}->{nxt}"
        print(
            f"{c:>6}  {arrow:<20}   "
            f"{freq_share:11.6f}   {mc_prob:9.6f}   "
            f"{cond_prob:14.6f}   {prefix_prob:11.6f}   "
            f"{chain_prob:11.8f}   "
            f"{score_mc:9.6f}   {score_chain:11.8f}   {score_mix:9.6f}   {mix_src}"
        )

        if csv_rows is not None:
            gram_str = "-".join(gram)
            csv_rows.append({
                "group": group,
                "N": N,
                "gram": gram_str,
                "count": c,
                "freq_share": freq_share,
                "mc_prob": mc_prob,
                "cond_prob": cond_prob,
                "prefix_prob": prefix_prob,
                "chain_prob": chain_prob,
                "score_mc": score_mc,
                "score_chain": score_chain,
                "score_mix": score_mix,
                "mix_src": mix_src,
            })


def group_sort_key(g):
    if g == "All":
        return (0, "")
    if g == "Unknown":
        return (2, "")
    return (1, g.lower())


# ------------------ main ------------------


def main():
    if len(sys.argv) < 4:
        print("python ngram_past_shifts_group.py [shift_file_or_dir] [setting_path_or_root] [N] [csv_path(optional)]")
        print("  単一病棟モード: past-shifts.lp と setting.lp を指定")
        print("  全病棟モード  : past-shifts ディレクトリ と group-settings ルートを指定")
        print(f"  ※ 期間指定: DATE_START={DATE_START}, DATE_END={DATE_END} をスクリプト内で編集")
        sys.exit(1)

    shift_arg = sys.argv[1]
    setting_arg = sys.argv[2]
    N = int(sys.argv[3])
    csv_path = sys.argv[4] if len(sys.argv) >= 5 else None

    # CSV 用の行を貯めておく
    csv_rows = []

    # -------------------------------------------------
    # 1) 全病棟モード（第1引数がディレクトリ）
    #    → 各病棟でカウントして「グループ別に集約」してから 1 回だけ出力
    # -------------------------------------------------
    if os.path.isdir(shift_arg):
        past_shifts_dir = shift_arg
        settings_root = setting_arg

        if not os.path.isdir(settings_root):
            print(f"[ERROR] settings_root として指定したパスがディレクトリではありません: {settings_root}",
                  file=sys.stderr)
            sys.exit(1)

        # 全病棟のグループタイムライン
        all_timelines = data_loader_all.load_all_staff_group_timelines(settings_root)
        print("# Wards found in settings_root:", ", ".join(sorted(all_timelines.keys())))

        # past-shifts ディレクトリ内の .lp を全部拾う
        ward_shift_files = {}
        for fname in sorted(os.listdir(past_shifts_dir)):
            if not fname.endswith(".lp"):
                continue
            ward_name = os.path.splitext(fname)[0]
            ward_shift_files[ward_name] = os.path.join(past_shifts_dir, fname)

        if not ward_shift_files:
            print(f"[ERROR] past-shifts-dir 内に .lp ファイルが見つかりません: {past_shifts_dir}",
                  file=sys.stderr)
            sys.exit(1)

        print("# Wards found in past_shifts_dir:", ", ".join(sorted(ward_shift_files.keys())))

        N_eff = max(1, N)

        # ★ 全病棟の結果をグループごとに集約するカウンタ
        aggregated_1   = defaultdict(Counter)  # group -> Counter for 1-gram
        aggregated_2   = defaultdict(Counter)  # group -> Counter for 2-gram
        aggregated_N   = defaultdict(Counter)  # group -> Counter for N-gram
        aggregated_Nm1 = defaultdict(Counter) if N_eff >= 2 else None

        any_ward_used = False

        # 病棟ごとに処理して aggregated_* に足していく
        for ward_name, shift_file in sorted(ward_shift_files.items()):
            if ward_name not in all_timelines:
                print(f"# [WARN] ward={ward_name} に対応する group-settings が見つからないのでスキップ")
                continue
            if not os.path.isfile(shift_file):
                print(f"# [WARN] shift_file not found for ward={ward_name}: {shift_file}")
                continue

            print(f"\n========== Ward=\"{ward_name}\" ==========")

            seqs = data_loader.load_past_shifts(shift_file)

            # ★ 期間フィルタを適用
            before = len(seqs)
            seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
            after = len(seqs)
            print(f"# [Ward={ward_name}] date filter: nurses {before} -> {after}")
            if not seqs:
                print(f"# [Ward={ward_name}] no shifts in specified period; skip.")
                continue

            group_timeline = all_timelines[ward_name]

            # 1-gram, 2-gram, N-gram, (N-1)-gram を病棟ごとに計算
            counters_1_ward = ngram_counts_by_group(seqs, group_timeline, 1)
            counters_2_ward = ngram_counts_by_group(seqs, group_timeline, 2)
            counters_N_ward = ngram_counts_by_group(seqs, group_timeline, N_eff)
            counters_Nm1_ward = (
                ngram_counts_by_group(seqs, group_timeline, N_eff - 1)
                if N_eff >= 2 else {}
            )

            # ★ グループごとに aggregated_* に加算
            for g, ctr in counters_1_ward.items():
                aggregated_1[g].update(ctr)
            for g, ctr in counters_2_ward.items():
                aggregated_2[g].update(ctr)
            for g, ctr in counters_N_ward.items():
                aggregated_N[g].update(ctr)
            if N_eff >= 2:
                for g, ctr in counters_Nm1_ward.items():
                    aggregated_Nm1[g].update(ctr)

            any_ward_used = True

        if not any_ward_used:
            print("# No ward has shifts in specified period. Abort.")
            return

        print("\n========== Aggregated over ALL wards ==========")

        # ★ 集約されたカウンタに対して、単一病棟モードと同じ出力形式で表示
        for g in sorted(aggregated_N.keys(), key=group_sort_key):
            if N_eff == 1:
                print_unigram_share(g, aggregated_N[g], csv_rows=csv_rows)
            elif N_eff == 2:
                uni = aggregated_1.get(g, Counter())
                bi  = aggregated_2.get(g, Counter())
                print_bigram_score(g, uni, bi, csv_rows=csv_rows)
            else:
                n_counter   = aggregated_N.get(g, Counter())
                nm1_counter = aggregated_Nm1.get(g, Counter()) if aggregated_Nm1 is not None else Counter()
                bi          = aggregated_2.get(g, Counter())
                print_ngramN_score(g, n_counter, nm1_counter, N_eff, bi, csv_rows=csv_rows)

        # CSV 書き出し
        if csv_path:
            fieldnames = [
                "group", "N", "gram", "count",
                "freq_share", "mc_prob", "cond_prob", "prefix_prob",
                "chain_prob", "score_mc", "score_chain", "score_mix", "mix_src",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"# CSV written to: {csv_path}")

        # 全病棟モードはここで終了
        return

    # -------------------------------------------------
    # 2) 単一病棟モード（従来の挙動）
    # -------------------------------------------------
    shift_file = shift_arg
    setting_path = setting_arg

    if not os.path.isfile(shift_file):
        print(f"[ERROR] past-shifts ファイルが見つかりません: {shift_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(setting_path):
        print(f"[ERROR] setting ファイルが見つかりません: {setting_path}", file=sys.stderr)
        sys.exit(1)

    # 1) データ読み込み
    seqs = data_loader.load_past_shifts(shift_file)

    # ★ 期間フィルタを適用
    before = len(seqs)
    seqs = filter_seqs_by_date(seqs, DATE_START, DATE_END)
    after = len(seqs)
    print(f"# [Single ward] date filter: nurses {before} -> {after}")
    if not seqs:
        print("# No shifts in specified period. Abort.")
        return

    # タイムライン（名前ごとに date->groups の変化）
    group_timeline = data_loader.load_staff_group_timeline(setting_path)

    # ついでに union 版（デバッグ用, name -> set(groups)）
    staff_to_groups_by_name = data_loader.load_staff_groups(setting_path)

    # 2) n-gram カウンタ作成（タイムラインを使う）
    N_eff = max(1, N)

    # 1-gram, 2-gram は常に作る（N>=3 の mc_prob にも使う）
    counters_1 = ngram_counts_by_group(seqs, group_timeline, 1)
    counters_2 = ngram_counts_by_group(seqs, group_timeline, 2)

    counters_N = ngram_counts_by_group(seqs, group_timeline, N_eff)
    counters_Nm1 = ngram_counts_by_group(seqs, group_timeline, N_eff - 1) if N_eff >= 2 else {}

    # 3) 出力
    for g in sorted(counters_N.keys(), key=group_sort_key):
        if N_eff == 1:
            print_unigram_share(g, counters_N[g], csv_rows=csv_rows)
        elif N_eff == 2:
            uni = counters_1.get(g, Counter())
            bi  = counters_2.get(g, Counter())
            print_bigram_score(g, uni, bi, csv_rows=csv_rows)
        else:
            n_counter   = counters_N.get(g, Counter())
            nm1_counter = counters_Nm1.get(g, Counter())
            bi          = counters_2.get(g, Counter())
            print_ngramN_score(g, n_counter, nm1_counter, N_eff, bi, csv_rows=csv_rows)

    # CSV 書き出し
    if csv_path:
        fieldnames = [
            "group", "N", "gram", "count",
            "freq_share", "mc_prob", "cond_prob", "prefix_prob",
            "chain_prob", "score_mc", "score_chain", "score_mix", "mix_src",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"# CSV written to: {csv_path}")

    # ---------------- DEBUG: タイムラインに応じたグループ確認 ----------------
    DEBUG_STAFF_GROUPS = False
    if DEBUG_STAFF_GROUPS:
        from data_loader import get_groups_for_date

        # まず past_shifts 側の name -> id 集合
        past_name_to_ids = defaultdict(set)
        for (nid, name) in seqs.keys():
            past_name_to_ids[name].add(nid)

        unique_staff_pairs = set(seqs.keys())          # (id, name) の組み合わせ
        unique_staff_ids = {nid for nid, _ in unique_staff_pairs}
        unique_staff_names = set(past_name_to_ids.keys())

        print("\n=== Unique staff summary ===")
        print(f"  - unique (id, name) pairs in past_shifts: {len(unique_staff_pairs)}")
        print(f"  - unique staff IDs in past_shifts       : {len(unique_staff_ids)}")
        print(f"  - unique names in past_shifts           : {len(unique_staff_names)}")

        # setting 側（タイムライン）に出てくる名前
        timeline_names = set(group_timeline.keys())
        past_names = set(past_name_to_ids.keys())

        unknown_names = sorted(past_names - timeline_names)

        print("\n=== Staff -> Timeline Groups (タイムライン基準) ===")
        for name in sorted(past_names | timeline_names):
            ids = sorted(past_name_to_ids.get(name, []))
            if not ids:
                id_str = "-"
            elif len(ids) == 1:
                id_str = ids[0]
            else:
                id_str = ",".join(ids)

            print(f"- {name} ({id_str})")

            # タイムラインそのもの
            if name in group_timeline:
                for start_date, groups in group_timeline[name]:
                    gstr = ", ".join(sorted(groups))
                    print(f"    [timeline] from {start_date}: {gstr}")
            else:
                print("    [timeline] <no setting entry>")

            # 実際の past_shifts に対して、日付ごとの effective group を確認
            all_seq = []
            for (nid2, name2), s in seqs.items():
                if name2 == name:
                    all_seq.extend(s)
            if all_seq:
                all_seq.sort(key=lambda t: t[0])  # 日付順
                for date, shift in all_seq[:20]:  # 出力しすぎ防止に先頭20件だけ
                    gset = get_groups_for_date(name, date, group_timeline)
                    if not gset:
                        gset = {"Unknown"}
                    gstr = ", ".join(sorted(gset))
                    print(f"    [date {date}] shift={shift} -> {gstr}")
            else:
                print("    [past_shifts] <no records>")

        # Unknown グループ用に名前一覧も出しておく
        if unknown_names:
            print("\n=== Names that appear only in past_shifts (Unknown) ===")
            for name in unknown_names:
                ids = sorted(past_name_to_ids.get(name, []))
                rep_id = ids[0] if len(ids) == 1 else (",".join(ids) if ids else "-")
                print(f"  - {name} ({rep_id})")


if __name__ == "__main__":
    main()
