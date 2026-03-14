#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【統合版】found-model(lp) を読み、n-gram を「Head/Other（+All）」で集計して表示する。
★追加: オプションで「看護師(staff_id)ごと」にも個別出力できる（--by-staff）
★追加: found-model 内の staff(...) から nurse_id/name/role を読み、by-staff の表示に付加する
★追加: ignored-ids で「どの nurse_id が無視されているか」を表示する（--show-ignored）
★追加: 集計期間を年月日(YYYYMMDD)で指定できる（--date-from / --date-to）
      ※期間でフィルタした結果、日付が連続しない部分は別セグメントとして扱い、
        n-gram がギャップを跨がないようにする。
★追加: 各 n-gram ブロックで「種類数(types)」と「総数(total)」も表示する

✅ 使い方（互換）
  1) ファイルモード:
       python ngram_found_shifts_group.py <found-model.lp> <N> [csv_path]

  2) ディレクトリモード:
       python ngram_found_shifts_group.py <found-model-dir/> <N> [csv_path]

✅ 追加オプション
  --by-staff
      staff_id ごとに個別集計して表示（Group="Staff:<sid>" として出力）
      さらに表示用ヘッダに nurse_id/name/role を付加する。

  --staff-ids 1,2,3
      指定した staff_id のみ出力（--by-staff と一緒に使う想定）

  --topk 80
      N>=2 の出力行を上位K件に制限（freq_share 降順）

  --date-from 20241101
  --date-to   20241130
      集計期間を YYYYMMDD で指定（両端含む）
      片方だけ指定も可（fromのみ→以降、toのみ→以前）

  --no-ignore
      loader の ignored-ids 適用を無効化

  --show-ignored
      ignored-ids により無視される nurse_id token と、
      それが解決される staff_id/name/role をモデルごとに表示する
      （集計処理自体は通常通り）

  --ignored-ids-dir /path/to/ignored-ids
      ignored-ids ディレクトリを明示指定（環境変数 IGNORED_IDS_DIR を上書き）
      ※ loader が見る ignored-ids の場所と一致させる用

出力:
  - N=1: 1-gram（勤務記号の割合）
  - N=2: 2-gram の freq_share と P(next|prefix)
  - N>=3: N-gram の freq_share と P(next|prefix)
  - 追加: 各ブロックで types と total を表示
  - CSV（任意）: model, group, N, gram, count, freq_share, cond_prob

グループ:
  - Head / Other の2値（+ All）

注意:
  - found-model の読み込みは ../foundmodel_data_loader.py を使用する
  - loader 側で ignored-ids が適用される（デフォルト有効）
  - このスクリプトの --show-ignored は「loader と同じ ignored-ids ディレクトリ」を見る必要があるので
    ここでも IGNORED_IDS_DIR / --ignored-ids-dir を使って合わせる
"""

import sys
import os
import csv
import re
import argparse
from collections import defaultdict, Counter
from datetime import date as _date


# -------------------------------------------------------------
# ★ loader（1個上のディレクトリ）
# -------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from foundmodel_data_loader import (  # noqa: E402
    load_found_model_ex,   # (shifts_by_staff, groups_by_staff, staff_info_by_sid)
    list_model_files,
)

# -------------------------------------------------------------
# ★ 任意：勤務シフト（10種）のみカウント
# -------------------------------------------------------------
VALID_SHIFTS = {
    "D", "LD", "EM", "LM", "SE", "SN", "E", "N",
    "WR", "PH"
}

# =============================================================
# ignored-ids show helper (align with loader policy)
# =============================================================
DATE_DIR_RE = re.compile(r"^\d{4}[_-]\d{2}[_-]\d{2}$")  # 2024_10_13 or 2024-10-13


def _default_ignored_ids_dir() -> str:
    """
    loader と同じデフォルト想定：
      <this_script_dir>/../2019-2025-data/ignored-ids
    例:
      /work/exp/statistics/ngram/ngram_found_shifts_group.py
      -> /work/exp/2019-2025-data/ignored-ids
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "2019-2025-data", "ignored-ids"))


def _ignored_ids_dir(explicit_dir: str | None = None) -> str:
    """
    優先順位:
      1) explicit_dir（--ignored-ids-dir）
      2) 環境変数 IGNORED_IDS_DIR
      3) デフォルト
    """
    if explicit_dir:
        return os.path.abspath(explicit_dir)

    d = os.environ.get("IGNORED_IDS_DIR", "").strip()
    if d:
        return os.path.abspath(d)

    return _default_ignored_ids_dir()


def _infer_ward_name_from_found_path(found_path: str) -> str:
    """
    found_path から「病棟名っぽいもの」を推定する。

    方針:
      - 親ディレクトリが日付っぽいなら、そのさらに親を ward とみなす
      - そうでなければ親ディレクトリ名を ward とみなす
      - それも無理ならファイル名 stem を ward とみなす（最終フォールバック）
    """
    parent = os.path.basename(os.path.dirname(found_path))
    if DATE_DIR_RE.match(parent):
        ward = os.path.basename(os.path.dirname(os.path.dirname(found_path)))
        if ward:
            return ward
    if parent:
        return parent
    stem = os.path.splitext(os.path.basename(found_path))[0]
    return stem or "UNKNOWN"


def _ignore_file_path(found_path: str, ignored_dir: str) -> tuple[str, str]:
    """
    ignore_file を探すパス計算（★固定ディレクトリ版：loader に合わせる）
    戻り: (ignore_file, ward_name)
    """
    ward_name = _infer_ward_name_from_found_path(found_path)
    ignore_file = os.path.join(ignored_dir, f"{ward_name}.txt")
    return ignore_file, ward_name


def _load_ignore_tokens(found_path: str, ignored_dir: str) -> tuple[str, str, list[str]]:
    """
    ignored-ids/<WARD>.txt を読んで token(文字列) のリストを返す
    ※ token はすべて NURSE_ID とみなす（互換なし）
    """
    ignore_file, ward = _ignore_file_path(found_path, ignored_dir)
    if not os.path.exists(ignore_file):
        return ignore_file, ward, []

    tokens: list[str] = []
    with open(ignore_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.replace(",", " ").split():
                tok = tok.strip()
                if tok:
                    tokens.append(tok)
    return ignore_file, ward, tokens


def _resolve_ignore_tokens(tokens: list[str], staff_info_by_sid: dict[int, dict]) -> tuple[list[tuple[str, int]], list[str]]:
    """
    tokens (NURSE_ID扱い) を staff_info_by_sid の nurse_id と突き合わせて
    token -> staff_id に解決する（loader と同じく先頭0落ちも救済）。
    """
    nurse_to_sid: dict[str, int] = {}
    for sid, info in staff_info_by_sid.items():
        nid = str(info.get("nurse_id", "")).strip()
        if nid:
            nurse_to_sid[nid] = sid

    resolved: list[tuple[str, int]] = []
    unresolved: list[str] = []

    for tok in tokens:
        if tok in nurse_to_sid:
            resolved.append((tok, nurse_to_sid[tok]))
            continue
        tok2 = tok.lstrip("0")
        if tok2 and tok2 in nurse_to_sid:
            # 先頭0落ち運用ブレ対策（不要なら削除してOK）
            resolved.append((tok, nurse_to_sid[tok2]))
            continue
        unresolved.append(tok)

    return resolved, unresolved


def show_ignored_summary(found_model_path: str, staff_info_by_sid: dict[int, dict], ignored_dir: str):
    """
    ignored-ids により無視される nurse_id token と、その解決結果を表示する。
    ※ staff_info_by_sid は「ignore無しで読んだもの」を渡すこと（重要）。
    """
    ignore_file, ward, tokens = _load_ignore_tokens(found_model_path, ignored_dir)

    print(f"# [show-ignored] ignored_ids_dir={ignored_dir}")
    print(f"# [show-ignored] ward={ward}")
    print(f"# [show-ignored] ignore_file={ignore_file}")

    if not tokens:
        if os.path.exists(ignore_file):
            print("# [show-ignored] ignore_file exists but empty")
        else:
            print("# [show-ignored] ignore_file not found")
        return

    resolved, unresolved = _resolve_ignore_tokens(tokens, staff_info_by_sid)

    print(f"# [show-ignored] tokens(NURSE_ID): {tokens}")

    if resolved:
        print("# [show-ignored] resolved (nurse_id -> staff_id):")
        for nurse_id, sid in resolved:
            info = staff_info_by_sid.get(sid, {})
            name = info.get("name", "")
            role = info.get("role", "")
            nurse_id2 = info.get("nurse_id", "")
            print(f"#   - nurse_id_token={nurse_id} -> staff_id={sid}  nurse_id={nurse_id2}  name={name}  role={role}")
    else:
        print("# [show-ignored] resolved: (none)")

    if unresolved:
        print(f"# [show-ignored] WARNING unresolved tokens: {unresolved}")


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


def merge_staff_counters(dst, src):
    """defaultdict(Counter) keyed by staff_label を加算マージ"""
    for k, c in src.items():
        dst[k].update(c)


def parse_staff_ids(s: str):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"Invalid staff id: {part}")
        out.add(int(part))
    return out if out else None


def staff_sort_key(k: str):
    # "Staff:123" を数値順
    m = re.match(r"^Staff:(\d+)$", k)
    if m:
        return (0, int(m.group(1)))
    return (9, k)


def _extract_sid_from_staff_label(label: str) -> int | None:
    m = re.match(r"^Staff:(\d+)$", label)
    return int(m.group(1)) if m else None


def format_staff_label(label: str, staff_info_by_sid: dict | None) -> str:
    """
    label = "Staff:<sid>"
    返す: 表示用ラベル（nurse_id/name/role を付加）
    """
    sid = _extract_sid_from_staff_label(label)
    if sid is None:
        return label

    info = (staff_info_by_sid or {}).get(sid, {})
    nurse_id = info.get("nurse_id", "")
    name = info.get("name", "")
    role = info.get("role", "")

    extra = [f"staff_id={sid}"]
    if nurse_id:
        extra.append(f"nurse_id={nurse_id}")
    if name:
        extra.append(f"name={name}")
    if role:
        extra.append(f"role={role}")

    return f"{label} ({', '.join(extra)})"


# -----------------------------
# Date range filtering helpers
# -----------------------------
DATE8_RE = re.compile(r"^\d{8}$")  # YYYYMMDD


def parse_yyyymmdd(s: str | None) -> int | None:
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
    seq_sorted: list[tuple[int, str]],
    date_from: int | None,
    date_to: int | None
) -> list[list[str]]:
    """
    seq_sorted: [(YYYYMMDD, shift), ...] 日付昇順前提
    1) [date_from, date_to] でフィルタ（Noneは無制限）
    2) 連続日付ごとに分割（ギャップがあれば別セグメント）
    返り値: [ [shift, shift, ...], [shift, ...], ... ]
    """
    filtered: list[tuple[int, str]] = []
    for d, sh in seq_sorted:
        if date_from is not None and d < date_from:
            continue
        if date_to is not None and d > date_to:
            continue
        filtered.append((d, sh))

    if not filtered:
        return []

    segments: list[list[str]] = []
    cur: list[str] = [filtered[0][1]]
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
    ※ groupnames が空でも Other
    """
    for g in groupnames:
        if is_head_groupname(g):
            return "Head"
    return "Other"


# =============================================================
# n-gram 集計（group版）
# =============================================================
def ngram_counts_by_group(seqs_by_staff, groups_by_staff, n, date_from=None, date_to=None):
    """
    n-gram 出現回数を Head/Other(+All) でカウントする。
    date_from/date_to (YYYYMMDD int) があれば、その期間内だけ集計する。
    期間フィルタで欠けた日があると n-gram が跨がないよう、連続日付ごとに分割して数える。
    """
    counters = defaultdict(Counter)
    if n <= 0:
        return counters

    for sid, seq in seqs_by_staff.items():
        if len(seq) < n:
            continue

        seq_sorted = sorted(seq, key=lambda t: t[0])
        segments = filter_and_split_by_consecutive_days(seq_sorted, date_from, date_to)

        g_bucket = bucket_group(groups_by_staff.get(sid, set()))
        gset = {g_bucket, "All"}

        for shifts in segments:
            if len(shifts) < n:
                continue
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i:i + n])
                if any(s not in VALID_SHIFTS for s in gram):
                    continue
                for g in gset:
                    counters[g][gram] += 1

    return counters


# =============================================================
# n-gram 集計（staff個別版）
# =============================================================
def ngram_counts_by_staff(seqs_by_staff, n, staff_filter=None, date_from=None, date_to=None):
    """
    n-gram 出現回数を staff_id ごとにカウントする。
      - staff_filter: set[int] or None
      - date_from/date_to: YYYYMMDD int or None
    期間フィルタで欠けた日があると n-gram が跨がないよう、連続日付ごとに分割して数える。
    返り値:
      counters: dict[str, Counter]  (keyは "Staff:<id>")
    """
    counters = defaultdict(Counter)
    if n <= 0:
        return counters

    for sid, seq in seqs_by_staff.items():
        if staff_filter is not None and sid not in staff_filter:
            continue
        if len(seq) < n:
            continue

        seq_sorted = sorted(seq, key=lambda t: t[0])
        segments = filter_and_split_by_consecutive_days(seq_sorted, date_from, date_to)

        k = f"Staff:{sid}"
        for shifts in segments:
            if len(shifts) < n:
                continue
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i:i + n])
                if any(s not in VALID_SHIFTS for s in gram):
                    continue
                counters[k][gram] += 1

    return counters


# =============================================================
# 出力（あなたの形式に合わせる）
# =============================================================
def print_unigram_share(model: str, group_label: str, uni_counter: Counter, csv_rows=None):
    total = sum(uni_counter.values())
    types = len(uni_counter)

    print(f'\n----- Model="{model}" | Group="{group_label}" | 1-gram（勤務記号の割合） -----')
    print(f"# [1-gram stats] types={types}  total={total}")
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
                "group": group_label,
                "N": 1,
                "gram": s,
                "count": c,
                "freq_share": share,
                "cond_prob": "",
            })


def print_bigram_score(model: str, group_label: str, uni_counter: Counter, bi_counter: Counter, csv_rows=None, topk=None):
    total_bi = sum(bi_counter.values())
    types_bi = len(bi_counter)

    print(f'\n----- Model="{model}" | Group="{group_label}" | 2-gram（freq_share と P(next|prefix)） -----')
    print(f"# [2-gram stats] types={types_bi}  total={total_bi}")
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
    if topk is not None and topk > 0:
        rows = rows[:topk]

    for (freq_share, c, gram, cond_prob) in rows:
        arrow = f"{gram[0]}->{gram[1]}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")
        if csv_rows is not None:
            csv_rows.append({
                "model": model,
                "group": group_label,
                "N": 2,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def print_ngramN_score(model: str, group_label: str, n_counter: Counter, nm1_counter: Counter, N: int, csv_rows=None, topk=None):
    total_N = sum(n_counter.values())
    types_N = len(n_counter)

    print(f'\n----- Model="{model}" | Group="{group_label}" | {N}-gram（freq_share と P(next|prefix)） -----')
    print(f"# [{N}-gram stats] types={types_N}  total={total_N}")
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
    if topk is not None and topk > 0:
        rows = rows[:topk]

    for (freq_share, c, gram, cond_prob) in rows:
        prefix_str = "-".join(gram[:-1])
        nxt = gram[-1]
        arrow = f"{prefix_str}->{nxt}"
        print(f"{c:>6}  {arrow:<20}   {freq_share:11.6f}   {cond_prob:14.6f}")
        if csv_rows is not None:
            csv_rows.append({
                "model": model,
                "group": group_label,
                "N": N,
                "gram": "-".join(gram),
                "count": c,
                "freq_share": freq_share,
                "cond_prob": cond_prob,
            })


def print_model_block_group(model_label: str, N_eff: int,
                            counters_1, counters_2, counters_N, counters_Nm1,
                            csv_rows=None, topk=None):
    """group(Head/Other/All) 出力"""
    for g in sorted(counters_N.keys(), key=group_sort_key):
        if g not in ("All", "Head", "Other"):
            continue

        if N_eff == 1:
            print_unigram_share(model_label, g, counters_N.get(g, Counter()), csv_rows=csv_rows)
        elif N_eff == 2:
            uni = counters_1.get(g, Counter())
            bi = counters_2.get(g, Counter())
            print_bigram_score(model_label, g, uni, bi, csv_rows=csv_rows, topk=topk)
        else:
            n_counter = counters_N.get(g, Counter())
            nm1_counter = counters_Nm1.get(g, Counter())
            print_ngramN_score(model_label, g, n_counter, nm1_counter, N_eff, csv_rows=csv_rows, topk=topk)


def print_model_block_staff(model_label: str, N_eff: int,
                            counters_1, counters_2, counters_N, counters_Nm1,
                            staff_info_by_sid: dict | None,
                            csv_rows=None, topk=None):
    """staff(Staff:<id>) 出力（表示ラベルに nurse_id/name/role を付加）"""
    for k in sorted(counters_N.keys(), key=staff_sort_key):
        pretty = format_staff_label(k, staff_info_by_sid)

        # CSV互換のため、group列は「表示ラベル pretty」をそのまま出す（既存方針維持）
        if N_eff == 1:
            print_unigram_share(model_label, pretty, counters_N.get(k, Counter()), csv_rows=csv_rows)
        elif N_eff == 2:
            uni = counters_1.get(k, Counter())
            bi = counters_2.get(k, Counter())
            print_bigram_score(model_label, pretty, uni, bi, csv_rows=csv_rows, topk=topk)
        else:
            n_counter = counters_N.get(k, Counter())
            nm1_counter = counters_Nm1.get(k, Counter())
            print_ngramN_score(model_label, pretty, n_counter, nm1_counter, N_eff, csv_rows=csv_rows, topk=topk)


# =============================================================
# main
# =============================================================
def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("target", help="<found-model.lp or dir>")
    parser.add_argument("N", type=int, help="n-gram N")
    parser.add_argument("csv_path", nargs="?", default=None, help="optional csv path")
    parser.add_argument("--by-staff", action="store_true",
                        help="看護師(staff_id)ごとに個別出力する（Group=Staff:<id>）")
    parser.add_argument("--staff-ids", default=None,
                        help="by-staff時に出す staff_id を限定（例: 1,2,3）")
    parser.add_argument("--topk", type=int, default=80,
                        help="N>=2 の出力行を上位K件に制限（デフォルト80, 0以下で制限なし）")
    parser.add_argument("--date-from", default=None,
                        help="集計期間の開始日（YYYYMMDD, 例: 20241101）")
    parser.add_argument("--date-to", default=None,
                        help="集計期間の終了日（YYYYMMDD, 例: 20241130）")
    parser.add_argument("--no-ignore", action="store_true",
                        help="ignored-ids を適用しない（loader側オプション）")
    parser.add_argument("--show-ignored", action="store_true",
                        help="ignored-ids で無視される nurse_id -> staff_id/name/role を表示する（集計前に出す）")
    parser.add_argument("--ignored-ids-dir", default=None,
                        help="ignored-ids ディレクトリを明示指定（例: /work/exp/2019-2025-data/ignored-ids）")

    args = parser.parse_args()

    target = args.target
    N_eff = max(1, int(args.N))
    csv_path = args.csv_path
    csv_rows = [] if csv_path else None

    staff_filter = None
    if args.staff_ids:
        staff_filter = parse_staff_ids(args.staff_ids)

    topk = args.topk if args.topk and args.topk > 0 else None
    apply_ignore = not args.no_ignore

    date_from = parse_yyyymmdd(args.date_from)
    date_to = parse_yyyymmdd(args.date_to)
    if date_from is not None and date_to is not None and date_to < date_from:
        raise ValueError(f"--date-to must be >= --date-from (from={date_from}, to={date_to})")

    # show-ignored の参照先を loader と合わせる
    ignored_dir = _ignored_ids_dir(args.ignored_ids_dir)

    # loader が参照する ignored-ids も合わせたい時は環境変数を設定しておく
    # ※ ただし "loader側に --ignored-ids-dir を渡す" という仕組みが無いので、ここで env を統一
    if args.ignored_ids_dir:
        os.environ["IGNORED_IDS_DIR"] = os.path.abspath(args.ignored_ids_dir)

    # =========================================================
    # ファイルモード
    # =========================================================
    if os.path.isfile(target):
        # 重要: ignored-ids で消される人の staff(...) 情報は
        # ignore適用後だと消えるので、必ず「ignore無し」で一度読む
        _seq0, _grp0, info0 = load_found_model_ex(target, apply_ignore_ids=False)
        if args.show_ignored:
            show_ignored_summary(target, info0, ignored_dir)

        seqs_by_staff, groups_by_staff, staff_info_by_sid = load_found_model_ex(
            target, apply_ignore_ids=apply_ignore
        )

        print(f'# [File mode] {target}')
        if date_from or date_to:
            print(f"# [Period] {date_from or 'MIN'} .. {date_to or 'MAX'} (YYYYMMDD)")

        if not args.by_staff:
            counters_1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 1, date_from=date_from, date_to=date_to)
            counters_2 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 2, date_from=date_from, date_to=date_to)
            counters_N = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff, date_from=date_from, date_to=date_to)
            counters_Nm1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff - 1, date_from=date_from, date_to=date_to) if N_eff >= 2 else defaultdict(Counter)

            for g in ("All", "Head", "Other"):
                counters_1.setdefault(g, Counter())
                counters_2.setdefault(g, Counter())
                counters_N.setdefault(g, Counter())
                if N_eff >= 2:
                    counters_Nm1.setdefault(g, Counter())

            print_model_block_group(os.path.basename(target), N_eff,
                                    counters_1, counters_2, counters_N, counters_Nm1,
                                    csv_rows=csv_rows, topk=topk)
        else:
            counters_1 = ngram_counts_by_staff(seqs_by_staff, 1, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            counters_2 = ngram_counts_by_staff(seqs_by_staff, 2, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            counters_N = ngram_counts_by_staff(seqs_by_staff, N_eff, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            counters_Nm1 = ngram_counts_by_staff(seqs_by_staff, N_eff - 1, staff_filter=staff_filter, date_from=date_from, date_to=date_to) if N_eff >= 2 else defaultdict(Counter)

            print_model_block_staff(os.path.basename(target), N_eff,
                                    counters_1, counters_2, counters_N, counters_Nm1,
                                    staff_info_by_sid=staff_info_by_sid,
                                    csv_rows=csv_rows, topk=topk)

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
    if date_from or date_to:
        print(f"# [Period] {date_from or 'MIN'} .. {date_to or 'MAX'} (YYYYMMDD)")
    print("# Models:")
    for p in model_files:
        print(f"#   - {os.path.basename(p)}")

    if not args.by_staff:
        counters_1_all = defaultdict(Counter)
        counters_2_all = defaultdict(Counter)
        counters_N_all = defaultdict(Counter)
        counters_Nm1_all = defaultdict(Counter) if N_eff >= 2 else defaultdict(Counter)

        for p in model_files:
            model_label = os.path.basename(p)

            # show-ignored は「ignore無しで一度読む」必要がある
            _seq0, _grp0, info0 = load_found_model_ex(p, apply_ignore_ids=False)
            if args.show_ignored:
                print(f"# [model] {model_label}")
                show_ignored_summary(p, info0, ignored_dir)

            seqs_by_staff, groups_by_staff, _staff_info = load_found_model_ex(
                p, apply_ignore_ids=apply_ignore
            )

            c1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 1, date_from=date_from, date_to=date_to)
            c2 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, 2, date_from=date_from, date_to=date_to)
            cN = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff, date_from=date_from, date_to=date_to)
            cNm1 = ngram_counts_by_group(seqs_by_staff, groups_by_staff, N_eff - 1, date_from=date_from, date_to=date_to) if N_eff >= 2 else defaultdict(Counter)

            for g in ("All", "Head", "Other"):
                c1.setdefault(g, Counter())
                c2.setdefault(g, Counter())
                cN.setdefault(g, Counter())
                if N_eff >= 2:
                    cNm1.setdefault(g, Counter())

            print_model_block_group(model_label, N_eff, c1, c2, cN, cNm1, csv_rows=csv_rows, topk=topk)

            merge_group_counters(counters_1_all, c1)
            merge_group_counters(counters_2_all, c2)
            merge_group_counters(counters_N_all, cN)
            if N_eff >= 2:
                merge_group_counters(counters_Nm1_all, cNm1)

        print("\n# =====================")
        print("# --- TOTAL (sum of all found-models) ---")
        print("# =====================")

        for g in ("All", "Head", "Other"):
            counters_1_all.setdefault(g, Counter())
            counters_2_all.setdefault(g, Counter())
            counters_N_all.setdefault(g, Counter())
            if N_eff >= 2:
                counters_Nm1_all.setdefault(g, Counter())

        print_model_block_group("TOTAL", N_eff,
                                counters_1_all, counters_2_all, counters_N_all, counters_Nm1_all,
                                csv_rows=csv_rows, topk=topk)

    else:
        counters_1_all = defaultdict(Counter)
        counters_2_all = defaultdict(Counter)
        counters_N_all = defaultdict(Counter)
        counters_Nm1_all = defaultdict(Counter) if N_eff >= 2 else defaultdict(Counter)

        # TOTAL 用に staff_info を集約（sid の情報をできるだけ保持）
        staff_info_total = {}

        for p in model_files:
            model_label = os.path.basename(p)

            # show-ignored は「ignore無しで一度読む」必要がある
            _seq0, _grp0, info0 = load_found_model_ex(p, apply_ignore_ids=False)
            if args.show_ignored:
                print(f"# [model] {model_label}")
                show_ignored_summary(p, info0, ignored_dir)

            seqs_by_staff, _groups_by_staff, staff_info_by_sid = load_found_model_ex(
                p, apply_ignore_ids=apply_ignore
            )

            # staff_info をマージ（同sidは後勝ち）
            for sid, info in staff_info_by_sid.items():
                staff_info_total[sid] = info

            c1 = ngram_counts_by_staff(seqs_by_staff, 1, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            c2 = ngram_counts_by_staff(seqs_by_staff, 2, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            cN = ngram_counts_by_staff(seqs_by_staff, N_eff, staff_filter=staff_filter, date_from=date_from, date_to=date_to)
            cNm1 = ngram_counts_by_staff(seqs_by_staff, N_eff - 1, staff_filter=staff_filter, date_from=date_from, date_to=date_to) if N_eff >= 2 else defaultdict(Counter)

            print_model_block_staff(model_label, N_eff, c1, c2, cN, cNm1,
                                    staff_info_by_sid=staff_info_by_sid,
                                    csv_rows=csv_rows, topk=topk)

            merge_staff_counters(counters_1_all, c1)
            merge_staff_counters(counters_2_all, c2)
            merge_staff_counters(counters_N_all, cN)
            if N_eff >= 2:
                merge_staff_counters(counters_Nm1_all, cNm1)

        print("\n# =====================")
        print("# --- TOTAL (sum of all found-models) ---")
        print("# =====================")

        print_model_block_staff("TOTAL", N_eff,
                                counters_1_all, counters_2_all, counters_N_all, counters_Nm1_all,
                                staff_info_by_sid=staff_info_total,
                                csv_rows=csv_rows, topk=topk)

    if csv_path:
        fieldnames = ["model", "group", "N", "gram", "count", "freq_share", "cond_prob"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"# CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
