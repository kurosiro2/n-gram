from __future__ import annotations

import os
import glob
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any


# -----------------------------
# regex
# -----------------------------
PAT_EXT = re.compile(r'^ext_assigned\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\)\.')
PAT_OUT3 = re.compile(r'^out_assigned\(\s*(\d+)\s*,\s*(\d{8})\s*,\s*"([^"]+)"\s*\)\.')
PAT_GROUP1 = re.compile(r'^staff_group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')
PAT_GROUP2 = re.compile(r'^group\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)\.')

# staff(1,"寺島　由美子","師長","0102025",0).
PAT_STAFF5 = re.compile(
    r'^staff\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(-?\d+)\s*\)\.'
)

DATE_DIR_RE = re.compile(r"^\d{4}[_-]\d{2}[_-]\d{2}$")  # 2024_10_13 or 2024-10-13


def strip_comment(line: str) -> str:
    """行末コメント（% または # で始まる部分）を削って両端 strip"""
    for mark in ("%", "#"):
        p = line.find(mark)
        if p != -1:
            line = line[:p]
    return line.strip()


# =============================================================
# ignored-ids (NURSE_ID only)
# =============================================================
def _default_ignored_ids_dir() -> str:
    """
    デフォルトの ignored-ids ディレクトリ:
      <this_file_dir>/../2019-2025-data/ignored-ids

    あなたの例:
      /work/exp/statistics/foundmodel_data_loader.py
      -> /work/exp/2019-2025-data/ignored-ids
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "2019-2025-data", "ignored-ids"))


def _ignored_ids_dir() -> str:
    """
    環境変数 IGNORED_IDS_DIR があればそれを優先。
    なければデフォルト位置。
    """
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


def _ignore_file_path(found_path: str) -> tuple[str, str, str]:
    """
    ignore_file を探すパス計算（★固定ディレクトリ版）
    戻り: (ignore_file, ward_name, ignore_dir)
    """
    ward_name = _infer_ward_name_from_found_path(found_path)
    ignore_dir = _ignored_ids_dir()
    ignore_file = os.path.join(ignore_dir, f"{ward_name}.txt")
    return ignore_file, ward_name, ignore_dir


def _load_ignore_tokens(found_path: str) -> tuple[str, str, str, List[str]]:
    """
    ignored-ids/<WARD>.txt を読んで token(文字列) のリストを返す
    ※ token はすべて NURSE_ID とみなす（互換なし）

    returns:
      ignore_file, ward_name, ignore_dir, tokens
    """
    ignore_file, ward_name, ignore_dir = _ignore_file_path(found_path)

    if not os.path.exists(ignore_file):
        return ignore_file, ward_name, ignore_dir, []

    tokens: List[str] = []
    with open(ignore_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in line.replace(",", " ").split():
                token = token.strip()
                if token:
                    tokens.append(token)
    return ignore_file, ward_name, ignore_dir, tokens


def _resolve_nurse_ids_to_staff_ids(
    nurse_id_tokens: List[str],
    staff_info_by_sid: Dict[int, Dict[str, str]],
) -> tuple[Set[int], List[tuple[str, int]], List[str]]:
    """
    nurse_id_tokens を NURSE_ID として staff_id に解決する。
    staff_id 互換は一切しない。

    returns:
      ignore_sids: set[int]
      resolved_pairs: list[(token, sid)]
      unresolved_tokens: list[token]
    """
    # nurse_id -> sid（文字列完全一致）
    nurse_to_sid: Dict[str, int] = {}
    for sid, info in staff_info_by_sid.items():
        nid = str(info.get("nurse_id", "")).strip()
        if nid:
            nurse_to_sid[nid] = sid

    ignore_sids: Set[int] = set()
    resolved: List[tuple[str, int]] = []
    unresolved: List[str] = []

    for tok in nurse_id_tokens:
        # 1) まず完全一致（先頭0も含む）
        if tok in nurse_to_sid:
            sid = nurse_to_sid[tok]
            ignore_sids.add(sid)
            resolved.append((tok, sid))
            continue

        # 2) 先頭0落ちでも念のため探す（運用ブレ対策。いらなければ削除OK）
        tok2 = tok.lstrip("0")
        if tok2 and tok2 in nurse_to_sid:
            sid = nurse_to_sid[tok2]
            ignore_sids.add(sid)
            resolved.append((tok, sid))
            continue

        unresolved.append(tok)

    return ignore_sids, resolved, unresolved


# =============================================================
# file listing
# =============================================================
def list_model_files(dir_path: str) -> List[str]:
    """
    dir直下の found-model*.lp を優先。
    無ければ *.lp を読む。
    """
    cand = sorted(glob.glob(os.path.join(dir_path, "found-model*.lp")))
    if cand:
        return cand
    return sorted(glob.glob(os.path.join(dir_path, "*.lp")))


# =============================================================
# core parser
# =============================================================
def _parse_found_model_file(path: str):
    """
    ignored-ids 適用前の素の読み込み。
    returns:
      shifts_by_staff, groups_by_staff, staff_info_by_sid
    """
    shifts_by_staff: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    groups_by_staff: Dict[int, Set[str]] = defaultdict(set)
    staff_info_by_sid: Dict[int, Dict[str, str]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue

            m = PAT_EXT.match(line)
            if m:
                sid = int(m.group(1))
                day = int(m.group(2))
                sh = m.group(3)
                shifts_by_staff[sid].append((day, sh))
                continue

            m = PAT_OUT3.match(line)
            if m:
                sid = int(m.group(1))
                date = int(m.group(2))  # yyyymmdd
                sh = m.group(3)
                shifts_by_staff[sid].append((date, sh))
                continue

            m = PAT_GROUP1.match(line)
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

            m = PAT_STAFF5.match(line)
            if m:
                sid = int(m.group(1))
                staff_info_by_sid[sid] = {
                    "name": m.group(2),
                    "role": m.group(3),
                    "nurse_id": m.group(4),
                    "pos": m.group(5),
                }
                continue

    for sid in shifts_by_staff:
        shifts_by_staff[sid].sort(key=lambda t: t[0])

    return shifts_by_staff, groups_by_staff, staff_info_by_sid


# =============================================================
# public helper for showing ignored-ids resolution
# =============================================================
def get_ignored_resolution(
    found_model_path: str,
    staff_info_by_sid: Dict[int, Dict[str, str]],
    *,
    warn_unresolved: bool = False,
) -> Dict[str, Any]:
    """
    ignored-ids に書かれた nurse_id token を、staff_info_by_sid の nurse_id と照合して
    token -> staff_id (+ staff info) を解決して返す（表示用）。

    注意:
      - staff_info_by_sid は「ignore無しで読んだもの」を渡すこと（重要）
      - ignored-ids の探索場所は loader の _ignored_ids_dir() に従う

    ★変更:
      - warn_unresolved=False がデフォルト（unresolved の WARNING を出さない）
    """
    ignore_file, ward_name, ignore_dir, tokens = _load_ignore_tokens(found_model_path)

    ignore_sids: Set[int] = set()
    resolved_pairs: List[tuple[str, int]] = []
    unresolved: List[str] = []

    if tokens:
        ignore_sids, resolved_pairs, unresolved = _resolve_nurse_ids_to_staff_ids(
            tokens, staff_info_by_sid
        )

    resolved_detailed = []
    for tok, sid in resolved_pairs:
        info = staff_info_by_sid.get(sid, {})
        resolved_detailed.append(
            {
                "token": tok,
                "staff_id": sid,
                "nurse_id": info.get("nurse_id", ""),
                "name": info.get("name", ""),
                "role": info.get("role", ""),
                "pos": info.get("pos", ""),
            }
        )

    if unresolved and warn_unresolved:
        # 表示用関数でも必要なら警告できる（デフォルトは静か）
        print(
            f"# [found_loader] WARNING: unresolved nurse_id tokens (no matching staff(..., nurse_id, ...) in this model): {unresolved}"
        )

    return {
        "ignored_ids_dir": ignore_dir,
        "ward_name": ward_name,
        "ignore_file": ignore_file,
        "tokens": tokens,
        "resolved_pairs": resolved_detailed,
        "unresolved_tokens": unresolved,
        "ignore_staff_ids": sorted(ignore_sids),
    }


# =============================================================
# public API
# =============================================================
def load_found_model_ex(
    path: str,
    *,
    apply_ignore_ids: bool = True,
    warn_unresolved: bool = False,
):
    """
    拡張版:
      - shifts_by_staff
      - groups_by_staff
      - staff_info_by_sid（nurse_id など）
    を返す

    ★変更:
      - warn_unresolved=False がデフォルト（unresolved の WARNING を出さない）
    """
    shifts_by_staff, groups_by_staff, staff_info_by_sid = _parse_found_model_file(path)

    if apply_ignore_ids:
        ignore_file, ward_name, ignore_dir, tokens = _load_ignore_tokens(path)

        if tokens:
            ignore_sids, resolved_pairs, unresolved = _resolve_nurse_ids_to_staff_ids(
                tokens, staff_info_by_sid
            )

            # ログ（原因追跡しやすいように常に出す）
            print(f"# [found_loader] ignored_ids_dir: {ignore_dir}")
            print(f"# [found_loader] ward_name: {ward_name}")
            print(f"# [found_loader] ignore file: {ignore_file}")
            print(f"# [found_loader] ignore NURSE_ID tokens: {tokens}")

            if resolved_pairs:
                pairs_str = ", ".join([f"{t}->{sid}" for (t, sid) in resolved_pairs])
                print(
                    f"# [found_loader] resolved ignore (nurse_id->staff_id): {pairs_str}"
                )
            else:
                print(f"# [found_loader] resolved ignore: (none)")

            # ★ここが要望: デフォルトで WARNING を出さない
            if unresolved and warn_unresolved:
                print(
                    f"# [found_loader] WARNING: unresolved nurse_id tokens (no matching staff(..., nurse_id, ...) in this model): {unresolved}"
                )

            if ignore_sids:
                before = len(shifts_by_staff)
                shifts_by_staff = {
                    sid: seq
                    for sid, seq in shifts_by_staff.items()
                    if sid not in ignore_sids
                }
                groups_by_staff = {
                    sid: gs
                    for sid, gs in groups_by_staff.items()
                    if sid not in ignore_sids
                }
                staff_info_by_sid = {
                    sid: info
                    for sid, info in staff_info_by_sid.items()
                    if sid not in ignore_sids
                }
                after = len(shifts_by_staff)
                print(
                    f"# [found_loader] filtered staff by ignore_sids: {before} -> {after}"
                )
        else:
            # ファイルが存在するのに空、または存在しないケースで分岐
            if os.path.exists(ignore_file):
                print(f"# [found_loader] ignored_ids_dir: {ignore_dir}")
                print(f"# [found_loader] ward_name: {ward_name}")
                print(f"# [found_loader] ignore file exists but empty: {ignore_file}")
            # 存在しない場合は静かでOK（必要ならログ追加してもよい）

    return shifts_by_staff, groups_by_staff, staff_info_by_sid


def load_found_model(
    path: str,
    *,
    apply_ignore_ids: bool = True,
    warn_unresolved: bool = False,
):
    """
    互換版（従来通り2戻り）:
      - shifts_by_staff
      - groups_by_staff

    ★変更:
      - warn_unresolved=False がデフォルト（unresolved の WARNING を出さない）
    """
    shifts_by_staff, groups_by_staff, _staff_info = load_found_model_ex(
        path,
        apply_ignore_ids=apply_ignore_ids,
        warn_unresolved=warn_unresolved,
    )
    return shifts_by_staff, groups_by_staff


def load_found_models_from_dir_ex(
    dir_path: str,
    *,
    apply_ignore_ids: bool = True,
    warn_unresolved: bool = False,
):
    """
    ディレクトリ直下の found-model*.lp（なければ *.lp）を全部読んで返す（拡張版）
    返り値:
      models: { model_filename: (shifts_by_staff, groups_by_staff, staff_info_by_sid) }

    ★変更:
      - warn_unresolved=False がデフォルト（unresolved の WARNING を出さない）
    """
    models = {}
    for p in list_model_files(dir_path):
        models[os.path.basename(p)] = load_found_model_ex(
            p,
            apply_ignore_ids=apply_ignore_ids,
            warn_unresolved=warn_unresolved,
        )
    return models


def load_found_models_from_dir(
    dir_path: str,
    *,
    apply_ignore_ids: bool = True,
    warn_unresolved: bool = False,
):
    """
    ディレクトリ直下の found-model*.lp（なければ *.lp）を全部読んで返す（互換版）
    返り値:
      models: { model_filename: (shifts_by_staff, groups_by_staff) }

    ★変更:
      - warn_unresolved=False がデフォルト（unresolved の WARNING を出さない）
    """
    models = {}
    for p in list_model_files(dir_path):
        models[os.path.basename(p)] = load_found_model(
            p,
            apply_ignore_ids=apply_ignore_ids,
            warn_unresolved=warn_unresolved,
        )
    return models


# -----------------------------
# tiny self-test (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="found-model.lp or dir")
    ap.add_argument("--no-ignore", action="store_true", help="ignored-ids を適用しない")
    ap.add_argument(
        "--ex", action="store_true", help="拡張版(ex)で読む（staff_infoも表示）"
    )
    ap.add_argument(
        "--show-ignored",
        action="store_true",
        help="ignored-ids の token->staff 解決結果を表示する（検証用）",
    )
    ap.add_argument(
        "--warn-unresolved",
        action="store_true",
        help="unresolved nurse_id tokens の WARNING を出す（デフォルトは出さない）",
    )
    args = ap.parse_args()

    apply_ignore = not args.no_ignore
    warn_unresolved = bool(args.warn_unresolved)

    if os.path.isfile(args.target):
        if args.ex:
            s, g, info = load_found_model_ex(
                args.target,
                apply_ignore_ids=apply_ignore,
                warn_unresolved=warn_unresolved,
            )
            print(f"# loaded(ex): {args.target}")
            print(f"# staff count: {len(s)}")
            print(f"# group info staff count: {len(g)}")
            print(f"# staff_info count: {len(info)}")
            keys = sorted(info.keys())[:5]
            for sid in keys:
                print(f"# staff_info[{sid}]: {info[sid]}")

            if args.show_ignored:
                # show-ignored は ignore無しで読む必要がある
                _s0, _g0, info0 = load_found_model_ex(
                    args.target, apply_ignore_ids=False, warn_unresolved=False
                )
                res = get_ignored_resolution(
                    args.target, info0, warn_unresolved=warn_unresolved
                )
                print("# [show-ignored] (self-test)")
                print(f"# ignored_ids_dir={res['ignored_ids_dir']}")
                print(f"# ward={res['ward_name']}")
                print(f"# ignore_file={res['ignore_file']}")
                print(f"# tokens={res['tokens']}")
                print(f"# ignore_staff_ids={res['ignore_staff_ids']}")
                if res["resolved_pairs"]:
                    for r in res["resolved_pairs"]:
                        print(f"#   - {r}")
                if res["unresolved_tokens"] and warn_unresolved:
                    print(f"# WARNING unresolved={res['unresolved_tokens']}")
        else:
            s, g = load_found_model(
                args.target,
                apply_ignore_ids=apply_ignore,
                warn_unresolved=warn_unresolved,
            )
            print(f"# loaded: {args.target}")
            print(f"# staff count: {len(s)}")
            print(f"# group info staff count: {len(g)}")
    else:
        if args.ex:
            models = load_found_models_from_dir_ex(
                args.target,
                apply_ignore_ids=apply_ignore,
                warn_unresolved=warn_unresolved,
            )
            print(f"# loaded dir(ex): {args.target}")
            for name, (s, g, info) in models.items():
                print(
                    f"# - {name}: staff={len(s)}, group_staff={len(g)}, staff_info={len(info)}"
                )
        else:
            models = load_found_models_from_dir(
                args.target,
                apply_ignore_ids=apply_ignore,
                warn_unresolved=warn_unresolved,
            )
            print(f"# loaded dir: {args.target}")
            for name, (s, g) in models.items():
                print(f"# - {name}: staff={len(s)}, group_staff={len(g)}")
