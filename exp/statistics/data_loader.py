# past_shiftとsettingの紐づけはnameのみで行う(datasetに同じidを持つ看護師がいるため)

#!/usr/bin/env python3
import os
import re
from collections import defaultdict

# True  : 最初に出てきたグループを過去にも遡って適用する
# False : setting に登場する前の日付はグループ不明（Unknown 扱い）
BACKFILL_EARLIEST_GROUP = False 

# -------------------------------------------------------------
# 共通ユーティリティ
# -------------------------------------------------------------

def strip_comment(line: str) -> str:
    """行末コメント（% または # で始まる）を除去して両端 strip"""
    for mark in ("%", "#"):
        p = line.find(mark)
        if p != -1:
            line = line[:p]
    return line.strip()


def normalize_id(nid: str) -> str:
    """
    ID を正規化する関数。
    - 先頭の 0 は無視（"0384563" → "384563"）
    - 数字以外が混ざっていればそのまま返す
    """
    s = nid.strip()
    if s.isdigit():
        s2 = s.lstrip("0")
        return s2 if s2 != "" else "0"
    return s


# -------------------------------------------------------------
# 1) past-shifts loader
# -------------------------------------------------------------

SHIFT_PATTERN = re.compile(
    r'shift_data\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*"([^"]+)"\)\.'
)


def _load_ignore_ids_for_past_shifts(shift_file: str):
    """
    shift_file のパスから病棟名を推定し、
    対応する ignored-ids/<病棟名>.txt を探して ID を読み込む。
    ファイルが無ければ空集合を返す。

    例:
      shift_file = exp/2019-2025-data/real-name/past-shifts/GCU.lp
        -> real-name/ignored-ids/GCU.txt
    """
    # 病棟名 = ファイル名 (拡張子除く)
    ward_name = os.path.splitext(os.path.basename(shift_file))[0]

    # .../real-name/past-shifts の一つ上が .../real-name
    past_shifts_dir = os.path.dirname(shift_file)       # .../real-name/past-shifts
    realname_dir    = os.path.dirname(past_shifts_dir)  # .../real-name

    ignore_dir  = os.path.join(realname_dir, "ignored-ids")
    ignore_file = os.path.join(ignore_dir, f"{ward_name}.txt")

    if not os.path.exists(ignore_file):
        return set()

    ids = set()
    with open(ignore_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # カンマ/空白区切りどっちでもOKにする
            for token in line.replace(",", " ").split():
                ids.add(normalize_id(token))

    print(f"# [data_loader] loaded ignore_ids from {ignore_file}: {sorted(ids)}")
    return ids


def load_past_shifts(path: str):
    """
    past-shifts.lp を読み込んで
      {(nurse_id, name): [(date, shift), ...]}
    の dict を返す。
    - nurse_id は normalize_id() 済み
    - date は int

    さらに:
      - 対応する ignored-ids/<病棟名>.txt が存在する場合、
        そこに書かれた ID（normalize_id 済み）に該当する看護師は
        自動的に除外する。
    """
    seqs = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue
            m = SHIFT_PATTERN.match(line)
            if not m:
                continue
            nurse_id_raw = m.group(1)
            nurse_id = normalize_id(nurse_id_raw)
            name = m.group(2)
            date = int(m.group(3))
            shift = m.group(4)
            seqs[(nurse_id, name)].append((date, shift))

    # 日付順ソート
    for key in seqs:
        seqs[key].sort(key=lambda t: t[0])

    # ignored-ids を自動適用
    ignore_ids = _load_ignore_ids_for_past_shifts(path)
    if ignore_ids:
        before = len(seqs)
        seqs = {
            (nid, name): s
            for (nid, name), s in seqs.items()
            if nid not in ignore_ids
        }
        after = len(seqs)
        print(f"# [data_loader] filtered nurses by ignore_ids: {before} -> {after}")

    return seqs


# -------------------------------------------------------------
# 2) setting loader（単一ファイル or ディレクトリ）: 「名前 → グループ集合」
# -------------------------------------------------------------

PAT_STAFF = re.compile(
    r'staff\((\d+),\s*"([^"]+)",\s*"[^"]+",\s*"([^"]+)"'
)
PAT_GROUP = re.compile(
    r'staff_group\("([^"]+)",\s*(\d+)\)\.'
)


def _load_single_setting_file_name_groups(path: str):
    """
    1つの setting.lp を読み込んで
      { name: set(groups) }
    を返すヘルパー。
    """
    local_idx_to_name = {}          # idx -> name
    name_to_groups = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue

            m1 = PAT_STAFF.match(line)
            if m1:
                sid = m1.group(1)
                name = m1.group(2)
                # nurse_id_raw = m1.group(3)  # 今回は使わないが、必要ならここで normalize_id 可
                local_idx_to_name[sid] = name
                continue

            m2 = PAT_GROUP.match(line)
            if m2:
                group = m2.group(1)
                sid = m2.group(2)
                name = local_idx_to_name.get(sid)
                if name is not None:
                    name_to_groups[name].add(group)

    return name_to_groups


def _collect_setting_files(setting_path: str):
    """
    setting_path が:
      - ファイル: [そのパス] を返す
      - ディレクトリ: 直下サブディレクトリ配下の setting.lp を全部集める
        例: 5階南病棟/2024-03-03/setting.lp など
    """
    if os.path.isfile(setting_path):
        return [setting_path]

    if os.path.isdir(setting_path):
        result = []
        for name in sorted(os.listdir(setting_path)):
            subdir = os.path.join(setting_path, name)
            if not os.path.isdir(subdir):
                continue
            cand = os.path.join(subdir, "setting.lp")
            if os.path.isfile(cand):
                result.append(cand)
        return result

    # どちらでもなければ空リスト
    return []


def load_staff_groups(setting_path: str):
    """
    setting_path（ファイル or ディレクトリ）から
      { name: set(groups) }
    を構築して返す。
    ※複数ファイルある場合は「全部 union」する（時系列を潰す）。
    """
    staff_to_groups = defaultdict(set)
    setting_files = _collect_setting_files(setting_path)
    if not setting_files:
        raise FileNotFoundError(f"setting files not found: {setting_path}")

    for spath in setting_files:
        local_map = _load_single_setting_file_name_groups(spath)
        for name, groups in local_map.items():
            staff_to_groups[name].update(groups)

    return staff_to_groups


# -------------------------------------------------------------
# 3) setting loader（ディレクトリ）: 「名前 → 時系列グループ」
# -------------------------------------------------------------

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date_from_dirname(dirname: str):
    """
    "YYYY-MM-DD" 形式のディレクトリ名から int YYYYMMDD を返す。
    それ以外なら None を返す。
    （※ index 付き "2025-09-14-2" はここでは無視するため、長さ 10 固定）
    """
    if DATE_DIR_RE.match(dirname):
        y, m, d = dirname.split("-")
        return int(y) * 10000 + int(m) * 100 + int(d)
    return None


def _collect_dated_setting_files(setting_dir: str):
    """
    setting_dir 配下の
      YYYY-MM-DD/setting.lp
    だけを集めて (date_int, path) のリストで返す。
    例:
      2024-03-03/setting.lp → (20240303, ".../2024-03-03/setting.lp")
    "2024-03-03-2" のような index 付きディレクトリは無視する。
    """
    result = []
    if not os.path.isdir(setting_dir):
        return result

    for name in sorted(os.listdir(setting_dir)):
        subdir = os.path.join(setting_dir, name)
        if not os.path.isdir(subdir):
            continue
        date_int = _parse_date_from_dirname(name)
        if date_int is None:
            continue
        cand = os.path.join(subdir, "setting.lp")
        if os.path.isfile(cand):
            result.append((date_int, cand))

    return result


def load_staff_group_timeline(setting_path: str):
    """
    setting_path が:
      - ディレクトリ: YYYY-MM-DD/setting.lp を日付順に読んで、
          { name: [ (start_date, set(groups)), ... ] }
        を返す。
        start_date は int YYYYMMDD。
      - ファイル: その1つだけのスナップショットとして
          start_date = 0 として返す。

    例:
      "5階南病棟" ディレクトリを渡した場合、
        2024-03-03/setting.lp
        2024-07-21/setting.lp
        2025-03-02/setting.lp
      → それぞれの日付でのグループがタイムラインとして name ごとに並ぶ。
    """
    timeline = defaultdict(list)

    # 単一ファイル指定の場合：時系列は 1 個だけ（start_date=0 とする）
    if os.path.isfile(setting_path):
        name_groups = _load_single_setting_file_name_groups(setting_path)
        for name, groups in name_groups.items():
            timeline[name].append((0, set(groups)))
        return timeline

    # ディレクトリの場合：YYYY-MM-DD/setting.lp を時系列で読む
    if os.path.isdir(setting_path):
        dated_files = _collect_dated_setting_files(setting_path)
        if not dated_files:
            raise FileNotFoundError(f"dated setting files not found under: {setting_path}")

        # 日付順に処理
        for date_int, spath in sorted(dated_files, key=lambda x: x[0]):
            name_groups = _load_single_setting_file_name_groups(spath)
            for name, groups in name_groups.items():
                # 同じ name について、その時点でのグループを追加
                timeline[name].append((date_int, set(groups)))

        return timeline

    raise FileNotFoundError(f"setting path not found: {setting_path}")


def get_groups_for_date(name: str, date: int, group_timeline):
    """
    指定した name, date に対して有効なグループ集合を返す。

    group_timeline[name] = [(start_date, set(groups)), ...] （start_date 昇順）

    - BACKFILL_EARLIEST_GROUP = True:
        date が最初の start_date より前なら、
        「最初の snapshot の groups をそのまま過去にも遡って適用」する。
    - BACKFILL_EARLIEST_GROUP = False:
        date が最初の start_date より前なら、所属不明（空集合）を返す。
    """
    entries = group_timeline.get(name)
    if not entries:
        # その名前が一度も setting に現れない
        return set()

    # entries: [(start_date, set(groups)), ...] 昇順
    first_date, first_groups = entries[0]

    # フラグで最初の snapshot を過去に遡って適用するか決める
    if date < first_date:
        if BACKFILL_EARLIEST_GROUP:
            return set(first_groups)
        else:
            return set()

    # それ以外は「start_date <= date の中で一番新しいもの」を探す
    chosen_groups = None
    for start_date, groups in entries:
        if start_date <= date:
            chosen_groups = groups
        else:
            break

    return set(chosen_groups) if chosen_groups is not None else set()
