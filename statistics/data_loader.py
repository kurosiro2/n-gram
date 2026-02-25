import os
import re
from collections import defaultdict

# True  : 最初に出てきたグループを過去にも遡って適用する
# False : setting に登場する前の日付はグループ不明（Unknown 扱い）
BACKFILL_EARLIEST_GROUP = False


def strip_comment(line: str) -> str:
    """行末コメント（% または # で始まる部分）を削って両端 strip"""
    for mark in ("%", "#"):
        p = line.find(mark)
        if p != -1:
            line = line[:p]
    return line.strip()


#  past-shifts loader
SHIFT_PATTERN = re.compile(
    r'shift_data\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*"([^"]+)"\)\.'
)


def _load_ignore_ids_for_past_shifts(shift_file: str):
    """
    shift_file のパスから病棟名を推定し、
    対応する ignored-ids/<病棟名>.txt を探して ID を読み込む。
    ファイルが無ければ空集合を返す。
    """
    ward_name = os.path.splitext(os.path.basename(shift_file))[0]

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
            for token in line.replace(",", " ").split():
                ids.add(token.strip())

    print(f"# [data_loader] loaded ignore_ids from {ignore_file}: {sorted(ids)}")
    return ids


def load_past_shifts(path: str):
    """
    past-shifts.lp を読み込んで
      {(nurse_id, name): [(date, shift), ...]}
    の dict を返す。
    - 対応する ignored-ids/<病棟名>.txt が存在する場合、
    そこに書かれた ID と一致する nurse_id を持つ看護師は除外する。
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
            nurse_id = m.group(1)       
            name = m.group(2)
            date = int(m.group(3))
            shift = m.group(4)
            seqs[(nurse_id, name)].append((date, shift))

    for key in seqs:
        seqs[key].sort(key=lambda t: t[0])

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


# 2) setting loader（単一ファイル or ディレクトリ）

PAT_STAFF = re.compile(
    r'staff\((\d+),\s*"([^"]+)",\s*"[^"]+",\s*"([^"]+)"'
)
PAT_GROUP = re.compile(
    r'staff_group\("([^"]+)",\s*(\d+)\)\.'
)


def _load_single_setting_file_idx_name_id_groups(path: str):
    """
    1つの setting.lp を読み込んで、
      - name_to_groups: { name: set(groups) }
      - key_to_groups : { (name, nurse_id): set(groups) }
    を返す。
    """
    local_idx_to_key = {}           
    name_to_groups = defaultdict(set)
    key_to_groups  = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue

            m1 = PAT_STAFF.match(line)
            if m1:
                sid = m1.group(1)
                name = m1.group(2)
                nurse_id = m1.group(3)  
                local_idx_to_key[sid] = (name, nurse_id)
                continue

            m2 = PAT_GROUP.match(line)
            if m2:
                group = m2.group(1)
                sid = m2.group(2)
                key = local_idx_to_key.get(sid)
                if key is not None:
                    name, nurse_id = key
                    name_to_groups[name].add(group)
                    key_to_groups[(name, nurse_id)].add(group)

    return name_to_groups, key_to_groups


def _load_single_setting_file_name_groups(path: str):
    """既存 API 用ヘルパー ({ name: set(groups) } だけ欲しいとき用)"""
    name_to_groups, _ = _load_single_setting_file_idx_name_id_groups(path)
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

    return []


def load_staff_groups(setting_path: str):
    """
    setting_path(ファイル or ディレクトリ）から
      { name: set(groups) }
    を構築して返す。
    ※要確認
    """
    staff_to_groups = defaultdict(set)
    setting_files = _collect_setting_files(setting_path)
    if not setting_files:
        raise FileNotFoundError(f"setting files not found: {setting_path}")

    for spath in setting_files:
        name_map, _ = _load_single_setting_file_idx_name_id_groups(spath)
        for name, groups in name_map.items():
            staff_to_groups[name].update(groups)

    return staff_to_groups


# 3) setting loader（ディレクトリ）: 「名前/名前+ID → 時系列グループ」
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date_from_dirname(dirname: str):
    if DATE_DIR_RE.match(dirname):
        y, m, d = dirname.split("-")
        return int(y) * 10000 + int(m) * 100 + int(d)
    return None


def _collect_dated_setting_files(setting_dir: str):
    """
    setting_dir 配下の
      YYYY-MM-DD/setting.lp
    だけを集めて (date_int, path) のリストで返す。
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
          {
            name: [(start_date, set(groups)), ...],
            (name, nurse_id): [(start_date, set(groups)), ...],
          }
        を返す。
      - ファイル: その1つだけのスナップショットとして
          start_date = 0 として同様の構造を返す。
    """
    timeline = defaultdict(list)

    # 単一ファイル指定の場合
    if os.path.isfile(setting_path):
        name_map, key_map = _load_single_setting_file_idx_name_id_groups(setting_path)
        for name, groups in name_map.items():
            timeline[name].append((0, set(groups)))
        for (name, nurse_id), groups in key_map.items():
            timeline[(name, nurse_id)].append((0, set(groups)))
        return timeline

    # ディレクトリの場合
    if os.path.isdir(setting_path):
        dated_files = _collect_dated_setting_files(setting_path)
        if not dated_files:
            raise FileNotFoundError(f"dated setting files not found under: {setting_path}")

        for date_int, spath in sorted(dated_files, key=lambda x: x[0]):
            name_map, key_map = _load_single_setting_file_idx_name_id_groups(spath)

            for name, groups in name_map.items():
                timeline[name].append((date_int, set(groups)))

            for key, groups in key_map.items():   # key は (name, nurse_id)
                timeline[key].append((date_int, set(groups)))

        return timeline

    raise FileNotFoundError(f"setting path not found: {setting_path}")



# 4) 日付からグループを引く関数
def _resolve_groups_for_entries(date: int, entries):
    """
    entries = [(start_date, set(groups)), ...] (start_date 昇順）
    の中から、指定した date に対応する groups を返す共通ロジック。
    """
    first_date, first_groups = entries[0]

    if date < first_date:
        if BACKFILL_EARLIEST_GROUP:
            return set(first_groups)
        else:
            return set()

    chosen_groups = None
    for start_date, groups in entries:
        if start_date <= date:
            chosen_groups = groups
        else:
            break

    return set(chosen_groups) if chosen_groups is not None else set()


def get_groups_for_date(name: str, date: int, group_timeline, nurse_id: str = None):
    """
    指定した name, (optionally nurse_id), date に対して有効なグループ集合を返す。

    group_timeline は load_staff_group_timeline() の戻り値を想定。

    - nurse_id を指定した場合:
        まず (name, nurse_id) キーで timeline を探し、
        見つかればそれを優先する。
        見つからなければ name のみで探す。

    - nurse_id を指定しない場合:
        従来通り name のみで探す。
    """
    
    if nurse_id is not None:
        entries = group_timeline.get((name, nurse_id))
        if entries:
            return _resolve_groups_for_entries(date, entries)

    
    entries = group_timeline.get(name)
    if not entries:
        return set()

    return _resolve_groups_for_entries(date, entries)
