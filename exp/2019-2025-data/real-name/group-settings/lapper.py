#!/usr/bin/env python3
import re
from pathlib import Path

# このスクリプトを group-settings/ に置く想定
BASE = Path(__file__).resolve().parent

# 例:
#   2階西病棟_2025-10-12
#   2階西病棟_2025-10-12-2
#   師長勤務表_2023-12-10-4
PAT = re.compile(r'^(.+?)_(\d{4}-\d{2}-\d{2})(-\d+)?$')


def main(dry_run: bool = True):
    # 最初に元のディレクトリ一覧を固定で取っておく
    dirs = [p for p in BASE.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)

    print(f"[INFO] Found {len(dirs)} entries in {BASE}")
    for d in dirs:
        name = d.name
        m = PAT.match(name)
        if not m:
            # GCU_2024-08-18 とかも全部このパターンにマッチするので、
            # ここに来るのは「本当に特殊な名前」だけ
            print(f"[SKIP] {name} (pattern not matched)")
            continue

        ward, date, suffix = m.groups()
        date_dir_name = date + (suffix or "")

        ward_dir = BASE / ward
        new_path = ward_dir / date_dir_name

        print(f"[MOVE] {name}  ->  {ward}/{date_dir_name}")

        if dry_run:
            continue

        ward_dir.mkdir(exist_ok=True)
        # すでに同じパスがあったら危ないので一応チェック
        if new_path.exists():
            print(f"  [WARN] {new_path} already exists, skipping")
            continue

        d.rename(new_path)


if __name__ == "__main__":
    # まず dry_run=True で動きを確認
    main(dry_run=False)
    # 問題なさそうなら下を有効にして本番
    # main(dry_run=False)
