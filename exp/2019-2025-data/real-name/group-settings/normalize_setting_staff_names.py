#!/usr/bin/env python3
import os
import re

# staff(Idx, "名前", "Role", "NurseId", Extra).
STAFF_RE = re.compile(
    r'^(?P<prefix>\s*staff\(\s*\d+\s*,\s*")(?P<name>[^"]*)("(?P<rest>,\s*".*)$)'
)

def normalize_name_spaces_in_line(line: str) -> str:
    m = STAFF_RE.match(line)
    if not m:
        return line  # staff 行じゃなければそのまま

    name = m.group("name")
    # 全角スペース + 半角スペースを削除
    new_name = name.replace("　", "").replace(" ", "")
    return f'{m.group("prefix")}{new_name}"{m.group("rest")}\n'


def main():
    root_dir = "."  # 今いる group-settings 以下を全部見る
    count_files = 0

    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname != "setting.lp":
                continue
            path = os.path.join(root, fname)
            print(f"[TARGET] {path}")
            count_files += 1

            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                new_lines.append(normalize_name_spaces_in_line(line.rstrip("\n")))

            with open(path, "w", encoding="utf-8") as f:
                for l in new_lines:
                    f.write(l)

    print(f"\nDone. Processed {count_files} setting.lp files.")


if __name__ == "__main__":
    main()
