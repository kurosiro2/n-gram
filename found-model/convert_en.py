#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import re
from pathlib import Path

def load_i18n(i18n_path: Path):
    spec = importlib.util.spec_from_file_location("i18n_mod", str(i18n_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load: {i18n_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# 1行 fact の軽いパーサ（末尾コメントは許容）
LINE_RE = re.compile(r'^\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*\.\s*(%.*)?$')
QUOTED_RE = re.compile(r'"([^"\n]*)"')

def split_args(argstr: str) -> list[str]:
    args, buf = [], []
    in_str = False
    i = 0
    while i < len(argstr):
        ch = argstr[i]
        if ch == '"':
            in_str = not in_str
            buf.append(ch)
        elif ch == ',' and not in_str:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        args.append("".join(buf).strip())
    return args

def is_quoted(s: str) -> bool:
    return len(s) >= 2 and s[0] == '"' and s[-1] == '"'

def unquote(s: str) -> str:
    return s[1:-1]

def quote(s: str) -> str:
    return f'"{s}"'

def map_general(i18n, s: str) -> str:
    """曜日以外（基本は shift/group/job/target_type/shift_def）"""
    # vertical target type
    try:
        return i18n.map_vertical_target_type(s, "en")
    except Exception:
        pass

    # group
    try:
        return i18n.map_group(s, "en")
    except Exception:
        pass

    # job
    try:
        return i18n.map_job(s, "en")
    except Exception:
        pass

    # shift (single)
    try:
        return i18n.map_shift(s, "en")
    except Exception:
        pass

    # shift_def (composite)
    if any(ch in s for ch in [",", "，", "+", "＋"]):
        try:
            return i18n.map_shift_def(s, "en")
        except Exception:
            pass

    return s

def map_dweek_only(i18n, s: str) -> str:
    try:
        return i18n.map_dweek(s, "en")
    except Exception:
        return s

def convert_line(i18n, line: str) -> str:
    m = LINE_RE.match(line.rstrip("\n"))
    if not m:
        # 形式が違う行は "..." を一般変換だけやる（安全側）
        def repl(mm: re.Match) -> str:
            inner = mm.group(1)
            return quote(map_general(i18n, inner))
        return QUOTED_RE.sub(repl, line)

    pred, argstr, comment = m.group(1), m.group(2), m.group(3)
    args = split_args(argstr)

    # out_date(Date,"日") の第2引数だけ dweek 扱い
    if pred == "out_date" and len(args) >= 2 and is_quoted(args[1]):
        inner = unquote(args[1])
        args[1] = quote(map_dweek_only(i18n, inner))

    # それ以外の "..." は general を適用（複数引数にも効く）
    for i, a in enumerate(args):
        if is_quoted(a):
            args[i] = quote(map_general(i18n, unquote(a)))

    new_line = f"{pred}({', '.join(args)})."
    if comment:
        new_line += f" {comment}"
    return new_line + "\n"

def convert_text(i18n, text: str) -> str:
    return "".join(convert_line(i18n, ln) for ln in text.splitlines(True))

def iter_target_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(path.rglob("found-model*.lp"))
        if found:
            return found
        return sorted(path.rglob("*.lp"))
    raise FileNotFoundError(str(path))

def main():
    ap = argparse.ArgumentParser(
        description="Convert found-model.lp (or a directory) ja->en based on i18n.py, in-place."
    )
    ap.add_argument("target", help="file or directory")
    ap.add_argument("--i18n", default="i18n.py", help="path to i18n.py (default: ./i18n.py)")
    ap.add_argument("--dry-run", action="store_true", help="do not write, only report")
    args = ap.parse_args()

    target = Path(args.target)
    i18n_path = Path(args.i18n)
    if not i18n_path.exists():
        raise FileNotFoundError(f"i18n.py not found: {i18n_path}")

    i18n = load_i18n(i18n_path)

    files = iter_target_files(target)
    if not files:
        print("No target files found.")
        return

    changed = 0
    for f in files:
        org = f.read_text(encoding="utf-8")
        new = convert_text(i18n, org)

        if new != org:
            changed += 1
            if args.dry_run:
                print(f"[DRY] would update: {f}")
            else:
                f.write_text(new, encoding="utf-8")
                print(f"updated: {f}")
        else:
            print(f"unchanged: {f}")

    print(f"done. files={len(files)} changed={changed}")

if __name__ == "__main__":
    main()
