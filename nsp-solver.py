#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw model NSP solver (no table / no Excel)

- clingo を実行して NSP を解く
- モデル(Answerの次行)をそのまま表示/保存する
- -m を付けると "v <literal>" 形式でリテラルを列挙
- -o FILE を付けると、モデルを FILE に .lp 形式で書き出す
  例: header("Answer: 1, Cost: 1 2 3, Elapsed: 12.3s").
      fact(...).
      fact(...).
"""

import sys
import os
import signal
import subprocess
import re
import time
from optparse import (OptionParser, BadOptionError, AmbiguousOptionError)

logs = []

# last captured (best/last) model info
last_answer_no = None
last_model_line = None          # raw model line (with # delimiter)
last_cost_line = None           # raw "Optimization: ..." line
last_elapsed = None             # seconds (int)


def log(s: str, verb: bool = True):
    if verb:
        logs.append(s)
        print(s)


def filter_shift_data_literals(literals):
    # shift_data は数万件あり、後処理には使わないので落とす（元コード踏襲）
    return [lit for lit in literals if lit and not lit.startswith("shift_data")]


def model_line_to_literals(model_line: str):
    # clingo --out-ifs=# により # 区切りで来る想定
    lits = model_line.split('#')
    return filter_shift_data_literals(lits)


def write_found_model_lp(path: str, answer_no, cost_line, elapsed_sec, model_line: str):
    # header の中身を作る
    parts = []
    if answer_no is not None:
        parts.append(f"Answer: {answer_no}")
    if cost_line is not None:
        # "Optimization: ..." から数値部分だけ抜く
        m = re.match(r"^Optimization:\s*(.*)\s*$", cost_line)
        if m:
            parts.append(f"Cost: {m.group(1).replace('#', ' ')}")
        else:
            parts.append(f"Cost: {cost_line.replace('#', ' ')}")
    if elapsed_sec is not None:
        parts.append(f"Elapsed: {elapsed_sec:.1f}s" if isinstance(elapsed_sec, float) else f"Elapsed: {elapsed_sec}s")

    header = ", ".join(parts) if parts else "Model"

    lits = model_line_to_literals(model_line)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'header("{header}").\n')
        for lit in lits:
            # 既に "." で終わってたらそのまま、そうでなければ "." を付ける
            lit2 = lit if lit.endswith('.') else (lit + ".")
            f.write(lit2 + "\n")


def print_model_lines_v(model_line: str):
    for lit in model_line_to_literals(model_line):
        print("v " + lit)


def parse_args(argv):
    myusage = "%prog [-o OUTPUT] INPUT1 INPUT2 ...\n" + \
              "  Solve given NSP instance. Clingo options can be specified as last args"
    parser = PassThroughOptionParser(usage=myusage)
    parser.add_option("-o", dest="out_file", default=None,
                      help="write found model as .lp file", metavar="FILE")
    parser.add_option("-m", action="store_true", dest="show_model", default=False,
                      help="print model literals (v ...)")
    parser.add_option("-q", action="store_false", dest="verb", default=True,
                      help="quiet mode")
    parser.add_option("--no-excel", action="store_false", dest="excel_output", default=True,
                      help="(ignored) no Excel output")

    (opts, args) = parser.parse_args(argv)
    args = list(args)
    args.pop(0)  # remove script name

    if len(args) == 0:
        parser.print_help()
        print()
        print("Error: at least one input file required")
        sys.exit(1)

    return opts, args


def split_files_and_clingo_opts(args):
    # 最初に '-' で始まるやつから clingo オプションとみなす
    clingo_opt_idx = -1
    for i, e in enumerate(args):
        if re.match(r'^-', e):
            clingo_opt_idx = i
            break

    # モデル行を1行にしやすいように out-ifs を付ける
    clingo_opts = ["--out-ifs=#", "--stats"]
    if clingo_opt_idx >= 0:
        files = args[:clingo_opt_idx]
        clingo_opts += args[clingo_opt_idx:]
    else:
        files = args

    return files, clingo_opts


def maybe_write_outfile(opts):
    # いま保持している last_* を使って書き出す
    global last_answer_no, last_cost_line, last_elapsed, last_model_line
    if opts.out_file and last_model_line:
        write_found_model_lp(
            opts.out_file,
            last_answer_no,
            last_cost_line,
            last_elapsed,
            last_model_line
        )


def signal_handler(signum, frame):
    # kill -USR1 で、いまの last model を書き出せるようにする
    # 表/Excelはないので、-o 指定があるときだけ保存
    # （-o が無いときは何もしない）
    # NOTE: signal handler では opts に触れられないので、出力はしない
    pass


def solve(files, clingo_opts, opts):
    global last_answer_no, last_model_line, last_cost_line, last_elapsed

    host = subprocess.run(["hostname", "-f"], capture_output=True, text=True).stdout.rstrip()
    date = subprocess.run(["date"], capture_output=True, text=True).stdout.rstrip()
    log(f"c Host:    {host}", opts.verb)
    log(f"c Date:    {date}", opts.verb)

    cmd = ["clingo"] + files + clingo_opts
    log(f"c Command: {' '.join(cmd)}", opts.verb)

    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # SIGUSR1 は互換のため登録だけ（この版では何もしない）
    signal.signal(signal.SIGUSR1, signal_handler)

    next_is_model = False

    for bline in proc.stdout:
        line = bline.decode("utf-8", errors="replace").rstrip()
        elapsed = time.time() - start
        head = f"c {int(elapsed)}  "

        if re.match(r'^Answer', line):
            # Answer: 1
            m = re.match(r'^Answer:\s*(\d+)\s*$', line)
            if m:
                last_answer_no = int(m.group(1))
            last_elapsed = int(elapsed)
            next_is_model = True
            if opts.verb:
                log(f"{head} {line}", opts.verb)
            continue

        if next_is_model:
            next_is_model = False
            last_model_line = line
            last_elapsed = int(elapsed)

            # 端末出力
            if opts.show_model:
                print_model_lines_v(line)
            else:
                # 1行で MODEL: として出す（shift_data は落とした版を表示）
                filtered = "#".join(model_line_to_literals(line))
                print(f"{head} MODEL: {filtered.replace('#', ' ')}")

            continue

        # Optimization を拾って保存（このタイミングで -o に書くと “Cost” 入りになる）
        if re.match(r'^Optimization:', line):
            last_cost_line = line
            last_elapsed = int(elapsed)
            if opts.verb:
                log(f"{head} {line.replace('#',' ')}", opts.verb)

            # -o があるならここで確定保存（-n 1 のケースにぴったり）
            maybe_write_outfile(opts)
            continue

        # SATISFIABLE / UNSAT など
        if re.match(r'^SATISFIABLE', line):
            last_elapsed = int(elapsed)
            if not opts.verb:
                print(f"{head}{line}")
            # Optimization が無い場合でも最後に保存したいのでここでも試す
            maybe_write_outfile(opts)
            continue

        if re.match(r'^UNSATISFIABLE', line):
            last_elapsed = int(elapsed)
            if not opts.verb:
                print(f"{head}{line}")
            continue

        if re.match(r'^OPTIMUM FOUND', line):
            last_elapsed = int(elapsed)
            if not opts.verb:
                print(f"{head}{line}")
            # 最終確定として保存
            maybe_write_outfile(opts)
            continue

        # その他ログ
        if opts.verb:
            log(f"{head} {line.replace('#', ' ')}", opts.verb)

    proc.wait()

    # 万一 Optimization が出ないパターンでも最後に保存
    maybe_write_outfile(opts)

    return proc.returncode


def main():
    opts, args = parse_args(sys.argv)
    files, clingo_opts = split_files_and_clingo_opts(args)
    rc = solve(files, clingo_opts, opts)
    sys.exit(rc)


# https://stackoverflow.com/questions/1885161/how-can-i-get-optparses-optionparser-to-ignore-invalid-options
class PassThroughOptionParser(OptionParser):
    """
    Unknown option pass-through implementation of OptionParser.
    """
    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                OptionParser._process_args(self, largs, rargs, values)
            except (BadOptionError, AmbiguousOptionError) as e:
                largs.append(e.opt_str)


if __name__ == "__main__":
    main()
