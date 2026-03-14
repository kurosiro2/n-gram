import os
import sys
import argparse
import glob
import re
import math
import io
import contextlib
from datetime import date as _date
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet

# -------------------------------------------------------------
# import path
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date
import foundmodel_data_loader as found_loader  # ★found 側ローダ（ignored-ids対応 + staff(...)）

# -------------------------------------------------------------
# constants
# -------------------------------------------------------------
VALID_SHIFTS = {"D", "LD", "EM", "LM", "SE", "SN", "E", "N", "WR", "PH"}
UNKNOWN_GROUP = "__UNKNOWN__"

PersonKey = Tuple[int, str]
SeqDict = Dict[PersonKey, List[Tuple[int, str]]]
Gram = Tuple[str, ...]


# =============================================================
# helpers
# =============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_seq(seq: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(d, s) for (d, s) in seq if s in VALID_SHIFTS]


def within_range(d: int, start: int, end: int) -> bool:
    return start <= d <= end


def get_groups_for_day(name: str, nid: int, date: int, timeline: dict) -> Set[str]:
    gs = data_loader.get_groups_for_date(name, date, timeline, nurse_id=nid)
    if not gs:
        return set()
    if isinstance(gs, (set, list, tuple)):
        return {str(x) for x in gs}
    return {str(gs)}


def group_set_contains(group_set: FrozenSet[str], target_group: str) -> bool:
    """MANUAL: heads_name exact match (case-insensitive) => Heads."""
    tg = (target_group or "").strip().lower()
    if not tg:
        return False
    return any((g or "").strip().lower() == tg for g in group_set)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# =============================================================
# MANUAL: segment builder (group-timeline aware)
# =============================================================
class Segment:
    __slots__ = ("groups", "seq")

    def __init__(self, groups: FrozenSet[str]):
        self.groups = groups
        self.seq: List[Tuple[int, str]] = []


def build_segments_for_person(
    seq: List[Tuple[int, str]],
    name: str,
    nid: int,
    timeline: dict,
) -> List["Segment"]:
    """
    Split segments whenever the group-set changes.
    Unknown => UNKNOWN_GROUP (goes to NonHeads later)
    """
    seq = normalize_seq(seq)
    if not seq:
        return []
    seq.sort(key=lambda t: t[0])

    segs: List[Segment] = []
    cur: Optional[Segment] = None

    for d, s in seq:
        gs = get_groups_for_day(name, nid, d, timeline)
        if not gs:
            gset = frozenset([UNKNOWN_GROUP])
        else:
            gset = frozenset(sorted(gs, key=lambda x: x.lower()))

        if cur is None or cur.groups != gset:
            cur = Segment(gset)
            segs.append(cur)

        cur.seq.append((d, s))

    return segs


def prebuild_all_segments(
    seqs: SeqDict, timeline: dict
) -> Dict[PersonKey, List[Segment]]:
    out: Dict[PersonKey, List[Segment]] = {}
    for (nid, name), seq in seqs.items():
        out[(nid, name)] = build_segments_for_person(seq, name, nid, timeline)
    return out


def count_ngrams_heads_nonheads_in_range(
    segs_by_person: Dict[PersonKey, List[Segment]],
    n: int,
    heads_name: str,
    date_start: int,
    date_end: int,
) -> Tuple[Counter, Counter]:
    """
    Count n-grams in [date_start, date_end] for Heads / NonHeads(+Unknown).
    Do not cross segment boundaries.
    """
    heads = Counter()
    nonheads = Counter()

    for _, segs in segs_by_person.items():
        for seg in segs:
            is_heads = group_set_contains(seg.groups, heads_name)

            sseq = [
                (d, s) for (d, s) in seg.seq if within_range(d, date_start, date_end)
            ]
            if len(sseq) < n:
                continue

            shifts = [s for _, s in sseq]
            for i in range(len(shifts) - n + 1):
                gram = tuple(shifts[i : i + n])
                if any(x not in VALID_SHIFTS for x in gram):
                    continue
                if is_heads:
                    heads[gram] += 1
                else:
                    nonheads[gram] += 1

    return heads, nonheads


# =============================================================
# AUTO(found): date filtering + split by consecutive days (★追加)
# =============================================================
DATE8_RE = re.compile(r"^\d{8}$")  # YYYYMMDD


def parse_yyyymmdd(s: Optional[str]) -> Optional[int]:
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
    seq_sorted: List[Tuple[int, str]],
    date_from: Optional[int],
    date_to: Optional[int],
) -> List[List[str]]:
    """
    seq_sorted: [(YYYYMMDD, shift), ...] 日付昇順前提
    1) [date_from, date_to] でフィルタ（Noneは無制限）
    2) 連続日付ごとに分割（ギャップがあれば別セグメント）
    返り値: [ [shift, shift, ...], [shift, ...], ... ]
    """
    filtered: List[Tuple[int, str]] = []
    for d, sh in seq_sorted:
        if date_from is not None and d < date_from:
            continue
        if date_to is not None and d > date_to:
            continue
        filtered.append((d, sh))

    if not filtered:
        return []

    segments: List[List[str]] = []
    cur: List[str] = [filtered[0][1]]
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
# AUTO(found): load + ngram counts (no timeline)
# =============================================================
def is_head_group_found(g: str) -> bool:
    if not g:
        return False
    gl = g.lower()
    return ("head" in gl) or ("師長" in g) or ("主任" in g)


def bucket_found(groups: Set[str], heads_name: str) -> str:
    """AUTO: heads_name exact match OR contains head-like keywords => Heads."""
    hn = (heads_name or "").strip().lower()
    for g in groups:
        if (g or "").strip().lower() == hn:
            return "Heads"
        if is_head_group_found(g):
            return "Heads"
    return "NonHeads"


def _pick_lp_files_in_dir(d: str) -> List[str]:
    fs = sorted(glob.glob(os.path.join(d, "found-model*.lp")))
    if fs:
        return fs
    return sorted(glob.glob(os.path.join(d, "*.lp")))


def collect_found_entries(found_path: str) -> List[Tuple[str, List[str]]]:
    """
    Returns [(label, [lp_files...]), ...]
    Same behavior as your heatmap script.
    """
    if os.path.isfile(found_path):
        label = os.path.basename(os.path.normpath(found_path))
        return [(label, [found_path])]

    if not os.path.isdir(found_path):
        return []

    found_path = os.path.normpath(found_path)

    direct = _pick_lp_files_in_dir(found_path)
    if direct:
        label = os.path.basename(found_path)
        return [(label, direct)]

    entries: List[Tuple[str, List[str]]] = []
    try:
        for name in sorted(os.listdir(found_path)):
            sub = os.path.join(found_path, name)
            if not os.path.isdir(sub):
                continue
            fs = _pick_lp_files_in_dir(sub)
            if fs:
                entries.append((os.path.basename(sub), fs))
    except OSError:
        pass

    if entries:
        return entries

    cand = sorted(
        glob.glob(os.path.join(found_path, "**", "found-model*.lp"), recursive=True)
    )
    if not cand:
        cand = sorted(glob.glob(os.path.join(found_path, "**", "*.lp"), recursive=True))
    if not cand:
        return []

    by_dir: Dict[str, List[str]] = defaultdict(list)
    for fp in cand:
        by_dir[os.path.dirname(fp)].append(fp)

    out: List[Tuple[str, List[str]]] = []
    for d in sorted(by_dir.keys()):
        files = sorted(by_dir[d])
        out.append((os.path.basename(d), files))
    return out


# -----------------------------
# found-model caching + single log
# -----------------------------
_FOUND_CACHE: Dict[
    str, Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]
] = {}
_PRINTED_IGNORE_LOG = False


def _suppress_stdout_call(fn, *args, **kwargs):
    """fn(...) の stdout を抑制して実行して戻り値だけ返す"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def load_found_model(
    path: str,
) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Set[str]]]:
    """
    foundmodel_data_loader を使って found-model.lp を読む（ignored-ids 対応）

    ★今回:
      - loader の verbose print をスクリプト側で抑制（nごとにログが出るのを防ぐ）
      - ignored-ids 適用情報は “全体で1回だけ” こちらが要約ログを出す
      - さらに結果をキャッシュして、同じファイルを n ごとに読み直さない
    """
    global _PRINTED_IGNORE_LOG

    p = os.path.abspath(path)
    if p in _FOUND_CACHE:
        return _FOUND_CACHE[p]

    # 1) ignore無しで staff_info を作って ignored-ids 解決結果を取得（この関数は基本printしない）
    _s0, _g0, info0 = _suppress_stdout_call(
        found_loader.load_found_model_ex, p, apply_ignore_ids=False
    )
    res = found_loader.get_ignored_resolution(p, info0, warn_unresolved=False)

    # 2) ignore適用版をロード（loaderのprintは抑制）
    seqs_by_staff, groups_by_staff = _suppress_stdout_call(
        found_loader.load_found_model, p, apply_ignore_ids=True
    )

    # 3) ignored-ids の要約ログを「全体で1回だけ」出す
    if (not _PRINTED_IGNORE_LOG) and res.get("tokens"):
        ignore_file = res.get("ignore_file", "")
        ward = res.get("ward_name", "")
        tokens = res.get("tokens", [])
        resolved_pairs = res.get("resolved_pairs", [])
        ignore_sids = res.get("ignore_staff_ids", [])
        unresolved = res.get("unresolved_tokens", [])

        resolved_str = "(none)"
        if resolved_pairs:
            resolved_str = ", ".join(
                [f"{r.get('token','')}->{r.get('staff_id','')}" for r in resolved_pairs]
            )

        print(
            "# [found_loader] ignored-ids applied: "
            f"ward={ward} file={ignore_file} "
            f"tokens={tokens} resolved={resolved_str} "
            f"ignore_staff_ids={ignore_sids} unresolved={unresolved}"
        )
        _PRINTED_IGNORE_LOG = True

    _FOUND_CACHE[p] = (seqs_by_staff, groups_by_staff)
    return seqs_by_staff, groups_by_staff


def count_ngrams_found_heads_nonheads(
    found_files: List[str],
    n: int,
    heads_name: str,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
) -> Tuple[Counter, Counter]:
    """
    found側: found_files（= 1 entry）を合算して Heads / NonHeads を数える（タイムライン無し）

    ★追加:
      - date_from/date_to があれば、その期間内だけ集計
      - 期間フィルタで欠けた日があると n-gram が跨がないよう、連続日付ごとに分割して数える
    """
    heads = Counter()
    nonheads = Counter()

    for fp in found_files:
        seqs_by_staff, groups_by_staff = load_found_model(fp)

        for sid, seq in seqs_by_staff.items():
            if len(seq) < n:
                continue

            seq_sorted = sorted(seq, key=lambda t: t[0])
            segments = filter_and_split_by_consecutive_days(
                seq_sorted, date_from, date_to
            )

            bucket = bucket_found(groups_by_staff.get(sid, set()), heads_name)

            for shifts in segments:
                if len(shifts) < n:
                    continue
                for i in range(len(shifts) - n + 1):
                    gram = tuple(shifts[i : i + n])
                    if any(x not in VALID_SHIFTS for x in gram):
                        continue
                    if bucket == "Heads":
                        heads[gram] += 1
                    else:
                        nonheads[gram] += 1

    return heads, nonheads


# =============================================================
# periods: half-year
# =============================================================
def make_halfyear_periods(start_year: int, end_year: int) -> List[Tuple[str, int, int]]:
    periods: List[Tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}H1", y * 10000 + 101, y * 10000 + 630))
        periods.append((f"{y}H2", y * 10000 + 701, y * 10000 + 1231))
    return periods


# =============================================================
# stats
# =============================================================
def _percentile_linear(xs_sorted: List[float], p: float) -> float:
    if not xs_sorted:
        return 0.0
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    pos = (len(xs_sorted) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    w = pos - lo
    return xs_sorted[lo] * (1.0 - w) + xs_sorted[hi] * w


def q1_q3(xs: List[float]) -> Tuple[float, float]:
    xs_sorted = sorted(xs)
    return _percentile_linear(xs_sorted, 0.25), _percentile_linear(xs_sorted, 0.75)


def median(xs: List[float]) -> float:
    xs_sorted = sorted(xs)
    return _percentile_linear(xs_sorted, 0.50)


def interval_distance(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


def gram_to_str(gram: Tuple[str, ...]) -> str:
    if len(gram) == 1:
        return gram[0]
    return ",".join(gram)


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


# =============================================================
# pretty print (fixed width)
# =============================================================
def _make_dash(ascii_dash: bool) -> str:
    return "-" if ascii_dash else "–"


def _fmt_iqr(q1: float, q3: float, ascii_dash: bool) -> str:
    d = _make_dash(ascii_dash)
    return f"{q1:.6f}{d}{q3:.6f}"


def _truncate(s: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: width - 1] + "…"


def print_table(
    rows: List[Tuple[int, str, float, int, str, float, float]],
    title: str,
    gram_width: int,
    ascii_dash: bool,
    full_gram: bool,
) -> None:
    # rows: (rank, gram, base_mean_cnt, found_cnt, manual_iqr_str, found_prob, diff)
    gram_width = _clamp_int(gram_width, 8, 200)

    h_rank = "rank"
    h_gram = "gram"
    h_base = "manual(mean)"
    h_found = "auto"
    h_iqr = "manual Q1-Q3" if ascii_dash else "manual Q1–Q3"
    h_fp = "auto_p"
    h_diff = "diff"

    print(title)
    print(
        f"{h_rank:>4}  "
        f"{h_gram:<{gram_width}}  "
        f"{h_base:>10}  "
        f"{h_found:>8}  "
        f"{h_iqr:<20}  "
        f"{h_fp:>10}  "
        f"{h_diff:>10}"
    )

    for rank, gram, base_mean_cnt, found_cnt, iqr, found_p, diff in rows:
        gram_show = gram if full_gram else _truncate(gram, gram_width)
        print(
            f"{rank:>4}  "
            f"{gram_show:<{gram_width}}  "
            f"{base_mean_cnt:>10.3f}  "
            f"{found_cnt:>8}  "
            f"{iqr:<20}  "
            f"{found_p:>10.6f}  "
            f"{diff:>10.6f}"
        )


# =============================================================
# main logic
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("past_shifts", help="past-shifts *.lp (ward file)")
    ap.add_argument("group_settings", help="group-settings dir (ward/)")
    ap.add_argument("found_path", help="found dir OR found-model.lp")
    ap.add_argument("--pstart-year", type=int, default=2019)
    ap.add_argument("--pend-year", type=int, default=2025)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--heads-name", default="Heads")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--outdir", default="out/gram_fix_candidates_freqdist")
    ap.add_argument(
        "--only-nonheads", action="store_true", help="output only NonHeads(+Unknown)"
    )
    ap.add_argument(
        "--print", dest="do_print", action="store_true", help="print topK to stdout"
    )

    # ---- print formatting options
    ap.add_argument(
        "--print-gram-width",
        type=int,
        default=28,
        help="stdout gram column width (ignored if --print-full-gram)",
    )
    ap.add_argument(
        "--print-full-gram",
        action="store_true",
        help="do not truncate gram in stdout (may wrap)",
    )
    ap.add_argument(
        "--print-ascii-dash",
        action="store_true",
        help="use ASCII '-' instead of '–' in stdout (CSV stays with '–')",
    )

    # ★追加: foundmodel側の期間指定
    ap.add_argument(
        "--fdate-from",
        default=None,
        help="found集計期間の開始日（YYYYMMDD, 例: 20241101）",
    )
    ap.add_argument(
        "--fdate-to",
        default=None,
        help="found集計期間の終了日（YYYYMMDD, 例: 20241130）",
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.pstart_year > args.pend_year:
        raise ValueError("--pstart-year must be <= --pend-year")
    if args.nmin <= 0 or args.nmax <= 0 or args.nmin > args.nmax:
        raise ValueError("--nmin/--nmax must satisfy 1 <= nmin <= nmax")
    if args.topk <= 0:
        raise ValueError("--topk must be >= 1")

    if not os.path.isfile(args.past_shifts):
        raise FileNotFoundError(f"past_shifts not found: {args.past_shifts}")
    if not os.path.isdir(args.group_settings):
        raise FileNotFoundError(f"group_settings not found: {args.group_settings}")

    found_entries = collect_found_entries(args.found_path)
    if not found_entries:
        raise FileNotFoundError(f"No found-model lp found under: {args.found_path}")

    # ★追加: found期間（int化）
    date_from = parse_yyyymmdd(args.fdate_from)
    date_to = parse_yyyymmdd(args.fdate_to)
    if date_from is not None and date_to is not None and date_to < date_from:
        raise ValueError(
            f"--date-to must be >= --date-from (from={date_from}, to={date_to})"
        )

    # ---- load MANUAL
    seqs = data_loader.load_past_shifts(args.past_shifts)
    timeline = data_loader.load_staff_group_timeline(args.group_settings)
    segs_by_person = prebuild_all_segments(seqs, timeline)

    # ---- periods
    periods = make_halfyear_periods(args.pstart_year, args.pend_year)
    if not periods:
        raise RuntimeError("No half-year periods generated.")

    ns = list(range(args.nmin, args.nmax + 1))

    # ---- Precompute MANUAL counts per period, per n, per bucket
    manual_counts: Dict[str, Dict[int, List[Counter]]] = {
        "Heads": {n: [] for n in ns},
        "NonHeads": {n: [] for n in ns},
    }
    manual_totals: Dict[str, Dict[int, List[int]]] = {
        "Heads": {n: [] for n in ns},
        "NonHeads": {n: [] for n in ns},
    }

    for _pkey, d1, d2 in periods:
        for n in ns:
            h_c, nh_c = count_ngrams_heads_nonheads_in_range(
                segs_by_person,
                n=n,
                heads_name=args.heads_name,
                date_start=d1,
                date_end=d2,
            )
            manual_counts["Heads"][n].append(h_c)
            manual_counts["NonHeads"][n].append(nh_c)
            manual_totals["Heads"][n].append(int(sum(h_c.values())))
            manual_totals["NonHeads"][n].append(int(sum(nh_c.values())))

    # ---- For each found entry, compute AUTO counts per n, per bucket
    for found_label, found_files in found_entries:
        auto_counts_by_n: Dict[int, Dict[str, Counter]] = {}
        auto_totals_by_n: Dict[int, Dict[str, int]] = {}
        for n in ns:
            # ★ここだけ変更: found側は期間指定 + 連続日付分割で数える
            h_f, nh_f = count_ngrams_found_heads_nonheads(
                found_files,
                n=n,
                heads_name=args.heads_name,
                date_from=date_from,
                date_to=date_to,
            )
            auto_counts_by_n[n] = {"Heads": h_f, "NonHeads": nh_f}
            auto_totals_by_n[n] = {
                "Heads": int(sum(h_f.values())),
                "NonHeads": int(sum(nh_f.values())),
            }

        # ---- produce reports for each bucket
        for bucket in ["NonHeads"] if args.only_nonheads else ["Heads", "NonHeads"]:
            for n in ns:
                # Build vocab = union of MANUAL grams across all periods + AUTO grams
                vocab: Set[Gram] = set()
                for c in manual_counts[bucket][n]:
                    vocab.update(c.keys())
                vocab.update(auto_counts_by_n[n][bucket].keys())

                out_csv = os.path.join(
                    args.outdir, f"fix_candidates_{bucket}_found-{found_label}_n{n}.csv"
                )

                if not vocab:
                    write_csv(
                        out_csv,
                        [
                            "n",
                            "gram",
                            "base_gram_mean_count",
                            "found_gram_count",
                            "manual_q1_q3",
                            "manual_median",
                            "found_prob",
                            "diff_from_iqr",
                        ],
                        [],
                    )
                    if args.do_print:
                        print(f"\n# {bucket} / FOUND={found_label} / n={n}: (no grams)")
                    print(f"# wrote: {out_csv}")
                    continue

                # Valid period count for "base mean count":
                valid_periods = sum(
                    1 for denom in manual_totals[bucket][n] if denom > 0
                )
                if valid_periods <= 0:
                    valid_periods = 1

                manual_probs_by_gram: Dict[Gram, List[float]] = {g: [] for g in vocab}
                for idx in range(len(periods)):
                    denom = manual_totals[bucket][n][idx]
                    if denom <= 0:
                        continue
                    c = manual_counts[bucket][n][idx]
                    for g in vocab:
                        manual_probs_by_gram[g].append(c.get(g, 0) / float(denom))

                # AUTO probs
                auto_total = auto_totals_by_n[n][bucket]
                auto_counter = auto_counts_by_n[n][bucket]
                if auto_total <= 0:

                    def auto_prob(g: Gram) -> float:
                        return 0.0

                else:

                    def auto_prob(g: Gram) -> float:
                        return auto_counter.get(g, 0) / float(auto_total)

                # Score each gram
                rows_scored: List[
                    Tuple[float, float, float, float, Gram, float, float, float, int]
                ] = []

                for g, probs in manual_probs_by_gram.items():
                    if not probs:
                        continue

                    q1, q3 = q1_q3(probs)
                    med = median(probs)
                    p_auto = auto_prob(g)
                    diff = interval_distance(p_auto, q1, q3)
                    span = q3 - q1

                    base_sum = 0
                    for idx in range(len(periods)):
                        if manual_totals[bucket][n][idx] <= 0:
                            continue
                        base_sum += manual_counts[bucket][n][idx].get(g, 0)
                    base_mean_cnt = base_sum / float(valid_periods)

                    found_cnt = int(auto_counter.get(g, 0))

                    rows_scored.append(
                        (diff, p_auto, med, span, g, q1, q3, base_mean_cnt, found_cnt)
                    )

                rows_scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)

                top = rows_scored[: args.topk]

                csv_rows: List[List[str]] = []
                for (
                    diff,
                    p_auto,
                    med,
                    _span,
                    g,
                    q1,
                    q3,
                    base_mean_cnt,
                    found_cnt,
                ) in top:
                    csv_rows.append(
                        [
                            str(n),
                            gram_to_str(g),
                            f"{base_mean_cnt:.6f}",
                            str(found_cnt),
                            f"{q1:.6f}–{q3:.6f}",
                            f"{med:.6f}",
                            f"{p_auto:.6f}",
                            f"{diff:.6f}",
                        ]
                    )

                write_csv(
                    out_csv,
                    [
                        "n",
                        "gram",
                        "base_gram_mean_count",
                        "found_gram_count",
                        "manual_q1_q3",
                        "manual_median",
                        "found_prob",
                        "diff_from_iqr",
                    ],
                    csv_rows,
                )

                if args.do_print:
                    dash_ascii = bool(args.print_ascii_dash)
                    gram_width = int(args.print_gram_width)

                    table_rows: List[Tuple[int, str, float, int, str, float, float]] = (
                        []
                    )
                    for i, (
                        diff,
                        p_auto,
                        _med,
                        _span,
                        g,
                        q1,
                        q3,
                        base_mean_cnt,
                        found_cnt,
                    ) in enumerate(top, start=1):
                        gram_str = gram_to_str(g)
                        iqr_str = _fmt_iqr(q1, q3, ascii_dash=dash_ascii)
                        table_rows.append(
                            (
                                i,
                                gram_str,
                                base_mean_cnt,
                                found_cnt,
                                iqr_str,
                                p_auto,
                                diff,
                            )
                        )

                    # 期間情報も一応見えるように（print時だけ）
                    if date_from is not None or date_to is not None:
                        print(
                            f"\n# [AUTO Period] {date_from or 'MIN'} .. {date_to or 'MAX'} (YYYYMMDD)"
                        )

                    print_table(
                        table_rows,
                        title=f"\n# {bucket} / FOUND={found_label} / n={n}  (top {args.topk})",
                        gram_width=gram_width,
                        ascii_dash=dash_ascii,
                        full_gram=bool(args.print_full_gram),
                    )

                print(f"# wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
