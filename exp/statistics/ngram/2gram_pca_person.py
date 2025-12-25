#!/usr/bin/env python3
import sys
import os
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 親ディレクトリを import パスに追加して data_loader を読めるようにする
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import data_loader  # load_past_shifts, load_staff_group_timeline, get_groups_for_date


def compute_unigram_bigram(seq):
    """1セグメント分のシフト列 -> (uni_counter, bi_counter)"""
    uni = Counter()
    bi  = Counter()
    if not seq:
        return uni, bi

    for _, s in seq:
        uni[(s,)] += 1
    for i in range(len(seq) - 1):
        s1 = seq[i][1]
        s2 = seq[i + 1][1]
        bi[(s1, s2)] += 1
    return uni, bi


def build_segments_with_timeline(seqs, group_timeline):
    """
    タイムラインに基づいて「看護師×期間」セグメントを作る。

    入力:
      - seqs: {(nid, name): [(date, shift), ...]}
      - group_timeline: { name: [(start_date, set(groups)), ...] }

    出力:
      - segments: list of dict
          {
            "unit_id":  "Name@0" など,
            "name":     名前,
            "groups":   set(groups)  ※"All" は除去済み。空なら {"Unknown"}。
            "seq":      [(date, shift), ...],
          }
    """
    from data_loader import get_groups_for_date

    segments = []

    # 名前順＋id順で安定させる
    for (nid, name), seq in sorted(seqs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if not seq:
            continue

        seq = sorted(seq, key=lambda t: t[0])  # 念のため日付順

        seg_idx      = 0
        curr_groups  = None
        curr_seq     = []

        for date, shift in seq:
            gset = get_groups_for_date(name, date, group_timeline)
            gset = set(gset) if gset else set()

            # "All" を削除し、残りが空なら Unknown
            if "All" in gset:
                gset.remove("All")
            if not gset:
                gset = {"Unknown"}

            if curr_groups is None:
                curr_groups = gset
                curr_seq    = [(date, shift)]
                continue

            if gset == curr_groups:
                curr_seq.append((date, shift))
            else:
                # グループ変化 → ここまでを1セグメントとして確定
                unit_id = f"{name}@{seg_idx}"
                segments.append({
                    "unit_id": unit_id,
                    "name":    name,
                    "groups":  curr_groups,
                    "seq":     curr_seq,
                })
                seg_idx += 1

                curr_groups = gset
                curr_seq    = [(date, shift)]

        # 最後のセグメント
        if curr_seq:
            unit_id = f"{name}@{seg_idx}"
            segments.append({
                "unit_id": unit_id,
                "name":    name,
                "groups":  curr_groups,
                "seq":     curr_seq,
            })

    return segments


def main():
    if len(sys.argv) < 3:
        print("python 2gram_top5_pca.py [shift_file] [setting_path]")
        sys.exit(1)

    shift_file   = sys.argv[1]
    setting_path = sys.argv[2]

    # ---------- 1) データ読み込み ----------
    seqs = data_loader.load_past_shifts(shift_file)
    group_timeline = data_loader.load_staff_group_timeline(setting_path)

    print(f"# past_shifts に出てくる (id,name) 組数: {len(seqs)}", file=sys.stderr)

    # ---------- 2) タイムラインに基づくセグメント化 ----------
    segments = build_segments_with_timeline(seqs, group_timeline)
    print(f"# 生成されたセグメント数: {len(segments)}", file=sys.stderr)

    # 名前ごとのセグメント数をデバッグ表示
    seg_count_by_name = Counter(seg["name"] for seg in segments)
    print("\n===== DEBUG: segment count per nurse =====", file=sys.stderr)
    for name, cnt in sorted(seg_count_by_name.items(), key=lambda x: x[0]):
        print(f"  {name}: {cnt} segments", file=sys.stderr)

    # ---------- 3) 各セグメントについて 2-gram スコア計算 ----------
    rows = []               # PCA 用 DataFrame 行
    unit_to_name   = {}     # unit_id -> name
    unit_to_groups = {}     # unit_id -> set(groups)

    print("\n===== DEBUG: Top 2-gram Scores per Segment =====", file=sys.stderr)

    skipped_no_bigram = 0

    for seg in segments:
        unit_id = seg["unit_id"]
        name    = seg["name"]
        groups  = seg["groups"]
        seq     = seg["seq"]

        uni, bi = compute_unigram_bigram(seq)
        total_bi = sum(bi.values())
        if total_bi == 0:
            skipped_no_bigram += 1
            continue

        unit_to_name[unit_id]   = name
        unit_to_groups[unit_id] = groups if groups else {"Unknown"}

        # prev -> 出現する2-gram総数（P(next|prev) 用）
        total_out = defaultdict(int)
        for (s1, s2), c in bi.items():
            total_out[s1] += c

        person_scores = []

        for (s1, s2), c in bi.items():
            base = total_out[s1]
            if base <= 0:
                continue

            prob       = c / base
            freq_share = c / total_bi
            score      = prob * freq_share

            rows.append({
                "unit_id": unit_id,
                "pair":    f"{s1}->{s2}",
                "score":   score,
            })

            person_scores.append((f"{s1}->{s2}", score))

        # セグメントごとの top3 デバッグ
        if person_scores:
            person_scores.sort(key=lambda x: x[1], reverse=True)
            top3 = person_scores[:3]
            print(
                f"\n# Segment: {unit_id} (name={name}, groups={sorted(groups)})",
                file=sys.stderr,
            )
            for pair, sc in top3:
                print(f"   {pair:6s}  score={sc:.6f}", file=sys.stderr)

    if skipped_no_bigram > 0:
        print(f"# 2-gram が1つも無くてスキップされたセグメント数: {skipped_no_bigram}", file=sys.stderr)

    if not rows:
        print("ERROR: PCA 用の2-gram行が 0 件です。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    feature = df.pivot_table(
        index="unit_id",
        columns="pair",
        values="score",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    unit_ids = list(feature.index)
    print(f"# PCA に使うセグメント（unit_id）数: {len(unit_ids)}", file=sys.stderr)

    # ---------- 4) グループ集合を集約 ----------
    all_groups = set()
    for uid in unit_ids:
        gset = unit_to_groups.get(uid, {"Unknown"})
        if not gset:
            gset = {"Unknown"}
        all_groups |= set(gset)
        unit_to_groups[uid] = gset

    all_groups = sorted(all_groups)
    print("# group ごとのセグメント点数:", file=sys.stderr)
    for g in all_groups:
        cnt = sum(1 for uid in unit_ids if g in unit_to_groups.get(uid, {}))
        print(f"  {g}: {cnt}", file=sys.stderr)
    print(f"  total segments: {len(unit_ids)}", file=sys.stderr)

    # ---------- 5) ALL の μ を計算し top5 抽出 ----------
    print("\n===== Selecting Top 5 2-grams (ALL-unit mean μ) =====")

    all_means = feature.mean(axis=0)
    top5_pairs = all_means.sort_values(ascending=False).head(5).index.tolist()

    print("Top5 pairs:")
    for p in top5_pairs:
        print(f"  {p}: μ={all_means[p]:.6f}")

    # ---------- 6) 各セグメントについて 5 次元ベクトル生成 ----------
    X = []
    for uid in unit_ids:
        vec = [feature.loc[uid, pair] if pair in feature.columns else 0.0
               for pair in top5_pairs]
        X.append(vec)
    X = np.array(X)

    # ---------- 7) PCA ----------
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    print("\nExplained variance ratio:", pca.explained_variance_ratio_, file=sys.stderr)

    print("\n===== PCA Eigen Info =====", file=sys.stderr)
    print("Eigenvalues:", pca.explained_variance_, file=sys.stderr)
    print("Eigenvalue ratio:", pca.explained_variance_ratio_, file=sys.stderr)
    print("Eigenvectors (PC1, PC2):", file=sys.stderr)
    for i, comp in enumerate(pca.components_):
        print(f"  PC{i+1}: {comp}", file=sys.stderr)

    # ---------- 8) 0 中心レンジ ----------
    max_abs_pc1 = np.max(np.abs(X_2d[:, 0]))
    max_abs_pc2 = np.max(np.abs(X_2d[:, 1]))
    radius = max(max_abs_pc1, max_abs_pc2) * 1.05  # 5% 余白

    x_min, x_max = -radius, radius
    y_min, y_max = -radius, radius

    print(f"# axis radius (PC1/PC2): {radius:.4f}", file=sys.stderr)

    # ---------- 9) プロット ----------
    def sanitize(s: str) -> str:
        return s.replace(" ", "_").replace("/", "_").replace("+", "_")

    # 9-1) グループ別 PNG（Heads, Mid-levels, Night, ...）
    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan",
    ]
    color_map = {g: colors[i % len(colors)] for i, g in enumerate(all_groups)}

    for g in all_groups:
        mask = np.array([g in unit_to_groups.get(uid, {}) for uid in unit_ids])
        if not mask.any():
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=40, alpha=0.8, c=color_map[g], label=g,
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Nurse Patterns PCA (Top5 2-grams, timeline segments) - {g}")
        plt.legend()
        plt.grid(True)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()

        fname = f"pca_top5_{sanitize(g)}.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        print(f"# saved: {fname}", file=sys.stderr)

    # 9-2) ALL_by_group（複合グループを 1 カテゴリとして扱う）
    combined_labels = []
    for uid in unit_ids:
        gset = unit_to_groups.get(uid, {"Unknown"})
        if not gset:
            gset = {"Unknown"}
        label = "+".join(sorted(gset))   # 例: "Night+Seniors"
        combined_labels.append(label)
    combined_labels = np.array(combined_labels)

    unique_combined = sorted(set(combined_labels))
    print("\n# combined-group ごとのセグメント点数:", file=sys.stderr)
    for cg in unique_combined:
        cnt = np.sum(combined_labels == cg)
        print(f"  {cg}: {cnt}", file=sys.stderr)

    # 複合グループ用の色マップ（単純に順番で割り当て）
    colors2 = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan",
    ]
    color_map_combined = {cg: colors2[i % len(colors2)]
                          for i, cg in enumerate(unique_combined)}

    plt.figure(figsize=(8, 6))
    for cg in unique_combined:
        mask = (combined_labels == cg)
        if not mask.any():
            continue
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=40, alpha=0.8, c=color_map_combined[cg], label=cg,
        )

        # 9-3) ALL_by_group（Unknown 除外）
    unique_no_unknown = [cg for cg in unique_combined if "Unknown" not in cg]

    if unique_no_unknown:
        plt.figure(figsize=(8, 6))
        for cg in unique_no_unknown:
            mask = (combined_labels == cg)
            if not mask.any():
                continue
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                s=40, alpha=0.8, c=color_map_combined[cg], label=cg,
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Nurse Patterns PCA (Top5 2-grams, timeline segments)\nALL by GroupColor (Unknown removed)")
        plt.legend()
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()

        fname_no_unknown = "pca_top5_ALL_by_group_no_unknown.png"
        plt.savefig(fname_no_unknown, dpi=300)
        plt.close()
        print(f"# saved: {fname_no_unknown}", file=sys.stderr)
    else:
        print("# (no groups remain after removing Unknown)", file=sys.stderr)


if __name__ == "__main__":
    main()
