#!/bin/bash

set -e

echo "===== ① solverで勤務表生成 ====="

for i in {1..6}; do
  timeout 3600 python ../nsp-solver.py \
  ../2019-2025-data/group-settings-encoding/GCU/2024_10_13/*.lp \
  -o ../found-model/GCU/2024-10-13/found-model$i.lp
done


echo "===== ② 日本語 → 英語変換 ====="

cd ../found-model
python convert_en.py GCU/2024-10-13/
cd ../exe


echo "===== ③ 手動勤務表と比較 ====="

python ../scripts/ngram/js_past_vs_found_freq.py \
../2019-2025-data/past-shifts/GCU.lp \
../2019-2025-data/group-settings/GCU/ \
../found-model/GCU/2024-10-13/ \
--pstart-year 2020 \
--pend-year 2025 \
--nmin 1 \
--nmax 5 \
--laplacek 1 \
--past-mode summary \
--fdate-from 20241013 \
--fdate-to 20241113 \
--past-base loo


echo "===== ④ 修正すべき n-gram を抽出 ====="

python ../scripts/ngram/correctp_past_vs_found_freq.py \
../2019-2025-data/past-shifts/GCU.lp \
../2019-2025-data/group-settings/GCU/ \
../found-model/GCU/2024-10-13/ \
--pstart-year 2020 \
--pend-year 2025 \
--nmin 1 \
--nmax 5 \
--topk 10 \
--fdate-from 20241013 \
--fdate-to 20241113 \
--print


echo "===== ⑤ settingファイルを手動修正してください ====="
echo "../2019-2025-data/group-settings-correctp/GCU/2024_10_13/"
read -p "修正後 Enter を押してください"


echo "===== ⑥ 修正版で再生成 ====="

for i in {1..6}; do
  timeout 3600 python ../nsp-solver.py \
  ../2019-2025-data/group-settings-correctp/GCU/2024_10_13/*.lp \
  -o ../found-model/correctp/GCU/2024-10-13/found-model$i.lp
done


echo "===== ⑦ 日本語 → 英語変換 ====="

cd ../found-model
python convert_en.py correctp/GCU/2024-10-13/
cd ../exe


echo "===== ⑧ 再比較 ====="

python ../scripts/ngram/js_past_vs_found_freq.py \
../2019-2025-data/past-shifts/GCU.lp \
../2019-2025-data/group-settings/GCU/ \
../found-model/correctp/GCU/2024-10-13/ \
--pstart-year 2020 \
--pend-year 2025 \
--nmin 1 \
--nmax 5 \
--laplacek 1 \
--past-mode summary \
--fdate-from 20241013 \
--fdate-to 20241113 \
--past-base loo


echo "===== 完了 ====="
