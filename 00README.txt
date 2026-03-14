#############################################################
環境構築
#############################################################
・インストール方法はなんでも良いのでdockerをインストールしてください
<!-- Dockerfileがあるディレクトリでビルド  -->
docker build -t nsp-dev . 
<!-- コンテナ起動 -->
docker run -dit --name nsp-dev -v "$(pwd)":/work nsp-dev
<!-- 次回から -->
docker start nsp-dev
docker exec -it nsp-dev bash
・起動時にrootユーザで入ってしまうのを防ぎたい場合
例）docker exec -it --user $(id -u):$(id -g) nsp-dev bash


############################################################
各ディレクトリ紹介（詳細は各サブディレクトリのREADME.txtを見てください）
############################################################
exe/
・実行ファイル
2019-2025-data/
・2019~2025年までの手動生成勤務表の勤務データ
found-model/
・求解した自動生成勤務表の勤務データ
out/
・n-gram分析で出力したグラフなど
scripts/
・n-gram分析に用いたスクリプト



##############################################################
基本的な使い方（各スクリプトの詳細は各サブディレクトリのREADME.txtを見てください）
###########################################################

①勤務表を6個自動生成する
for i in {1..6}; do
  timeout 3600 python nsp-solver.py 2019-2025-data/group-settings-encoding/GCU/2024_10_13/*.lp \
  -o found-model/GCU/2024-10-13/found-model$i.lp
done

②日本語から英語に変換
cd found-model
python convert_en.py GCU/2024-10-13/


③手動生成勤務表と比較する
python scripts/ngram/js_past_vs_found_freq.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ found-model/GCU/2024-10-13/ --pstart-year 2020 --pend-year 2025 --nmin 1 --nmax 5 --laplacek 1 --past-mode summary --fdate-from 20241013 --fdate-to 20241113 --past-base loo

④修正すべきn-gramを見つける
python scripts/ngram/correctp_past_vs_found_freq.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ found-model/GCU/2024-10-13/ --pstart-year 2020 --pend-year 2025 --nmin 1 --nmax 5 --topk 10 --fdate-from 20241013 --fdate-to 20241113 --print

⑤n-gram出現数を調整する
2019-2025-data/group-settings-correctp/GCU/2024_10_13/を直接編集

⑥調整したsettingファイルで再生成する（同じ条件で）
for i in {1..6}; do
  timeout 3600 python nsp-solver.py 2019-2025-data/group-settings-correctp/GCU/2024_10_13/*.lp \
  -o found-model/correctp/GCU/2024-10-13/found-model$i.lp
done

⑦日本語から英語に変換
cd found-model
python convert_en.py correctp/GCU/2024-10-13/


⑧再生した自動生成勤務表を手動生成勤務表と比較する
python scripts/ngram/js_past_vs_found_freq.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ found-model/correctp/GCU/2024-10-13/ --pstart-year 2020 --pend-year 2025 --nmin 1 --nmax 5 --laplacek 1 --past-mode summary --fdate-from 20241013 --fdate-to 20241113 --past-base loo




