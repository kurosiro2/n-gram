##########################
past_shifts_group.py
#########################
・過去の手動生成勤務表のn-gram勤務データを読み込む
・ファイルを直接修正することで期間修正可
例）GCU病棟の1-gramデータを集計する
python scripts/ngram/past_shifts_group.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ 1



##########################
found_shifts_group.py
#########################
・自動生成勤務表のn-gram勤務データを読み込む
・オプションで期間指定(ASPにおける自動生成区間base_dateに合わせるため)
・ディレクトリ直下のfound-model*.lp（1~6）を読み込みます
例）GCU病棟の1-gramデータを20241013~20241113において集計する
python scripts/ngram/found_shifts_group.py found-model/GCU/2024-10-13/ 1 --date-from 20241013 --date-to 20241113

###############################
correctp_past_vs_found_freq.py
###############################
・自動生成と手動生成をn-gram頻度分布を比較し修正すべきn-gram上位topKを表示するスクリプト
．修正すべきn-gramについてどのように集計するかは過去の発表資料を参照
例）2020~2025の手動生成勤務表（半期ごとの四分位範囲）と20241013~20241113の自動生成勤務表(6個分)を1~5gramで比較し修正すべきn-gramのtop10をconsoleに出力するスクリプト
python scripts/ngram/correctp_past_vs_found_freq.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ found-model/GCU/2024-10-13/ --pstart-year 2020 --pend-year 2025 --nmin 1 --nmax 5 --topk 10 --fdate-from 20241013 --fdate-to 20241113 --print

###############################
det_next_pairs.py
############################
・決定的後続パターンが手動生成勤務表にどの程度出現するか列挙するスクリプト
・freq det_pattern [nurse_id]が出力される
例）python scripts/ngram/det_next_pairs.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ 2 5

#####################################
js_past_vs_found_freq.py 
#####################################
・自動生成と手動生成をn-gram頻度分布比較
例） 2020~2025の手動生成勤務表（半期ごとの四分位範囲）と20241013~20241113の自動生成勤務表(6個分)を1~5gramで比較しそのヒートマップを出力
python scripts/ngram/js_past_vs_found_freq.py 2019-2025-data/past-shifts/GCU.lp 2019-2025-data/group-settings/GCU/ found-model/GCU/2024-10-13/ --pstart-year 2020 --pend-year 2025 --nmin 1 --nmax 5 --laplacek 1 --past-mode full --fdate-from 20241013 --fdate-to 20241113 --past-base full
＊＊オプションについて＊＊
--past-mode ｛full,summary｝
・fullは手動生成の半期ごとのjsdをすべて表示
・summaryは第一四分位数から第三四分位数としてまとめて表示
--past-base {full,loo}
・looは比較する半期を比較先つまり(2020~2025の手動生成勤務表)から除外（デフォルト）
・fullは含める
--laplacek
・円滑化定数（デフォルト＝１）
--box-plot
・箱ひげ図で表示（卒論発表資料の形式）

#########################################
js_group_freq.py
js_group_pnext.py
####################################
・卒論発表資料の補助資料p20を出力するスクリプト

#######################################
js_ward_freq.py
js_ward_pnext.py
###############################
．卒論発表資料の補助資料p19を出力するスクリプト

####################################
js_period_freq.py
#################################
・卒論発表資料の補助資料p21を出力するスクリプト


########################################
who_in_which_group.py
#################################
・どの看護師がどの看護師グループに所属するか調べるスクリプト