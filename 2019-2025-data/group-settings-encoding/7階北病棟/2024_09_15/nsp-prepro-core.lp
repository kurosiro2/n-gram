%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% シフトの種類
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 勤務シフト
working_shift("日";"N";"J";"S";"★";"☆";"A";"P").
% 夜勤シフト
night_working_shift("J";"S";"★";"☆").
% 週休シフト
weekly_holiday_shift("○").
% 祝日シフト
national_holiday_shift("◎"). 
% ソフト休暇シフト
soft_holiday_shift("年";"健").
% ハード休暇シフト（そもそも部署にいない，週休調整の対象）
% もし異動などで 14 日間 / ならば，週休も 4
hard_holiday_shift("特";"/"). % 「特」はここが適当？
% ビジネスシフト（そもそも部署にいない，週休調整の対象外）
business_shift("研";"病";"産";"育").
% 休日・祝日シフト
regular_holiday_shift(S) :- weekly_holiday_shift(S).
regular_holiday_shift(S) :- national_holiday_shift(S).
% 拡張休暇シフト
valid_holiday_shift(S) :- regular_holiday_shift(S).
valid_holiday_shift(S) :- soft_holiday_shift(S).
% 翌日勤務シフトがあるシフト
consecutive_working_shift(S) :-
    succ_shift(S, _), working_shift(NS) : succ_shift(S, NS).  % : = such that

% 作成するシフト表の横幅（日数）
table_width(X) :- X = { date(_,_) }.
% 週休の数
default_weekly_holidays(8).
% 祝日の数
num_national_holidays(X) :- X = { national_holiday(D) : date(D) }.

% 平日
weekday("月";"火";"水";"木";"金").
% 週末
weekend("土";"日").

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 各看護師・各シフトごとの上限・下限の算出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 看護師ごとの勤務日数の上下限
working_dates_bounds(N, LB, UB) :-
    staff(N),
    table_width(WIDTH),
    weekly_holiday_bounds(N, WH_LB, WH_UB),
    num_national_holiday_dates(N, NH),
    num_soft_holiday_dates(N, SH),      % 年，健
    num_hard_holiday_dates(N, HH),      % 幅固定：特, /
    num_business_dates(N, B),           % 幅固定：研, 病, 産
    UB = WIDTH - WH_LB - HH - B,
    LB = WIDTH - WH_UB - HH - B - NH - SH.

% 看護師ごとの週休の数
% TODO: 場合によっては設定ファイルで入力してもらったほうがよい（異動前に週休をとった回数が規定通りとは限らない）
weekly_holiday_bounds(N, LB, UB) :-
    staff(N), table_width(WIDTH), 
    num_hard_holiday_dates(N, HH),
    default_weekly_holidays(DWH),
    LB = @weekly_holidays_lb(WIDTH, HH, DWH),
    CUB = @weekly_holidays_ub(WIDTH, HH, DWH),
    RWH = { staff_request(N, D, "○") : date(D) },
    UB = @max(CUB, RWH).     % 希望シフトで週休が CUB を超えていた場合は，それを優先
    %UB = #max {CUB; RWH}.     % 希望シフトで週休が CUB を超えていた場合は，それを優先

% 看護師ごとの祝日の数
national_holiday_date(N, D, W) :- 
    staff(N), date(D, W), national_holiday(D, W), 
    not soft_holiday_date(N, D, W), not hard_holiday_date(N, D, W).
num_national_holiday_dates(N, X) :-
    staff(N), X = { national_holiday_date(N, D, W) }.

% 看護師ごとのソフト休暇シフトの数
soft_holiday_date(N, D, W) :- 
    staff(N), date(D, W), staff_request(N, D, S), soft_holiday_shift(S).
num_soft_holiday_dates(N, X) :-
    staff(N), X = { soft_holiday_date(N, D, W) }.

% 看護師ごとのハード休暇シフトの数
hard_holiday_date(N, D, W) :- 
    staff(N), date(D, W), staff_request(N, D, S), hard_holiday_shift(S).
num_hard_holiday_dates(N, X) :-
    staff(N), X = { hard_holiday_date(N, D, W) }.

% 看護師ごとのビジネスシフトの数
business_date(N, D, W) :- 
    staff(N), date(D, W), staff_request(N, D, S), business_shift(S).
num_business_dates(N, X) :-
    staff(N), X = { business_date(N, D, W) }.

% 勤務シフトを割り当て可能な日
workable_date(N, D, W) :-
    staff(N), date(D, W), not hard_holiday_date(N, D, W), not business_date(N, D, W).

% 確実に勤務シフトを割り当て可能な日（workable から soft_holiday と national_holiday を除いたもの）
strict_workable_date(N, D, W) :-
    staff(N), workable_date(N, D, W), not soft_holiday_date(N, D, W), not national_holiday_date(N, D, W).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 曜日に関する制約を日付に展開
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 人数制限
%%% 平日
staff_hard_lb(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), staff_dweek_hard_lb(G, S, "平日", B).
staff_soft_lb(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), staff_dweek_soft_lb(G, S, "平日", B).
staff_soft_ub(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), staff_dweek_soft_ub(G, S, "平日", B).
staff_hard_ub(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), staff_dweek_hard_ub(G, S, "平日", B).
%%% 土日
staff_hard_lb(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), staff_dweek_hard_lb(G, S, "土日", B).
staff_soft_lb(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), staff_dweek_soft_lb(G, S, "土日", B).
staff_soft_ub(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), staff_dweek_soft_ub(G, S, "土日", B).
staff_hard_ub(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), staff_dweek_hard_ub(G, S, "土日", B).
%%% 祝日
staff_hard_lb(G, S, D, B) :-
    date(D,_), national_holiday(D), staff_dweek_hard_lb(G, S, "祝日", B).
staff_soft_lb(G, S, D, B) :-
    date(D,_), national_holiday(D), staff_dweek_soft_lb(G, S, "祝日", B).
staff_soft_ub(G, S, D, B) :-
    date(D,_), national_holiday(D), staff_dweek_soft_ub(G, S, "祝日", B).
staff_hard_ub(G, S, D, B) :-
    date(D,_), national_holiday(D), staff_dweek_hard_ub(G, S, "祝日", B).

%% 点数制限
%%% 平日
point_hard_lb(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), point_dweek_hard_lb(G, S, "平日", B).
point_soft_lb(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), point_dweek_soft_lb(G, S, "平日", B).
point_soft_ub(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), point_dweek_soft_ub(G, S, "平日", B).
point_hard_ub(G, S, D, B) :-
    date(D,W), weekday(W), not national_holiday(D), point_dweek_hard_ub(G, S, "平日", B).
%%% 土日
point_hard_lb(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), point_dweek_hard_lb(G, S, "土日", B).
point_soft_lb(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), point_dweek_soft_lb(G, S, "土日", B).
point_soft_ub(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), point_dweek_soft_ub(G, S, "土日", B).
point_hard_ub(G, S, D, B) :-
    date(D,W), weekend(W), not national_holiday(D), point_dweek_hard_ub(G, S, "土日", B).
%%% 祝日
point_hard_lb(G, S, D, B) :-
    date(D,_), national_holiday(D), point_dweek_hard_lb(G, S, "祝日", B).
point_soft_lb(G, S, D, B) :-
    date(D,_), national_holiday(D), point_dweek_soft_lb(G, S, "祝日", B).
point_soft_ub(G, S, D, B) :-
    date(D,_), national_holiday(D), point_dweek_soft_ub(G, S, "祝日", B).
point_hard_ub(G, S, D, B) :-
    date(D,_), national_holiday(D), point_dweek_hard_ub(G, S, "祝日", B).

%% 担当可能・不能なシフト
pos_def_request(N, D, S) :- 
    pos_shift(N, W, S), date(D, W), not national_holiday(D).
pos_def_request(N, D, S) :-
    pos_shift(N, "祝", S), date(D), national_holiday(D).
neg_def_request(N, D, S) :- 
    neg_shift(N, W, S), date(D, W). %, not national_holiday(D). 担当不能シフトは祝日であっても，指定曜日は担当不可とする
neg_def_request(N, D, S) :-
    neg_shift(N, "祝", S), date(D), national_holiday(D).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 簡略形の定義
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 曜日なし日付の定義
base_date(D) :- base_date(D, _, W).
prev_date(D) :- prev_date(D, W).
date(D) :- date(D, W).
next_date(D) :- next_date(D, W).
national_holiday(D) :- national_holiday(D, W).
workable_date(N, D) :- workable_date(N, D, W).
strict_workable_date(N, D) :- strict_workable_date(N, D, W).
hard_holiday_date(N, D) :- hard_holiday_date(N, D, W).
soft_holiday_date(N, D) :- soft_holiday_date(N, D, W).
date2fulldate(D, FD) :- base_date(D, FD, _).

% 作成するシフト表の開始日と最終日
first_date(X) :- X = #min { D : date(D, _) }.
last_date(X)  :- X = #max { D : date(D, _) }.
first_full_date(FD) :- base_date(0, FD, _).

% スタッフ ID, 点数
staff(ID) :- staff(ID, _,  _, _, _).
point(ID, P) :- staff(ID,  _, _, _, P).

% スタッフの希望シフトがある日
staff_request(N, D) :- staff_request(N, D, _).
% スタッフのソフト休暇シフトがある日
staff_soft_holiday_request(N, D) :- staff_request(N, D, S), soft_holiday_shift(S).
% スタッフの希望休暇シフトがある日
staff_valid_holiday_request(N, D) :- staff_request(N, D, S), valid_holiday_shift(S).
% スタッフの希望ビジネスシフトがある日
staff_business_request(N, D) :- staff_request(N, D, S), business_shift(S).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% シフトデータ（希望シフト含む）から前月・当月・翌月シフトデータの生成
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 前月シフトを ext_assigned/3 として定義
ext_assigned(N, D, S) :- staff(N, _, _, CD, _), prev_date(D), shift_data(CD, _, FD, S), base_date(D, FD, _).

% 前月に working shift があるならば ext_assigned/2 を定義
%ext_assigned(N, D) :- staff(N), prev_date(D), ext_assigned(N, D, S), working_shift(S).

% 当月シフトを staff_request として定義
staff_request(N, D, S) :- staff(N, _, _, CD, _), date(D), shift_data(CD, _, FD, S), base_date(D, FD, _).

% 翌月シフトを ext_assigned/3 として定義
ext_assigned(N, D, S) :- staff(N, _, _, CD, _), next_date(D), shift_data(CD, _, FD, S), base_date(D, FD, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 過去のシフトの担当回数データの生成
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 看護師 N がシフト S を担当した回数 X
num_past_shifts(N, S, X) :- 
    staff(N, _, _, CD, _), shift(S), 
    X = { shift_data(CD, _, FD, S) : first_full_date(FFD), FD < FFD }.

% 看護師 N の週末休暇数 
num_past_weekend_offs(N, X) :-
    staff(N, _, _, CD, _), 
    X = { shift_data(CD, _, FD1, S1) : shift_data(CD, _, FD2, S2), weekend(FD1, FD2), first_full_date(FFD), FD2 < FFD, regular_holiday_shift(S1), regular_holiday_shift(S2) }.
% 週末休暇数の最大値
max_past_weekend_offs(X) :- X = #max { Y : num_past_weekend_offs(N, Y), staff(N) }.

% 看護師 N の祝日休暇数 
num_past_national_holiday_offs(N, X) :-
    staff(N, _, _, CD, _), 
    X = { shift_data(CD, _, FD, "◎") : first_full_date(FFD), FD < FFD, national_holiday(FD) }.
% 祝日休暇数の最大値
max_past_national_holiday_offs(X) :- X = #max { Y : num_past_national_holiday_offs(N, Y), staff(N) }.

% 看護師 N の２連続休暇数（理想の上限は看護師ごとに異なるため，平等化が大変．いったん保留）
% num_past_double_reg_holidays(N, X) :-
%     staff(N, _, _, CD, _), 
%     X = { shift_data(CD, _, FD1, S1) : regular_holiday_shift(S1), next_date(FD1, FD2), shift_data(CD, _, FD2, S2), regular_holiday_shift(S2), first_full_date(FFD), FD2 < FFD }.
% 看護師 N の連続休暇数（２連休以上）の回数算出
% TODO: 所属日数に応じた正規化が必要
num_past_consecutive_holidays(N, X) :-
    staff(N, _, _, CD, _), 
    X = { shift_data(CD, _, FD1, S1) :             
            next_date(FD1, FD2),
            next_date(FD2, FD3),
            first_full_date(FFD), FD3 < FFD,

            shift_data(CD, _, FD2, S2),
            shift_data(CD, _, FD3, S3),

            working_shift(S1),
            regular_holiday_shift(S2),
            regular_holiday_shift(S3)            
        }.
% 連続休暇数の最大値
max_past_consecutive_holidays(X) :- X = #max { Y : num_past_consecutive_holidays(N, Y), staff(N) }.