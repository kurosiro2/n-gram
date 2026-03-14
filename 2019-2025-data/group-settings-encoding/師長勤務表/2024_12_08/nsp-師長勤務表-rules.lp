%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 師長勤務表独自の設定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --------------------------------------------------------------------
% 要望：年間を通して火曜日N、Jの回数を均等に振り分けたい
% --------------------------------------------------------------------

% 過去の火曜日のシフト N, J の割当数
num_past_tue_nj_shifts(N, X) :- 
    staff(N, _, _, CD, _), X = { shift_data(CD, _, FD, S) : S = ("N";"J"), first_full_date(FFD), FD < FFD, date_data(FD, "火") }.

% 当月の火曜日のシフト N, J の割当数
num_curr_tue_nj_shifts(N, X) :-
    staff(N), X = { assigned(N, D, S) : S = ("N";"J"), date(D, "火") }.

% 当月までの火曜日のシフト N, J の割当数
num_tue_nj_shifts(N, X+Y) :-
    num_past_tue_nj_shifts(N, X), num_curr_tue_nj_shifts(N, Y).
%#show num_tue_nj_shifts/3.   % for debug with -m

% N,J の割当回数差に応じてペナルティ（N,J,S は熟練１グループにしか割り当てがない）
penalty(diff_tue_shifts(MIN, MAX), staffs(MIN_S, MAX_S), tue_nj_shifts_penalty * D * D, tue_nj_shifts_priority) :-
    MIN = #min{ V : num_tue_nj_shifts(N, V), staff(N), group("熟練１", N) },
    MAX = #max{ V : num_tue_nj_shifts(N, V), staff(N), group("熟練１", N) },
    MIN_S = #min{ N : num_tue_nj_shifts(N, MIN), staff(N), group("熟練１", N) },  % 割当数最小のスタッフ（ID最小の人のみ）
    MAX_S = #min{ N : num_tue_nj_shifts(N, MAX), staff(N), group("熟練１", N) },  % 割当数最大のスタッフ（ID最小の人のみ）
    D = MAX - MIN,
    D > 0.

% 出力用
out_stats(num_tue_nj_shifts(N, P, T-P, T)) :- num_past_tue_nj_shifts(N, P), num_tue_nj_shifts(N, T).

% --------------------------------------------------------------------
% 要望：火曜日N、Jの回数は1回まで
% --------------------------------------------------------------------
:- staff(N), num_curr_tue_nj_shifts(N, X), X > 1, not soft_all.

% --------------------------------------------------------------------
% 要望：各師長ペアの土日の勤務の重複は最大2回まで(2023/6/7)
% --------------------------------------------------------------------
num_weekend_shifts(N1, N2, X) :-
    staff(N1), staff(N2), N1 < N2, X > 0,
    X = { assigned(N1, D, S1) : assigned(N2, D, S2), S1 = ("N";"J";"S"), S2 = ("N";"J";"S"), date(D, ("土";"日")) }.
%#show num_weekend_shifts/3.

:- num_weekend_shifts(N1, N2, X), X > 2, not soft_all.

% --------------------------------------------------------------------
% 要望：NJS の割当間隔を13日以上にしたい(2023/11/21)
% --------------------------------------------------------------------

% D～D+12 間の N の割当回数
num_n_shifts_per_week(N, D, X) :-
    staff(N), base_date(D), ext_assigned(N, D, "J"), X = { ext_assigned(N, DD, "J") :  D <= DD, DD <= D + 12 }.

penalty(short_nsj_interval(N, FD), table(N, D), short_nsj_interval_penalty, short_nsj_interval_priority) :-
    num_n_shifts_per_week(N, D, X), staff(N), base_date(D), X > 1, date2fulldate(D, FD).

%---------------------------------------------------------------------
% 前日・翌日可能シフトに関する制約
%---------------------------------------------------------------------
 
% J, S, ★, ☆ の前日シフトをハード化
%hard_pred_shift("J";"S";"★";"☆").
% 2023-09-17 N◯を許すという要求より，Jの前のシフトのハード化を解除
hard_pred_shift("S";"★";"☆").

%hard_pred_shift("★";"☆").  % 6/25 は前月末シフトのため J,S をハードにできない
% S, J, ★, ☆ の翌日シフトをハード化
hard_succ_shift("S";"J";"★";"☆").
%hard_succ_shift("★";"☆").  % 6/25 は前月末シフトのため J,S をハードにできない

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ソフト制約の優先度と重み
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 看護師の希望シフトは優先的に充足する
priority("staff_request", 4).

%==2023-09-17 追加===================
staff_lb_priority("全員", "N", 3).
staff_lb_priority("全員", "J", 3).
staff_lb_priority("全員", "S", 3).

staff_lb_priority("熟練１", "N", 3).
staff_lb_priority("熟練１", "J", 3).
staff_lb_priority("熟練１", "S", 3).

priority("shift_lb", 3).
priority("shift_ub", 3).

priority("invalid_patterns", 2).
%====================================

% 固有：火曜日の N,J シフトバランス違反
#const tue_nj_shifts_penalty = 1.
#const tue_nj_shifts_priority = 2.

% 固有：火曜日の N,J シフトバランス違反
#const short_nsj_interval_penalty = 1.
#const short_nsj_interval_priority = 2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 週間業務分担表生成のためのデータ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_rel("深夜勤", ("S";"☆")).
shift_rel("長日勤", "N").
shift_rel("日勤", "日").
shift_rel("早出", "A").
shift_rel("遅出", "P").
shift_rel("準夜勤", ("J";"★")).

shift_order(1, "深夜勤").
shift_order(2, "長日勤").
shift_order(3, "日勤").
shift_order(4, "早出").
shift_order(5, "遅出").
shift_order(6, "準夜勤").

% 線で区切るグループの定義
group_rel("深夜勤", ("熟練１";"熟練２")).
group_rel("長日勤", ("熟練１";"熟練２")).
group_rel("日勤", ("熟練１";"熟練２")).
group_rel("早出", ("熟練１";"熟練２")).
group_rel("遅出", ("熟練１";"熟練２")).
group_rel("準夜勤", ("熟練１";"熟練２")).

group_order(1, "熟練１").
group_order(2, "熟練２").

section("夜勤師長").

#show group_rel/2.
#show group_order/2.
#show section/1.


