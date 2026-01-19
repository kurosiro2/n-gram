%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ICU 独自の設定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------------------------------------------------------------
% 要望：新採用者の夜勤回数を同一にしたい
%---------------------------------------------------------------------
% 夜勤回数をJSの回数とする

% シフト J, S の割当数
num_js_shifts(N, X) :-
    staff(N), group("New hires", N), 
    X = { assigned(N, D, S) : S = ("SE";"SN"), date(D) }.
% % 割当数が同一であること
% 新採用者が３グループ（上限が異なるグループが２つ，病欠１名）いるので無効化
% :- num_js_shifts(N1, X1), num_js_shifts(N2, X2), 
%     staff(N1), staff(N2), N1 != N2, X1 != X2, not soft_all.

%---------------------------------------------------------------------
% 提案：6連勤はできる限り避ける (2023-06-25)
%---------------------------------------------------------------------

% ６連続勤務にペナルティ
penalty(over_work_period(N, FBD, ED - BD + 1), table(N, BD), PNL, PRI) :-
    full_work_period(N, BD, ED), date(ED),
    date2fulldate(BD, FBD),
    _penalty("over_work_period", PNL), _priority("over_work_period", PRI).

%---------------------------------------------------------------------
% 要望：新採用者は6連勤禁止 (2023-06-25)
%---------------------------------------------------------------------

:- staff(N), group("New hires", N), full_work_period(N, _, ED) : date(ED); not soft_all.

%---------------------------------------------------------------------
% 要望：新採用者は JSJS 禁止 (2023-06-25)
%---------------------------------------------------------------------

%:- staff(N), group("New hires", N), pattern_occured("JSJS", N, BE, ED), date(ED), not soft_all.

%---------------------------------------------------------------------
% 要望：長期休暇は７日まで (2023-08-03)
%---------------------------------------------------------------------

% ８日連続の休みがあり，うち希望シフトではない週休・祝日休があることを禁止
:- staff(N), date(BD), ED = BD + 7,
    valid_holiday_off(N, D) : D = BD..ED;
    1 { not staff_holiday_request(N, D) : D = BD..ED }, not soft_all.

%---------------------------------------------------------------------
% 提案：シフトパターン違反は１人あたり２回まで
%---------------------------------------------------------------------
% :- staff(N), 
%     3 { penalty(pred_shift(N, _, _), table(N, D), _, _) : date(D) }, not soft_all. % succ_shift はハード化されており違反はない

%---------------------------------------------------------------------
% 提案：孤立勤務は１人あたり２回まで
%---------------------------------------------------------------------
% :- staff(N), 
%    3 { isolated_working_day(N, D) : date(D+1) }, not soft_all. 
% 孤立勤務のチェックなし
allow_isolated_working_day.

% 下記を追加すると夜勤８名の確保が難しくなる
% %---------------------------------------------------------------------
% % 提案：孤立休日は１人あたり４回まで
% %---------------------------------------------------------------------
% :- staff(N), 
%    5 { isolated_weekly_holiday(N, D) : date(D+1) }. 

%---------------------------------------------------------------------
% 前日・翌日可能シフトに関する制約
%---------------------------------------------------------------------

% J, S, ★, ☆ の前日シフトをハード化
%hard_pred_shift("SE";"SN";"E";"N").
hard_pred_shift("SN";"N";"TR").
% J, S, ★, ☆ の翌日シフトをハード化
hard_succ_shift("SE";"SN";"E";"N").

%---------------------------------------------------------------------
% 希望シフトに関する制約
%---------------------------------------------------------------------

% 各スタッフの希望シフトをハード化
hard_staff_request(N) :- staff(N).

%---------------------------------------------------------------------
% 希望休暇の連続日数が3日以上の場合は前後に休暇を割り当てる必要はない (2023/6/7)
%---------------------------------------------------------------------
adjacent_holiday_limit(3).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ソフト制約の優先度と重み
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 看護師の希望シフトは最優先で充足する
priority("staff_request", 6).
priority("neg_def_request", 6).

% ６連続勤務を避ける
priority("over_work_period", 5).
% パターン割当数の上限を優先的に充足
priority("patterns_ub", 4).

% 若手以上グループの上限・下限を優先的に充足
staff_ub_priority("Juniors+", ("D+LD";"SE+E";"SN+N"), 4).
staff_lb_priority("Juniors+", ("D+LD";"SE+E";"SN+N"), 6).

% 全員グループの上限を優先的に充足
staff_ub_priority("All", ("SE+E";"SN+N"), 4).

% 下記の下限も優先的に充足
staff_lb_priority("Newcomers+", "LD+LM+E", 3).
staff_lb_priority("Weekend Leaders", "D+LD", 3).
staff_lb_priority("Day Leaders", "D+LD", 3).
staff_lb_priority("Night Leaders", ("SE+E";"SN+N"), 3).
staff_lb_priority("Juniors", ("D+LD";"SE+E";"SN+N"), 3).

% 各看護師のソフト休暇前後の週休を優先的に割り当てる
priority("adjacent_holiday", 2).
penalty("adjacent_holiday", 10).  
% 連続休暇も優先的に割り当てる
priority("consecutive_holidays", 2).
% priority("weekend_offs", 2).
% priority("national_holiday_off", 2).
% priority("isolated_working_day", 2).
% priority("isolated_weekly_holiday", 2).

% シフト担当回数を若干優先する
priority("shift_lb", 2).
priority("shift_ub", 4).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 週間業務分担表生成のためのデータ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_rel("深夜勤", ("SN";"N")).
shift_rel("日勤", ("Su";"LD")).
shift_rel("早出", "EM").
shift_rel("遅出", "LM").
shift_rel("準夜勤", ("SE";"E")).

shift_order(1, "深夜勤").
shift_order(2, "日勤").
shift_order(3, "早出").
shift_order(4, "遅出").
shift_order(5, "準夜勤").

% 線で区切るグループの定義
group_rel("日勤", ("Day Leaders";"Juniors";"Newcomers";"New hires")).
group_rel("早出", "All").
group_rel("遅出", "All").
group_rel("準夜勤", ("Newcomers+";"New hires")).   
group_rel("深夜勤", ("Newcomers+";"New hires")).   

group_order(1, "All").
group_order(2, "Newcomers+").
group_order(3, "Day Leaders").
group_order(4, "Juniors").
group_order(5, "Newcomers").
group_order(6, "New hires").

section("集中治療室").

#show group_rel/2.
#show group_order/2.
#show section/1.