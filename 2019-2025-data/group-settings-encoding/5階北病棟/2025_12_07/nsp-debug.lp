%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 看護師ごとのシフトの上下限制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ソフト版：勤務シフトを指定回数割り当てる
{ assigned(N,D) : date(D) } :- staff(N), target(N), soft_all.  % soft_all 指定時

penalty(working_dates_lb(N, LB, X), staff(N), PNL * (LB - X), PRI) :-
    staff(N), working_dates_bounds(N, LB, UB), X = { assigned(N,D) : workable_date(N,D)}, X < LB,
    _penalty("working_dates", PNL), _priority("working_dates", PRI), soft_all.
penalty(working_dates_ub(N, UB, X), staff(N), PNL * (X - UB), PRI) :-
    staff(N), working_dates_bounds(N, LB, UB), X = { assigned(N,D) : workable_date(N,D)}, UB < X,
    _penalty("working_dates", PNL), _priority("working_dates", PRI), soft_all.

% ソフト版：週休シフトを指定回数割り当てる
penalty(weekly_holidays_lb(N, LB, X), staff(N), PNL * (LB - X), PRI) :-
    staff(N), weekly_holiday_bounds(N, LB, UB), X = { not assigned(N,D) : strict_workable_date(N,D) }, X < LB,
    _penalty("weekly_holidays", PNL), _priority("weekly_holidays", PRI), soft_all.
penalty(weekly_holidays_ub(N, UB, X), staff(N), PNL * (X - UB), PRI) :-
    staff(N), weekly_holiday_bounds(N, LB, UB), X = { not assigned(N,D) : strict_workable_date(N,D) }, UB < X,
    _penalty("weekly_holidays", PNL), _priority("weekly_holidays", PRI), soft_all.

% ソフト版：ハード休暇シフト
penalty(hard_holiday(N, FD, S, T), table(N, D), PNL, PRI) :-
    staff_request(N, D, S), hard_holiday_shift(S), assigned(N, D, T), date2fulldate(D, FD),
    _penalty("hard_holidays", PNL), _priority("hard_holidays", PRI), soft_all.

% ソフト版：ビジネスシフト
penalty(business_dates(N, FD, S, T), table(N, D), PNL, PRI) :-
    staff_request(N, D, S), business_shift(S), assigned(N, D, T), S != T, date2fulldate(D, FD),
    _penalty("business_dates", PNL), _priority("business_dates", PRI), soft_all.

% ペナルティチェック時に希望が満たされていないことを検出
penalty(staff_request(N, FD, S), table(N, D), PNL, PRI) :-
    staff_request(N, D, S), not ext_assigned(N, D, _), date2fulldate(D, FD),
    _penalty("staff_request", PNL), _priority("staff_request", PRI), soft_all.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 縦制約：各日 D，各シフト S，各看護師グループ G ごとの人数制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 各日 D，各シフト S，各看護師グループ G に属する看護師数の下限制約
penalty(staff_hard_lb(FD, SS, G, LB, X), num_staffs(D, SS, G), PNL * (LB - X) * (LB - X), PRI)
    :- staff_hard_lb(G, SS, D, LB), date2fulldate(D, FD), X = { assigned(N, D, S) : group(G, N), shifts(SS, S) }, X < LB,
       _penalty("staff_hard_lb", PNL), _priority("staff_hard_lb", PRI), soft_all.

% 各日 D，各シフト S，各看護師グループ G に属する看護師数の上限制約
penalty(staff_hard_ub(FD, SS, G, UB, X), num_staffs(D, SS, G), PNL * (X - UB) * (X - UB), PRI)
    :- staff_hard_ub(G, SS, D, UB), date2fulldate(D, FD), X = { assigned(N, D, S) : group(G, N), shifts(SS, S) }, UB < X,
       _penalty("staff_hard_ub", PNL), _priority("staff_hard_lb", PRI), soft_all.

%*
% デバッグ時に点数制約が大きな違反コストを持ち探索が進まないのでいったんコメントアウト
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 縦制約：各日 D，各シフト S，各看護師グループ G ごとの点数制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 各日 D，各シフト S，各看護師グループ G に属する看護師の点数和の下限制約
penalty(point_hard_lb(FD, SS, G, LB, X), num_points(D, SS, G), PNL * (LB - X) * (LB - X), PRI)
    :- point_hard_lb(G, SS, D, LB), date2fulldate(D, FD), X = #sum{ P, N : point(N, P), assigned(N, D, S), group(G, N), shifts(SS, S) }, X < LB,
       _penalty("point_hard_lb", PNL), _priority("point_hard_lb", PRI), soft_all.

% 各日 D，各シフト S，各看護師グループ G に属する看護師の点数和の上限制約
penalty(point_hard_ub(FD, SS, G, UB, X), num_points(D, SS, G), PNL * (X - UB) * (X - UB), PRI)
    :- point_hard_ub(G, SS, D, UB), date2fulldate(D, FD), X = #sum{ P, N : point(N, P), assigned(N, D, S), group(G, N), shifts(SS, S) }, UB < X,
       _penalty("point_hard_ub", PNL), _priority("point_hard_ub", PRI), soft_all.
 *%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 横制約：各看護師の担当するシフト数に関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 各看護師 N，各シフト S の担当回数の下限制約
penalty(shifts_hard_lb(N, SS, LB, X), num_shifts(N, SS), PNL * (LB - X) * (LB - X), PRI) :-
    shifts_hard_lb(N, SS, LB), X = { assigned(N, D, S) : shifts(SS, S) }, X < LB,
    _penalty("shift_hard_lb", PNL), _priority("shift_hard_lb", PRI), soft_all.

penalty(shifts_hard_ub(N, SS, UB, X), num_shifts(N, SS), PNL * (X - UB) * (X - UB), PRI) :-
    shifts_hard_ub(N, SS, UB), X = { assigned(N, D, S) : shifts(SS, S) }, UB < X,
    _penalty("shift_hard_ub", PNL), _priority("shift_hard_ub", PRI), soft_all.

% ソフト版：各看護師 N，シフト N, J の担当回数の同一に
penalty(diff_nj_shifts(N, NN, NJ), staff(N), PNL, PRI) :-
    staff(N),
    NN = { assigned(N, D, "N") : date(D) },
    NJ = { assigned(N, D, "J") : date(D) },
    NN != NJ,
    _penalty("diff_nj_shifts", PNL), _priority("diff_nj_shifts", PRI), soft_all.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 連続勤務回数に関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ソフト版：連続勤務回数を超えてはならない
penalty(over_work_period(N, FBD, ED - BD + 2), table(N, BD), PNL, PRI) :-
    full_work_period(N, BD, ED), assigned(N, ED+1), date(ED+1),
    date2fulldate(BD, FBD),
    _penalty("over_work_period", PNL), _priority("over_work_period", PRI), soft_all.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 禁止シフトパターンに関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ソフト版：禁止パターンが出現してはならない（希望シフトによるものを除く）
penalty(forbidden_pattern(N, PID, FBD), table(N, BD), PNL, PRI) :-
    forbidden_pattern(PID), pattern_occured(PID, N, BD, ED), not requested_pattern(PID, N, BD, ED),
    date((BD;ED)), date2fulldate(BD, FBD),
    _penalty("forbidden_pattern", PNL), _priority("forbidden_pattern", PRI), soft_all.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 前日・翌日可能シフトに関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ハード要求していた前日可能シフト違反
penalty(pred_shift(N, DD, S), table(N, D), PNL, PRI) :-
    hard_pred_shift(S), pred_base_shift(S), ext_assigned(N, D, S),
    first_date(FD), last_date(LD), FD <= D, D <= LD+1, date2fulldate(D, DD), % 当月頭から翌月頭までが対象
    not 1 { ext_assigned(N, D-1, PS) : pred_shift(PS, S), not valid_holiday_shift(PS) ;
            ext_valid_holiday_off(N, D-1) : pred_shift("○", S) } 1,
    _penalty("invalid_hard_patterns", PNL), _priority("invalid_hard_patterns", PRI).

% ハード要求していた翌日可能シフト違反
penalty(succ_shift(N, DD, S), table(N, D), PNL, PRI) :-
    hard_succ_shift(S), succ_base_shift(S), ext_assigned(N, D, S),
    first_date(FD), last_date(LD), FD-1 <= D, D <= LD,  date2fulldate(D, DD), % 前月末から当月末までが対象
    not 1 { ext_assigned(N, D+1, SS) : succ_shift(S, SS), not valid_holiday_shift(SS) ;
            ext_valid_holiday_off(N, D+1) : succ_shift(S, "○") ;
            staff_request(N, D+1) : D = FD-1, ext_assigned(N, D+1, SS), not succ_shift(S, SS) ;
            not staff_request(N, D+1) : last_date(D) },
    _penalty("invalid_hard_patterns", PNL), _priority("invalid_hard_patterns", PRI).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% パターン割当数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

penalty(patterns_hard_lb(N, PS, LB, X), horizontal(N), PNL * (LB - X) * (LB - X), PRI) :-
    patterns_hard_lb(PS, LB), staff(N), soft_all,
    X = { pattern_occured(P, N, BD, ED) : patterns(PS, P), date(ED) }, X < LB, % 今月に終わるパターンの頻度を数える
    _penalty("patterns_hard_lb", PNL), _priority("patterns_hard_lb", PRI).

penalty(patterns_hard_ub(N, PS, UB, X), horizontal(N), PNL * (UB - X) * (UB - X), PRI) :-
    patterns_hard_ub(PS, UB), staff(N), soft_all,
    X = { pattern_occured(P, N, BD, ED) : patterns(PS, P), date(ED) }, UB < X,  % 今月に終わるパターンの頻度を数える
    _penalty("patterns_hard_ub", PNL), _priority("patterns_hard_ub", PRI).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 夜勤禁止ペアに関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ソフト版：夜勤禁止ペアは同じ夜勤シフトを割り当てない
penalty(forbidden_night_pair(N1, N2, FD, S), table(N1, D), PNL, PRI) :-
    forbidden_night_pair(N1, N2), night_working_shift(S),
    assigned(N1, D, S), assigned(N2, D, S),
    date2fulldate(D, FD),
    _penalty("forbidden_night_pair", PNL), _priority("forbidden_night_pair", PRI), soft_all.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 看護師ごとの希望シフト・不可シフトに関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 希望シフト以外が割り当てられていると矛盾
penalty(pos_request(N, FD), table(N, D), PNL, PRI) :-
    pos_request(N, D),
    not 1 { ext_assigned(N, D, S) : pos_request(N, D, S) } 1,
    date2fulldate(D, FD),
    _penalty("staff_request", PNL), _priority("staff_request", PRI), soft_all.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 看護師ごとのデフォルト割当可能シフト・割当不可シフトに関する制約
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ソフト版：シフト希望がない場合にポジティブシフト以外が割り当てられていると矛盾
penalty(pos_def_request(N, FD), table(N, D), PNL, PRI) :-
    pos_def_request(N, D), not staff_request(N, D),
    not 1 { ext_assigned(N, D, S) : pos_def_request(N, D, S) } 1,
    date2fulldate(D, FD),
    _penalty("pos_def_request", PNL), _priority("pos_def_request", PRI), soft_all.
