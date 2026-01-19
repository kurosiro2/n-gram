%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6階北病棟 病棟独自の設定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------------------------------------------------------------
% 前日・翌日可能シフトに関する制約
%---------------------------------------------------------------------

% S, ★, ☆ の前日シフトをハード化
% J の前日はソフトでよい
hard_pred_shift("SN";"E";"N").

% S, J, ★, ☆ の翌日シフトをハード化
hard_succ_shift("SN";"SE";"E";"N").

%---------------------------------------------------------------------
% 希望シフトに関する制約
%---------------------------------------------------------------------

% 各スタッフの希望シフトをハード化
hard_staff_request(N) :- staff(N).

%---------------------------------------------------------------------
% 提案：6連勤はできる限り避ける
%---------------------------------------------------------------------

% ６連続勤務にペナルティ
penalty(over_work_period(N, FBD, ED - BD + 1), table(N, BD), PNL, PRI) :-
    full_work_period(N, BD, ED), date(ED),
    date2fulldate(BD, FBD),
    _penalty("over_work_period", PNL), _priority("over_work_period", PRI).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ソフト制約の優先度と重み
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 看護師の希望シフトは優先的に充足する
priority("staff_request", 20).
priority("pos_def_request", 20).
priority("neg_def_request", 20).
% 超過連続勤務を避ける
priority("over_work_period", 19).

% 不正パターンの抑制を優先する
priority("invalid_patterns", 8).
% パターン割当数を優先する
priority("patterns_lb", 7).
priority("patterns_ub", 7).
% NJS の隣接を優先する
priority("isolated_nj_shift", 7).

% 全員，夜勤の NJS シフトの上限・下限を優先
staff_ub_priority("All", ("LD";"SE";"SN"), 6).
staff_lb_priority("All", ("LD";"SE";"SN"), 6).
staff_ub_priority("Night", ("LD";"SE";"SN"), 6).
staff_lb_priority("Night", ("LD";"SE";"SN"), 6).

% リーダー, A, B, C の NJS シフトの上限・下限を優先
staff_ub_priority("Leaders", ("LD";"SE";"SN"), 5).
staff_lb_priority("Leaders", ("LD";"SE";"SN"), 5).
staff_ub_priority("EM", ("LD";"SE";"SN"), 5).
staff_lb_priority("EM", ("LD";"SE";"SN"), 5).
staff_ub_priority("B", ("LD";"SE";"SN"), 5).
staff_lb_priority("B", ("LD";"SE";"SN"), 5).
staff_ub_priority("C", ("LD";"SE";"SN"), 5).
staff_lb_priority("C", ("LD";"SE";"SN"), 5).

% 次に優先するシフト
staff_ub_priority("All", ("Su";"LM"), 4).
staff_lb_priority("All", ("Su";"LM"), 4).
staff_ub_priority("代行", ("Su";"LD"), 4).
staff_lb_priority("代行", ("Su";"LD"), 4).
staff_ub_priority("平日代行", ("Su";"LD"), 4).
staff_lb_priority("平日代行", ("Su";"LD"), 4).

% その他のシフトの優先度
staff_ub_priority("Newcomers", ("SE";"SN"), 3).
staff_lb_priority("Newcomers", ("SE";"SN"), 3).

% シフト担当回数の優先度
priority("shift_lb", 3).
priority("shift_ub", 3).

% 孤立した勤務日・隣接した週休の優先度
priority("isolated_working_day", 2).
priority("adjacent_holiday", 2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 週間業務分担表生成のためのデータ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_rel("深夜勤", ("SN";"N")).
shift_rel("長日勤", "LD").
shift_rel("日勤", "Su").
shift_rel("早出", "EM").
shift_rel("遅出", "LM").
shift_rel("準夜勤", ("SE";"E")).

shift_order(1, "深夜勤").
shift_order(2, "長日勤").
shift_order(3, "日勤").
shift_order(4, "早出").
shift_order(5, "遅出").
shift_order(6, "準夜勤").

% 線で区切るグループの定義
group_rel("深夜勤", ("EM";"B";"Newcomers")).
group_rel("長日勤", ("EM";"B";"Newcomers")).
group_rel("日勤", ("EM";"B";"Newcomers")).
group_rel("早出", ("All")).
group_rel("遅出", ("All")).
group_rel("準夜勤", ("EM";"B";"Newcomers")).

group_order(1, "All").
group_order(2, "代行").
group_order(3, "平日代行").
group_order(4, "Leaders").
group_order(5, "EM").
group_order(6, "B").
group_order(7, "Newcomers").
group_order(8, "Night").

section("6階北病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


