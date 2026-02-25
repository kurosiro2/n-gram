%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7階北病棟独自の設定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------------------------------------------------------------
% 前日・翌日可能シフトに関する制約
%---------------------------------------------------------------------

% S, ★, ☆ の前日シフトをハード化
% J の前日はソフトでよい
hard_pred_shift("S";"★";"☆").

% S, J, ★, ☆ の翌日シフトをハード化
hard_succ_shift("S";"J";"★";"☆").

%---------------------------------------------------------------------
% 希望シフトに関する制約
%---------------------------------------------------------------------

% 各スタッフの希望シフトをハード化
hard_staff_request(N) :- staff(N).

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
staff_ub_priority("全員", ("N";"J";"S"), 6).
staff_lb_priority("全員", ("N";"J";"S"), 6).
staff_ub_priority("夜勤", ("N";"J";"S"), 6).
staff_lb_priority("夜勤", ("N";"J";"S"), 6).

% リーダー, A, B, C の NJS シフトの上限・下限を優先
staff_ub_priority("リーダー", ("N";"J";"S"), 5).
staff_lb_priority("リーダー", ("N";"J";"S"), 5).
staff_ub_priority("A", ("N";"J";"S"), 5).
staff_lb_priority("A", ("N";"J";"S"), 5).
staff_ub_priority("B", ("N";"J";"S"), 5).
staff_lb_priority("B", ("N";"J";"S"), 5).
staff_ub_priority("C", ("N";"J";"S"), 5).
staff_lb_priority("C", ("N";"J";"S"), 5).

% 次に優先するシフト
staff_ub_priority("全員", ("日";"P"), 4).
staff_lb_priority("全員", ("日";"P"), 4).
staff_ub_priority("代行", ("日";"N"), 4).
staff_lb_priority("代行", ("日";"N"), 4).
staff_ub_priority("平日代行", ("日";"N"), 4).
staff_lb_priority("平日代行", ("日";"N"), 4).

% その他のシフトの優先度
staff_ub_priority("新人", ("J";"S"), 3).
staff_lb_priority("新人", ("J";"S"), 3).

% シフト担当回数の優先度
priority("shift_lb", 3).
priority("shift_ub", 3).

% 孤立した勤務日・隣接した週休の優先度
priority("isolated_working_day", 2).
priority("adjacent_holiday", 2).

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
group_rel("深夜勤", ("師長";"A";"B";"C")).
group_rel("長日勤", ("師長";"A";"B";"C")).
group_rel("日勤", ("師長";"A";"B";"C")).
group_rel("早出", ("師長";"A";"B";"C")).
group_rel("遅出", ("師長";"A";"B";"C")).
group_rel("準夜勤", ("師長";"A";"B";"C")).

group_order(1, "師長").
group_order(2, "A").
group_order(3, "B").
group_order(4, "C").
group_order(5, "全員").

section("5階南病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


