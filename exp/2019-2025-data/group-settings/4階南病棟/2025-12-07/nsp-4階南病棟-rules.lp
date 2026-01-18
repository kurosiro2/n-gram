%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4階南病棟独自の設定
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
priority("invalid_patterns", 7).
% パターン割当数を優先する
priority("patterns_lb", 6).
priority("patterns_ub", 6).
% NJS の隣接を優先する
priority("isolated_nj_shift", 6).

% 全員グループの上限・下限を優先的に充足
staff_ub_priority("All", ("Su";"LD";"SE";"SN"), 5).
staff_lb_priority("All", ("Su";"LD";"SE";"SN"), 5).

% 代行・平日代行・リーダーグループの上限・下限を優先的に充足
staff_ub_priority("代行", ("Su";"LD"), 4).
staff_lb_priority("代行", ("Su";"LD"), 4).
staff_ub_priority("平日代行", "Su", 4).
staff_lb_priority("平日代行", "Su", 4).
staff_ub_priority("Leaders", ("LD";"SE";"SN"), 4).
staff_lb_priority("Leaders", ("LD";"SE";"SN"), 4).

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
group_rel("深夜勤", "All").
group_rel("長日勤", "All").
group_rel("日勤", "All").
group_rel("早出", "All").
group_rel("遅出", "All").
group_rel("準夜勤", "All").

group_order(1, "All").
group_order(2, "代行").
group_order(3, "Leaders").
group_order(4, "平日代行").
group_order(5, "Newcomers").
group_order(6, "Night").
section("4階南病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


