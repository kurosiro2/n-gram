%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ２階西病棟独自の設定
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
priority("staff_request", 5).

% 超過連続勤務を避ける
priority("over_work_period", 5).

% 不正パターンの抑制を優先する
priority("invalid_patterns", 4).
% パターン割当数を優先する
priority("patterns_lb", 4).
priority("patterns_ub", 4).

% 夜勤グループの上限・下限を優先的に充足
staff_ub_priority("Night", ("SE+E";"SN+N"), 3).
staff_lb_priority("Night", ("SE+E";"SN+N"), 3).

% スタッフ数とシフト担当回数を同程度優先する
staff_ub_priority(G, SS, 2) :- group(G), shifts(SS).
staff_lb_priority(G, SS, 2) :- group(G), shifts(SS).
priority("shift_lb", 2).
priority("shift_ub", 2).

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
group_rel("深夜勤", ("Heads";"Seniors";"Mid-levels";"Newcomers")).
group_rel("長日勤", ("Heads";"Seniors";"Mid-levels";"Newcomers")).
group_rel("日勤", ("Heads";"Seniors";"Mid-levels";"Newcomers")).
group_rel("早出", ("All";"Heads")).
group_rel("遅出", ("All";"Heads")).
group_rel("準夜勤", ("Heads";"Seniors";"Mid-levels";"Newcomers")).

group_order(1, "All").
group_order(2, "Heads").
group_order(3, "Seniors").
group_order(4, "Mid-levels").
group_order(5, "Newcomers").
group_order(6, "Night").

section("2階西病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


