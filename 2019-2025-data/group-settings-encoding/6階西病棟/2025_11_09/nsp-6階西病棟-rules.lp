%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6階西病棟 病棟独自の設定
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
priority("staff_request", 15).
% 超過連続勤務を避ける
priority("over_work_period", 15).

% 不正パターンの抑制を優先する
priority("invalid_patterns", 6).
% パターン割当数を優先する
priority("patterns_lb", 5).
priority("patterns_ub", 5).

% リーダーグループの上限・下限を優先的に充足
staff_ub_priority("リーダー", ("N";"J";"S"), 4).
staff_lb_priority("リーダー", ("N";"J";"S"), 4).

% A, B, C, 夜勤グループの上限・下限を優先的に充足
staff_ub_priority("A", ("N";"J";"S"), 3).
staff_lb_priority("A", ("N";"J";"S"), 3).
staff_ub_priority("B", ("N";"J";"S"), 3).
staff_lb_priority("B", ("N";"J";"S"), 3).
staff_ub_priority("C", ("N";"J";"S"), 3).
staff_lb_priority("C", ("N";"J";"S"), 3).
staff_ub_priority("夜勤", ("J";"S"), 4).
staff_lb_priority("夜勤", ("J";"S"), 4).

% スタッフ数とシフト担当回数を同程度優先する
staff_ub_priority(G, SS, 2) :- group(G), shifts(SS).
staff_lb_priority(G, SS, 2) :- group(G), shifts(SS).
priority("shift_lb", 2).
priority("shift_ub", 2).

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
group_rel("深夜勤", ("A";"B";"C")).
group_rel("長日勤", ("A";"B";"C")).
group_rel("日勤", ("A";"B";"C")).
group_rel("早出", ("A";"B";"C")).
group_rel("遅出", ("A";"B";"C")).
group_rel("準夜勤", ("A";"B";"C")).

group_order(1, "A").
group_order(2, "B").
group_order(3, "C").

section("6階西病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


