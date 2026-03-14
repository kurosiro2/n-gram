%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5階北病棟 病棟独自の設定
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
priority("staff_request", 15).
% 超過連続勤務を避ける
priority("over_work_period", 15).

% 不正パターンの抑制を優先する
priority("invalid_patterns", 6).
% パターン割当数を優先する
priority("patterns_lb", 5).
priority("patterns_ub", 5).

% A, B, 夜勤グループの上限・下限を優先的に充足
staff_ub_priority("A", ("N";"J";"S"), 4).
staff_lb_priority("A", ("N";"J";"S"), 4).
staff_ub_priority("B", ("N";"J";"S"), 4).
staff_lb_priority("B", ("N";"J";"S"), 4).
staff_ub_priority("夜勤", ("J";"S"), 4).
staff_lb_priority("夜勤", ("J";"S"), 4).

% リーダーグループの上限・下限を優先的に充足
staff_ub_priority("リーダー", ("N";"J";"S"), 4).
staff_lb_priority("リーダー", ("N";"J";"S"), 4).

% 代行・平日代行の上限・下限を優先的に充足
staff_ub_priority("代行", "日", 3).
staff_lb_priority("代行", "日", 3).
staff_ub_priority("平日代行", "日", 3).
staff_lb_priority("平日代行", "日", 3).

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
group_rel("深夜勤", ("A";"B";"新人")).
group_rel("長日勤", ("A";"B";"新人")).
group_rel("日勤", ("A";"B";"新人")).
group_rel("早出", ("全員")).
group_rel("遅出", ("全員")).
group_rel("準夜勤", ("A";"B";"新人")).

group_order(1, "全員").
group_order(2, "代行").
group_order(3, "平日代行").
group_order(4, "リーダー").
group_order(5, "A").
group_order(6, "B").
group_order(7, "新人").
group_order(8, "夜勤").

section("5階北病棟").

#show group_rel/2.
#show group_order/2.
#show section/1.


