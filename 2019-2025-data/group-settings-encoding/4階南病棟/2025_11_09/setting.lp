% Staffs ---------------------------------------------------
staff(1, "金子　春美", "師長", "0031213", 0).
staff(2, "上原　良江", "副師長", "0021574", 0).
staff(3, "望月　沙織", "副師長", "0054940", 0).
staff(4, "大久保　香織", "副師長", "0102231", 0).
staff(5, "松山　和佳", "看護師", "0056471", 0).
staff(6, "大木　貴絵", "看護師", "0058352", 0).
staff(7, "藤巻　夕香里", "看護師", "0123486", 0).
staff(8, "佐藤　恵理香", "看護師", "0354841", 0).
staff(9, "稲垣　愛", "看護師", "0279884", 0).
staff(10, "田上　貴士", "看護師", "0340514", 0).
staff(11, "雨宮　周実", "看護師", "0384233", 0).
staff(12, "穂坂　樹里", "看護師", "0384723", 0).
staff(13, "伊東　未緒", "看護師", "0349720", 0).
staff(14, "中野　佑麻", "看護師", "0399960", 0).
staff(15, "岩波　みさと", "看護師", "0400072", 0).
staff(16, "芳賀　友香", "看護師", "0400265", 0).
staff(17, "高井　彩", "看護師", "0357890", 0).
staff(18, "秋山　温輝", "看護師", "0469361", 0).
staff(19, "鈴木　優果", "看護師", "0469565", 0).
staff(20, "金田　莉里花", "看護師", "0469485", 0).
staff(21, "門西　知香", "看護師", "0125262", 0).
staff(22, "塚田　瑶", "看護師", "0357935", 0).
staff(23, "櫻井　千聖", "看護師", "0385804", 0).
staff(24, "㓛刀　七海", "看護師", "0606220", 0).
staff(25, "土橋　美佳", "看護師", "0341036", 0).
% Groups ---------------------------------------------------
group("全員").
group("全員", 1).
group("全員", 2).
group("全員", 3).
group("全員", 4).
group("全員", 5).
group("全員", 6).
group("全員", 7).
group("全員", 8).
group("全員", 9).
group("全員", 10).
group("全員", 11).
group("全員", 13).
group("全員", 14).
group("全員", 15).
group("全員", 16).
group("全員", 17).
group("全員", 18).
group("全員", 19).
group("全員", 20).
group("全員", 21).
group("全員", 22).
group("全員", 23).
group("全員", 24).
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("リーダー").
group("リーダー", 2).
group("リーダー", 3).
group("リーダー", 4).
group("リーダー", 5).
group("リーダー", 7).
group("リーダー", 8).
group("リーダー", 10).
group("リーダー", 13).
group("リーダー", 14).
group("リーダー", 15).
group("リーダー", 16).
group("平日代行").
group("平日代行", 2).
group("平日代行", 3).
group("平日代行", 4).
group("新人").
group("新人", 22).
group("新人", 23).
group("新人", 24).
group("夜勤").
group("夜勤", 2).
group("夜勤", 3).
group("夜勤", 4).
group("夜勤", 5).
group("夜勤", 6).
group("夜勤", 7).
group("夜勤", 8).
group("夜勤", 9).
group("夜勤", 10).
group("夜勤", 11).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 19).
group("夜勤", 20).
group("夜勤", 22).
group("夜勤", 23).
group("夜勤", 24).
% Shifts ---------------------------------------------------
shift("日").
shift("N").
shift("J").
shift("S").
shift("★").
shift("☆").
shift("A").
shift("P").
shift("○").
shift("◎").
shift("研").
shift("年").
shift("健").
shift("特").
shift("/").
% Shift constraints ----------------------------------------
% 連続勤務可能日数
consecutive_work_ub(5).
% 禁止パターン
forbidden_pattern("NN").
pattern("NN", 2).
pattern("NN", 0, "N").
pattern("NN", 1, "N").
forbidden_pattern("JSJS").
pattern("JSJS", 4).
pattern("JSJS", 0, "J").
pattern("JSJS", 1, "S").
pattern("JSJS", 2, "J").
pattern("JSJS", 3, "S").
forbidden_pattern("N日").
pattern("N日", 2).
pattern("N日", 0, "N").
pattern("N日", 1, "日").
forbidden_pattern("PN").
pattern("PN", 2).
pattern("PN", 0, "P").
pattern("PN", 1, "N").
forbidden_pattern("P日").
pattern("P日", 2).
pattern("P日", 0, "P").
pattern("P日", 1, "日").
forbidden_pattern("日日日日日N").
pattern("日日日日日N", 6).
pattern("日日日日日N", 0, "日").
pattern("日日日日日N", 1, "日").
pattern("日日日日日N", 2, "日").
pattern("日日日日日N", 3, "日").
pattern("日日日日日N", 4, "日").
pattern("日日日日日N", 5, "N").
forbidden_pattern("日日日日NP").
pattern("日日日日NP", 6).
pattern("日日日日NP", 0, "日").
pattern("日日日日NP", 1, "日").
pattern("日日日日NP", 2, "日").
pattern("日日日日NP", 3, "日").
pattern("日日日日NP", 4, "N").
pattern("日日日日NP", 5, "P").
forbidden_pattern("日日日日NJ").
pattern("日日日日NJ", 6).
pattern("日日日日NJ", 0, "日").
pattern("日日日日NJ", 1, "日").
pattern("日日日日NJ", 2, "日").
pattern("日日日日NJ", 3, "日").
pattern("日日日日NJ", 4, "N").
pattern("日日日日NJ", 5, "J").
forbidden_pattern("日日NJSJS").
pattern("日日NJSJS", 7).
pattern("日日NJSJS", 0, "日").
pattern("日日NJSJS", 1, "日").
pattern("日日NJSJS", 2, "N").
pattern("日日NJSJS", 3, "J").
pattern("日日NJSJS", 4, "S").
pattern("日日NJSJS", 5, "J").
pattern("日日NJSJS", 6, "S").
forbidden_pattern("日日日日日日").
pattern("日日日日日日", 6).
pattern("日日日日日日", 0, "日").
pattern("日日日日日日", 1, "日").
pattern("日日日日日日", 2, "日").
pattern("日日日日日日", 3, "日").
pattern("日日日日日日", 4, "日").
pattern("日日日日日日", 5, "日").
% 前日・翌日可能シフト
pred_shift("J", "S").
succ_shift("S", "J").
succ_shift("S", "○").
succ_shift("S", "◎").
pred_shift("N", "J").
pred_shift("P", "J").
pred_shift("日", "J").
pred_shift("○", "J").
pred_shift("◎", "J").
succ_shift("J", "S").
pred_shift("N", "★").
pred_shift("P", "★").
pred_shift("日", "★").
pred_shift("○", "★").
pred_shift("◎", "★").
succ_shift("★", "☆").
pred_shift("★", "☆").
succ_shift("☆", "○").
succ_shift("☆", "◎").
pred_shift("日", "日").
pred_shift("○", "日").
pred_shift("◎", "日").
succ_shift("日", "N").
succ_shift("日", "P").
succ_shift("日", "日").
succ_shift("日", "○").
succ_shift("日", "◎").
succ_shift("日", "J").
succ_shift("日", "★").
pred_shift("日", "N").
pred_shift("○", "N").
pred_shift("◎", "N").
succ_shift("N", "P").
succ_shift("N", "○").
succ_shift("N", "◎").
succ_shift("N", "J").
succ_shift("N", "★").
pred_shift("N", "P").
pred_shift("日", "P").
pred_shift("○", "P").
pred_shift("◎", "P").
succ_shift("P", "○").
succ_shift("P", "◎").
succ_shift("P", "J").
succ_shift("P", "★").
% パターン割当数
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
patterns_hard_lb("○○", 2).
patterns("N日", "N日").
patterns_soft_ub("N日", 0).
patterns("P日", "P日").
patterns_soft_ub("P日", 0).
patterns("PN", "PN").
patterns_soft_ub("PN", 0).
patterns("PP", "PP").
pattern("PP", 2).
pattern("PP", 0, "P").
pattern("PP", 1, "P").
patterns_soft_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("日").
shifts("日", "日").
staff_dweek_hard_lb("全員", "日", "平日", 3).
staff_dweek_hard_lb("全員", "日", "土日", 2).
staff_dweek_hard_lb("全員", "日", "祝日", 2).
staff_dweek_soft_lb("全員", "日", "平日", 3).
staff_dweek_soft_lb("全員", "日", "土日", 2).
staff_dweek_soft_lb("全員", "日", "祝日", 2).
staff_dweek_soft_ub("全員", "日", "平日", 7).
staff_dweek_soft_ub("全員", "日", "土日", 3).
staff_dweek_soft_ub("全員", "日", "祝日", 3).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 3).
staff_dweek_hard_lb("全員", "N", "土日", 3).
staff_dweek_hard_lb("全員", "N", "祝日", 3).
staff_dweek_soft_lb("全員", "N", "平日", 3).
staff_dweek_soft_lb("全員", "N", "土日", 3).
staff_dweek_soft_lb("全員", "N", "祝日", 3).
staff_dweek_soft_ub("全員", "N", "平日", 3).
staff_dweek_soft_ub("全員", "N", "土日", 3).
staff_dweek_soft_ub("全員", "N", "祝日", 3).
staff_dweek_hard_ub("全員", "N", "平日", 3).
staff_dweek_hard_ub("全員", "N", "土日", 3).
staff_dweek_hard_ub("全員", "N", "祝日", 3).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 3).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 3).
staff_dweek_hard_lb("全員", "J", "土日", 3).
staff_dweek_hard_lb("全員", "S", "土日", 3).
staff_dweek_hard_lb("全員", "J", "祝日", 3).
staff_dweek_hard_lb("全員", "S", "祝日", 3).
staff_dweek_soft_lb("全員", "J", "平日", 3).
staff_dweek_soft_lb("全員", "S", "平日", 3).
staff_dweek_soft_lb("全員", "J", "土日", 3).
staff_dweek_soft_lb("全員", "S", "土日", 3).
staff_dweek_soft_lb("全員", "J", "祝日", 3).
staff_dweek_soft_lb("全員", "S", "祝日", 3).
staff_dweek_soft_ub("全員", "J", "平日", 3).
staff_dweek_soft_ub("全員", "S", "平日", 3).
staff_dweek_soft_ub("全員", "J", "土日", 3).
staff_dweek_soft_ub("全員", "S", "土日", 3).
staff_dweek_soft_ub("全員", "J", "祝日", 3).
staff_dweek_soft_ub("全員", "S", "祝日", 3).
staff_dweek_hard_ub("全員", "J", "平日", 3).
staff_dweek_hard_ub("全員", "S", "平日", 3).
staff_dweek_hard_ub("全員", "J", "土日", 3).
staff_dweek_hard_ub("全員", "S", "土日", 3).
staff_dweek_hard_ub("全員", "J", "祝日", 3).
staff_dweek_hard_ub("全員", "S", "祝日", 3).
shifts("P").
shifts("P", "P").
staff_dweek_hard_lb("全員", "P", "平日", 0).
staff_dweek_hard_lb("全員", "P", "土日", 0).
staff_dweek_hard_lb("全員", "P", "祝日", 0).
staff_dweek_soft_lb("全員", "P", "平日", 0).
staff_dweek_soft_lb("全員", "P", "土日", 0).
staff_dweek_soft_lb("全員", "P", "祝日", 0).
staff_dweek_soft_ub("全員", "P", "平日", 0).
staff_dweek_soft_ub("全員", "P", "土日", 0).
staff_dweek_soft_ub("全員", "P", "祝日", 0).
staff_dweek_hard_ub("全員", "P", "平日", 0).
staff_dweek_hard_ub("全員", "P", "土日", 0).
staff_dweek_hard_ub("全員", "P", "祝日", 0).
staff_dweek_hard_lb("代行", "日", "平日", 1).
staff_dweek_hard_lb("代行", "N", "平日", 1).
staff_dweek_hard_lb("代行", "日", "土日", 1).
staff_dweek_hard_lb("代行", "N", "土日", 1).
staff_dweek_hard_lb("代行", "日", "祝日", 1).
staff_dweek_hard_lb("代行", "N", "祝日", 1).
staff_dweek_soft_lb("代行", "日", "平日", 1).
staff_dweek_soft_lb("代行", "N", "平日", 1).
staff_dweek_soft_lb("代行", "日", "土日", 1).
staff_dweek_soft_lb("代行", "N", "土日", 1).
staff_dweek_soft_lb("代行", "日", "祝日", 1).
staff_dweek_soft_lb("代行", "N", "祝日", 1).
staff_dweek_soft_ub("代行", "日", "平日", 1).
staff_dweek_soft_ub("代行", "N", "平日", 1).
staff_dweek_soft_ub("代行", "日", "土日", 1).
staff_dweek_soft_ub("代行", "N", "土日", 1).
staff_dweek_soft_ub("代行", "日", "祝日", 1).
staff_dweek_soft_ub("代行", "N", "祝日", 1).
staff_dweek_hard_ub("代行", "日", "平日", 1).
staff_dweek_hard_ub("代行", "N", "平日", 1).
staff_dweek_hard_ub("代行", "日", "土日", 1).
staff_dweek_hard_ub("代行", "N", "土日", 1).
staff_dweek_hard_ub("代行", "日", "祝日", 1).
staff_dweek_hard_ub("代行", "N", "祝日", 1).
staff_dweek_hard_lb("リーダー", "N", "平日", 1).
staff_dweek_hard_lb("リーダー", "J", "平日", 1).
staff_dweek_hard_lb("リーダー", "S", "平日", 1).
staff_dweek_hard_lb("リーダー", "N", "土日", 1).
staff_dweek_hard_lb("リーダー", "J", "土日", 1).
staff_dweek_hard_lb("リーダー", "S", "土日", 1).
staff_dweek_hard_lb("リーダー", "N", "祝日", 1).
staff_dweek_hard_lb("リーダー", "J", "祝日", 1).
staff_dweek_hard_lb("リーダー", "S", "祝日", 1).
staff_dweek_soft_lb("リーダー", "N", "平日", 1).
staff_dweek_soft_lb("リーダー", "J", "平日", 1).
staff_dweek_soft_lb("リーダー", "S", "平日", 1).
staff_dweek_soft_lb("リーダー", "N", "土日", 1).
staff_dweek_soft_lb("リーダー", "J", "土日", 1).
staff_dweek_soft_lb("リーダー", "S", "土日", 1).
staff_dweek_soft_lb("リーダー", "N", "祝日", 1).
staff_dweek_soft_lb("リーダー", "J", "祝日", 1).
staff_dweek_soft_lb("リーダー", "S", "祝日", 1).
staff_dweek_soft_ub("リーダー", "N", "平日", 1).
staff_dweek_soft_ub("リーダー", "J", "平日", 1).
staff_dweek_soft_ub("リーダー", "S", "平日", 1).
staff_dweek_soft_ub("リーダー", "N", "土日", 1).
staff_dweek_soft_ub("リーダー", "J", "土日", 1).
staff_dweek_soft_ub("リーダー", "S", "土日", 1).
staff_dweek_soft_ub("リーダー", "N", "祝日", 1).
staff_dweek_soft_ub("リーダー", "J", "祝日", 1).
staff_dweek_soft_ub("リーダー", "S", "祝日", 1).
staff_dweek_hard_ub("リーダー", "N", "平日", 2).
staff_dweek_hard_ub("リーダー", "J", "平日", 2).
staff_dweek_hard_ub("リーダー", "S", "平日", 2).
staff_dweek_hard_ub("リーダー", "N", "土日", 2).
staff_dweek_hard_ub("リーダー", "J", "土日", 2).
staff_dweek_hard_ub("リーダー", "S", "土日", 2).
staff_dweek_hard_ub("リーダー", "N", "祝日", 2).
staff_dweek_hard_ub("リーダー", "J", "祝日", 2).
staff_dweek_hard_ub("リーダー", "S", "祝日", 2).
staff_dweek_hard_lb("平日代行", "日", "平日", 1).
staff_dweek_hard_lb("平日代行", "日", "土日", 0).
staff_dweek_hard_lb("平日代行", "日", "祝日", 0).
staff_dweek_soft_lb("平日代行", "日", "平日", 1).
staff_dweek_soft_lb("平日代行", "日", "土日", 0).
staff_dweek_soft_lb("平日代行", "日", "祝日", 0).
staff_dweek_soft_ub("平日代行", "日", "平日", 1).
staff_dweek_soft_ub("平日代行", "日", "土日", 0).
staff_dweek_soft_ub("平日代行", "日", "祝日", 0).
staff_dweek_hard_ub("平日代行", "日", "平日", 1).
staff_dweek_hard_ub("平日代行", "日", "土日", 0).
staff_dweek_hard_ub("平日代行", "日", "祝日", 0).
staff_dweek_hard_lb("新人", "J", "平日", 0).
staff_dweek_hard_lb("新人", "S", "平日", 0).
staff_dweek_hard_lb("新人", "J", "土日", 0).
staff_dweek_hard_lb("新人", "S", "土日", 0).
staff_dweek_hard_lb("新人", "J", "祝日", 0).
staff_dweek_hard_lb("新人", "S", "祝日", 0).
staff_dweek_soft_lb("新人", "J", "平日", 1).
staff_dweek_soft_lb("新人", "S", "平日", 1).
staff_dweek_soft_lb("新人", "J", "土日", 1).
staff_dweek_soft_lb("新人", "S", "土日", 1).
staff_dweek_soft_lb("新人", "J", "祝日", 1).
staff_dweek_soft_lb("新人", "S", "祝日", 1).
staff_dweek_soft_ub("新人", "J", "平日", 1).
staff_dweek_soft_ub("新人", "S", "平日", 1).
staff_dweek_soft_ub("新人", "J", "土日", 1).
staff_dweek_soft_ub("新人", "S", "土日", 1).
staff_dweek_soft_ub("新人", "J", "祝日", 1).
staff_dweek_soft_ub("新人", "S", "祝日", 1).
staff_dweek_hard_ub("新人", "J", "平日", 1).
staff_dweek_hard_ub("新人", "S", "平日", 1).
staff_dweek_hard_ub("新人", "J", "土日", 1).
staff_dweek_hard_ub("新人", "S", "土日", 1).
staff_dweek_hard_ub("新人", "J", "祝日", 1).
staff_dweek_hard_ub("新人", "S", "祝日", 1).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("代行","日","人数").
vertical_constraint_type("代行","N","人数").
vertical_constraint_type("リーダー","N","人数").
vertical_constraint_type("リーダー","J","人数").
vertical_constraint_type("リーダー","S","人数").
vertical_constraint_type("平日代行","日","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts("★").
shifts("★", "★").
shifts_hard_lb(1, "★", 0).
shifts("☆").
shifts("☆", "☆").
shifts_hard_lb(1, "☆", 0).
shifts_soft_lb(1, "★", 0).
shifts_soft_lb(1, "☆", 0).
shifts_soft_ub(1, "★", 0).
shifts_soft_ub(1, "☆", 0).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_hard_lb(1, "P", 0).
shifts_soft_lb(1, "P", 0).
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_lb(1, "A", 0).
shifts_soft_lb(1, "A", 0).
shifts_soft_ub(1, "A", 0).
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_hard_lb(2, "N", 1).
shifts_hard_lb(2, "J", 1).
shifts_hard_lb(2, "S", 1).
shifts_soft_lb(2, "N", 2).
shifts_soft_lb(2, "J", 2).
shifts_soft_lb(2, "S", 2).
shifts_soft_ub(2, "N", 3).
shifts_soft_ub(2, "J", 3).
shifts_soft_ub(2, "S", 3).
shifts_hard_ub(2, "N", 3).
shifts_hard_ub(2, "J", 3).
shifts_hard_ub(2, "S", 3).
shifts_hard_lb(2, "★", 0).
shifts_hard_lb(2, "☆", 0).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_ub(2, "★", 0).
shifts_soft_ub(2, "☆", 0).
shifts_hard_ub(2, "★", 0).
shifts_hard_ub(2, "☆", 0).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "P", 0).
shifts_hard_ub(2, "P", 0).
shifts_hard_lb(2, "A", 0).
shifts_soft_lb(2, "A", 0).
shifts_soft_ub(2, "A", 0).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_hard_lb(3, "N", 1).
shifts_hard_lb(3, "J", 1).
shifts_hard_lb(3, "S", 1).
shifts_soft_lb(3, "N", 2).
shifts_soft_lb(3, "J", 2).
shifts_soft_lb(3, "S", 2).
shifts_soft_ub(3, "N", 3).
shifts_soft_ub(3, "J", 3).
shifts_soft_ub(3, "S", 3).
shifts_hard_ub(3, "N", 3).
shifts_hard_ub(3, "J", 3).
shifts_hard_ub(3, "S", 3).
shifts_hard_lb(3, "★", 0).
shifts_hard_lb(3, "☆", 0).
shifts_soft_lb(3, "★", 0).
shifts_soft_lb(3, "☆", 0).
shifts_soft_ub(3, "★", 0).
shifts_soft_ub(3, "☆", 0).
shifts_hard_ub(3, "★", 0).
shifts_hard_ub(3, "☆", 0).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "P", 0).
shifts_hard_lb(3, "A", 0).
shifts_soft_lb(3, "A", 0).
shifts_soft_ub(3, "A", 0).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_hard_lb(4, "N", 1).
shifts_hard_lb(4, "J", 1).
shifts_hard_lb(4, "S", 1).
shifts_soft_lb(4, "N", 2).
shifts_soft_lb(4, "J", 2).
shifts_soft_lb(4, "S", 2).
shifts_soft_ub(4, "N", 3).
shifts_soft_ub(4, "J", 3).
shifts_soft_ub(4, "S", 3).
shifts_hard_ub(4, "N", 3).
shifts_hard_ub(4, "J", 3).
shifts_hard_ub(4, "S", 3).
shifts_hard_lb(4, "★", 0).
shifts_hard_lb(4, "☆", 0).
shifts_soft_lb(4, "★", 0).
shifts_soft_lb(4, "☆", 0).
shifts_soft_ub(4, "★", 0).
shifts_soft_ub(4, "☆", 0).
shifts_hard_ub(4, "★", 0).
shifts_hard_ub(4, "☆", 0).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "P", 0).
shifts_hard_ub(4, "P", 0).
shifts_hard_lb(4, "A", 0).
shifts_soft_lb(4, "A", 0).
shifts_soft_ub(4, "A", 0).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_hard_lb(5, "N", 1).
shifts_hard_lb(5, "J", 1).
shifts_hard_lb(5, "S", 1).
shifts_soft_lb(5, "N", 3).
shifts_soft_lb(5, "J", 3).
shifts_soft_lb(5, "S", 3).
shifts_soft_ub(5, "N", 4).
shifts_soft_ub(5, "J", 4).
shifts_soft_ub(5, "S", 4).
shifts_hard_ub(5, "N", 6).
shifts_hard_ub(5, "J", 6).
shifts_hard_ub(5, "S", 6).
shifts_hard_lb(5, "★", 0).
shifts_hard_lb(5, "☆", 0).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_ub(5, "★", 0).
shifts_soft_ub(5, "☆", 0).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "P", 0).
shifts_hard_lb(5, "A", 0).
shifts_soft_lb(5, "A", 0).
shifts_soft_ub(5, "A", 0).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_hard_lb(6, "N", 1).
shifts_hard_lb(6, "J", 1).
shifts_hard_lb(6, "S", 1).
shifts_soft_lb(6, "N", 3).
shifts_soft_lb(6, "J", 3).
shifts_soft_lb(6, "S", 3).
shifts_soft_ub(6, "N", 4).
shifts_soft_ub(6, "J", 4).
shifts_soft_ub(6, "S", 4).
shifts_hard_ub(6, "N", 6).
shifts_hard_ub(6, "J", 6).
shifts_hard_ub(6, "S", 6).
shifts_hard_lb(6, "★", 0).
shifts_hard_lb(6, "☆", 0).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_ub(6, "★", 0).
shifts_soft_ub(6, "☆", 0).
shifts_hard_ub(6, "★", 0).
shifts_hard_ub(6, "☆", 0).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "P", 0).
shifts_hard_lb(6, "A", 0).
shifts_soft_lb(6, "A", 0).
shifts_soft_ub(6, "A", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_hard_lb(7, "N", 1).
shifts_hard_lb(7, "J", 1).
shifts_hard_lb(7, "S", 1).
shifts_soft_lb(7, "N", 2).
shifts_soft_lb(7, "J", 2).
shifts_soft_lb(7, "S", 2).
shifts_soft_ub(7, "N", 2).
shifts_soft_ub(7, "J", 2).
shifts_soft_ub(7, "S", 2).
shifts_hard_ub(7, "N", 2).
shifts_hard_ub(7, "J", 2).
shifts_hard_ub(7, "S", 2).
shifts_hard_lb(7, "★", 0).
shifts_hard_lb(7, "☆", 0).
shifts_soft_lb(7, "★", 0).
shifts_soft_lb(7, "☆", 0).
shifts_soft_ub(7, "★", 0).
shifts_soft_ub(7, "☆", 0).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "P", 0).
shifts_hard_lb(7, "A", 0).
shifts_soft_lb(7, "A", 0).
shifts_soft_ub(7, "A", 0).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_hard_lb(8, "N", 1).
shifts_hard_lb(8, "J", 1).
shifts_hard_lb(8, "S", 1).
shifts_soft_lb(8, "N", 2).
shifts_soft_lb(8, "J", 2).
shifts_soft_lb(8, "S", 2).
shifts_soft_ub(8, "N", 3).
shifts_soft_ub(8, "J", 3).
shifts_soft_ub(8, "S", 3).
shifts_hard_ub(8, "N", 4).
shifts_hard_ub(8, "J", 4).
shifts_hard_ub(8, "S", 4).
shifts_hard_lb(8, "★", 0).
shifts_hard_lb(8, "☆", 0).
shifts_soft_lb(8, "★", 0).
shifts_soft_lb(8, "☆", 0).
shifts_soft_ub(8, "★", 0).
shifts_soft_ub(8, "☆", 0).
shifts_hard_ub(8, "★", 0).
shifts_hard_ub(8, "☆", 0).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "P", 0).
shifts_hard_lb(8, "A", 0).
shifts_soft_lb(8, "A", 0).
shifts_soft_ub(8, "A", 0).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_hard_lb(9, "N", 1).
shifts_hard_lb(9, "J", 1).
shifts_hard_lb(9, "S", 1).
shifts_soft_lb(9, "N", 2).
shifts_soft_lb(9, "J", 2).
shifts_soft_lb(9, "S", 2).
shifts_soft_ub(9, "N", 2).
shifts_soft_ub(9, "J", 2).
shifts_soft_ub(9, "S", 2).
shifts_hard_ub(9, "N", 2).
shifts_hard_ub(9, "J", 2).
shifts_hard_ub(9, "S", 2).
shifts_hard_lb(9, "★", 0).
shifts_hard_lb(9, "☆", 0).
shifts_soft_lb(9, "★", 0).
shifts_soft_lb(9, "☆", 0).
shifts_soft_ub(9, "★", 0).
shifts_soft_ub(9, "☆", 0).
shifts_hard_ub(9, "★", 0).
shifts_hard_ub(9, "☆", 0).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "P", 0).
shifts_hard_ub(9, "P", 0).
shifts_hard_lb(9, "A", 0).
shifts_soft_lb(9, "A", 0).
shifts_soft_ub(9, "A", 0).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_hard_lb(10, "N", 1).
shifts_hard_lb(10, "J", 1).
shifts_hard_lb(10, "S", 1).
shifts_soft_lb(10, "N", 3).
shifts_soft_lb(10, "J", 3).
shifts_soft_lb(10, "S", 3).
shifts_soft_ub(10, "N", 4).
shifts_soft_ub(10, "J", 4).
shifts_soft_ub(10, "S", 4).
shifts_hard_ub(10, "N", 6).
shifts_hard_ub(10, "J", 6).
shifts_hard_ub(10, "S", 6).
shifts_hard_lb(10, "★", 0).
shifts_hard_lb(10, "☆", 0).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_ub(10, "★", 0).
shifts_soft_ub(10, "☆", 0).
shifts_hard_ub(10, "★", 0).
shifts_hard_ub(10, "☆", 0).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "P", 0).
shifts_hard_ub(10, "P", 0).
shifts_hard_lb(10, "A", 0).
shifts_soft_lb(10, "A", 0).
shifts_soft_ub(10, "A", 0).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_hard_lb(11, "N", 1).
shifts_hard_lb(11, "J", 1).
shifts_hard_lb(11, "S", 1).
shifts_soft_lb(11, "N", 3).
shifts_soft_lb(11, "J", 3).
shifts_soft_lb(11, "S", 3).
shifts_soft_ub(11, "N", 4).
shifts_soft_ub(11, "J", 4).
shifts_soft_ub(11, "S", 4).
shifts_hard_ub(11, "N", 6).
shifts_hard_ub(11, "J", 6).
shifts_hard_ub(11, "S", 6).
shifts_hard_lb(11, "★", 0).
shifts_hard_lb(11, "☆", 0).
shifts_soft_lb(11, "★", 0).
shifts_soft_lb(11, "☆", 0).
shifts_soft_ub(11, "★", 0).
shifts_soft_ub(11, "☆", 0).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "P", 0).
shifts_hard_lb(11, "A", 0).
shifts_soft_lb(11, "A", 0).
shifts_soft_ub(11, "A", 0).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_hard_lb(12, "★", 0).
shifts_hard_lb(12, "☆", 0).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_ub(12, "★", 0).
shifts_soft_ub(12, "☆", 0).
shifts_hard_ub(12, "★", 0).
shifts_hard_ub(12, "☆", 0).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "P", 0).
shifts_hard_ub(12, "P", 0).
shifts_hard_lb(12, "A", 0).
shifts_soft_lb(12, "A", 0).
shifts_soft_ub(12, "A", 0).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_hard_lb(13, "N", 1).
shifts_hard_lb(13, "J", 1).
shifts_hard_lb(13, "S", 1).
shifts_soft_lb(13, "N", 3).
shifts_soft_lb(13, "J", 3).
shifts_soft_lb(13, "S", 3).
shifts_soft_ub(13, "N", 4).
shifts_soft_ub(13, "J", 4).
shifts_soft_ub(13, "S", 4).
shifts_hard_ub(13, "N", 6).
shifts_hard_ub(13, "J", 6).
shifts_hard_ub(13, "S", 6).
shifts_hard_lb(13, "★", 0).
shifts_hard_lb(13, "☆", 0).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_ub(13, "★", 0).
shifts_soft_ub(13, "☆", 0).
shifts_hard_ub(13, "★", 0).
shifts_hard_ub(13, "☆", 0).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "P", 0).
shifts_hard_ub(13, "P", 0).
shifts_hard_lb(13, "A", 0).
shifts_soft_lb(13, "A", 0).
shifts_soft_ub(13, "A", 0).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_hard_lb(14, "N", 1).
shifts_hard_lb(14, "J", 1).
shifts_hard_lb(14, "S", 1).
shifts_soft_lb(14, "N", 3).
shifts_soft_lb(14, "J", 3).
shifts_soft_lb(14, "S", 3).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_hard_ub(14, "N", 6).
shifts_hard_ub(14, "J", 6).
shifts_hard_ub(14, "S", 6).
shifts_hard_lb(14, "★", 0).
shifts_hard_lb(14, "☆", 0).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_ub(14, "★", 0).
shifts_soft_ub(14, "☆", 0).
shifts_hard_ub(14, "★", 0).
shifts_hard_ub(14, "☆", 0).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "P", 0).
shifts_hard_ub(14, "P", 0).
shifts_hard_lb(14, "A", 0).
shifts_soft_lb(14, "A", 0).
shifts_soft_ub(14, "A", 0).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_hard_lb(15, "N", 1).
shifts_hard_lb(15, "J", 1).
shifts_hard_lb(15, "S", 1).
shifts_soft_lb(15, "N", 3).
shifts_soft_lb(15, "J", 3).
shifts_soft_lb(15, "S", 3).
shifts_soft_ub(15, "N", 4).
shifts_soft_ub(15, "J", 4).
shifts_soft_ub(15, "S", 4).
shifts_hard_ub(15, "N", 6).
shifts_hard_ub(15, "J", 6).
shifts_hard_ub(15, "S", 6).
shifts_hard_lb(15, "★", 0).
shifts_hard_lb(15, "☆", 0).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_ub(15, "★", 0).
shifts_soft_ub(15, "☆", 0).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "P", 0).
shifts_hard_lb(15, "A", 0).
shifts_soft_lb(15, "A", 0).
shifts_soft_ub(15, "A", 0).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_hard_lb(16, "N", 1).
shifts_hard_lb(16, "J", 1).
shifts_hard_lb(16, "S", 1).
shifts_soft_lb(16, "N", 3).
shifts_soft_lb(16, "J", 3).
shifts_soft_lb(16, "S", 3).
shifts_soft_ub(16, "N", 4).
shifts_soft_ub(16, "J", 4).
shifts_soft_ub(16, "S", 4).
shifts_hard_ub(16, "N", 6).
shifts_hard_ub(16, "J", 6).
shifts_hard_ub(16, "S", 6).
shifts_hard_lb(16, "★", 0).
shifts_hard_lb(16, "☆", 0).
shifts_soft_lb(16, "★", 0).
shifts_soft_lb(16, "☆", 0).
shifts_soft_ub(16, "★", 0).
shifts_soft_ub(16, "☆", 0).
shifts_hard_ub(16, "★", 0).
shifts_hard_ub(16, "☆", 0).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "P", 0).
shifts_hard_ub(16, "P", 0).
shifts_hard_lb(16, "A", 0).
shifts_soft_lb(16, "A", 0).
shifts_soft_ub(16, "A", 0).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_hard_lb(17, "N", 1).
shifts_hard_lb(17, "J", 1).
shifts_hard_lb(17, "S", 1).
shifts_soft_lb(17, "N", 3).
shifts_soft_lb(17, "J", 3).
shifts_soft_lb(17, "S", 3).
shifts_soft_ub(17, "N", 4).
shifts_soft_ub(17, "J", 4).
shifts_soft_ub(17, "S", 4).
shifts_hard_ub(17, "N", 6).
shifts_hard_ub(17, "J", 6).
shifts_hard_ub(17, "S", 6).
shifts_hard_lb(17, "★", 0).
shifts_hard_lb(17, "☆", 0).
shifts_soft_lb(17, "★", 0).
shifts_soft_lb(17, "☆", 0).
shifts_soft_ub(17, "★", 0).
shifts_soft_ub(17, "☆", 0).
shifts_hard_ub(17, "★", 0).
shifts_hard_ub(17, "☆", 0).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "P", 0).
shifts_hard_ub(17, "P", 0).
shifts_hard_lb(17, "A", 0).
shifts_soft_lb(17, "A", 0).
shifts_soft_ub(17, "A", 0).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_hard_lb(18, "N", 1).
shifts_hard_lb(18, "J", 1).
shifts_hard_lb(18, "S", 1).
shifts_soft_lb(18, "N", 3).
shifts_soft_lb(18, "J", 3).
shifts_soft_lb(18, "S", 3).
shifts_soft_ub(18, "N", 4).
shifts_soft_ub(18, "J", 4).
shifts_soft_ub(18, "S", 4).
shifts_hard_ub(18, "N", 6).
shifts_hard_ub(18, "J", 6).
shifts_hard_ub(18, "S", 6).
shifts_hard_lb(18, "★", 0).
shifts_hard_lb(18, "☆", 0).
shifts_soft_lb(18, "★", 0).
shifts_soft_lb(18, "☆", 0).
shifts_soft_ub(18, "★", 0).
shifts_soft_ub(18, "☆", 0).
shifts_hard_ub(18, "★", 0).
shifts_hard_ub(18, "☆", 0).
shifts_hard_lb(18, "P", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "P", 0).
shifts_hard_ub(18, "P", 0).
shifts_hard_lb(18, "A", 0).
shifts_soft_lb(18, "A", 0).
shifts_soft_ub(18, "A", 0).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_hard_lb(19, "N", 1).
shifts_hard_lb(19, "J", 1).
shifts_hard_lb(19, "S", 1).
shifts_soft_lb(19, "N", 3).
shifts_soft_lb(19, "J", 3).
shifts_soft_lb(19, "S", 3).
shifts_soft_ub(19, "N", 4).
shifts_soft_ub(19, "J", 4).
shifts_soft_ub(19, "S", 4).
shifts_hard_ub(19, "N", 6).
shifts_hard_ub(19, "J", 6).
shifts_hard_ub(19, "S", 6).
shifts_hard_lb(19, "★", 0).
shifts_hard_lb(19, "☆", 0).
shifts_soft_lb(19, "★", 0).
shifts_soft_lb(19, "☆", 0).
shifts_soft_ub(19, "★", 0).
shifts_soft_ub(19, "☆", 0).
shifts_hard_ub(19, "★", 0).
shifts_hard_ub(19, "☆", 0).
shifts_hard_lb(19, "P", 0).
shifts_soft_lb(19, "P", 0).
shifts_soft_ub(19, "P", 0).
shifts_hard_ub(19, "P", 0).
shifts_hard_lb(19, "A", 0).
shifts_soft_lb(19, "A", 0).
shifts_soft_ub(19, "A", 0).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_hard_lb(20, "N", 1).
shifts_hard_lb(20, "J", 1).
shifts_hard_lb(20, "S", 1).
shifts_soft_lb(20, "N", 3).
shifts_soft_lb(20, "J", 3).
shifts_soft_lb(20, "S", 3).
shifts_soft_ub(20, "N", 4).
shifts_soft_ub(20, "J", 4).
shifts_soft_ub(20, "S", 4).
shifts_hard_ub(20, "N", 6).
shifts_hard_ub(20, "J", 6).
shifts_hard_ub(20, "S", 6).
shifts_hard_lb(20, "★", 0).
shifts_hard_lb(20, "☆", 0).
shifts_soft_lb(20, "★", 0).
shifts_soft_lb(20, "☆", 0).
shifts_soft_ub(20, "★", 0).
shifts_soft_ub(20, "☆", 0).
shifts_hard_ub(20, "★", 0).
shifts_hard_ub(20, "☆", 0).
shifts_hard_lb(20, "P", 0).
shifts_soft_lb(20, "P", 0).
shifts_soft_ub(20, "P", 0).
shifts_hard_ub(20, "P", 0).
shifts_hard_lb(20, "A", 0).
shifts_soft_lb(20, "A", 0).
shifts_soft_ub(20, "A", 0).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_hard_lb(21, "★", 0).
shifts_hard_lb(21, "☆", 0).
shifts_soft_lb(21, "★", 0).
shifts_soft_lb(21, "☆", 0).
shifts_soft_ub(21, "★", 0).
shifts_soft_ub(21, "☆", 0).
shifts_hard_ub(21, "★", 0).
shifts_hard_ub(21, "☆", 0).
shifts_hard_lb(21, "P", 0).
shifts_soft_lb(21, "P", 0).
shifts_soft_ub(21, "P", 0).
shifts_hard_ub(21, "P", 0).
shifts_hard_lb(21, "A", 0).
shifts_soft_lb(21, "A", 0).
shifts_soft_ub(21, "A", 0).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_hard_lb(22, "N", 1).
shifts_hard_lb(22, "J", 1).
shifts_hard_lb(22, "S", 1).
shifts_soft_lb(22, "N", 2).
shifts_soft_lb(22, "J", 2).
shifts_soft_lb(22, "S", 2).
shifts_soft_ub(22, "N", 3).
shifts_soft_ub(22, "J", 3).
shifts_soft_ub(22, "S", 3).
shifts_hard_ub(22, "N", 3).
shifts_hard_ub(22, "J", 3).
shifts_hard_ub(22, "S", 3).
shifts_hard_lb(22, "★", 0).
shifts_hard_lb(22, "☆", 0).
shifts_soft_lb(22, "★", 0).
shifts_soft_lb(22, "☆", 0).
shifts_soft_ub(22, "★", 0).
shifts_soft_ub(22, "☆", 0).
shifts_hard_ub(22, "★", 0).
shifts_hard_ub(22, "☆", 0).
shifts_hard_lb(22, "P", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "P", 0).
shifts_hard_ub(22, "P", 0).
shifts_hard_lb(22, "A", 0).
shifts_soft_lb(22, "A", 0).
shifts_soft_ub(22, "A", 0).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_hard_lb(23, "N", 1).
shifts_hard_lb(23, "J", 1).
shifts_hard_lb(23, "S", 1).
shifts_soft_lb(23, "N", 2).
shifts_soft_lb(23, "J", 2).
shifts_soft_lb(23, "S", 2).
shifts_soft_ub(23, "N", 3).
shifts_soft_ub(23, "J", 3).
shifts_soft_ub(23, "S", 3).
shifts_hard_ub(23, "N", 3).
shifts_hard_ub(23, "J", 3).
shifts_hard_ub(23, "S", 3).
shifts_hard_lb(23, "★", 0).
shifts_hard_lb(23, "☆", 0).
shifts_soft_lb(23, "★", 0).
shifts_soft_lb(23, "☆", 0).
shifts_soft_ub(23, "★", 0).
shifts_soft_ub(23, "☆", 0).
shifts_hard_ub(23, "★", 0).
shifts_hard_ub(23, "☆", 0).
shifts_hard_lb(23, "P", 0).
shifts_soft_lb(23, "P", 0).
shifts_soft_ub(23, "P", 0).
shifts_hard_ub(23, "P", 0).
shifts_hard_lb(23, "A", 0).
shifts_soft_lb(23, "A", 0).
shifts_soft_ub(23, "A", 0).
shifts_hard_ub(23, "A", 0).
% Staff 24:
shifts_hard_lb(24, "N", 1).
shifts_hard_lb(24, "J", 1).
shifts_hard_lb(24, "S", 1).
shifts_soft_lb(24, "N", 2).
shifts_soft_lb(24, "J", 2).
shifts_soft_lb(24, "S", 2).
shifts_soft_ub(24, "N", 3).
shifts_soft_ub(24, "J", 3).
shifts_soft_ub(24, "S", 3).
shifts_hard_ub(24, "N", 3).
shifts_hard_ub(24, "J", 3).
shifts_hard_ub(24, "S", 3).
shifts_hard_lb(24, "★", 0).
shifts_hard_lb(24, "☆", 0).
shifts_soft_lb(24, "★", 0).
shifts_soft_lb(24, "☆", 0).
shifts_soft_ub(24, "★", 0).
shifts_soft_ub(24, "☆", 0).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_hard_lb(24, "P", 0).
shifts_soft_lb(24, "P", 0).
shifts_soft_ub(24, "P", 0).
shifts_hard_ub(24, "P", 0).
shifts_hard_lb(24, "A", 0).
shifts_soft_lb(24, "A", 0).
shifts_soft_ub(24, "A", 0).
shifts_hard_ub(24, "A", 0).
% Staff 25:
horizontal_constraint_type("日").
horizontal_constraint_type("N").
horizontal_constraint_type("J").
horizontal_constraint_type("S").
horizontal_constraint_type("★").
horizontal_constraint_type("☆").
horizontal_constraint_type("P").
horizontal_constraint_type("A").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
