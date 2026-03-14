% Staffs ---------------------------------------------------
staff(1, "寺島　由美子", "師長", "0102025", 0).
staff(2, "杉本　美貴", "副師長", "0051525", 0).
staff(3, "小池　友佳梨", "看護師", "0283402", 0).
staff(4, "築野　梨奈", "看護師", "0312840", 0).
staff(5, "砂田　早紀", "看護師", "0326562", 0).
staff(6, "川村　友実", "看護師", "0341003", 0).
staff(7, "林　愛海", "看護師", "0370817", 0).
staff(8, "加賀美　里奈", "看護師", "0418450", 0).
staff(9, "稲村　結花", "看護師", "0315843", 0).
staff(10, "池亀　莉菜", "看護師", "0606127", 0).
staff(11, "手塚　浩美", "副師長", "0192093", 0).
staff(12, "舘田　安希子", "看護師", "0031188", 0).
staff(13, "立川　理奈子", "看護師", "0326607", 0).
staff(14, "古屋　愛有香", "看護師", "0312920", 0).
staff(15, "原　砂貴子", "看護師", "0341218", 0).
staff(16, "森本　柚実", "看護師", "0355705", 0).
staff(17, "梶原　葉南", "看護師", "0418448", 0).
staff(18, "清水　りな", "看護師", "0606344", 0).
staff(19, "内藤　聖七", "看護師", "0418585", 0).
staff(20, "剣持　友利亜", "看護師", "0312678", 0).
% Groups ---------------------------------------------------
group("全員").
group("師長").
group("A").
group("B").
group("新人").
group("夜勤").
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
consecutive_work_ub(6).
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
patterns_hard_lb("○○", 3).
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
staff_dweek_hard_lb("全員", "日", "平日", 1).
staff_dweek_hard_lb("全員", "日", "土日", 1).
staff_dweek_hard_lb("全員", "日", "祝日", 1).
staff_dweek_soft_lb("全員", "日", "平日", 2).
staff_dweek_soft_lb("全員", "日", "土日", 2).
staff_dweek_soft_lb("全員", "日", "祝日", 2).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 2).
staff_dweek_hard_lb("全員", "N", "土日", 2).
staff_dweek_hard_lb("全員", "N", "祝日", 2).
staff_dweek_soft_lb("全員", "N", "平日", 3).
staff_dweek_soft_lb("全員", "N", "土日", 3).
staff_dweek_soft_lb("全員", "N", "祝日", 3).
staff_dweek_soft_ub("全員", "N", "平日", 3).
staff_dweek_soft_ub("全員", "N", "土日", 3).
staff_dweek_soft_ub("全員", "N", "祝日", 3).
staff_dweek_hard_ub("全員", "N", "平日", 4).
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
shifts("★").
shifts("★", "★").
staff_dweek_soft_ub("全員", "★", "平日", 0).
shifts("☆").
shifts("☆", "☆").
staff_dweek_soft_ub("全員", "☆", "平日", 0).
staff_dweek_soft_ub("全員", "★", "土日", 0).
staff_dweek_soft_ub("全員", "☆", "土日", 0).
staff_dweek_soft_ub("全員", "★", "祝日", 0).
staff_dweek_soft_ub("全員", "☆", "祝日", 0).
staff_dweek_hard_ub("全員", "★", "平日", 1).
staff_dweek_hard_ub("全員", "☆", "平日", 1).
staff_dweek_hard_ub("全員", "★", "土日", 0).
staff_dweek_hard_ub("全員", "☆", "土日", 0).
staff_dweek_hard_ub("全員", "★", "祝日", 0).
staff_dweek_hard_ub("全員", "☆", "祝日", 0).
shifts("J+★").
shifts("J+★", "J").
shifts("J+★", "★").
staff_dweek_hard_ub("全員", "J+★", "平日", 3).
staff_dweek_hard_ub("全員", "J+★", "土日", 0).
staff_dweek_hard_ub("全員", "J+★", "祝日", 0).
shifts("P").
shifts("P", "P").
staff_dweek_hard_ub("全員", "P", "平日", 1).
staff_dweek_hard_ub("全員", "P", "土日", 0).
staff_dweek_hard_ub("全員", "P", "祝日", 0).
shifts("N+P+★").
shifts("N+P+★", "N").
shifts("N+P+★", "P").
shifts("N+P+★", "★").
staff_dweek_hard_lb("全員", "N+P+★", "平日", 3).
staff_dweek_hard_lb("全員", "N+P+★", "土日", 3).
staff_dweek_hard_lb("全員", "N+P+★", "祝日", 3).
staff_dweek_soft_ub("全員", "N+P+★", "平日", 5).
staff_dweek_soft_ub("全員", "N+P+★", "土日", 4).
staff_dweek_soft_ub("全員", "N+P+★", "祝日", 4).
staff_dweek_hard_lb("A", "N", "平日", 1).
staff_dweek_hard_lb("A", "J", "平日", 1).
staff_dweek_hard_lb("A", "S", "平日", 1).
staff_dweek_hard_lb("A", "N", "土日", 1).
staff_dweek_hard_lb("A", "J", "土日", 1).
staff_dweek_hard_lb("A", "S", "土日", 1).
staff_dweek_hard_lb("A", "N", "祝日", 1).
staff_dweek_hard_lb("A", "J", "祝日", 1).
staff_dweek_hard_lb("A", "S", "祝日", 1).
staff_dweek_soft_lb("A", "N", "平日", 2).
staff_dweek_soft_lb("A", "J", "平日", 2).
staff_dweek_soft_lb("A", "S", "平日", 2).
staff_dweek_soft_lb("A", "N", "土日", 2).
staff_dweek_soft_lb("A", "J", "土日", 2).
staff_dweek_soft_lb("A", "S", "土日", 2).
staff_dweek_soft_lb("A", "N", "祝日", 2).
staff_dweek_soft_lb("A", "J", "祝日", 2).
staff_dweek_soft_lb("A", "S", "祝日", 2).
staff_dweek_soft_ub("A", "N", "平日", 5).
staff_dweek_soft_ub("A", "J", "平日", 5).
staff_dweek_soft_ub("A", "S", "平日", 5).
staff_dweek_soft_ub("A", "N", "土日", 5).
staff_dweek_soft_ub("A", "J", "土日", 5).
staff_dweek_soft_ub("A", "S", "土日", 5).
staff_dweek_soft_ub("A", "N", "祝日", 5).
staff_dweek_soft_ub("A", "J", "祝日", 5).
staff_dweek_soft_ub("A", "S", "祝日", 5).
staff_dweek_soft_lb("師長", "日", "平日", 1).
staff_dweek_soft_lb("師長", "日", "土日", 0).
staff_dweek_soft_lb("師長", "日", "祝日", 0).
staff_dweek_soft_ub("師長", "日", "平日", 1).
staff_dweek_soft_ub("師長", "日", "土日", 0).
staff_dweek_soft_ub("師長", "日", "祝日", 0).
staff_dweek_hard_lb("B", "N", "平日", 1).
staff_dweek_hard_lb("B", "J", "平日", 1).
staff_dweek_hard_lb("B", "S", "平日", 1).
staff_dweek_hard_lb("B", "N", "土日", 1).
staff_dweek_hard_lb("B", "J", "土日", 1).
staff_dweek_hard_lb("B", "S", "土日", 1).
staff_dweek_hard_lb("B", "N", "祝日", 1).
staff_dweek_hard_lb("B", "J", "祝日", 1).
staff_dweek_hard_lb("B", "S", "祝日", 1).
staff_dweek_soft_lb("B", "N", "平日", 2).
staff_dweek_soft_lb("B", "J", "平日", 2).
staff_dweek_soft_lb("B", "S", "平日", 2).
staff_dweek_soft_lb("B", "N", "土日", 2).
staff_dweek_soft_lb("B", "J", "土日", 2).
staff_dweek_soft_lb("B", "S", "土日", 2).
staff_dweek_soft_lb("B", "N", "祝日", 2).
staff_dweek_soft_lb("B", "J", "祝日", 2).
staff_dweek_soft_lb("B", "S", "祝日", 2).
staff_dweek_soft_ub("B", "N", "平日", 5).
staff_dweek_soft_ub("B", "J", "平日", 5).
staff_dweek_soft_ub("B", "S", "平日", 5).
staff_dweek_soft_ub("B", "N", "土日", 5).
staff_dweek_soft_ub("B", "J", "土日", 5).
staff_dweek_soft_ub("B", "S", "土日", 5).
staff_dweek_soft_ub("B", "N", "祝日", 5).
staff_dweek_soft_ub("B", "J", "祝日", 5).
staff_dweek_soft_ub("B", "S", "祝日", 5).
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
vertical_constraint_type("全員","★","人数").
vertical_constraint_type("全員","☆","人数").
vertical_constraint_type("全員","J+★","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("A","N","人数").
vertical_constraint_type("A","J","人数").
vertical_constraint_type("A","S","人数").
vertical_constraint_type("師長","日","人数").
vertical_constraint_type("B","N","人数").
vertical_constraint_type("B","J","人数").
vertical_constraint_type("B","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
% Staff 3:
% Staff 4:
% Staff 5:
% Staff 6:
% Staff 7:
% Staff 8:
% Staff 9:
% Staff 10:
% Staff 11:
% Staff 12:
% Staff 13:
% Staff 14:
% Staff 15:
% Staff 16:
% Staff 17:
% Staff 18:
% Staff 19:
% Staff 20:
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
