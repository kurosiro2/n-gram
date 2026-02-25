% Staffs ---------------------------------------------------
staff(2, "杉本　美貴", "副師長", "0051525", 5).
staff(4, "舘田　安希子", "看護師", "0031188", 5).
staff(6, "築野　梨奈", "看護師", "0312840", 5).
staff(7, "古屋　愛有香", "看護師", "0312920", 2).
staff(9, "原　砂貴子", "看護師", "0341218", 5).
staff(10, "斉藤　麻未", "看護師", "0355465", 2).
staff(11, "森本　柚実", "看護師", "0355705", 3).
staff(12, "小林　紅葉", "看護師", "0370418", 0).
staff(13, "大久保　茉耶", "看護師", "0370657", 3).
staff(14, "田邉　愛海", "看護師", "0370817", 3).
staff(15, "中川　菜々子", "看護師", "0384563", 3).
staff(16, "小林　結菜", "看護師", "0418437", 3).
staff(17, "梶原　葉南", "副師長", "0418448", 3).
staff(18, "加賀美　里奈", "看護師", "0418450", 2).
staff(3, "中込　美幸", "副師長", "0058294", 5).
staff(1, "寺島　由美子", "師長", "0102025", 0).
staff(5, "小池　友佳梨", "看護師", "0283402", 5).
staff(8, "立川　理奈子", "看護師", "0326607", 2).
% Groups ---------------------------------------------------
group("全員").
group("全員", 2).
group("全員", 4).
group("全員", 6).
group("全員", 7).
group("全員", 9).
group("全員", 10).
group("全員", 11).
group("全員", 13).
group("全員", 14).
group("全員", 15).
group("全員", 16).
group("全員", 17).
group("全員", 18).
group("全員", 3).
group("全員", 5).
group("全員", 8).
group("師長").
group("師長", 1).
group("熟練").
group("熟練", 2).
group("熟練", 4).
group("熟練", 6).
group("熟練", 7).
group("熟練", 10).
group("熟練", 13).
group("熟練", 14).
group("熟練", 3).
group("熟練", 5).
group("中堅").
group("中堅", 11).
group("中堅", 15).
group("新人").
group("新人", 9).
group("新人", 16).
group("新人", 17).
group("新人", 18).
group("夜勤").
group("夜勤", 2).
group("夜勤", 4).
group("夜勤", 6).
group("夜勤", 7).
group("夜勤", 9).
group("夜勤", 10).
group("夜勤", 11).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 3).
group("夜勤", 5).
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
succ_shift("J", "S").
succ_shift("★", "☆").
pred_shift("★", "☆").
succ_shift("☆", "○").
succ_shift("☆", "◎").
% パターン割当数
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
patterns_hard_lb("○○", 2).
patterns("N日", "N日").
patterns_soft_ub("N日", 1).
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
staff_dweek_hard_lb("全員", "日", "土日", 0).
staff_dweek_hard_lb("全員", "日", "祝日", 0).
staff_dweek_soft_lb("全員", "日", "土日", 1).
staff_dweek_soft_lb("全員", "日", "祝日", 1).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 2).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 2).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 2).
staff_dweek_hard_lb("全員", "N", "土日", 2).
staff_dweek_hard_lb("全員", "J", "土日", 2).
staff_dweek_hard_lb("全員", "S", "土日", 2).
staff_dweek_hard_lb("全員", "N", "祝日", 2).
staff_dweek_hard_lb("全員", "J", "祝日", 2).
staff_dweek_hard_lb("全員", "S", "祝日", 2).
staff_dweek_soft_lb("全員", "N", "平日", 3).
staff_dweek_soft_lb("全員", "J", "平日", 3).
staff_dweek_soft_lb("全員", "S", "平日", 3).
staff_dweek_soft_lb("全員", "N", "土日", 3).
staff_dweek_soft_lb("全員", "J", "土日", 3).
staff_dweek_soft_lb("全員", "S", "土日", 3).
staff_dweek_soft_lb("全員", "N", "祝日", 3).
staff_dweek_soft_lb("全員", "J", "祝日", 3).
staff_dweek_soft_lb("全員", "S", "祝日", 3).
staff_dweek_soft_ub("全員", "N", "平日", 3).
staff_dweek_soft_ub("全員", "J", "平日", 3).
staff_dweek_soft_ub("全員", "S", "平日", 3).
staff_dweek_soft_ub("全員", "N", "土日", 3).
staff_dweek_soft_ub("全員", "J", "土日", 3).
staff_dweek_soft_ub("全員", "S", "土日", 3).
staff_dweek_soft_ub("全員", "N", "祝日", 3).
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
shifts("P").
shifts("P", "P").
staff_dweek_hard_ub("全員", "P", "平日", 1).
staff_dweek_hard_ub("全員", "P", "土日", 1).
staff_dweek_hard_ub("全員", "P", "祝日", 1).
shifts("N+P+★").
shifts("N+P+★", "N").
shifts("N+P+★", "P").
shifts("N+P+★", "★").
staff_dweek_soft_lb("全員", "N+P+★", "平日", 2).
staff_dweek_soft_lb("全員", "N+P+★", "土日", 2).
staff_dweek_soft_lb("全員", "N+P+★", "祝日", 2).
staff_dweek_soft_ub("全員", "N+P+★", "平日", 3).
staff_dweek_soft_ub("全員", "N+P+★", "土日", 3).
staff_dweek_soft_ub("全員", "N+P+★", "祝日", 3).
staff_dweek_hard_lb("熟練", "N", "平日", 0).
staff_dweek_hard_lb("熟練", "J", "平日", 0).
staff_dweek_hard_lb("熟練", "S", "平日", 0).
staff_dweek_hard_lb("熟練", "N", "土日", 0).
staff_dweek_hard_lb("熟練", "J", "土日", 0).
staff_dweek_hard_lb("熟練", "S", "土日", 0).
staff_dweek_hard_lb("熟練", "N", "祝日", 0).
staff_dweek_hard_lb("熟練", "J", "祝日", 0).
staff_dweek_hard_lb("熟練", "S", "祝日", 0).
staff_dweek_soft_lb("熟練", "N", "平日", 2).
staff_dweek_soft_lb("熟練", "J", "平日", 2).
staff_dweek_soft_lb("熟練", "S", "平日", 2).
staff_dweek_soft_lb("熟練", "N", "土日", 2).
staff_dweek_soft_lb("熟練", "J", "土日", 2).
staff_dweek_soft_lb("熟練", "S", "土日", 2).
staff_dweek_soft_lb("熟練", "N", "祝日", 2).
staff_dweek_soft_lb("熟練", "J", "祝日", 2).
staff_dweek_soft_lb("熟練", "S", "祝日", 2).
staff_dweek_soft_ub("熟練", "N", "平日", 2).
staff_dweek_soft_ub("熟練", "J", "平日", 2).
staff_dweek_soft_ub("熟練", "S", "平日", 2).
staff_dweek_soft_ub("熟練", "N", "土日", 2).
staff_dweek_soft_ub("熟練", "J", "土日", 2).
staff_dweek_soft_ub("熟練", "S", "土日", 2).
staff_dweek_soft_ub("熟練", "N", "祝日", 2).
staff_dweek_soft_ub("熟練", "J", "祝日", 2).
staff_dweek_soft_ub("熟練", "S", "祝日", 2).
staff_dweek_hard_ub("熟練", "N", "平日", 2).
staff_dweek_hard_ub("熟練", "J", "平日", 2).
staff_dweek_hard_ub("熟練", "S", "平日", 2).
staff_dweek_hard_ub("熟練", "N", "土日", 2).
staff_dweek_hard_ub("熟練", "J", "土日", 2).
staff_dweek_hard_ub("熟練", "S", "土日", 2).
staff_dweek_hard_ub("熟練", "N", "祝日", 2).
staff_dweek_hard_ub("熟練", "J", "祝日", 2).
staff_dweek_hard_ub("熟練", "S", "祝日", 2).
staff_dweek_soft_lb("師長", "日", "平日", 1).
staff_dweek_soft_lb("師長", "日", "土日", 0).
staff_dweek_soft_lb("師長", "日", "祝日", 0).
staff_dweek_soft_ub("師長", "日", "平日", 1).
staff_dweek_soft_ub("師長", "日", "土日", 0).
staff_dweek_soft_ub("師長", "日", "祝日", 0).
staff_dweek_hard_lb("中堅", "J", "平日", 0).
staff_dweek_hard_lb("中堅", "S", "平日", 0).
staff_dweek_hard_lb("中堅", "J", "土日", 0).
staff_dweek_hard_lb("中堅", "S", "土日", 0).
staff_dweek_hard_lb("中堅", "J", "祝日", 0).
staff_dweek_hard_lb("中堅", "S", "祝日", 0).
staff_dweek_soft_lb("中堅", "J", "平日", 0).
staff_dweek_soft_lb("中堅", "S", "平日", 0).
staff_dweek_soft_lb("中堅", "J", "土日", 0).
staff_dweek_soft_lb("中堅", "S", "土日", 0).
staff_dweek_soft_lb("中堅", "J", "祝日", 0).
staff_dweek_soft_lb("中堅", "S", "祝日", 0).
staff_dweek_soft_ub("中堅", "J", "平日", 1).
staff_dweek_soft_ub("中堅", "S", "平日", 1).
staff_dweek_soft_ub("中堅", "J", "土日", 1).
staff_dweek_soft_ub("中堅", "S", "土日", 1).
staff_dweek_soft_ub("中堅", "J", "祝日", 1).
staff_dweek_soft_ub("中堅", "S", "祝日", 1).
staff_dweek_hard_ub("中堅", "J", "平日", 1).
staff_dweek_hard_ub("中堅", "S", "平日", 1).
staff_dweek_hard_ub("中堅", "J", "土日", 1).
staff_dweek_hard_ub("中堅", "S", "土日", 1).
staff_dweek_hard_ub("中堅", "J", "祝日", 1).
staff_dweek_hard_ub("中堅", "S", "祝日", 1).
staff_dweek_hard_lb("新人", "J", "平日", 0).
staff_dweek_hard_lb("新人", "S", "平日", 0).
staff_dweek_hard_lb("新人", "J", "土日", 0).
staff_dweek_hard_lb("新人", "S", "土日", 0).
staff_dweek_hard_lb("新人", "J", "祝日", 0).
staff_dweek_hard_lb("新人", "S", "祝日", 0).
staff_dweek_soft_lb("新人", "J", "平日", 0).
staff_dweek_soft_lb("新人", "S", "平日", 0).
staff_dweek_soft_lb("新人", "J", "土日", 0).
staff_dweek_soft_lb("新人", "S", "土日", 0).
staff_dweek_soft_lb("新人", "J", "祝日", 0).
staff_dweek_soft_lb("新人", "S", "祝日", 0).
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
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("熟練","N","人数").
vertical_constraint_type("熟練","J","人数").
vertical_constraint_type("熟練","S","人数").
vertical_constraint_type("師長","日","人数").
vertical_constraint_type("中堅","J","人数").
vertical_constraint_type("中堅","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 2:
shifts_soft_lb(2, "N", 3).
shifts_soft_lb(2, "J", 3).
shifts_soft_lb(2, "S", 3).
shifts_soft_ub(2, "N", 4).
shifts_soft_ub(2, "J", 4).
shifts_soft_ub(2, "S", 4).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "P", 1).
shifts_hard_ub(2, "P", 0).
% Staff 4:
shifts_soft_lb(4, "N", 3).
shifts_soft_lb(4, "J", 3).
shifts_soft_lb(4, "S", 3).
shifts_soft_ub(4, "N", 4).
shifts_soft_ub(4, "J", 4).
shifts_soft_ub(4, "S", 4).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "P", 1).
shifts_hard_ub(4, "P", 0).
% Staff 6:
shifts_soft_lb(6, "N", 3).
shifts_soft_lb(6, "J", 3).
shifts_soft_lb(6, "S", 3).
shifts_soft_ub(6, "N", 4).
shifts_soft_ub(6, "J", 4).
shifts_soft_ub(6, "S", 4).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "P", 1).
shifts_hard_ub(6, "P", 0).
% Staff 7:
shifts_soft_lb(7, "N", 3).
shifts_soft_lb(7, "J", 3).
shifts_soft_lb(7, "S", 3).
shifts_soft_ub(7, "N", 4).
shifts_soft_ub(7, "J", 4).
shifts_soft_ub(7, "S", 4).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "P", 1).
shifts_hard_ub(7, "P", 0).
% Staff 9:
shifts_soft_lb(9, "N", 3).
shifts_soft_lb(9, "J", 3).
shifts_soft_lb(9, "S", 3).
shifts_soft_ub(9, "N", 4).
shifts_soft_ub(9, "J", 4).
shifts_soft_ub(9, "S", 4).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "P", 1).
shifts_hard_ub(9, "P", 0).
% Staff 10:
shifts_soft_lb(10, "N", 3).
shifts_soft_lb(10, "J", 3).
shifts_soft_lb(10, "S", 3).
shifts_soft_ub(10, "N", 4).
shifts_soft_ub(10, "J", 4).
shifts_soft_ub(10, "S", 4).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "P", 1).
shifts_hard_ub(10, "P", 0).
% Staff 11:
shifts_soft_lb(11, "N", 3).
shifts_soft_lb(11, "J", 3).
shifts_soft_lb(11, "S", 3).
shifts_soft_ub(11, "N", 4).
shifts_soft_ub(11, "J", 4).
shifts_soft_ub(11, "S", 4).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "P", 0).
% Staff 12:
shifts_soft_lb(12, "N", 3).
shifts_soft_lb(12, "J", 3).
shifts_soft_lb(12, "S", 3).
shifts_soft_ub(12, "N", 4).
shifts_soft_ub(12, "J", 4).
shifts_soft_ub(12, "S", 4).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "P", 1).
shifts_hard_ub(12, "P", 0).
% Staff 13:
shifts_soft_lb(13, "N", 3).
shifts_soft_lb(13, "J", 3).
shifts_soft_lb(13, "S", 3).
shifts_soft_ub(13, "N", 4).
shifts_soft_ub(13, "J", 4).
shifts_soft_ub(13, "S", 4).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "P", 1).
shifts_hard_ub(13, "P", 0).
% Staff 14:
shifts_soft_lb(14, "N", 3).
shifts_soft_lb(14, "J", 3).
shifts_soft_lb(14, "S", 3).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "P", 1).
shifts_hard_ub(14, "P", 0).
% Staff 15:
shifts_soft_lb(15, "N", 3).
shifts_soft_lb(15, "J", 3).
shifts_soft_lb(15, "S", 3).
shifts_soft_ub(15, "N", 4).
shifts_soft_ub(15, "J", 4).
shifts_soft_ub(15, "S", 4).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "P", 1).
shifts_hard_ub(15, "P", 0).
% Staff 16:
shifts_soft_lb(16, "N", 3).
shifts_soft_lb(16, "J", 3).
shifts_soft_lb(16, "S", 3).
shifts_soft_ub(16, "N", 4).
shifts_soft_ub(16, "J", 4).
shifts_soft_ub(16, "S", 4).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "P", 1).
shifts_hard_ub(16, "P", 0).
% Staff 17:
shifts_soft_lb(17, "N", 3).
shifts_soft_lb(17, "J", 3).
shifts_soft_lb(17, "S", 3).
shifts_soft_ub(17, "N", 4).
shifts_soft_ub(17, "J", 4).
shifts_soft_ub(17, "S", 4).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "P", 1).
shifts_hard_ub(17, "P", 0).
% Staff 18:
shifts_soft_lb(18, "N", 3).
shifts_soft_lb(18, "J", 3).
shifts_soft_lb(18, "S", 3).
shifts_soft_ub(18, "N", 4).
shifts_soft_ub(18, "J", 4).
shifts_soft_ub(18, "S", 4).
shifts_hard_lb(18, "P", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "P", 1).
shifts_hard_ub(18, "P", 0).
% Staff 3:
shifts_soft_lb(3, "N", 1).
shifts_soft_lb(3, "J", 1).
shifts_soft_lb(3, "S", 1).
shifts_soft_ub(3, "N", 2).
shifts_soft_ub(3, "J", 2).
shifts_soft_ub(3, "S", 2).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "P", 0).
% Staff 1:
shifts_soft_lb(1, "N", 0).
shifts_soft_lb(1, "J", 0).
shifts_soft_lb(1, "S", 0).
shifts_soft_ub(1, "N", 0).
shifts_soft_ub(1, "J", 0).
shifts_soft_ub(1, "S", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "J", 0).
shifts_hard_ub(1, "S", 0).
shifts_hard_lb(1, "P", 0).
shifts_soft_lb(1, "P", 0).
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "P", 0).
% Staff 5:
shifts_soft_lb(5, "N", 0).
shifts_soft_lb(5, "J", 0).
shifts_soft_lb(5, "S", 0).
shifts_soft_ub(5, "N", 0).
shifts_soft_ub(5, "J", 0).
shifts_soft_ub(5, "S", 0).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "P", 0).
% Staff 8:
shifts_soft_lb(8, "N", 0).
shifts_soft_lb(8, "J", 0).
shifts_soft_lb(8, "S", 0).
shifts_soft_ub(8, "N", 0).
shifts_soft_ub(8, "J", 0).
shifts_soft_ub(8, "S", 0).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "P", 0).
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
