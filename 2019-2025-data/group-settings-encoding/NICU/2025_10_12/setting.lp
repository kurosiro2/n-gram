% Staffs ---------------------------------------------------
staff(1, "清水　陽子", "師長", "0104392", 0).
staff(2, "山本　明奈", "副師長", "0177562", 0).
staff(3, "佐藤　貴広", "看護師", "0222258", 0).
staff(4, "中沢　大輔", "看護師", "0418415", 0).
staff(5, "中込　彩香", "看護師", "0184732", 0).
staff(6, "濵﨑　有香", "看護師", "0245234", 0).
staff(7, "丸山　璃子", "看護師", "0340753", 0).
staff(8, "清水　優日", "看護師", "0418426", 0).
staff(9, "望月　葵", "看護師", "0492770", 0).
staff(10, "金子　友里恵", "看護師", "0370602", 0).
staff(11, "齊藤　璃音", "看護師", "0606311", 0).
staff(12, "髙村　彩夏", "看護師", "0606435", 0).
staff(13, "藤井　愛", "副師長", "0354624", 0).
staff(14, "奥山　藍", "看護師", "0123021", 0).
staff(15, "雨宮　茉友子", "看護師", "0400185", 0).
staff(16, "永関　茉里", "看護師", "0191968", 0).
staff(17, "市川　乃愛", "看護師", "0357662", 0).
staff(18, "白鳥　美佳", "看護師", "0282536", 0).
staff(19, "大曽根　美青", "看護師", "0384324", 0).
staff(20, "入倉　みな美", "助産師", "0272465", 0).
staff(21, "大森　妃奈子", "看護師", "0355603", 0).
staff(22, "関　美南", "看護師", "0606424", 0).
staff(23, "中込　朝日", "看護師", "0606014", 0).
staff(24, "補）久保　孝子", "補助者", "0344212", 0).
% Groups ---------------------------------------------------
group("全員").
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
group("全員", 12).
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
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 13).
group("代行", 14).
group("代行", 19).
group("代行", 21).
group("リーダー").
group("リーダー", 2).
group("リーダー", 3).
group("リーダー", 4).
group("リーダー", 7).
group("リーダー", 13).
group("リーダー", 14).
group("リーダー", 16).
group("リーダー", 19).
group("リーダー", 21).
group("メンバー").
group("メンバー", 5).
group("メンバー", 6).
group("メンバー", 8).
group("メンバー", 9).
group("メンバー", 10).
group("メンバー", 15).
group("新人").
group("新人", 11).
group("新人", 12).
group("新人", 17).
group("新人", 18).
group("新人", 20).
group("新人", 22).
group("新人", 23).
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
group("夜勤", 12).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 19).
group("夜勤", 20).
group("夜勤", 21).
group("夜勤", 22).
group("夜勤", 23).
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
staff_dweek_hard_lb("全員", "日", "平日", 2).
staff_dweek_hard_lb("全員", "日", "土日", 2).
staff_dweek_hard_lb("全員", "日", "祝日", 2).
staff_dweek_soft_lb("全員", "日", "平日", 2).
staff_dweek_soft_lb("全員", "日", "土日", 2).
staff_dweek_soft_lb("全員", "日", "祝日", 2).
staff_dweek_soft_ub("全員", "日", "平日", 3).
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
staff_dweek_soft_ub("全員", "N", "平日", 4).
staff_dweek_soft_ub("全員", "N", "土日", 4).
staff_dweek_soft_ub("全員", "N", "祝日", 4).
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
staff_dweek_soft_ub("全員", "J", "平日", 4).
staff_dweek_soft_ub("全員", "S", "平日", 4).
staff_dweek_soft_ub("全員", "J", "土日", 4).
staff_dweek_soft_ub("全員", "S", "土日", 4).
staff_dweek_soft_ub("全員", "J", "祝日", 4).
staff_dweek_soft_ub("全員", "S", "祝日", 4).
shifts("★").
shifts("★", "★").
staff_dweek_hard_lb("全員", "★", "平日", 0).
shifts("☆").
shifts("☆", "☆").
staff_dweek_hard_lb("全員", "☆", "平日", 0).
staff_dweek_hard_lb("全員", "★", "土日", 0).
staff_dweek_hard_lb("全員", "☆", "土日", 0).
staff_dweek_hard_lb("全員", "★", "祝日", 0).
staff_dweek_hard_lb("全員", "☆", "祝日", 0).
staff_dweek_soft_lb("全員", "★", "平日", 0).
staff_dweek_soft_lb("全員", "☆", "平日", 0).
staff_dweek_soft_lb("全員", "★", "土日", 0).
staff_dweek_soft_lb("全員", "☆", "土日", 0).
staff_dweek_soft_lb("全員", "★", "祝日", 0).
staff_dweek_soft_lb("全員", "☆", "祝日", 0).
staff_dweek_soft_ub("全員", "★", "平日", 0).
staff_dweek_soft_ub("全員", "☆", "平日", 0).
staff_dweek_soft_ub("全員", "★", "土日", 0).
staff_dweek_soft_ub("全員", "☆", "土日", 0).
staff_dweek_soft_ub("全員", "★", "祝日", 0).
staff_dweek_soft_ub("全員", "☆", "祝日", 0).
staff_dweek_hard_ub("全員", "★", "平日", 0).
staff_dweek_hard_ub("全員", "☆", "平日", 0).
staff_dweek_hard_ub("全員", "★", "土日", 0).
staff_dweek_hard_ub("全員", "☆", "土日", 0).
staff_dweek_hard_ub("全員", "★", "祝日", 0).
staff_dweek_hard_ub("全員", "☆", "祝日", 0).
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
staff_dweek_hard_lb("リーダー", "N", "平日", 1).
staff_dweek_hard_lb("リーダー", "J", "平日", 1).
staff_dweek_hard_lb("リーダー", "S", "平日", 1).
staff_dweek_hard_lb("リーダー", "N", "土日", 1).
staff_dweek_hard_lb("リーダー", "J", "土日", 1).
staff_dweek_hard_lb("リーダー", "S", "土日", 1).
staff_dweek_hard_lb("リーダー", "N", "祝日", 1).
staff_dweek_hard_lb("リーダー", "J", "祝日", 1).
staff_dweek_hard_lb("リーダー", "S", "祝日", 1).
staff_dweek_soft_lb("リーダー", "N", "平日", 2).
staff_dweek_soft_lb("リーダー", "J", "平日", 2).
staff_dweek_soft_lb("リーダー", "S", "平日", 2).
staff_dweek_soft_lb("リーダー", "N", "土日", 2).
staff_dweek_soft_lb("リーダー", "J", "土日", 2).
staff_dweek_soft_lb("リーダー", "S", "土日", 2).
staff_dweek_soft_lb("リーダー", "N", "祝日", 2).
staff_dweek_soft_lb("リーダー", "J", "祝日", 2).
staff_dweek_soft_lb("リーダー", "S", "祝日", 2).
staff_dweek_soft_ub("リーダー", "N", "平日", 2).
staff_dweek_soft_ub("リーダー", "J", "平日", 2).
staff_dweek_soft_ub("リーダー", "S", "平日", 2).
staff_dweek_soft_ub("リーダー", "N", "土日", 2).
staff_dweek_soft_ub("リーダー", "J", "土日", 2).
staff_dweek_soft_ub("リーダー", "S", "土日", 2).
staff_dweek_soft_ub("リーダー", "N", "祝日", 2).
staff_dweek_soft_ub("リーダー", "J", "祝日", 2).
staff_dweek_soft_ub("リーダー", "S", "祝日", 2).
staff_dweek_soft_lb("代行", "日", "平日", 0).
staff_dweek_soft_lb("代行", "日", "土日", 1).
staff_dweek_soft_lb("代行", "日", "祝日", 1).
staff_dweek_soft_ub("代行", "日", "平日", 0).
staff_dweek_soft_ub("代行", "日", "土日", 1).
staff_dweek_soft_ub("代行", "日", "祝日", 1).
staff_dweek_hard_lb("メンバー", "N", "平日", 1).
staff_dweek_hard_lb("メンバー", "J", "平日", 1).
staff_dweek_hard_lb("メンバー", "S", "平日", 1).
staff_dweek_hard_lb("メンバー", "N", "土日", 1).
staff_dweek_hard_lb("メンバー", "J", "土日", 1).
staff_dweek_hard_lb("メンバー", "S", "土日", 1).
staff_dweek_hard_lb("メンバー", "N", "祝日", 1).
staff_dweek_hard_lb("メンバー", "J", "祝日", 1).
staff_dweek_hard_lb("メンバー", "S", "祝日", 1).
staff_dweek_soft_lb("メンバー", "N", "平日", 2).
staff_dweek_soft_lb("メンバー", "J", "平日", 2).
staff_dweek_soft_lb("メンバー", "S", "平日", 2).
staff_dweek_soft_lb("メンバー", "N", "土日", 2).
staff_dweek_soft_lb("メンバー", "J", "土日", 2).
staff_dweek_soft_lb("メンバー", "S", "土日", 2).
staff_dweek_soft_lb("メンバー", "N", "祝日", 2).
staff_dweek_soft_lb("メンバー", "J", "祝日", 2).
staff_dweek_soft_lb("メンバー", "S", "祝日", 2).
staff_dweek_soft_ub("メンバー", "N", "平日", 2).
staff_dweek_soft_ub("メンバー", "J", "平日", 2).
staff_dweek_soft_ub("メンバー", "S", "平日", 2).
staff_dweek_soft_ub("メンバー", "N", "土日", 2).
staff_dweek_soft_ub("メンバー", "J", "土日", 2).
staff_dweek_soft_ub("メンバー", "S", "土日", 2).
staff_dweek_soft_ub("メンバー", "N", "祝日", 2).
staff_dweek_soft_ub("メンバー", "J", "祝日", 2).
staff_dweek_soft_ub("メンバー", "S", "祝日", 2).
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
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("リーダー","N","人数").
vertical_constraint_type("リーダー","J","人数").
vertical_constraint_type("リーダー","S","人数").
vertical_constraint_type("代行","日","人数").
vertical_constraint_type("メンバー","N","人数").
vertical_constraint_type("メンバー","J","人数").
vertical_constraint_type("メンバー","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_soft_lb(2, "N", 1).
shifts_soft_lb(2, "J", 1).
shifts_soft_lb(2, "S", 1).
shifts_soft_ub(2, "N", 2).
shifts_soft_ub(2, "J", 2).
shifts_soft_ub(2, "S", 2).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "P", 0).
shifts_hard_ub(2, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_lb(2, "A", 0).
shifts_soft_lb(2, "A", 0).
shifts_soft_ub(2, "A", 0).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_soft_lb(3, "N", 5).
shifts_soft_lb(3, "J", 5).
shifts_soft_lb(3, "S", 5).
shifts_soft_ub(3, "N", 6).
shifts_soft_ub(3, "J", 6).
shifts_soft_ub(3, "S", 6).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "P", 0).
shifts_hard_lb(3, "A", 0).
shifts_soft_lb(3, "A", 0).
shifts_soft_ub(3, "A", 0).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_soft_lb(4, "N", 5).
shifts_soft_lb(4, "J", 5).
shifts_soft_lb(4, "S", 5).
shifts_soft_ub(4, "N", 6).
shifts_soft_ub(4, "J", 6).
shifts_soft_ub(4, "S", 6).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "P", 0).
shifts_hard_ub(4, "P", 0).
shifts_hard_lb(4, "A", 0).
shifts_soft_lb(4, "A", 0).
shifts_soft_ub(4, "A", 0).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_soft_lb(5, "N", 1).
shifts_soft_lb(5, "J", 1).
shifts_soft_lb(5, "S", 1).
shifts_soft_ub(5, "N", 2).
shifts_soft_ub(5, "J", 2).
shifts_soft_ub(5, "S", 2).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "P", 0).
shifts_hard_lb(5, "A", 0).
shifts_soft_lb(5, "A", 0).
shifts_soft_ub(5, "A", 0).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_soft_lb(6, "N", 2).
shifts_soft_lb(6, "J", 2).
shifts_soft_lb(6, "S", 2).
shifts_soft_ub(6, "N", 2).
shifts_soft_ub(6, "J", 2).
shifts_soft_ub(6, "S", 2).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "P", 0).
shifts_hard_lb(6, "A", 0).
shifts_soft_lb(6, "A", 0).
shifts_soft_ub(6, "A", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_soft_lb(7, "N", 5).
shifts_soft_lb(7, "J", 5).
shifts_soft_lb(7, "S", 5).
shifts_soft_ub(7, "N", 6).
shifts_soft_ub(7, "J", 6).
shifts_soft_ub(7, "S", 6).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "P", 0).
shifts_hard_lb(7, "A", 0).
shifts_soft_lb(7, "A", 0).
shifts_soft_ub(7, "A", 0).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_soft_lb(8, "N", 5).
shifts_soft_lb(8, "J", 5).
shifts_soft_lb(8, "S", 5).
shifts_soft_ub(8, "N", 6).
shifts_soft_ub(8, "J", 6).
shifts_soft_ub(8, "S", 6).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "P", 0).
shifts_hard_lb(8, "A", 0).
shifts_soft_lb(8, "A", 0).
shifts_soft_ub(8, "A", 0).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_soft_lb(9, "N", 5).
shifts_soft_lb(9, "J", 5).
shifts_soft_lb(9, "S", 5).
shifts_soft_ub(9, "N", 6).
shifts_soft_ub(9, "J", 6).
shifts_soft_ub(9, "S", 6).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "P", 0).
shifts_hard_ub(9, "P", 0).
shifts_hard_lb(9, "A", 0).
shifts_soft_lb(9, "A", 0).
shifts_soft_ub(9, "A", 0).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_soft_lb(10, "N", 5).
shifts_soft_lb(10, "J", 5).
shifts_soft_lb(10, "S", 5).
shifts_soft_ub(10, "N", 6).
shifts_soft_ub(10, "J", 6).
shifts_soft_ub(10, "S", 6).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "P", 0).
shifts_hard_ub(10, "P", 0).
shifts_hard_lb(10, "A", 0).
shifts_soft_lb(10, "A", 0).
shifts_soft_ub(10, "A", 0).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_soft_lb(11, "N", 4).
shifts_soft_lb(11, "J", 4).
shifts_soft_lb(11, "S", 4).
shifts_soft_ub(11, "N", 4).
shifts_soft_ub(11, "J", 4).
shifts_soft_ub(11, "S", 4).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "P", 0).
shifts_hard_lb(11, "A", 0).
shifts_soft_lb(11, "A", 0).
shifts_soft_ub(11, "A", 0).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_soft_lb(12, "N", 4).
shifts_soft_lb(12, "J", 4).
shifts_soft_lb(12, "S", 4).
shifts_soft_ub(12, "N", 5).
shifts_soft_ub(12, "J", 5).
shifts_soft_ub(12, "S", 5).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "P", 0).
shifts_hard_ub(12, "P", 0).
shifts_hard_lb(12, "A", 0).
shifts_soft_lb(12, "A", 0).
shifts_soft_ub(12, "A", 0).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_soft_lb(13, "N", 5).
shifts_soft_lb(13, "J", 5).
shifts_soft_lb(13, "S", 5).
shifts_soft_ub(13, "N", 6).
shifts_soft_ub(13, "J", 6).
shifts_soft_ub(13, "S", 6).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "P", 0).
shifts_hard_ub(13, "P", 0).
shifts_hard_lb(13, "A", 0).
shifts_soft_lb(13, "A", 0).
shifts_soft_ub(13, "A", 0).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_soft_lb(14, "N", 3).
shifts_soft_lb(14, "J", 3).
shifts_soft_lb(14, "S", 3).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "P", 0).
shifts_hard_ub(14, "P", 0).
shifts_hard_lb(14, "A", 0).
shifts_soft_lb(14, "A", 0).
shifts_soft_ub(14, "A", 0).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_soft_lb(15, "N", 5).
shifts_soft_lb(15, "J", 5).
shifts_soft_lb(15, "S", 5).
shifts_soft_ub(15, "N", 6).
shifts_soft_ub(15, "J", 6).
shifts_soft_ub(15, "S", 6).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "P", 0).
shifts_hard_lb(15, "A", 0).
shifts_soft_lb(15, "A", 0).
shifts_soft_ub(15, "A", 0).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_soft_lb(16, "N", 2).
shifts_soft_lb(16, "J", 2).
shifts_soft_lb(16, "S", 2).
shifts_soft_ub(16, "N", 3).
shifts_soft_ub(16, "J", 3).
shifts_soft_ub(16, "S", 3).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "P", 0).
shifts_hard_ub(16, "P", 0).
shifts_hard_lb(16, "A", 0).
shifts_soft_lb(16, "A", 0).
shifts_soft_ub(16, "A", 0).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_soft_lb(17, "N", 5).
shifts_soft_lb(17, "J", 5).
shifts_soft_lb(17, "S", 5).
shifts_soft_ub(17, "N", 6).
shifts_soft_ub(17, "J", 6).
shifts_soft_ub(17, "S", 6).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "P", 0).
shifts_hard_ub(17, "P", 0).
shifts_hard_lb(17, "A", 0).
shifts_soft_lb(17, "A", 0).
shifts_soft_ub(17, "A", 0).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_soft_lb(18, "N", 5).
shifts_soft_lb(18, "J", 5).
shifts_soft_lb(18, "S", 5).
shifts_soft_ub(18, "N", 6).
shifts_soft_ub(18, "J", 6).
shifts_soft_ub(18, "S", 6).
shifts_hard_lb(18, "P", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "P", 0).
shifts_hard_ub(18, "P", 0).
shifts_hard_lb(18, "A", 0).
shifts_soft_lb(18, "A", 0).
shifts_soft_ub(18, "A", 0).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_soft_lb(19, "N", 5).
shifts_soft_lb(19, "J", 5).
shifts_soft_lb(19, "S", 5).
shifts_soft_ub(19, "N", 6).
shifts_soft_ub(19, "J", 6).
shifts_soft_ub(19, "S", 6).
shifts_hard_lb(19, "P", 0).
shifts_soft_lb(19, "P", 0).
shifts_soft_ub(19, "P", 0).
shifts_hard_ub(19, "P", 0).
shifts_hard_lb(19, "A", 0).
shifts_soft_lb(19, "A", 0).
shifts_soft_ub(19, "A", 0).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_soft_lb(20, "N", 4).
shifts_soft_lb(20, "J", 4).
shifts_soft_lb(20, "S", 4).
shifts_soft_ub(20, "N", 6).
shifts_soft_ub(20, "J", 6).
shifts_soft_ub(20, "S", 6).
shifts_hard_lb(20, "P", 0).
shifts_soft_lb(20, "P", 0).
shifts_soft_ub(20, "P", 0).
shifts_hard_ub(20, "P", 0).
shifts_hard_lb(20, "A", 0).
shifts_soft_lb(20, "A", 0).
shifts_soft_ub(20, "A", 0).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_soft_lb(21, "N", 5).
shifts_soft_lb(21, "J", 5).
shifts_soft_lb(21, "S", 5).
shifts_soft_ub(21, "N", 6).
shifts_soft_ub(21, "J", 6).
shifts_soft_ub(21, "S", 6).
shifts_hard_lb(21, "P", 0).
shifts_soft_lb(21, "P", 0).
shifts_soft_ub(21, "P", 0).
shifts_hard_ub(21, "P", 0).
shifts_hard_lb(21, "A", 0).
shifts_soft_lb(21, "A", 0).
shifts_soft_ub(21, "A", 0).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_soft_lb(22, "N", 4).
shifts_soft_lb(22, "J", 4).
shifts_soft_lb(22, "S", 4).
shifts_soft_ub(22, "N", 6).
shifts_soft_ub(22, "J", 6).
shifts_soft_ub(22, "S", 6).
shifts_hard_lb(22, "P", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "P", 0).
shifts_hard_ub(22, "P", 0).
shifts_hard_lb(22, "A", 0).
shifts_soft_lb(22, "A", 0).
shifts_soft_ub(22, "A", 0).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_soft_lb(23, "N", 4).
shifts_soft_lb(23, "J", 4).
shifts_soft_lb(23, "S", 4).
shifts_soft_ub(23, "N", 6).
shifts_soft_ub(23, "J", 6).
shifts_soft_ub(23, "S", 6).
shifts_hard_lb(23, "P", 0).
shifts_soft_lb(23, "P", 0).
shifts_soft_ub(23, "P", 0).
shifts_hard_ub(23, "P", 0).
shifts_hard_lb(23, "A", 0).
shifts_soft_lb(23, "A", 0).
shifts_soft_ub(23, "A", 0).
shifts_hard_ub(23, "A", 0).
% Staff 24:
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
