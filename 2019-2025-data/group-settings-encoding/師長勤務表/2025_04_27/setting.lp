% Staffs ---------------------------------------------------
staff(1, "茶谷　直子", "師長", "4444488", 2).
staff(2, "金丸　紀子", "師長", "55555582", 2).
staff(3, "長澤良美", "師長", "5555501", 2).
staff(4, "山本瑠実", "師長", "7777701", 2).
staff(5, "金子春美", "師長", "55555590", 2).
staff(6, "山中浩代", "師長", "7777703", 2).
staff(7, "山本秀美", "師長", "5555568", 2).
staff(8, "清水美紀", "師長", "7777705", 2).
staff(9, "河西　典子", "師長", "4444489", 2).
staff(10, "牧野　基美", "師長", "66666664", 2).
staff(11, "伊藤由香", "師長", "44444441", 2).
staff(12, "相川真弓", "師長", "7777704", 2).
staff(13, "内田純子", "師長", "44444442", 2).
staff(14, "寺島由美子", "師長", "44444444", 2).
staff(15, "戸栗宏子", "師長", "66666900", 2).
staff(16, "清水陽子", "師長", "7777702", 2).
staff(17, "磯野絵美", "師長", "44444443", 2).
staff(18, "坂野雅子", "師長", "66666902", 2).
staff(19, "山本　ゆかり", "師長", "55555580", 2).
staff(20, "大芝まゆみ", "師長", "55555594", 2).
staff(21, "田邉玲子", "師長", "55555550", 2).
staff(22, "鈴木聖美", "師長", "55555591", 2).
staff(23, "谷戸るみ", "師長", "55555600", 2).
staff(24, "櫻本かおり", "師長", "55555551", 2).
staff(25, "大森ゆかり", "師長", "66666671", 2).
% Groups ---------------------------------------------------
group("全員").
group("全員", 1).
group("全員", 2).
group("全員", 3).
group("全員", 5).
group("全員", 7).
group("全員", 9).
group("全員", 10).
group("全員", 11).
group("全員", 13).
group("全員", 14).
group("全員", 15).
group("全員", 17).
group("全員", 18).
group("全員", 19).
group("全員", 20).
group("全員", 21).
group("全員", 22).
group("全員", 23).
group("全員", 24).
group("全員", 25).
group("師長").
group("熟練").
group("熟練１").
group("熟練１", 1).
group("熟練１", 2).
group("熟練１", 3).
group("熟練１", 5).
group("熟練１", 7).
group("熟練１", 9).
group("熟練１", 10).
group("熟練１", 11).
group("熟練１", 13).
group("熟練１", 14).
group("熟練１", 15).
group("熟練１", 17).
group("熟練１", 18).
group("熟練１", 19).
group("熟練１", 20).
group("熟練１", 21).
group("熟練１", 22).
group("熟練１", 23).
group("熟練１", 24).
group("熟練１", 25).
group("熟練２").
group("中堅").
group("新人").
group("新人１").
group("新人２").
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
consecutive_work_ub(5).
% 禁止パターン
forbidden_pattern("NN").
pattern("NN", 2).
pattern("NN", 0, "N").
pattern("NN", 1, "N").
forbidden_pattern("N日").
pattern("N日", 2).
pattern("N日", 0, "N").
pattern("N日", 1, "日").
forbidden_pattern("P日").
pattern("P日", 2).
pattern("P日", 0, "P").
pattern("P日", 1, "日").
forbidden_pattern("PN").
pattern("PN", 2).
pattern("PN", 0, "P").
pattern("PN", 1, "N").
forbidden_pattern("JSJS").
pattern("JSJS", 4).
pattern("JSJS", 0, "J").
pattern("JSJS", 1, "S").
pattern("JSJS", 2, "J").
pattern("JSJS", 3, "S").
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
patterns_hard_ub("N日", 1).
patterns("P日", "P日").
patterns_hard_ub("P日", 0).
patterns("PN", "PN").
patterns_hard_ub("PN", 0).
patterns("PP", "PP").
pattern("PP", 2).
pattern("PP", 0, "P").
pattern("PP", 1, "P").
patterns_hard_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("日").
shifts("日", "日").
staff_dweek_soft_lb("全員", "日", "平日", 1).
staff_dweek_soft_lb("全員", "日", "土日", 1).
staff_dweek_soft_lb("全員", "日", "祝日", 1).
shifts("N").
shifts("N", "N").
staff_dweek_soft_lb("全員", "N", "平日", 1).
shifts("J").
shifts("J", "J").
staff_dweek_soft_lb("全員", "J", "平日", 1).
shifts("S").
shifts("S", "S").
staff_dweek_soft_lb("全員", "S", "平日", 1).
staff_dweek_soft_lb("全員", "N", "土日", 1).
staff_dweek_soft_lb("全員", "J", "土日", 1).
staff_dweek_soft_lb("全員", "S", "土日", 1).
staff_dweek_soft_lb("全員", "N", "祝日", 1).
staff_dweek_soft_lb("全員", "J", "祝日", 1).
staff_dweek_soft_lb("全員", "S", "祝日", 1).
staff_dweek_soft_ub("全員", "N", "平日", 1).
staff_dweek_soft_ub("全員", "J", "平日", 1).
staff_dweek_soft_ub("全員", "S", "平日", 1).
staff_dweek_soft_ub("全員", "N", "土日", 1).
staff_dweek_soft_ub("全員", "J", "土日", 1).
staff_dweek_soft_ub("全員", "S", "土日", 1).
staff_dweek_soft_ub("全員", "N", "祝日", 1).
staff_dweek_soft_ub("全員", "J", "祝日", 1).
staff_dweek_soft_ub("全員", "S", "祝日", 1).
staff_dweek_soft_lb("熟練１", "N", "平日", 1).
staff_dweek_soft_lb("熟練１", "J", "平日", 1).
staff_dweek_soft_lb("熟練１", "S", "平日", 1).
staff_dweek_soft_lb("熟練１", "N", "土日", 1).
staff_dweek_soft_lb("熟練１", "J", "土日", 1).
staff_dweek_soft_lb("熟練１", "S", "土日", 1).
staff_dweek_soft_lb("熟練１", "N", "祝日", 1).
staff_dweek_soft_lb("熟練１", "J", "祝日", 1).
staff_dweek_soft_lb("熟練１", "S", "祝日", 1).
staff_dweek_soft_ub("熟練１", "N", "平日", 1).
staff_dweek_soft_ub("熟練１", "J", "平日", 1).
staff_dweek_soft_ub("熟練１", "S", "平日", 1).
staff_dweek_soft_ub("熟練１", "N", "土日", 1).
staff_dweek_soft_ub("熟練１", "J", "土日", 1).
staff_dweek_soft_ub("熟練１", "S", "土日", 1).
staff_dweek_soft_ub("熟練１", "N", "祝日", 1).
staff_dweek_soft_ub("熟練１", "J", "祝日", 1).
staff_dweek_soft_ub("熟練１", "S", "祝日", 1).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("熟練１","N","人数").
vertical_constraint_type("熟練１","J","人数").
vertical_constraint_type("熟練１","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "N", 1).
shifts_soft_lb(1, "J", 1).
shifts_soft_lb(1, "S", 1).
shifts_soft_ub(1, "N", 2).
shifts_soft_ub(1, "J", 2).
shifts_soft_ub(1, "S", 2).
shifts("★").
shifts("★", "★").
shifts_soft_ub(1, "★", 0).
shifts("☆").
shifts("☆", "☆").
shifts_soft_ub(1, "☆", 0).
shifts("A").
shifts("A", "A").
shifts_soft_ub(1, "A", 0).
shifts("P").
shifts("P", "P").
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_hard_ub(1, "A", 0).
shifts_hard_ub(1, "P", 0).
% Staff 2:
shifts_soft_lb(2, "N", 1).
shifts_soft_lb(2, "J", 1).
shifts_soft_lb(2, "S", 1).
shifts_soft_ub(2, "N", 2).
shifts_soft_ub(2, "J", 2).
shifts_soft_ub(2, "S", 2).
shifts_soft_ub(2, "★", 0).
shifts_soft_ub(2, "☆", 0).
shifts_soft_ub(2, "A", 0).
shifts_soft_ub(2, "P", 0).
shifts_hard_ub(2, "★", 0).
shifts_hard_ub(2, "☆", 0).
shifts_hard_ub(2, "A", 0).
shifts_hard_ub(2, "P", 0).
% Staff 3:
shifts_soft_lb(3, "N", 1).
shifts_soft_lb(3, "J", 1).
shifts_soft_lb(3, "S", 1).
shifts_soft_ub(3, "N", 2).
shifts_soft_ub(3, "J", 2).
shifts_soft_ub(3, "S", 2).
shifts_soft_ub(3, "★", 0).
shifts_soft_ub(3, "☆", 0).
shifts_soft_ub(3, "A", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "★", 0).
shifts_hard_ub(3, "☆", 0).
shifts_hard_ub(3, "A", 0).
shifts_hard_ub(3, "P", 0).
% Staff 4:
% Staff 5:
shifts_soft_lb(5, "N", 1).
shifts_soft_lb(5, "J", 1).
shifts_soft_lb(5, "S", 1).
shifts_soft_ub(5, "N", 2).
shifts_soft_ub(5, "J", 2).
shifts_soft_ub(5, "S", 2).
shifts_soft_ub(5, "★", 0).
shifts_soft_ub(5, "☆", 0).
shifts_soft_ub(5, "A", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_hard_ub(5, "A", 0).
shifts_hard_ub(5, "P", 0).
% Staff 6:
% Staff 7:
shifts_soft_lb(7, "N", 1).
shifts_soft_lb(7, "J", 1).
shifts_soft_lb(7, "S", 1).
shifts_soft_ub(7, "N", 2).
shifts_soft_ub(7, "J", 2).
shifts_soft_ub(7, "S", 2).
shifts_soft_ub(7, "★", 0).
shifts_soft_ub(7, "☆", 0).
shifts_soft_ub(7, "A", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_hard_ub(7, "A", 0).
shifts_hard_ub(7, "P", 0).
% Staff 8:
% Staff 9:
shifts_soft_lb(9, "N", 1).
shifts_soft_lb(9, "J", 1).
shifts_soft_lb(9, "S", 1).
shifts_soft_ub(9, "N", 2).
shifts_soft_ub(9, "J", 2).
shifts_soft_ub(9, "S", 2).
shifts_soft_ub(9, "★", 0).
shifts_soft_ub(9, "☆", 0).
shifts_soft_ub(9, "A", 0).
shifts_soft_ub(9, "P", 0).
shifts_hard_ub(9, "★", 0).
shifts_hard_ub(9, "☆", 0).
shifts_hard_ub(9, "A", 0).
shifts_hard_ub(9, "P", 0).
% Staff 10:
shifts_soft_lb(10, "N", 1).
shifts_soft_lb(10, "J", 1).
shifts_soft_lb(10, "S", 1).
shifts_soft_ub(10, "N", 2).
shifts_soft_ub(10, "J", 2).
shifts_soft_ub(10, "S", 2).
shifts_soft_ub(10, "★", 0).
shifts_soft_ub(10, "☆", 0).
shifts_soft_ub(10, "A", 0).
shifts_soft_ub(10, "P", 0).
shifts_hard_ub(10, "★", 0).
shifts_hard_ub(10, "☆", 0).
shifts_hard_ub(10, "A", 0).
shifts_hard_ub(10, "P", 0).
% Staff 11:
shifts_soft_lb(11, "N", 1).
shifts_soft_lb(11, "J", 1).
shifts_soft_lb(11, "S", 1).
shifts_soft_ub(11, "N", 2).
shifts_soft_ub(11, "J", 2).
shifts_soft_ub(11, "S", 2).
shifts_soft_ub(11, "★", 0).
shifts_soft_ub(11, "☆", 0).
shifts_soft_ub(11, "A", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_hard_ub(11, "A", 0).
shifts_hard_ub(11, "P", 0).
% Staff 12:
% Staff 13:
shifts_soft_lb(13, "N", 1).
shifts_soft_lb(13, "J", 1).
shifts_soft_lb(13, "S", 1).
shifts_soft_ub(13, "N", 2).
shifts_soft_ub(13, "J", 2).
shifts_soft_ub(13, "S", 2).
shifts_soft_ub(13, "★", 0).
shifts_soft_ub(13, "☆", 0).
shifts_soft_ub(13, "A", 0).
shifts_soft_ub(13, "P", 0).
shifts_hard_ub(13, "★", 0).
shifts_hard_ub(13, "☆", 0).
shifts_hard_ub(13, "A", 0).
shifts_hard_ub(13, "P", 0).
% Staff 14:
shifts_soft_lb(14, "N", 1).
shifts_soft_lb(14, "J", 1).
shifts_soft_lb(14, "S", 1).
shifts_soft_ub(14, "N", 2).
shifts_soft_ub(14, "J", 2).
shifts_soft_ub(14, "S", 2).
shifts_soft_ub(14, "★", 0).
shifts_soft_ub(14, "☆", 0).
shifts_soft_ub(14, "A", 0).
shifts_soft_ub(14, "P", 0).
shifts_hard_ub(14, "★", 0).
shifts_hard_ub(14, "☆", 0).
shifts_hard_ub(14, "A", 0).
shifts_hard_ub(14, "P", 0).
% Staff 15:
shifts_soft_lb(15, "N", 1).
shifts_soft_lb(15, "J", 1).
shifts_soft_lb(15, "S", 1).
shifts_soft_ub(15, "N", 2).
shifts_soft_ub(15, "J", 2).
shifts_soft_ub(15, "S", 2).
shifts_soft_ub(15, "★", 0).
shifts_soft_ub(15, "☆", 0).
shifts_soft_ub(15, "A", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_hard_ub(15, "A", 0).
shifts_hard_ub(15, "P", 0).
% Staff 16:
% Staff 17:
shifts_soft_lb(17, "N", 1).
shifts_soft_lb(17, "J", 1).
shifts_soft_lb(17, "S", 1).
shifts_soft_ub(17, "N", 2).
shifts_soft_ub(17, "J", 2).
shifts_soft_ub(17, "S", 2).
shifts_soft_ub(17, "★", 0).
shifts_soft_ub(17, "☆", 0).
shifts_soft_ub(17, "A", 0).
shifts_soft_ub(17, "P", 0).
shifts_hard_ub(17, "★", 0).
shifts_hard_ub(17, "☆", 0).
shifts_hard_ub(17, "A", 0).
shifts_hard_ub(17, "P", 0).
% Staff 18:
shifts_soft_lb(18, "N", 1).
shifts_soft_lb(18, "J", 1).
shifts_soft_lb(18, "S", 1).
shifts_soft_ub(18, "N", 2).
shifts_soft_ub(18, "J", 2).
shifts_soft_ub(18, "S", 2).
shifts_soft_ub(18, "★", 0).
shifts_soft_ub(18, "☆", 0).
shifts_soft_ub(18, "A", 0).
shifts_soft_ub(18, "P", 0).
shifts_hard_ub(18, "★", 0).
shifts_hard_ub(18, "☆", 0).
shifts_hard_ub(18, "A", 0).
shifts_hard_ub(18, "P", 0).
% Staff 19:
shifts_soft_lb(19, "N", 1).
shifts_soft_lb(19, "J", 1).
shifts_soft_lb(19, "S", 1).
shifts_soft_ub(19, "N", 2).
shifts_soft_ub(19, "J", 2).
shifts_soft_ub(19, "S", 2).
shifts_soft_ub(19, "★", 0).
shifts_soft_ub(19, "☆", 0).
shifts_soft_ub(19, "A", 0).
shifts_soft_ub(19, "P", 0).
shifts_hard_ub(19, "★", 0).
shifts_hard_ub(19, "☆", 0).
shifts_hard_ub(19, "A", 0).
shifts_hard_ub(19, "P", 0).
% Staff 20:
shifts_soft_lb(20, "N", 1).
shifts_soft_lb(20, "J", 1).
shifts_soft_lb(20, "S", 1).
shifts_soft_ub(20, "N", 2).
shifts_soft_ub(20, "J", 2).
shifts_soft_ub(20, "S", 2).
shifts_soft_ub(20, "★", 0).
shifts_soft_ub(20, "☆", 0).
shifts_soft_ub(20, "A", 0).
shifts_soft_ub(20, "P", 0).
shifts_hard_ub(20, "★", 0).
shifts_hard_ub(20, "☆", 0).
shifts_hard_ub(20, "A", 0).
shifts_hard_ub(20, "P", 0).
% Staff 21:
shifts_soft_lb(21, "N", 1).
shifts_soft_lb(21, "J", 1).
shifts_soft_lb(21, "S", 1).
shifts_soft_ub(21, "N", 2).
shifts_soft_ub(21, "J", 2).
shifts_soft_ub(21, "S", 2).
shifts_soft_ub(21, "★", 0).
shifts_soft_ub(21, "☆", 0).
shifts_soft_ub(21, "A", 0).
shifts_soft_ub(21, "P", 0).
shifts_hard_ub(21, "★", 0).
shifts_hard_ub(21, "☆", 0).
shifts_hard_ub(21, "A", 0).
shifts_hard_ub(21, "P", 0).
% Staff 22:
shifts_soft_lb(22, "N", 1).
shifts_soft_lb(22, "J", 1).
shifts_soft_lb(22, "S", 1).
shifts_soft_ub(22, "N", 2).
shifts_soft_ub(22, "J", 2).
shifts_soft_ub(22, "S", 2).
shifts_soft_ub(22, "★", 0).
shifts_soft_ub(22, "☆", 0).
shifts_soft_ub(22, "A", 0).
shifts_soft_ub(22, "P", 0).
shifts_hard_ub(22, "★", 0).
shifts_hard_ub(22, "☆", 0).
shifts_hard_ub(22, "A", 0).
shifts_hard_ub(22, "P", 0).
% Staff 23:
shifts_soft_lb(23, "N", 1).
shifts_soft_lb(23, "J", 1).
shifts_soft_lb(23, "S", 1).
shifts_soft_ub(23, "N", 2).
shifts_soft_ub(23, "J", 2).
shifts_soft_ub(23, "S", 2).
shifts_soft_ub(23, "★", 0).
shifts_soft_ub(23, "☆", 0).
shifts_soft_ub(23, "A", 0).
shifts_soft_ub(23, "P", 0).
shifts_hard_ub(23, "★", 0).
shifts_hard_ub(23, "☆", 0).
shifts_hard_ub(23, "A", 0).
shifts_hard_ub(23, "P", 0).
% Staff 24:
shifts_soft_lb(24, "N", 1).
shifts_soft_lb(24, "J", 1).
shifts_soft_lb(24, "S", 1).
shifts_soft_ub(24, "N", 2).
shifts_soft_ub(24, "J", 2).
shifts_soft_ub(24, "S", 2).
shifts_soft_ub(24, "★", 0).
shifts_soft_ub(24, "☆", 0).
shifts_soft_ub(24, "A", 0).
shifts_soft_ub(24, "P", 0).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_hard_ub(24, "A", 0).
shifts_hard_ub(24, "P", 0).
% Staff 25:
shifts_soft_lb(25, "N", 1).
shifts_soft_lb(25, "J", 1).
shifts_soft_lb(25, "S", 1).
shifts_soft_ub(25, "N", 2).
shifts_soft_ub(25, "J", 2).
shifts_soft_ub(25, "S", 2).
shifts_soft_ub(25, "★", 0).
shifts_soft_ub(25, "☆", 0).
shifts_soft_ub(25, "A", 0).
shifts_soft_ub(25, "P", 0).
shifts_hard_ub(25, "★", 0).
shifts_hard_ub(25, "☆", 0).
shifts_hard_ub(25, "A", 0).
shifts_hard_ub(25, "P", 0).
horizontal_constraint_type("日").
horizontal_constraint_type("N").
horizontal_constraint_type("J").
horizontal_constraint_type("S").
horizontal_constraint_type("★").
horizontal_constraint_type("☆").
horizontal_constraint_type("A").
horizontal_constraint_type("P").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
