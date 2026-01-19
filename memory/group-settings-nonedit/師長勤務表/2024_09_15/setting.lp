% Staffs ---------------------------------------------------
staff(1, "内田　純子", "師長", "44444442", 2).
staff(2, "牧野　基美", "師長", "66666664", 2).
staff(3, "伊藤　由香", "師長", "44444441", 2).
staff(4, "山本　秀美", "師長", "5555568", 2).
staff(5, "金丸　紀子", "師長", "55555582", 2).
staff(6, "茶谷　直子", "師長", "4444488", 2).
staff(7, "金子  春美", "師長", "55555590", 2).
staff(8, "穴水　美和", "師長", "4444443", 2).
staff(9, "河西　典子", "師長", "4444489", 2).
staff(10, "髙橋　里香", "師長", "66666901", 2).
staff(11, "田邉　玲子", "師長", "55555550", 2).
staff(12, "櫻本　かおり", "師長", "55555551", 2).
staff(13, "寺島　由美子", "師長", "44444444", 2).
staff(14, "矢崎　正浩", "師長", "55667783", 2).
staff(15, "鈴木 聖美", "師長", "55555591", 2).
staff(16, "戸栗　宏子", "師長", "66666900", 2).
staff(17, "坂野　雅子", "師長", "66666902", 2).
staff(18, "萩原　千代子", "師長", "55667782", 2).
staff(19, "大芝 まゆみ", "師長", "55555594", 2).
staff(20, "磯野　絵美", "師長", "44444443", 2).
staff(21, "竹田  礼子", "師長", "55555592", 2).
staff(22, "大森　ゆかり", "師長", "66666671", 2).
staff(23, "山本  智子", "師長", "55555593", 2).
staff(24, "蓮沼　知津子", "師長", "5555566", 2).
staff(25, "渡邉　理映子", "師長", "66666620", 0).
staff(26, "金丸　明美", "師長", "66666651", 0).
staff(27, "窪川  佳世", "師長", "55555595", 0).
staff(28, "三平　まゆみ", "師長", "5555559", 0).
staff(29, "岩澤　久美", "師長", "6666662", 0).
staff(30, "永田　明子", "師長", "4444442", 0).
staff(31, "谷戸　るみ", "師長", "55555600", 0).
staff(32, "山本　ゆかり", "師長", "55555580", 0).
staff(33, "松土　裕子", "師長", "44444445", 0).
staff(34, "青木　真里", "師長", "44444446", 0).
staff(35, "武田　陽子", "師長", "66666670", 0).
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
group("全員", 24).
group("師長").
group("熟練").
group("熟練１").
group("熟練１", 1).
group("熟練１", 3).
group("熟練１", 4).
group("熟練１", 5).
group("熟練１", 6).
group("熟練１", 7).
group("熟練１", 8).
group("熟練１", 9).
group("熟練１", 10).
group("熟練１", 11).
group("熟練１", 12).
group("熟練１", 13).
group("熟練１", 14).
group("熟練１", 15).
group("熟練１", 16).
group("熟練１", 17).
group("熟練１", 18).
group("熟練１", 19).
group("熟練１", 20).
group("熟練１", 21).
group("熟練１", 22).
group("熟練１", 23).
group("熟練１", 24).
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
shifts("N+P+★").
shifts("N+P+★", "N").
shifts("N+P+★", "P").
shifts("N+P+★", "★").
staff_dweek_soft_ub("全員", "N+P+★", "平日", 1).
staff_dweek_soft_ub("全員", "N+P+★", "土日", 1).
staff_dweek_soft_ub("全員", "N+P+★", "祝日", 1).
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
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("熟練１","N","人数").
vertical_constraint_type("熟練１","J","人数").
vertical_constraint_type("熟練１","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "N", 1).
shifts_soft_lb(1, "J", 1).
shifts_soft_lb(1, "S", 1).
shifts_soft_ub(1, "N", 1).
shifts_soft_ub(1, "J", 1).
shifts_soft_ub(1, "S", 1).
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
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "J", 0).
shifts_soft_lb(2, "S", 0).
shifts_soft_ub(2, "N", 0).
shifts_soft_ub(2, "J", 0).
shifts_soft_ub(2, "S", 0).
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
shifts_soft_ub(3, "N", 1).
shifts_soft_ub(3, "J", 1).
shifts_soft_ub(3, "S", 1).
shifts_soft_ub(3, "★", 0).
shifts_soft_ub(3, "☆", 0).
shifts_soft_ub(3, "A", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "★", 0).
shifts_hard_ub(3, "☆", 0).
shifts_hard_ub(3, "A", 0).
shifts_hard_ub(3, "P", 0).
% Staff 4:
shifts_soft_lb(4, "N", 1).
shifts_soft_lb(4, "J", 1).
shifts_soft_lb(4, "S", 1).
shifts_soft_ub(4, "N", 1).
shifts_soft_ub(4, "J", 1).
shifts_soft_ub(4, "S", 1).
shifts_soft_ub(4, "★", 0).
shifts_soft_ub(4, "☆", 0).
shifts_soft_ub(4, "A", 0).
shifts_soft_ub(4, "P", 0).
shifts_hard_ub(4, "★", 0).
shifts_hard_ub(4, "☆", 0).
shifts_hard_ub(4, "A", 0).
shifts_hard_ub(4, "P", 0).
% Staff 5:
shifts_soft_lb(5, "N", 1).
shifts_soft_lb(5, "J", 1).
shifts_soft_lb(5, "S", 1).
shifts_soft_ub(5, "N", 1).
shifts_soft_ub(5, "J", 1).
shifts_soft_ub(5, "S", 1).
shifts_soft_ub(5, "★", 0).
shifts_soft_ub(5, "☆", 0).
shifts_soft_ub(5, "A", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_hard_ub(5, "A", 0).
shifts_hard_ub(5, "P", 0).
% Staff 6:
shifts_soft_lb(6, "N", 1).
shifts_soft_lb(6, "J", 1).
shifts_soft_lb(6, "S", 1).
shifts_soft_ub(6, "N", 1).
shifts_soft_ub(6, "J", 1).
shifts_soft_ub(6, "S", 1).
shifts_soft_ub(6, "★", 0).
shifts_soft_ub(6, "☆", 0).
shifts_soft_ub(6, "A", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "★", 0).
shifts_hard_ub(6, "☆", 0).
shifts_hard_ub(6, "A", 0).
shifts_hard_ub(6, "P", 0).
% Staff 7:
shifts_soft_lb(7, "N", 1).
shifts_soft_lb(7, "J", 1).
shifts_soft_lb(7, "S", 1).
shifts_soft_ub(7, "N", 1).
shifts_soft_ub(7, "J", 1).
shifts_soft_ub(7, "S", 1).
shifts_soft_ub(7, "★", 0).
shifts_soft_ub(7, "☆", 0).
shifts_soft_ub(7, "A", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_hard_ub(7, "A", 0).
shifts_hard_ub(7, "P", 0).
% Staff 8:
shifts_soft_lb(8, "N", 1).
shifts_soft_lb(8, "J", 1).
shifts_soft_lb(8, "S", 1).
shifts_soft_ub(8, "N", 1).
shifts_soft_ub(8, "J", 1).
shifts_soft_ub(8, "S", 1).
shifts_soft_ub(8, "★", 0).
shifts_soft_ub(8, "☆", 0).
shifts_soft_ub(8, "A", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "★", 0).
shifts_hard_ub(8, "☆", 0).
shifts_hard_ub(8, "A", 0).
shifts_hard_ub(8, "P", 0).
% Staff 9:
shifts_soft_lb(9, "N", 1).
shifts_soft_lb(9, "J", 1).
shifts_soft_lb(9, "S", 1).
shifts_soft_ub(9, "N", 1).
shifts_soft_ub(9, "J", 1).
shifts_soft_ub(9, "S", 1).
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
shifts_soft_ub(10, "N", 1).
shifts_soft_ub(10, "J", 1).
shifts_soft_ub(10, "S", 1).
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
shifts_soft_ub(11, "N", 1).
shifts_soft_ub(11, "J", 1).
shifts_soft_ub(11, "S", 1).
shifts_soft_ub(11, "★", 0).
shifts_soft_ub(11, "☆", 0).
shifts_soft_ub(11, "A", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_hard_ub(11, "A", 0).
shifts_hard_ub(11, "P", 0).
% Staff 12:
shifts_soft_lb(12, "N", 1).
shifts_soft_lb(12, "J", 1).
shifts_soft_lb(12, "S", 1).
shifts_soft_ub(12, "N", 1).
shifts_soft_ub(12, "J", 1).
shifts_soft_ub(12, "S", 1).
shifts_soft_ub(12, "★", 0).
shifts_soft_ub(12, "☆", 0).
shifts_soft_ub(12, "A", 0).
shifts_soft_ub(12, "P", 0).
shifts_hard_ub(12, "★", 0).
shifts_hard_ub(12, "☆", 0).
shifts_hard_ub(12, "A", 0).
shifts_hard_ub(12, "P", 0).
% Staff 13:
shifts_soft_lb(13, "N", 1).
shifts_soft_lb(13, "J", 1).
shifts_soft_lb(13, "S", 1).
shifts_soft_ub(13, "N", 1).
shifts_soft_ub(13, "J", 1).
shifts_soft_ub(13, "S", 1).
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
shifts_soft_ub(14, "N", 1).
shifts_soft_ub(14, "J", 1).
shifts_soft_ub(14, "S", 1).
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
shifts_soft_ub(15, "N", 1).
shifts_soft_ub(15, "J", 1).
shifts_soft_ub(15, "S", 1).
shifts_soft_ub(15, "★", 0).
shifts_soft_ub(15, "☆", 0).
shifts_soft_ub(15, "A", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_hard_ub(15, "A", 0).
shifts_hard_ub(15, "P", 0).
% Staff 16:
shifts_soft_lb(16, "N", 1).
shifts_soft_lb(16, "J", 1).
shifts_soft_lb(16, "S", 1).
shifts_soft_ub(16, "N", 1).
shifts_soft_ub(16, "J", 1).
shifts_soft_ub(16, "S", 1).
shifts_soft_ub(16, "★", 0).
shifts_soft_ub(16, "☆", 0).
shifts_soft_ub(16, "A", 0).
shifts_soft_ub(16, "P", 0).
shifts_hard_ub(16, "★", 0).
shifts_hard_ub(16, "☆", 0).
shifts_hard_ub(16, "A", 0).
shifts_hard_ub(16, "P", 0).
% Staff 17:
shifts_soft_lb(17, "N", 1).
shifts_soft_lb(17, "J", 1).
shifts_soft_lb(17, "S", 1).
shifts_soft_ub(17, "N", 1).
shifts_soft_ub(17, "J", 1).
shifts_soft_ub(17, "S", 1).
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
shifts_soft_ub(18, "N", 1).
shifts_soft_ub(18, "J", 1).
shifts_soft_ub(18, "S", 1).
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
shifts_soft_ub(19, "N", 1).
shifts_soft_ub(19, "J", 1).
shifts_soft_ub(19, "S", 1).
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
shifts_soft_ub(20, "N", 1).
shifts_soft_ub(20, "J", 1).
shifts_soft_ub(20, "S", 1).
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
shifts_soft_ub(21, "N", 1).
shifts_soft_ub(21, "J", 1).
shifts_soft_ub(21, "S", 1).
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
shifts_soft_ub(22, "N", 1).
shifts_soft_ub(22, "J", 1).
shifts_soft_ub(22, "S", 1).
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
shifts_soft_ub(23, "N", 1).
shifts_soft_ub(23, "J", 1).
shifts_soft_ub(23, "S", 1).
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
shifts_soft_ub(24, "N", 1).
shifts_soft_ub(24, "J", 1).
shifts_soft_ub(24, "S", 1).
shifts_soft_ub(24, "★", 0).
shifts_soft_ub(24, "☆", 0).
shifts_soft_ub(24, "A", 0).
shifts_soft_ub(24, "P", 0).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_hard_ub(24, "A", 0).
shifts_hard_ub(24, "P", 0).
% Staff 25:
% Staff 26:
% Staff 27:
% Staff 28:
% Staff 29:
% Staff 30:
% Staff 31:
% Staff 32:
% Staff 33:
% Staff 34:
% Staff 35:
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
