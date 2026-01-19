% Staffs ---------------------------------------------------
staff(1, "櫻本　かおり", "師長", "0102275", 0).
staff(2, "上原　良江", "副師長", "0021574", 5).
staff(3, "齊藤　渚", "副師長", "0163361", 5).
staff(4, "伊藤　祥子", "副師長", "0163521", 5).
staff(5, "松山　和佳", "看護師", "0056417", 0).
staff(6, "藤巻　夕香里", "看護師", "0123486", 0).
staff(7, "佐藤　恵理香", "看護師", "0354841", 0).
staff(8, "甘蔗　華子", "看護師", "0250721", 0).
staff(9, "菅原　梨沙", "看護師", "0269483", 0).
staff(10, "田上　貴士", "看護師", "0340514", 0).
staff(11, "土橋　美佳", "看護師", "0341036", 0).
staff(12, "大畠　綾香", "看護師", "0370522", 0).
staff(13, "伊東　美来", "看護師", "0370668", 0).
staff(14, "萩原　万友佳", "看護師", "0315796", 0).
staff(15, "雨宮　周実", "看護師", "0384233", 0).
staff(16, "渡邉　樹里", "看護師", "0384723", 0).
staff(17, "伊東　未緒", "看護師", "0349720", 0).
staff(18, "中野　佑麻", "看護師", "0399960", 0).
staff(19, "岩波　みさと", "看護師", "0400072", 0).
staff(20, "芳賀　友香", "看護師", "0400265", 0).
staff(21, "高井　彩", "看護師", "0357890", 0).
staff(22, "渡邊　陽子", "看護師", "0418860", 0).
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
group("師長").
group("師長", 1).
group("熟練").
group("熟練１").
group("熟練１", 3).
group("熟練１", 4).
group("熟練１", 5).
group("熟練１", 7).
group("熟練１", 8).
group("熟練1").
group("熟練1", 3).
group("熟練1", 4).
group("熟練1", 5).
group("熟練1", 7).
group("熟練1", 8).
group("中堅").
group("中堅", 2).
group("中堅", 6).
group("中堅", 9).
group("中堅", 10).
group("中堅", 11).
group("中堅", 12).
group("中堅", 13).
group("中堅", 14).
group("中堅", 15).
group("中堅", 16).
group("中堅", 17).
group("中堅", 18).
group("中堅", 19).
group("中堅", 20).
group("新人").
group("新人", 21).
group("新人", 22).
group("新人１").
group("新人１", 21).
group("新人１", 22).
group("新人２").
group("新人２", 21).
group("新人２", 22).
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
group("夜勤", 12).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 19).
group("夜勤", 20).
group("夜勤", 21).
group("夜勤", 22).
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
forbidden_pattern("JSJSJS").
pattern("JSJSJS", 6).
pattern("JSJSJS", 0, "J").
pattern("JSJSJS", 1, "S").
pattern("JSJSJS", 2, "J").
pattern("JSJSJS", 3, "S").
pattern("JSJSJS", 4, "J").
pattern("JSJSJS", 5, "S").
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
patterns("JSJSJS", "JSJSJS").
patterns_soft_ub("JSJSJS", 0).
% Vertical constraints -------------------------------------
shifts("日").
shifts("日", "日").
staff_dweek_hard_lb("全員", "日", "平日", 4).
staff_dweek_hard_lb("全員", "日", "土日", 1).
staff_dweek_hard_lb("全員", "日", "祝日", 1).
staff_dweek_soft_lb("全員", "日", "平日", 4).
staff_dweek_soft_lb("全員", "日", "土日", 2).
staff_dweek_soft_lb("全員", "日", "祝日", 2).
staff_dweek_soft_ub("全員", "日", "平日", 13).
staff_dweek_soft_ub("全員", "日", "土日", 3).
staff_dweek_soft_ub("全員", "日", "祝日", 3).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 3).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 3).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 3).
staff_dweek_hard_lb("全員", "N", "土日", 3).
staff_dweek_hard_lb("全員", "J", "土日", 3).
staff_dweek_hard_lb("全員", "S", "土日", 3).
staff_dweek_hard_lb("全員", "N", "祝日", 3).
staff_dweek_hard_lb("全員", "J", "祝日", 3).
staff_dweek_hard_lb("全員", "S", "祝日", 3).
staff_dweek_hard_ub("全員", "N", "平日", 3).
staff_dweek_hard_ub("全員", "J", "平日", 3).
staff_dweek_hard_ub("全員", "S", "平日", 3).
staff_dweek_hard_ub("全員", "N", "土日", 3).
staff_dweek_hard_ub("全員", "J", "土日", 3).
staff_dweek_hard_ub("全員", "S", "土日", 3).
staff_dweek_hard_ub("全員", "N", "祝日", 3).
staff_dweek_hard_ub("全員", "J", "祝日", 3).
staff_dweek_hard_ub("全員", "S", "祝日", 3).
shifts("★").
shifts("★", "★").
staff_dweek_hard_ub("全員", "★", "平日", 0).
shifts("☆").
shifts("☆", "☆").
staff_dweek_hard_ub("全員", "☆", "平日", 0).
staff_dweek_hard_ub("全員", "★", "土日", 0).
staff_dweek_hard_ub("全員", "☆", "土日", 0).
staff_dweek_hard_ub("全員", "★", "祝日", 0).
staff_dweek_hard_ub("全員", "☆", "祝日", 0).
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
staff_dweek_hard_lb("熟練１", "N", "平日", 0).
staff_dweek_hard_lb("熟練１", "J", "平日", 0).
staff_dweek_hard_lb("熟練１", "S", "平日", 0).
staff_dweek_hard_lb("熟練１", "N", "土日", 0).
staff_dweek_hard_lb("熟練１", "J", "土日", 0).
staff_dweek_hard_lb("熟練１", "S", "土日", 0).
staff_dweek_hard_lb("熟練１", "N", "祝日", 0).
staff_dweek_hard_lb("熟練１", "J", "祝日", 0).
staff_dweek_hard_lb("熟練１", "S", "祝日", 0).
staff_dweek_soft_lb("熟練１", "N", "平日", 0).
staff_dweek_soft_lb("熟練１", "J", "平日", 0).
staff_dweek_soft_lb("熟練１", "S", "平日", 0).
staff_dweek_soft_lb("熟練１", "N", "土日", 0).
staff_dweek_soft_lb("熟練１", "J", "土日", 0).
staff_dweek_soft_lb("熟練１", "S", "土日", 0).
staff_dweek_soft_lb("熟練１", "N", "祝日", 0).
staff_dweek_soft_lb("熟練１", "J", "祝日", 0).
staff_dweek_soft_lb("熟練１", "S", "祝日", 0).
staff_dweek_soft_ub("熟練１", "N", "平日", 3).
staff_dweek_soft_ub("熟練１", "J", "平日", 3).
staff_dweek_soft_ub("熟練１", "S", "平日", 3).
staff_dweek_soft_ub("熟練１", "N", "土日", 3).
staff_dweek_soft_ub("熟練１", "J", "土日", 3).
staff_dweek_soft_ub("熟練１", "S", "土日", 3).
staff_dweek_soft_ub("熟練１", "N", "祝日", 3).
staff_dweek_soft_ub("熟練１", "J", "祝日", 3).
staff_dweek_soft_ub("熟練１", "S", "祝日", 3).
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
staff_dweek_soft_lb("中堅", "J", "平日", 1).
staff_dweek_soft_lb("中堅", "S", "平日", 1).
staff_dweek_soft_lb("中堅", "J", "土日", 1).
staff_dweek_soft_lb("中堅", "S", "土日", 1).
staff_dweek_soft_lb("中堅", "J", "祝日", 1).
staff_dweek_soft_lb("中堅", "S", "祝日", 1).
staff_dweek_soft_ub("中堅", "J", "平日", 3).
staff_dweek_soft_ub("中堅", "S", "平日", 3).
staff_dweek_soft_ub("中堅", "J", "土日", 3).
staff_dweek_soft_ub("中堅", "S", "土日", 3).
staff_dweek_soft_ub("中堅", "J", "祝日", 3).
staff_dweek_soft_ub("中堅", "S", "祝日", 3).
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
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("全員","★","人数").
vertical_constraint_type("全員","☆","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("熟練１","N","人数").
vertical_constraint_type("熟練１","J","人数").
vertical_constraint_type("熟練１","S","人数").
vertical_constraint_type("師長","日","人数").
vertical_constraint_type("中堅","J","人数").
vertical_constraint_type("中堅","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
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
shifts_soft_lb(1, "★", 0).
shifts_soft_lb(1, "☆", 0).
shifts_soft_lb(1, "P", 0).
shifts_soft_ub(1, "★", 0).
shifts_soft_ub(1, "☆", 0).
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_hard_ub(1, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_soft_lb(2, "N", 1).
shifts_soft_lb(2, "J", 1).
shifts_soft_lb(2, "S", 1).
shifts_soft_ub(2, "N", 3).
shifts_soft_ub(2, "J", 3).
shifts_soft_ub(2, "S", 3).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "★", 1).
shifts_soft_ub(2, "☆", 1).
shifts_soft_ub(2, "P", 1).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_soft_lb(3, "N", 0).
shifts_soft_lb(3, "J", 0).
shifts_soft_lb(3, "S", 0).
shifts_soft_ub(3, "N", 1).
shifts_soft_ub(3, "J", 1).
shifts_soft_ub(3, "S", 1).
shifts_hard_ub(3, "N", 2).
shifts_hard_ub(3, "J", 2).
shifts_hard_ub(3, "S", 2).
shifts_soft_lb(3, "★", 0).
shifts_soft_lb(3, "☆", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "★", 1).
shifts_soft_ub(3, "☆", 1).
shifts_soft_ub(3, "P", 1).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_soft_lb(4, "N", 1).
shifts_soft_lb(4, "J", 1).
shifts_soft_lb(4, "S", 1).
shifts_soft_ub(4, "N", 3).
shifts_soft_ub(4, "J", 3).
shifts_soft_ub(4, "S", 3).
shifts_soft_lb(4, "★", 0).
shifts_soft_lb(4, "☆", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "★", 1).
shifts_soft_ub(4, "☆", 1).
shifts_soft_ub(4, "P", 1).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_soft_lb(5, "N", 1).
shifts_soft_lb(5, "J", 1).
shifts_soft_lb(5, "S", 1).
shifts_soft_ub(5, "N", 4).
shifts_soft_ub(5, "J", 4).
shifts_soft_ub(5, "S", 4).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "★", 1).
shifts_soft_ub(5, "☆", 1).
shifts_soft_ub(5, "P", 1).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_soft_lb(6, "N", 3).
shifts_soft_lb(6, "J", 3).
shifts_soft_lb(6, "S", 3).
shifts_soft_ub(6, "N", 3).
shifts_soft_ub(6, "J", 3).
shifts_soft_ub(6, "S", 3).
shifts_hard_ub(6, "N", 3).
shifts_hard_ub(6, "J", 3).
shifts_hard_ub(6, "S", 3).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "★", 0).
shifts_soft_ub(6, "☆", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_soft_lb(7, "N", 1).
shifts_soft_lb(7, "J", 1).
shifts_soft_lb(7, "S", 1).
shifts_soft_ub(7, "N", 4).
shifts_soft_ub(7, "J", 4).
shifts_soft_ub(7, "S", 4).
shifts_soft_lb(7, "★", 0).
shifts_soft_lb(7, "☆", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "★", 1).
shifts_soft_ub(7, "☆", 1).
shifts_soft_ub(7, "P", 1).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_soft_lb(8, "N", 1).
shifts_soft_lb(8, "J", 1).
shifts_soft_lb(8, "S", 1).
shifts_soft_ub(8, "N", 4).
shifts_soft_ub(8, "J", 4).
shifts_soft_ub(8, "S", 4).
shifts_soft_lb(8, "★", 0).
shifts_soft_lb(8, "☆", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "★", 1).
shifts_soft_ub(8, "☆", 1).
shifts_soft_ub(8, "P", 1).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_soft_lb(9, "N", 1).
shifts_soft_lb(9, "J", 1).
shifts_soft_lb(9, "S", 1).
shifts_soft_ub(9, "N", 4).
shifts_soft_ub(9, "J", 4).
shifts_soft_ub(9, "S", 4).
shifts_soft_lb(9, "★", 0).
shifts_soft_lb(9, "☆", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "★", 1).
shifts_soft_ub(9, "☆", 1).
shifts_soft_ub(9, "P", 1).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_soft_lb(10, "N", 1).
shifts_soft_lb(10, "J", 1).
shifts_soft_lb(10, "S", 1).
shifts_soft_ub(10, "N", 4).
shifts_soft_ub(10, "J", 4).
shifts_soft_ub(10, "S", 4).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "★", 1).
shifts_soft_ub(10, "☆", 1).
shifts_soft_ub(10, "P", 1).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_soft_lb(11, "N", 0).
shifts_soft_lb(11, "J", 0).
shifts_soft_lb(11, "S", 0).
shifts_soft_ub(11, "N", 0).
shifts_soft_ub(11, "J", 0).
shifts_soft_ub(11, "S", 0).
shifts_hard_ub(11, "N", 0).
shifts_hard_ub(11, "J", 0).
shifts_hard_ub(11, "S", 0).
shifts_soft_lb(11, "★", 0).
shifts_soft_lb(11, "☆", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "★", 1).
shifts_soft_ub(11, "☆", 1).
shifts_soft_ub(11, "P", 1).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_soft_lb(12, "N", 1).
shifts_soft_lb(12, "J", 1).
shifts_soft_lb(12, "S", 1).
shifts_soft_ub(12, "N", 4).
shifts_soft_ub(12, "J", 4).
shifts_soft_ub(12, "S", 4).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "★", 1).
shifts_soft_ub(12, "☆", 1).
shifts_soft_ub(12, "P", 1).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_soft_lb(13, "N", 0).
shifts_soft_lb(13, "J", 0).
shifts_soft_lb(13, "S", 0).
shifts_soft_ub(13, "N", 4).
shifts_soft_ub(13, "J", 4).
shifts_soft_ub(13, "S", 4).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "★", 1).
shifts_soft_ub(13, "☆", 1).
shifts_soft_ub(13, "P", 1).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_soft_lb(14, "N", 0).
shifts_soft_lb(14, "J", 0).
shifts_soft_lb(14, "S", 0).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "★", 1).
shifts_soft_ub(14, "☆", 1).
shifts_soft_ub(14, "P", 1).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_soft_lb(15, "N", 0).
shifts_soft_lb(15, "J", 0).
shifts_soft_lb(15, "S", 0).
shifts_soft_ub(15, "N", 0).
shifts_soft_ub(15, "J", 0).
shifts_soft_ub(15, "S", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_ub(15, "J", 0).
shifts_hard_ub(15, "S", 0).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "★", 1).
shifts_soft_ub(15, "☆", 1).
shifts_soft_ub(15, "P", 1).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_soft_lb(16, "N", 0).
shifts_soft_lb(16, "J", 0).
shifts_soft_lb(16, "S", 0).
shifts_soft_ub(16, "N", 2).
shifts_soft_ub(16, "J", 2).
shifts_soft_ub(16, "S", 2).
shifts_hard_ub(16, "N", 3).
shifts_hard_ub(16, "J", 3).
shifts_hard_ub(16, "S", 3).
shifts_soft_lb(16, "★", 0).
shifts_soft_lb(16, "☆", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "★", 1).
shifts_soft_ub(16, "☆", 1).
shifts_soft_ub(16, "P", 1).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_soft_lb(17, "N", 1).
shifts_soft_lb(17, "J", 1).
shifts_soft_lb(17, "S", 1).
shifts_soft_ub(17, "N", 4).
shifts_soft_ub(17, "J", 4).
shifts_soft_ub(17, "S", 4).
shifts_soft_lb(17, "★", 0).
shifts_soft_lb(17, "☆", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "★", 1).
shifts_soft_ub(17, "☆", 1).
shifts_soft_ub(17, "P", 1).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_soft_lb(18, "N", 1).
shifts_soft_lb(18, "J", 1).
shifts_soft_lb(18, "S", 1).
shifts_soft_ub(18, "N", 4).
shifts_soft_ub(18, "J", 4).
shifts_soft_ub(18, "S", 4).
shifts_soft_lb(18, "★", 0).
shifts_soft_lb(18, "☆", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "★", 1).
shifts_soft_ub(18, "☆", 1).
shifts_soft_ub(18, "P", 1).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_soft_lb(19, "N", 1).
shifts_soft_lb(19, "J", 1).
shifts_soft_lb(19, "S", 1).
shifts_soft_ub(19, "N", 4).
shifts_soft_ub(19, "J", 4).
shifts_soft_ub(19, "S", 4).
shifts_soft_lb(19, "★", 0).
shifts_soft_lb(19, "☆", 0).
shifts_soft_lb(19, "P", 0).
shifts_soft_ub(19, "★", 1).
shifts_soft_ub(19, "☆", 1).
shifts_soft_ub(19, "P", 1).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_soft_lb(20, "N", 1).
shifts_soft_lb(20, "J", 1).
shifts_soft_lb(20, "S", 1).
shifts_soft_ub(20, "N", 4).
shifts_soft_ub(20, "J", 4).
shifts_soft_ub(20, "S", 4).
shifts_soft_lb(20, "★", 0).
shifts_soft_lb(20, "☆", 0).
shifts_soft_lb(20, "P", 0).
shifts_soft_ub(20, "★", 1).
shifts_soft_ub(20, "☆", 1).
shifts_soft_ub(20, "P", 1).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_soft_lb(21, "N", 1).
shifts_soft_lb(21, "J", 1).
shifts_soft_lb(21, "S", 1).
shifts_soft_ub(21, "N", 3).
shifts_soft_ub(21, "J", 3).
shifts_soft_ub(21, "S", 3).
shifts_hard_ub(21, "N", 4).
shifts_hard_ub(21, "J", 4).
shifts_hard_ub(21, "S", 4).
shifts_soft_lb(21, "★", 0).
shifts_soft_lb(21, "☆", 0).
shifts_soft_lb(21, "P", 0).
shifts_soft_ub(21, "★", 1).
shifts_soft_ub(21, "☆", 1).
shifts_soft_ub(21, "P", 1).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_soft_lb(22, "N", 1).
shifts_soft_lb(22, "J", 1).
shifts_soft_lb(22, "S", 1).
shifts_soft_ub(22, "N", 3).
shifts_soft_ub(22, "J", 3).
shifts_soft_ub(22, "S", 3).
shifts_hard_ub(22, "N", 4).
shifts_hard_ub(22, "J", 4).
shifts_hard_ub(22, "S", 4).
shifts_soft_lb(22, "★", 0).
shifts_soft_lb(22, "☆", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "★", 1).
shifts_soft_ub(22, "☆", 1).
shifts_soft_ub(22, "P", 1).
shifts_hard_ub(22, "A", 0).
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
