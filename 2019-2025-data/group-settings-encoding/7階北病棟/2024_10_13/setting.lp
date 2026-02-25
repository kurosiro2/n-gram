% Staffs ---------------------------------------------------
staff(1, "相川　真弓", "副師長", "0058237", 0).
staff(2, "秋山　友梨", "副師長", "0191662", 0).
staff(3, "織田　茉莉恵", "副師長", "0196122", 0).
staff(4, "浅野　ともみ", "看護師", "0207181", 0).
staff(5, "石川　美空", "看護師長", "0254667", 0).
staff(6, "松木　美枝子", "副看護師長", "0282970", 0).
staff(7, "岡部　和歌菜", "看護師", "0326425", 0).
staff(8, "牧野　佳奈", "看護師長", "0326755", 0).
staff(9, "山下　美紀", "副看護師長", "0326870", 0).
staff(10, "大久保　沙都", "看護師", "0329133", 0).
staff(11, "櫻井　ちか", "看護師長", "0355215", 0).
staff(12, "長野　加琳", "副看護師長", "0355647", 0).
staff(13, "御手洗　芽衣", "看護師", "0358058", 0).
staff(14, "横田　那央", "看護師長", "0358127", 0).
staff(15, "牧野　英司", "副看護師長", "0367870", 0).
staff(16, "松野　汐里", "看護師", "0370704", 0).
staff(17, "雨宮　真弥", "看護師長", "0384244", 0).
staff(18, "小俣　日菜子", "副看護師長", "0384381", 0).
staff(19, "向山　優花", "看護師", "0384665", 0).
staff(20, "三島木　珠里", "看護師長", "0399798", 0).
staff(21, "村松　加菜", "副看護師長", "0399845", 0).
staff(22, "佐藤　信之介", "看護師", "0399903", 0).
staff(23, "河野　花凜", "看護師長", "0400298", 0).
staff(24, "保坂　奏芽", "副看護師長", "0418574", 0).
staff(25, "櫻本　あみ", "看護師", "0418608", 0).
staff(26, "藤井　嵐士", "看護師長", "0459448", 0).
staff(27, "遠藤　あいり", "副看護師長", "0469428", 0).
staff(28, "丹澤　一瑳", "看護師", "0469587", 0).
staff(29, "古屋　優衣", "看護師長", "0469645", 0).
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
group("全員", 25).
group("全員", 26).
group("全員", 27).
group("全員", 28).
group("全員", 29).
group("師長").
group("熟練").
group("熟練", 1).
group("熟練", 2).
group("熟練", 3).
group("熟練", 4).
group("熟練", 5).
group("熟練", 6).
group("中堅").
group("中堅", 7).
group("中堅", 8).
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
group("中堅", 21).
group("中堅", 22).
group("中堅", 23).
group("中堅", 24).
group("中堅", 25).
group("中堅", 26).
group("新人").
group("新人", 27).
group("新人", 28).
group("新人", 29).
group("夜勤").
group("夜勤", 1).
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
group("夜勤", 24).
group("夜勤", 25).
group("夜勤", 26).
group("夜勤", 27).
group("夜勤", 28).
group("夜勤", 29).
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
staff_dweek_hard_lb("全員", "日", "平日", 4).
staff_dweek_hard_lb("全員", "日", "土日", 1).
staff_dweek_hard_lb("全員", "日", "祝日", 1).
staff_dweek_soft_lb("全員", "日", "平日", 6).
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
staff_dweek_soft_lb("全員", "N", "平日", 4).
staff_dweek_soft_lb("全員", "J", "平日", 4).
staff_dweek_soft_lb("全員", "S", "平日", 4).
staff_dweek_soft_lb("全員", "N", "土日", 4).
staff_dweek_soft_lb("全員", "J", "土日", 4).
staff_dweek_soft_lb("全員", "S", "土日", 4).
staff_dweek_soft_lb("全員", "N", "祝日", 4).
staff_dweek_soft_lb("全員", "J", "祝日", 4).
staff_dweek_soft_lb("全員", "S", "祝日", 4).
staff_dweek_soft_ub("全員", "N", "平日", 5).
staff_dweek_soft_ub("全員", "J", "平日", 5).
staff_dweek_soft_ub("全員", "S", "平日", 5).
staff_dweek_soft_ub("全員", "N", "土日", 5).
staff_dweek_soft_ub("全員", "J", "土日", 5).
staff_dweek_soft_ub("全員", "S", "土日", 5).
staff_dweek_soft_ub("全員", "N", "祝日", 5).
staff_dweek_soft_ub("全員", "J", "祝日", 5).
staff_dweek_soft_ub("全員", "S", "祝日", 5).
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
staff_dweek_soft_lb("師長", "日", "平日", 1).
staff_dweek_soft_lb("師長", "日", "土日", 0).
staff_dweek_soft_lb("師長", "日", "祝日", 0).
staff_dweek_soft_ub("師長", "日", "平日", 1).
staff_dweek_soft_ub("師長", "日", "土日", 0).
staff_dweek_soft_ub("師長", "日", "祝日", 0).
staff_dweek_hard_lb("熟練", "N", "平日", 1).
staff_dweek_hard_lb("熟練", "J", "平日", 1).
staff_dweek_hard_lb("熟練", "S", "平日", 1).
staff_dweek_hard_lb("熟練", "N", "土日", 1).
staff_dweek_hard_lb("熟練", "J", "土日", 1).
staff_dweek_hard_lb("熟練", "S", "土日", 1).
staff_dweek_hard_lb("熟練", "N", "祝日", 1).
staff_dweek_hard_lb("熟練", "J", "祝日", 1).
staff_dweek_hard_lb("熟練", "S", "祝日", 1).
staff_dweek_soft_lb("熟練", "N", "平日", 1).
staff_dweek_soft_lb("熟練", "J", "平日", 1).
staff_dweek_soft_lb("熟練", "S", "平日", 1).
staff_dweek_soft_lb("熟練", "N", "土日", 1).
staff_dweek_soft_lb("熟練", "J", "土日", 1).
staff_dweek_soft_lb("熟練", "S", "土日", 1).
staff_dweek_soft_lb("熟練", "N", "祝日", 1).
staff_dweek_soft_lb("熟練", "J", "祝日", 1).
staff_dweek_soft_lb("熟練", "S", "祝日", 1).
staff_dweek_soft_ub("熟練", "N", "平日", 4).
staff_dweek_soft_ub("熟練", "J", "平日", 4).
staff_dweek_soft_ub("熟練", "S", "平日", 4).
staff_dweek_soft_ub("熟練", "N", "土日", 4).
staff_dweek_soft_ub("熟練", "J", "土日", 4).
staff_dweek_soft_ub("熟練", "S", "土日", 4).
staff_dweek_soft_ub("熟練", "N", "祝日", 4).
staff_dweek_soft_ub("熟練", "J", "祝日", 4).
staff_dweek_soft_ub("熟練", "S", "祝日", 4).
staff_dweek_hard_lb("中堅", "N", "平日", 1).
staff_dweek_hard_lb("中堅", "J", "平日", 1).
staff_dweek_hard_lb("中堅", "S", "平日", 1).
staff_dweek_hard_lb("中堅", "N", "土日", 1).
staff_dweek_hard_lb("中堅", "J", "土日", 1).
staff_dweek_hard_lb("中堅", "S", "土日", 1).
staff_dweek_hard_lb("中堅", "N", "祝日", 1).
staff_dweek_hard_lb("中堅", "J", "祝日", 1).
staff_dweek_hard_lb("中堅", "S", "祝日", 1).
staff_dweek_soft_lb("中堅", "N", "平日", 2).
staff_dweek_soft_lb("中堅", "J", "平日", 2).
staff_dweek_soft_lb("中堅", "S", "平日", 2).
staff_dweek_soft_lb("中堅", "N", "土日", 2).
staff_dweek_soft_lb("中堅", "J", "土日", 2).
staff_dweek_soft_lb("中堅", "S", "土日", 2).
staff_dweek_soft_lb("中堅", "N", "祝日", 2).
staff_dweek_soft_lb("中堅", "J", "祝日", 2).
staff_dweek_soft_lb("中堅", "S", "祝日", 2).
staff_dweek_soft_ub("中堅", "N", "平日", 4).
staff_dweek_soft_ub("中堅", "J", "平日", 4).
staff_dweek_soft_ub("中堅", "S", "平日", 4).
staff_dweek_soft_ub("中堅", "N", "土日", 4).
staff_dweek_soft_ub("中堅", "J", "土日", 4).
staff_dweek_soft_ub("中堅", "S", "土日", 4).
staff_dweek_soft_ub("中堅", "N", "祝日", 4).
staff_dweek_soft_ub("中堅", "J", "祝日", 4).
staff_dweek_soft_ub("中堅", "S", "祝日", 4).
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
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("師長","日","人数").
vertical_constraint_type("熟練","N","人数").
vertical_constraint_type("熟練","J","人数").
vertical_constraint_type("熟練","S","人数").
vertical_constraint_type("中堅","N","人数").
vertical_constraint_type("中堅","J","人数").
vertical_constraint_type("中堅","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "N", 4).
shifts_soft_lb(1, "J", 4).
shifts_soft_lb(1, "S", 4).
shifts_soft_ub(1, "N", 4).
shifts_soft_ub(1, "J", 4).
shifts_soft_ub(1, "S", 4).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_soft_lb(1, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_soft_lb(2, "N", 3).
shifts_soft_lb(2, "J", 3).
shifts_soft_lb(2, "S", 3).
shifts_soft_ub(2, "N", 3).
shifts_soft_ub(2, "J", 3).
shifts_soft_ub(2, "S", 3).
shifts_hard_ub(2, "★", 0).
shifts_hard_ub(2, "☆", 0).
shifts_soft_lb(2, "P", 0).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_soft_lb(3, "N", 3).
shifts_soft_lb(3, "J", 3).
shifts_soft_lb(3, "S", 3).
shifts_soft_ub(3, "N", 3).
shifts_soft_ub(3, "J", 3).
shifts_soft_ub(3, "S", 3).
shifts_hard_ub(3, "★", 0).
shifts_hard_ub(3, "☆", 0).
shifts_soft_lb(3, "P", 0).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_soft_lb(4, "N", 4).
shifts_soft_lb(4, "J", 4).
shifts_soft_lb(4, "S", 4).
shifts_soft_ub(4, "N", 4).
shifts_soft_ub(4, "J", 4).
shifts_soft_ub(4, "S", 4).
shifts_hard_ub(4, "★", 0).
shifts_hard_ub(4, "☆", 0).
shifts_soft_lb(4, "P", 0).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_soft_lb(5, "N", 5).
shifts_soft_lb(5, "J", 5).
shifts_soft_lb(5, "S", 5).
shifts_soft_ub(5, "N", 5).
shifts_soft_ub(5, "J", 5).
shifts_soft_ub(5, "S", 5).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_soft_lb(5, "P", 0).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_soft_lb(6, "N", 6).
shifts_soft_lb(6, "J", 6).
shifts_soft_lb(6, "S", 6).
shifts_soft_ub(6, "N", 6).
shifts_soft_ub(6, "J", 6).
shifts_soft_ub(6, "S", 6).
shifts_hard_ub(6, "★", 0).
shifts_hard_ub(6, "☆", 0).
shifts_soft_lb(6, "P", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_soft_lb(7, "N", 3).
shifts_soft_lb(7, "J", 3).
shifts_soft_lb(7, "S", 3).
shifts_soft_ub(7, "N", 3).
shifts_soft_ub(7, "J", 3).
shifts_soft_ub(7, "S", 3).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_soft_lb(7, "P", 0).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_soft_lb(8, "N", 7).
shifts_soft_lb(8, "J", 7).
shifts_soft_lb(8, "S", 7).
shifts_soft_ub(8, "N", 7).
shifts_soft_ub(8, "J", 7).
shifts_soft_ub(8, "S", 7).
shifts_hard_ub(8, "★", 0).
shifts_hard_ub(8, "☆", 0).
shifts_soft_lb(8, "P", 0).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_soft_lb(9, "N", 5).
shifts_soft_lb(9, "J", 5).
shifts_soft_lb(9, "S", 5).
shifts_soft_ub(9, "N", 5).
shifts_soft_ub(9, "J", 5).
shifts_soft_ub(9, "S", 5).
shifts_hard_ub(9, "★", 0).
shifts_hard_ub(9, "☆", 0).
shifts_soft_lb(9, "P", 0).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_soft_lb(10, "N", 6).
shifts_soft_lb(10, "J", 6).
shifts_soft_lb(10, "S", 6).
shifts_soft_ub(10, "N", 6).
shifts_soft_ub(10, "J", 6).
shifts_soft_ub(10, "S", 6).
shifts_hard_ub(10, "★", 0).
shifts_hard_ub(10, "☆", 0).
shifts_soft_lb(10, "P", 0).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_soft_lb(11, "N", 5).
shifts_soft_lb(11, "J", 5).
shifts_soft_lb(11, "S", 5).
shifts_soft_ub(11, "N", 5).
shifts_soft_ub(11, "J", 5).
shifts_soft_ub(11, "S", 5).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_soft_lb(11, "P", 0).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_soft_lb(12, "N", 4).
shifts_soft_lb(12, "J", 4).
shifts_soft_lb(12, "S", 4).
shifts_soft_ub(12, "N", 4).
shifts_soft_ub(12, "J", 4).
shifts_soft_ub(12, "S", 4).
shifts_hard_ub(12, "★", 0).
shifts_hard_ub(12, "☆", 0).
shifts_soft_lb(12, "P", 0).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_soft_lb(13, "N", 5).
shifts_soft_lb(13, "J", 5).
shifts_soft_lb(13, "S", 5).
shifts_soft_ub(13, "N", 5).
shifts_soft_ub(13, "J", 5).
shifts_soft_ub(13, "S", 5).
shifts_hard_ub(13, "★", 0).
shifts_hard_ub(13, "☆", 0).
shifts_soft_lb(13, "P", 0).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_soft_lb(14, "N", 6).
shifts_soft_lb(14, "J", 6).
shifts_soft_lb(14, "S", 6).
shifts_soft_ub(14, "N", 6).
shifts_soft_ub(14, "J", 6).
shifts_soft_ub(14, "S", 6).
shifts_hard_ub(14, "★", 0).
shifts_hard_ub(14, "☆", 0).
shifts_soft_lb(14, "P", 0).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_soft_lb(15, "N", 3).
shifts_soft_lb(15, "J", 3).
shifts_soft_lb(15, "S", 3).
shifts_soft_ub(15, "N", 3).
shifts_soft_ub(15, "J", 3).
shifts_soft_ub(15, "S", 3).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_soft_lb(15, "P", 0).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_soft_lb(16, "N", 6).
shifts_soft_lb(16, "J", 6).
shifts_soft_lb(16, "S", 6).
shifts_soft_ub(16, "N", 6).
shifts_soft_ub(16, "J", 6).
shifts_soft_ub(16, "S", 6).
shifts_hard_ub(16, "★", 0).
shifts_hard_ub(16, "☆", 0).
shifts_soft_lb(16, "P", 0).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_soft_lb(17, "N", 4).
shifts_soft_lb(17, "J", 4).
shifts_soft_lb(17, "S", 4).
shifts_soft_ub(17, "N", 4).
shifts_soft_ub(17, "J", 4).
shifts_soft_ub(17, "S", 4).
shifts_hard_ub(17, "★", 0).
shifts_hard_ub(17, "☆", 0).
shifts_soft_lb(17, "P", 0).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_soft_lb(18, "N", 1).
shifts_soft_lb(18, "J", 1).
shifts_soft_lb(18, "S", 1).
shifts_soft_ub(18, "N", 1).
shifts_soft_ub(18, "J", 1).
shifts_soft_ub(18, "S", 1).
shifts_hard_ub(18, "★", 3).
shifts_hard_ub(18, "☆", 3).
shifts_soft_lb(18, "P", 0).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_soft_lb(19, "N", 5).
shifts_soft_lb(19, "J", 5).
shifts_soft_lb(19, "S", 5).
shifts_soft_ub(19, "N", 5).
shifts_soft_ub(19, "J", 5).
shifts_soft_ub(19, "S", 5).
shifts_hard_ub(19, "★", 0).
shifts_hard_ub(19, "☆", 0).
shifts_soft_lb(19, "P", 0).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_soft_lb(20, "N", 6).
shifts_soft_lb(20, "J", 6).
shifts_soft_lb(20, "S", 6).
shifts_soft_ub(20, "N", 6).
shifts_soft_ub(20, "J", 6).
shifts_soft_ub(20, "S", 6).
shifts_hard_ub(20, "★", 0).
shifts_hard_ub(20, "☆", 0).
shifts_soft_lb(20, "P", 0).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_soft_lb(21, "N", 7).
shifts_soft_lb(21, "J", 7).
shifts_soft_lb(21, "S", 7).
shifts_soft_ub(21, "N", 7).
shifts_soft_ub(21, "J", 7).
shifts_soft_ub(21, "S", 7).
shifts_hard_ub(21, "★", 0).
shifts_hard_ub(21, "☆", 0).
shifts_soft_lb(21, "P", 0).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_soft_lb(22, "N", 4).
shifts_soft_lb(22, "J", 4).
shifts_soft_lb(22, "S", 4).
shifts_soft_ub(22, "N", 4).
shifts_soft_ub(22, "J", 4).
shifts_soft_ub(22, "S", 4).
shifts_hard_ub(22, "★", 0).
shifts_hard_ub(22, "☆", 0).
shifts_soft_lb(22, "P", 0).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_soft_lb(23, "N", 4).
shifts_soft_lb(23, "J", 4).
shifts_soft_lb(23, "S", 4).
shifts_soft_ub(23, "N", 4).
shifts_soft_ub(23, "J", 4).
shifts_soft_ub(23, "S", 4).
shifts_hard_ub(23, "★", 0).
shifts_hard_ub(23, "☆", 0).
shifts_soft_lb(23, "P", 0).
shifts_hard_ub(23, "A", 0).
% Staff 24:
shifts_soft_lb(24, "N", 5).
shifts_soft_lb(24, "J", 5).
shifts_soft_lb(24, "S", 5).
shifts_soft_ub(24, "N", 5).
shifts_soft_ub(24, "J", 5).
shifts_soft_ub(24, "S", 5).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_soft_lb(24, "P", 0).
shifts_hard_ub(24, "A", 0).
% Staff 25:
shifts_soft_lb(25, "N", 5).
shifts_soft_lb(25, "J", 5).
shifts_soft_lb(25, "S", 5).
shifts_soft_ub(25, "N", 5).
shifts_soft_ub(25, "J", 5).
shifts_soft_ub(25, "S", 5).
shifts_hard_ub(25, "★", 0).
shifts_hard_ub(25, "☆", 0).
shifts_soft_lb(25, "P", 0).
shifts_hard_ub(25, "A", 0).
% Staff 26:
shifts_soft_lb(26, "N", 6).
shifts_soft_lb(26, "J", 6).
shifts_soft_lb(26, "S", 6).
shifts_soft_ub(26, "N", 6).
shifts_soft_ub(26, "J", 6).
shifts_soft_ub(26, "S", 6).
shifts_hard_ub(26, "★", 0).
shifts_hard_ub(26, "☆", 0).
shifts_soft_lb(26, "P", 0).
shifts_hard_ub(26, "A", 0).
% Staff 27:
shifts_soft_lb(27, "N", 1).
shifts_soft_lb(27, "J", 1).
shifts_soft_lb(27, "S", 1).
shifts_soft_ub(27, "N", 1).
shifts_soft_ub(27, "J", 1).
shifts_soft_ub(27, "S", 1).
shifts_hard_ub(27, "★", 0).
shifts_hard_ub(27, "☆", 0).
shifts_soft_lb(27, "P", 0).
shifts_hard_ub(27, "A", 0).
% Staff 28:
shifts_soft_lb(28, "N", 1).
shifts_soft_lb(28, "J", 1).
shifts_soft_lb(28, "S", 1).
shifts_soft_ub(28, "N", 1).
shifts_soft_ub(28, "J", 1).
shifts_soft_ub(28, "S", 1).
shifts_hard_ub(28, "★", 0).
shifts_hard_ub(28, "☆", 0).
shifts_soft_lb(28, "P", 0).
shifts_hard_ub(28, "A", 0).
% Staff 29:
shifts_soft_lb(29, "N", 1).
shifts_soft_lb(29, "J", 1).
shifts_soft_lb(29, "S", 1).
shifts_soft_ub(29, "N", 1).
shifts_soft_ub(29, "J", 1).
shifts_soft_ub(29, "S", 1).
shifts_hard_ub(29, "★", 0).
shifts_hard_ub(29, "☆", 0).
shifts_soft_lb(29, "P", 0).
shifts_hard_ub(29, "A", 0).
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
