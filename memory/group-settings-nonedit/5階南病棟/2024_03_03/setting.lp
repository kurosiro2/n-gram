% Staffs ---------------------------------------------------
staff(1, "髙橋　里香", "師長", "0123555", 0).
staff(2, "清水　美紀", "副師長", "0123544", 0).
staff(3, "神宮司　文", "副師長", "0146427", 0).
staff(4, "小林　伊吹", "看護師", "0355727", 0).
staff(5, "小佐野　まりな", "看護師", "0312576", 0).
staff(6, "篠原　梨花", "看護師", "0355454", 0).
staff(7, "成田　彩月", "看護師", "0355556", 0).
staff(8, "石橋　朋華", "看護師", "0370761", 0).
staff(9, "南　かれん", "看護師", "0370588", 0).
staff(10, "髙山　りえ", "看護師", "0329337", 0).
staff(11, "矢崎　優華", "看護師", "0384687", 0).
staff(12, "矢野　杏藍", "看護師", "0399641", 0).
staff(13, "笹本　茉衣", "看護師", "0341710", 0).
staff(14, "細井　一成", "看護師", "0418791", 0).
staff(15, "清水　優花", "看護師", "0418814", 0).
staff(16, "金野　舞子", "看護師", "0418825", 0).
staff(17, "深沢　泉", "副師長", "0192253", 0).
staff(18, "中村　香織", "看護師", "0283504", 0).
staff(19, "赤池　詩織", "看護師", "0232821", 0).
staff(20, "武藤　裕紀", "看護師", "0370954", 0).
staff(21, "本田　愛梨香", "看護師", "0313054", 0).
staff(22, "岡本　のどか", "看護師", "0340913", 0).
staff(23, "一瀬　妃菜", "看護師", "0355294", 0).
staff(24, "堀内　美希", "看護師", "0370340", 0).
staff(25, "石黒　鈴葉", "看護師", "0315411", 0).
staff(26, "三井　歩夢", "看護師", "0384643", 0).
staff(27, "渡邉　栞里", "看護師", "0384712", 0).
staff(28, "志村　実優", "看護師", "0399801", 0).
staff(29, "阿部　知依奈", "看護師", "0350460", 0).
staff(30, "岩本　麻希", "看護師", "0399663", 0).
staff(31, "中島　夏希", "看護師", "0341708", 0).
staff(32, "鈴木　百花", "看護師", "0418803", 0).
staff(33, "信田　舞奈", "看護師", "0418404", 0).
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
group("全員", 24).
group("全員", 25).
group("全員", 26).
group("全員", 27).
group("全員", 28).
group("全員", 29).
group("全員", 30).
group("全員", 31).
group("全員", 32).
group("全員", 33).
group("師長").
group("師長", 1).
group("熟練").
group("熟練", 2).
group("熟練", 3).
group("熟練", 4).
group("熟練", 5).
group("熟練", 6).
group("熟練", 17).
group("熟練", 18).
group("熟練", 19).
group("熟練", 20).
group("熟練", 21).
group("中堅").
group("中堅", 7).
group("中堅", 8).
group("中堅", 9).
group("メンバー").
group("メンバー", 10).
group("メンバー", 11).
group("メンバー", 12).
group("メンバー", 13).
group("メンバー", 14).
group("メンバー", 22).
group("メンバー", 23).
group("メンバー", 24).
group("メンバー", 25).
group("メンバー", 26).
group("メンバー", 27).
group("メンバー", 28).
group("メンバー", 29).
group("メンバー", 30).
group("メンバー", 31).
group("新人").
group("新人", 15).
group("新人", 16).
group("新人", 32).
group("新人", 33).
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
group("夜勤", 24).
group("夜勤", 25).
group("夜勤", 26).
group("夜勤", 27).
group("夜勤", 28).
group("夜勤", 29).
group("夜勤", 30).
group("夜勤", 31).
group("夜勤", 32).
group("夜勤", 33).
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
forbidden_pattern("日日日日日").
pattern("日日日日日", 5).
pattern("日日日日日", 0, "日").
pattern("日日日日日", 1, "日").
pattern("日日日日日", 2, "日").
pattern("日日日日日", 3, "日").
pattern("日日日日日", 4, "日").
forbidden_pattern("日日日NPJS").
pattern("日日日NPJS", 7).
pattern("日日日NPJS", 0, "日").
pattern("日日日NPJS", 1, "日").
pattern("日日日NPJS", 2, "日").
pattern("日日日NPJS", 3, "N").
pattern("日日日NPJS", 4, "P").
pattern("日日日NPJS", 5, "J").
pattern("日日日NPJS", 6, "S").
forbidden_pattern("PP").
pattern("PP", 2).
pattern("PP", 0, "P").
pattern("PP", 1, "P").
forbidden_pattern("日日日日N").
pattern("日日日日N", 5).
pattern("日日日日N", 0, "日").
pattern("日日日日N", 1, "日").
pattern("日日日日N", 2, "日").
pattern("日日日日N", 3, "日").
pattern("日日日日N", 4, "N").
forbidden_pattern("日日日日P").
pattern("日日日日P", 5).
pattern("日日日日P", 0, "日").
pattern("日日日日P", 1, "日").
pattern("日日日日P", 2, "日").
pattern("日日日日P", 3, "日").
pattern("日日日日P", 4, "P").
forbidden_pattern("日日日日JSJS").
pattern("日日日日JSJS", 8).
pattern("日日日日JSJS", 0, "日").
pattern("日日日日JSJS", 1, "日").
pattern("日日日日JSJS", 2, "日").
pattern("日日日日JSJS", 3, "日").
pattern("日日日日JSJS", 4, "J").
pattern("日日日日JSJS", 5, "S").
pattern("日日日日JSJS", 6, "J").
pattern("日日日日JSJS", 7, "S").
% 前日・翌日可能シフト
pred_shift("J", "S").
succ_shift("S", "J").
succ_shift("S", "○").
succ_shift("S", "◎").
pred_shift("N", "J").
pred_shift("P", "J").
pred_shift("日", "J").
pred_shift("○", "J").
succ_shift("J", "S").
% パターン割当数
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
patterns_soft_ub("○○", 2).
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
staff_dweek_hard_lb("全員", "N", "平日", 4).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 4).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 4).
staff_dweek_hard_lb("全員", "N", "土日", 4).
staff_dweek_hard_lb("全員", "J", "土日", 4).
staff_dweek_hard_lb("全員", "S", "土日", 4).
staff_dweek_hard_lb("全員", "N", "祝日", 4).
staff_dweek_hard_lb("全員", "J", "祝日", 4).
staff_dweek_hard_lb("全員", "S", "祝日", 4).
staff_dweek_soft_lb("全員", "N", "平日", 4).
staff_dweek_soft_lb("全員", "J", "平日", 4).
staff_dweek_soft_lb("全員", "S", "平日", 4).
staff_dweek_soft_lb("全員", "N", "土日", 4).
staff_dweek_soft_lb("全員", "J", "土日", 4).
staff_dweek_soft_lb("全員", "S", "土日", 4).
staff_dweek_soft_lb("全員", "N", "祝日", 4).
staff_dweek_soft_lb("全員", "J", "祝日", 4).
staff_dweek_soft_lb("全員", "S", "祝日", 4).
staff_dweek_soft_ub("全員", "N", "平日", 4).
staff_dweek_soft_ub("全員", "J", "平日", 4).
staff_dweek_soft_ub("全員", "S", "平日", 4).
staff_dweek_soft_ub("全員", "N", "土日", 4).
staff_dweek_soft_ub("全員", "J", "土日", 4).
staff_dweek_soft_ub("全員", "S", "土日", 4).
staff_dweek_soft_ub("全員", "N", "祝日", 4).
staff_dweek_soft_ub("全員", "J", "祝日", 4).
staff_dweek_soft_ub("全員", "S", "祝日", 4).
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
staff_dweek_soft_lb("全員", "N+P+★", "平日", 4).
staff_dweek_soft_lb("全員", "N+P+★", "土日", 3).
staff_dweek_soft_lb("全員", "N+P+★", "祝日", 3).
staff_dweek_soft_ub("全員", "N+P+★", "平日", 5).
staff_dweek_soft_ub("全員", "N+P+★", "土日", 4).
staff_dweek_soft_ub("全員", "N+P+★", "祝日", 4).
staff_dweek_hard_lb("熟練", "N", "平日", 1).
staff_dweek_hard_lb("熟練", "J", "平日", 1).
staff_dweek_hard_lb("熟練", "S", "平日", 1).
staff_dweek_hard_lb("熟練", "N", "土日", 1).
staff_dweek_hard_lb("熟練", "J", "土日", 1).
staff_dweek_hard_lb("熟練", "S", "土日", 1).
staff_dweek_hard_lb("熟練", "N", "祝日", 1).
staff_dweek_hard_lb("熟練", "J", "祝日", 1).
staff_dweek_hard_lb("熟練", "S", "祝日", 1).
staff_dweek_soft_lb("熟練", "N", "平日", 2).
staff_dweek_soft_lb("熟練", "J", "平日", 2).
staff_dweek_soft_lb("熟練", "S", "平日", 2).
staff_dweek_soft_lb("熟練", "N", "土日", 2).
staff_dweek_soft_lb("熟練", "J", "土日", 2).
staff_dweek_soft_lb("熟練", "S", "土日", 2).
staff_dweek_soft_lb("熟練", "N", "祝日", 2).
staff_dweek_soft_lb("熟練", "J", "祝日", 2).
staff_dweek_soft_lb("熟練", "S", "祝日", 2).
staff_dweek_soft_ub("熟練", "N", "平日", 3).
staff_dweek_soft_ub("熟練", "J", "平日", 3).
staff_dweek_soft_ub("熟練", "S", "平日", 3).
staff_dweek_soft_ub("熟練", "N", "土日", 3).
staff_dweek_soft_ub("熟練", "J", "土日", 3).
staff_dweek_soft_ub("熟練", "S", "土日", 3).
staff_dweek_soft_ub("熟練", "N", "祝日", 3).
staff_dweek_soft_ub("熟練", "J", "祝日", 3).
staff_dweek_soft_ub("熟練", "S", "祝日", 3).
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
staff_dweek_soft_ub("中堅", "J", "平日", 4).
staff_dweek_soft_ub("中堅", "S", "平日", 4).
staff_dweek_soft_ub("中堅", "J", "土日", 4).
staff_dweek_soft_ub("中堅", "S", "土日", 4).
staff_dweek_soft_ub("中堅", "J", "祝日", 4).
staff_dweek_soft_ub("中堅", "S", "祝日", 4).
staff_dweek_hard_lb("メンバー", "J", "平日", 0).
staff_dweek_hard_lb("メンバー", "S", "平日", 0).
staff_dweek_hard_lb("メンバー", "J", "土日", 0).
staff_dweek_hard_lb("メンバー", "S", "土日", 0).
staff_dweek_hard_lb("メンバー", "J", "祝日", 0).
staff_dweek_hard_lb("メンバー", "S", "祝日", 0).
staff_dweek_soft_lb("メンバー", "J", "平日", 1).
staff_dweek_soft_lb("メンバー", "S", "平日", 1).
staff_dweek_soft_lb("メンバー", "J", "土日", 1).
staff_dweek_soft_lb("メンバー", "S", "土日", 1).
staff_dweek_soft_lb("メンバー", "J", "祝日", 1).
staff_dweek_soft_lb("メンバー", "S", "祝日", 1).
staff_dweek_soft_ub("メンバー", "J", "平日", 4).
staff_dweek_soft_ub("メンバー", "S", "平日", 4).
staff_dweek_soft_ub("メンバー", "J", "土日", 4).
staff_dweek_soft_ub("メンバー", "S", "土日", 4).
staff_dweek_soft_ub("メンバー", "J", "祝日", 4).
staff_dweek_soft_ub("メンバー", "S", "祝日", 4).
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
shifts("J+★").
shifts("J+★", "J").
shifts("J+★", "★").
staff_dweek_soft_lb("夜勤", "J+★", "平日", 4).
shifts("S+☆").
shifts("S+☆", "S").
shifts("S+☆", "☆").
staff_dweek_soft_lb("夜勤", "S+☆", "平日", 4).
staff_dweek_soft_lb("夜勤", "J+★", "土日", 4).
staff_dweek_soft_lb("夜勤", "S+☆", "土日", 4).
staff_dweek_soft_lb("夜勤", "J+★", "祝日", 4).
staff_dweek_soft_lb("夜勤", "S+☆", "祝日", 4).
staff_dweek_soft_ub("夜勤", "J+★", "平日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "平日", 4).
staff_dweek_soft_ub("夜勤", "J+★", "土日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "土日", 4).
staff_dweek_soft_ub("夜勤", "J+★", "祝日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "祝日", 4).
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
vertical_constraint_type("メンバー","J","人数").
vertical_constraint_type("メンバー","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
vertical_constraint_type("夜勤","J+★","人数").
vertical_constraint_type("夜勤","S+☆","人数").
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
shifts_soft_ub(2, "N", 2).
shifts_soft_ub(2, "J", 2).
shifts_soft_ub(2, "S", 2).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "★", 1).
shifts_soft_ub(2, "☆", 1).
shifts_soft_ub(2, "P", 1).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_soft_lb(3, "N", 1).
shifts_soft_lb(3, "J", 1).
shifts_soft_lb(3, "S", 1).
shifts_soft_ub(3, "N", 3).
shifts_soft_ub(3, "J", 3).
shifts_soft_ub(3, "S", 3).
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
shifts_soft_ub(5, "N", 3).
shifts_soft_ub(5, "J", 3).
shifts_soft_ub(5, "S", 3).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "★", 1).
shifts_soft_ub(5, "☆", 1).
shifts_soft_ub(5, "P", 1).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_soft_lb(6, "N", 1).
shifts_soft_lb(6, "J", 1).
shifts_soft_lb(6, "S", 1).
shifts_soft_ub(6, "N", 3).
shifts_soft_ub(6, "J", 3).
shifts_soft_ub(6, "S", 3).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "★", 1).
shifts_soft_ub(6, "☆", 1).
shifts_soft_ub(6, "P", 1).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_soft_lb(7, "N", 1).
shifts_soft_lb(7, "J", 1).
shifts_soft_lb(7, "S", 1).
shifts_soft_ub(7, "N", 3).
shifts_soft_ub(7, "J", 3).
shifts_soft_ub(7, "S", 3).
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
shifts_soft_ub(8, "N", 3).
shifts_soft_ub(8, "J", 3).
shifts_soft_ub(8, "S", 3).
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
shifts_soft_ub(9, "N", 3).
shifts_soft_ub(9, "J", 3).
shifts_soft_ub(9, "S", 3).
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
shifts_soft_ub(10, "N", 3).
shifts_soft_ub(10, "J", 3).
shifts_soft_ub(10, "S", 3).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "★", 1).
shifts_soft_ub(10, "☆", 1).
shifts_soft_ub(10, "P", 1).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_soft_lb(11, "N", 1).
shifts_soft_lb(11, "J", 1).
shifts_soft_lb(11, "S", 1).
shifts_soft_ub(11, "N", 3).
shifts_soft_ub(11, "J", 3).
shifts_soft_ub(11, "S", 3).
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
shifts_soft_ub(12, "N", 3).
shifts_soft_ub(12, "J", 3).
shifts_soft_ub(12, "S", 3).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "★", 1).
shifts_soft_ub(12, "☆", 1).
shifts_soft_ub(12, "P", 1).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_soft_lb(13, "N", 1).
shifts_soft_lb(13, "J", 1).
shifts_soft_lb(13, "S", 1).
shifts_soft_ub(13, "N", 3).
shifts_soft_ub(13, "J", 3).
shifts_soft_ub(13, "S", 3).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "★", 1).
shifts_soft_ub(13, "☆", 1).
shifts_soft_ub(13, "P", 1).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_soft_lb(14, "N", 1).
shifts_soft_lb(14, "J", 1).
shifts_soft_lb(14, "S", 1).
shifts_soft_ub(14, "N", 3).
shifts_soft_ub(14, "J", 3).
shifts_soft_ub(14, "S", 3).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "★", 1).
shifts_soft_ub(14, "☆", 1).
shifts_soft_ub(14, "P", 1).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_soft_lb(15, "N", 1).
shifts_soft_lb(15, "J", 1).
shifts_soft_lb(15, "S", 1).
shifts_soft_ub(15, "N", 3).
shifts_soft_ub(15, "J", 3).
shifts_soft_ub(15, "S", 3).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "★", 1).
shifts_soft_ub(15, "☆", 1).
shifts_soft_ub(15, "P", 1).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_soft_lb(16, "N", 1).
shifts_soft_lb(16, "J", 1).
shifts_soft_lb(16, "S", 1).
shifts_soft_ub(16, "N", 3).
shifts_soft_ub(16, "J", 3).
shifts_soft_ub(16, "S", 3).
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
shifts_soft_ub(17, "N", 2).
shifts_soft_ub(17, "J", 2).
shifts_soft_ub(17, "S", 2).
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
shifts_soft_ub(18, "N", 3).
shifts_soft_ub(18, "J", 3).
shifts_soft_ub(18, "S", 3).
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
shifts_soft_ub(19, "N", 3).
shifts_soft_ub(19, "J", 3).
shifts_soft_ub(19, "S", 3).
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
shifts_soft_ub(20, "N", 3).
shifts_soft_ub(20, "J", 3).
shifts_soft_ub(20, "S", 3).
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
shifts_soft_lb(22, "★", 0).
shifts_soft_lb(22, "☆", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "★", 1).
shifts_soft_ub(22, "☆", 1).
shifts_soft_ub(22, "P", 1).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_soft_lb(23, "N", 1).
shifts_soft_lb(23, "J", 1).
shifts_soft_lb(23, "S", 1).
shifts_soft_ub(23, "N", 3).
shifts_soft_ub(23, "J", 3).
shifts_soft_ub(23, "S", 3).
shifts_soft_lb(23, "★", 0).
shifts_soft_lb(23, "☆", 0).
shifts_soft_lb(23, "P", 0).
shifts_soft_ub(23, "★", 1).
shifts_soft_ub(23, "☆", 1).
shifts_soft_ub(23, "P", 1).
shifts_hard_ub(23, "A", 0).
% Staff 24:
shifts_soft_lb(24, "N", 1).
shifts_soft_lb(24, "J", 1).
shifts_soft_lb(24, "S", 1).
shifts_soft_ub(24, "N", 3).
shifts_soft_ub(24, "J", 3).
shifts_soft_ub(24, "S", 3).
shifts_soft_lb(24, "★", 0).
shifts_soft_lb(24, "☆", 0).
shifts_soft_lb(24, "P", 0).
shifts_soft_ub(24, "★", 1).
shifts_soft_ub(24, "☆", 1).
shifts_soft_ub(24, "P", 1).
shifts_hard_ub(24, "A", 0).
% Staff 25:
shifts_soft_lb(25, "N", 1).
shifts_soft_lb(25, "J", 1).
shifts_soft_lb(25, "S", 1).
shifts_soft_ub(25, "N", 3).
shifts_soft_ub(25, "J", 3).
shifts_soft_ub(25, "S", 3).
shifts_soft_lb(25, "★", 0).
shifts_soft_lb(25, "☆", 0).
shifts_soft_lb(25, "P", 0).
shifts_soft_ub(25, "★", 1).
shifts_soft_ub(25, "☆", 1).
shifts_soft_ub(25, "P", 1).
shifts_hard_ub(25, "A", 0).
% Staff 26:
shifts_soft_lb(26, "N", 1).
shifts_soft_lb(26, "J", 1).
shifts_soft_lb(26, "S", 1).
shifts_soft_ub(26, "N", 3).
shifts_soft_ub(26, "J", 3).
shifts_soft_ub(26, "S", 3).
shifts_soft_lb(26, "★", 0).
shifts_soft_lb(26, "☆", 0).
shifts_soft_lb(26, "P", 0).
shifts_soft_ub(26, "★", 1).
shifts_soft_ub(26, "☆", 1).
shifts_soft_ub(26, "P", 1).
shifts_hard_ub(26, "A", 0).
% Staff 27:
shifts_soft_lb(27, "N", 1).
shifts_soft_lb(27, "J", 1).
shifts_soft_lb(27, "S", 1).
shifts_soft_ub(27, "N", 3).
shifts_soft_ub(27, "J", 3).
shifts_soft_ub(27, "S", 3).
shifts_soft_lb(27, "★", 0).
shifts_soft_lb(27, "☆", 0).
shifts_soft_lb(27, "P", 0).
shifts_soft_ub(27, "★", 1).
shifts_soft_ub(27, "☆", 1).
shifts_soft_ub(27, "P", 1).
shifts_hard_ub(27, "A", 0).
% Staff 28:
shifts_soft_lb(28, "N", 1).
shifts_soft_lb(28, "J", 1).
shifts_soft_lb(28, "S", 1).
shifts_soft_ub(28, "N", 3).
shifts_soft_ub(28, "J", 3).
shifts_soft_ub(28, "S", 3).
shifts_soft_lb(28, "★", 0).
shifts_soft_lb(28, "☆", 0).
shifts_soft_lb(28, "P", 0).
shifts_soft_ub(28, "★", 1).
shifts_soft_ub(28, "☆", 1).
shifts_soft_ub(28, "P", 1).
shifts_hard_ub(28, "A", 0).
% Staff 29:
shifts_soft_lb(29, "N", 1).
shifts_soft_lb(29, "J", 1).
shifts_soft_lb(29, "S", 1).
shifts_soft_ub(29, "N", 3).
shifts_soft_ub(29, "J", 3).
shifts_soft_ub(29, "S", 3).
shifts_soft_lb(29, "★", 0).
shifts_soft_lb(29, "☆", 0).
shifts_soft_lb(29, "P", 0).
shifts_soft_ub(29, "★", 1).
shifts_soft_ub(29, "☆", 1).
shifts_soft_ub(29, "P", 1).
shifts_hard_ub(29, "A", 0).
% Staff 30:
shifts_soft_lb(30, "N", 1).
shifts_soft_lb(30, "J", 1).
shifts_soft_lb(30, "S", 1).
shifts_soft_ub(30, "N", 3).
shifts_soft_ub(30, "J", 3).
shifts_soft_ub(30, "S", 3).
shifts_soft_lb(30, "★", 0).
shifts_soft_lb(30, "☆", 0).
shifts_soft_lb(30, "P", 0).
shifts_soft_ub(30, "★", 1).
shifts_soft_ub(30, "☆", 1).
shifts_soft_ub(30, "P", 1).
shifts_hard_ub(30, "A", 0).
% Staff 31:
shifts_soft_lb(31, "N", 1).
shifts_soft_lb(31, "J", 1).
shifts_soft_lb(31, "S", 1).
shifts_soft_ub(31, "N", 3).
shifts_soft_ub(31, "J", 3).
shifts_soft_ub(31, "S", 3).
shifts_soft_lb(31, "★", 0).
shifts_soft_lb(31, "☆", 0).
shifts_soft_lb(31, "P", 0).
shifts_soft_ub(31, "★", 1).
shifts_soft_ub(31, "☆", 1).
shifts_soft_ub(31, "P", 1).
shifts_hard_ub(31, "A", 0).
% Staff 32:
shifts_soft_lb(32, "N", 1).
shifts_soft_lb(32, "J", 1).
shifts_soft_lb(32, "S", 1).
shifts_soft_ub(32, "N", 3).
shifts_soft_ub(32, "J", 3).
shifts_soft_ub(32, "S", 3).
shifts_soft_lb(32, "★", 0).
shifts_soft_lb(32, "☆", 0).
shifts_soft_lb(32, "P", 0).
shifts_soft_ub(32, "★", 1).
shifts_soft_ub(32, "☆", 1).
shifts_soft_ub(32, "P", 1).
shifts_hard_ub(32, "A", 0).
% Staff 33:
shifts_soft_lb(33, "N", 1).
shifts_soft_lb(33, "J", 1).
shifts_soft_lb(33, "S", 1).
shifts_soft_ub(33, "N", 3).
shifts_soft_ub(33, "J", 3).
shifts_soft_ub(33, "S", 3).
shifts_soft_lb(33, "★", 0).
shifts_soft_lb(33, "☆", 0).
shifts_soft_lb(33, "P", 0).
shifts_soft_ub(33, "★", 1).
shifts_soft_ub(33, "☆", 1).
shifts_soft_ub(33, "P", 1).
shifts_hard_ub(33, "A", 0).
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
