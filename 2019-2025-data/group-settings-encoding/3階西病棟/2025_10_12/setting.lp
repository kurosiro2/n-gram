% Staffs ---------------------------------------------------
staff(1, "金丸　紀子", "師長", "0039405", 0).
staff(2, "辻　稔", "副師長", "0017897", 0).
staff(3, "山下　湧大", "看護師", "0299547", 0).
staff(4, "桒原　鈴加", "看護師", "0383232", 0).
staff(5, "渡邉　南実", "看護師", "0418916", 0).
staff(6, "山本　龍", "看護師", "0358116", 0).
staff(7, "堀内　彩香", "看護師", "0469656", 0).
staff(8, "千葉　知輝", "看護師", "0606470", 0).
staff(9, "手塚　絵里子", "副師長", "0102355", 0).
staff(10, "上田　愛実", "看護師", "0299605", 0).
staff(11, "芦原　佳奈", "看護師", "0370407", 0).
staff(14, "小田切　美葵", "看護師", "0469452", 0).
staff(15, "阿部　葉月", "看護師", "0385622", 0).
staff(16, "長田　和子", "副師長", "0123475", 0).
staff(17, "勝村　美紀", "看護師", "0299423", 0).
staff(18, "樋口　希良莉", "看護師", "0315821", 0).
staff(19, "竹村　小春", "看護師", "0345895", 0).
staff(21, "長澤　吏桜", "看護師", "0365263", 0).
staff(22, "長瀬　光", "看護師", "0385974", 0).
staff(24, "田中　紳子", "サポート", "0020106", 0).
staff(25, "補）名取　真由美", "補助者", "0251378", 0).
staff(26, "派）船木　貴子", "補助者", "0397342", 0).
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
group("全員", 14).
group("全員", 15).
group("全員", 16).
group("全員", 17).
group("全員", 18).
group("全員", 19).
group("全員", 21).
group("全員", 22).
group("全員", 24).
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 9).
group("代行", 10).
group("代行", 11).
group("代行", 16).
group("代行", 17).
group("代行", 18).
group("A").
group("B").
group("新人").
group("新人", 8).
group("新人", 15).
group("新人", 22).
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
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 19).
group("夜勤", 21).
group("夜勤", 22).
group("夜勤", 24).
group("リーダー").
group("リーダー", 2).
group("リーダー", 3).
group("リーダー", 4).
group("リーダー", 5).
group("リーダー", 10).
group("リーダー", 11).
group("リーダー", 16).
group("リーダー", 17).
group("リーダー", 18).
group("リーダー", 19).
group("平日代行").
group("平日代行", 2).
group("平日代行", 9).
group("平日代行", 16).
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
staff_dweek_hard_lb("全員", "日", "平日", 5).
staff_dweek_hard_lb("全員", "日", "土日", 3).
staff_dweek_hard_lb("全員", "日", "祝日", 3).
staff_dweek_soft_lb("全員", "日", "平日", 6).
staff_dweek_soft_lb("全員", "日", "土日", 4).
staff_dweek_soft_lb("全員", "日", "祝日", 4).
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
staff_dweek_soft_lb("代行", "日", "平日", 1).
staff_dweek_soft_lb("代行", "日", "土日", 1).
staff_dweek_soft_lb("代行", "日", "祝日", 1).
staff_dweek_soft_ub("代行", "日", "平日", 2).
staff_dweek_soft_ub("代行", "日", "土日", 2).
staff_dweek_soft_ub("代行", "日", "祝日", 2).
staff_dweek_hard_lb("新人", "N", "平日", 1).
staff_dweek_hard_lb("新人", "J", "平日", 1).
staff_dweek_hard_lb("新人", "S", "平日", 1).
staff_dweek_hard_lb("新人", "N", "土日", 1).
staff_dweek_hard_lb("新人", "J", "土日", 1).
staff_dweek_hard_lb("新人", "S", "土日", 1).
staff_dweek_hard_lb("新人", "N", "祝日", 1).
staff_dweek_hard_lb("新人", "J", "祝日", 1).
staff_dweek_hard_lb("新人", "S", "祝日", 1).
staff_dweek_soft_lb("新人", "N", "平日", 1).
staff_dweek_soft_lb("新人", "J", "平日", 1).
staff_dweek_soft_lb("新人", "S", "平日", 1).
staff_dweek_soft_lb("新人", "N", "土日", 1).
staff_dweek_soft_lb("新人", "J", "土日", 1).
staff_dweek_soft_lb("新人", "S", "土日", 1).
staff_dweek_soft_lb("新人", "N", "祝日", 1).
staff_dweek_soft_lb("新人", "J", "祝日", 1).
staff_dweek_soft_lb("新人", "S", "祝日", 1).
staff_dweek_soft_ub("新人", "N", "平日", 1).
staff_dweek_soft_ub("新人", "J", "平日", 1).
staff_dweek_soft_ub("新人", "S", "平日", 1).
staff_dweek_soft_ub("新人", "N", "土日", 1).
staff_dweek_soft_ub("新人", "J", "土日", 1).
staff_dweek_soft_ub("新人", "S", "土日", 1).
staff_dweek_soft_ub("新人", "N", "祝日", 1).
staff_dweek_soft_ub("新人", "J", "祝日", 1).
staff_dweek_soft_ub("新人", "S", "祝日", 1).
staff_dweek_hard_ub("新人", "N", "平日", 1).
staff_dweek_hard_ub("新人", "J", "平日", 1).
staff_dweek_hard_ub("新人", "S", "平日", 1).
staff_dweek_hard_ub("新人", "N", "土日", 1).
staff_dweek_hard_ub("新人", "J", "土日", 1).
staff_dweek_hard_ub("新人", "S", "土日", 1).
staff_dweek_hard_ub("新人", "N", "祝日", 1).
staff_dweek_hard_ub("新人", "J", "祝日", 1).
staff_dweek_hard_ub("新人", "S", "祝日", 1).
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
staff_dweek_soft_ub("リーダー", "N", "平日", 2).
staff_dweek_soft_ub("リーダー", "J", "平日", 2).
staff_dweek_soft_ub("リーダー", "S", "平日", 2).
staff_dweek_soft_ub("リーダー", "N", "土日", 2).
staff_dweek_soft_ub("リーダー", "J", "土日", 2).
staff_dweek_soft_ub("リーダー", "S", "土日", 2).
staff_dweek_soft_ub("リーダー", "N", "祝日", 2).
staff_dweek_soft_ub("リーダー", "J", "祝日", 2).
staff_dweek_soft_ub("リーダー", "S", "祝日", 2).
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
staff_dweek_hard_lb("平日代行", "日", "土日", 1).
staff_dweek_hard_lb("平日代行", "日", "祝日", 1).
staff_dweek_soft_lb("平日代行", "日", "平日", 1).
staff_dweek_soft_lb("平日代行", "日", "土日", 1).
staff_dweek_soft_lb("平日代行", "日", "祝日", 1).
staff_dweek_soft_ub("平日代行", "日", "平日", 2).
staff_dweek_soft_ub("平日代行", "日", "土日", 2).
staff_dweek_soft_ub("平日代行", "日", "祝日", 2).
staff_dweek_hard_ub("平日代行", "日", "平日", 2).
staff_dweek_hard_ub("平日代行", "日", "土日", 2).
staff_dweek_hard_ub("平日代行", "日", "祝日", 2).
staff_dweek_hard_lb("夜勤", "J", "平日", 3).
staff_dweek_hard_lb("夜勤", "J", "土日", 3).
staff_dweek_hard_lb("夜勤", "J", "祝日", 3).
staff_dweek_soft_lb("夜勤", "J", "平日", 3).
staff_dweek_soft_lb("夜勤", "J", "土日", 3).
staff_dweek_soft_lb("夜勤", "J", "祝日", 3).
staff_dweek_soft_ub("夜勤", "J", "平日", 3).
staff_dweek_soft_ub("夜勤", "J", "土日", 3).
staff_dweek_soft_ub("夜勤", "J", "祝日", 3).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("代行","日","人数").
vertical_constraint_type("新人","N","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
vertical_constraint_type("リーダー","N","人数").
vertical_constraint_type("リーダー","J","人数").
vertical_constraint_type("リーダー","S","人数").
vertical_constraint_type("平日代行","日","人数").
vertical_constraint_type("夜勤","J","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_soft_ub(2, "日", 20).
shifts_hard_lb(2, "N", 1).
shifts_hard_lb(2, "J", 1).
shifts_hard_lb(2, "S", 1).
shifts_soft_lb(2, "N", 2).
shifts_soft_lb(2, "J", 2).
shifts_soft_lb(2, "S", 2).
shifts_soft_ub(2, "N", 2).
shifts_soft_ub(2, "J", 2).
shifts_soft_ub(2, "S", 2).
shifts_hard_ub(2, "N", 2).
shifts_hard_ub(2, "J", 2).
shifts_hard_ub(2, "S", 2).
% Staff 3:
shifts_soft_ub(3, "日", 20).
shifts_hard_lb(3, "N", 3).
shifts_hard_lb(3, "J", 3).
shifts_hard_lb(3, "S", 3).
shifts_soft_lb(3, "N", 4).
shifts_soft_lb(3, "J", 4).
shifts_soft_lb(3, "S", 4).
shifts_soft_ub(3, "N", 4).
shifts_soft_ub(3, "J", 4).
shifts_soft_ub(3, "S", 4).
shifts_hard_ub(3, "N", 5).
shifts_hard_ub(3, "J", 5).
shifts_hard_ub(3, "S", 5).
% Staff 4:
shifts_soft_ub(4, "日", 20).
shifts_hard_lb(4, "N", 3).
shifts_hard_lb(4, "J", 3).
shifts_hard_lb(4, "S", 3).
shifts_soft_lb(4, "N", 4).
shifts_soft_lb(4, "J", 4).
shifts_soft_lb(4, "S", 4).
shifts_soft_ub(4, "N", 4).
shifts_soft_ub(4, "J", 4).
shifts_soft_ub(4, "S", 4).
shifts_hard_ub(4, "N", 5).
shifts_hard_ub(4, "J", 5).
shifts_hard_ub(4, "S", 5).
% Staff 5:
shifts_soft_ub(5, "日", 20).
shifts_hard_lb(5, "N", 3).
shifts_hard_lb(5, "J", 3).
shifts_hard_lb(5, "S", 3).
shifts_soft_lb(5, "N", 4).
shifts_soft_lb(5, "J", 4).
shifts_soft_lb(5, "S", 4).
shifts_soft_ub(5, "N", 4).
shifts_soft_ub(5, "J", 4).
shifts_soft_ub(5, "S", 4).
shifts_hard_ub(5, "N", 5).
shifts_hard_ub(5, "J", 5).
shifts_hard_ub(5, "S", 5).
% Staff 6:
shifts_soft_ub(6, "日", 20).
shifts_hard_lb(6, "N", 3).
shifts_hard_lb(6, "J", 3).
shifts_hard_lb(6, "S", 3).
shifts_soft_lb(6, "N", 4).
shifts_soft_lb(6, "J", 4).
shifts_soft_lb(6, "S", 4).
shifts_soft_ub(6, "N", 4).
shifts_soft_ub(6, "J", 4).
shifts_soft_ub(6, "S", 4).
shifts_hard_ub(6, "N", 5).
shifts_hard_ub(6, "J", 5).
shifts_hard_ub(6, "S", 5).
% Staff 7:
shifts_soft_ub(7, "日", 20).
shifts_hard_lb(7, "N", 3).
shifts_hard_lb(7, "J", 3).
shifts_hard_lb(7, "S", 3).
shifts_soft_lb(7, "N", 4).
shifts_soft_lb(7, "J", 4).
shifts_soft_lb(7, "S", 4).
shifts_soft_ub(7, "N", 4).
shifts_soft_ub(7, "J", 4).
shifts_soft_ub(7, "S", 4).
shifts_hard_ub(7, "N", 5).
shifts_hard_ub(7, "J", 5).
shifts_hard_ub(7, "S", 5).
% Staff 8:
shifts_soft_ub(8, "日", 20).
shifts_hard_lb(8, "N", 1).
shifts_hard_lb(8, "J", 1).
shifts_hard_lb(8, "S", 1).
shifts_soft_lb(8, "N", 2).
shifts_soft_lb(8, "J", 2).
shifts_soft_lb(8, "S", 2).
shifts_soft_ub(8, "N", 2).
shifts_soft_ub(8, "J", 2).
shifts_soft_ub(8, "S", 2).
shifts_hard_ub(8, "N", 2).
shifts_hard_ub(8, "J", 2).
shifts_hard_ub(8, "S", 2).
% Staff 9:
shifts_soft_ub(9, "日", 20).
shifts_hard_lb(9, "N", 1).
shifts_hard_lb(9, "J", 1).
shifts_hard_lb(9, "S", 1).
shifts_soft_lb(9, "N", 1).
shifts_soft_lb(9, "J", 1).
shifts_soft_lb(9, "S", 1).
shifts_soft_ub(9, "N", 1).
shifts_soft_ub(9, "J", 1).
shifts_soft_ub(9, "S", 1).
shifts_hard_ub(9, "N", 1).
shifts_hard_ub(9, "J", 1).
shifts_hard_ub(9, "S", 1).
% Staff 10:
shifts_soft_ub(10, "日", 20).
shifts_hard_lb(10, "N", 3).
shifts_hard_lb(10, "J", 3).
shifts_hard_lb(10, "S", 3).
shifts_soft_lb(10, "N", 4).
shifts_soft_lb(10, "J", 4).
shifts_soft_lb(10, "S", 4).
shifts_soft_ub(10, "N", 4).
shifts_soft_ub(10, "J", 4).
shifts_soft_ub(10, "S", 4).
shifts_hard_ub(10, "N", 5).
shifts_hard_ub(10, "J", 5).
shifts_hard_ub(10, "S", 5).
% Staff 11:
shifts_soft_ub(11, "日", 20).
shifts_hard_lb(11, "N", 3).
shifts_hard_lb(11, "J", 3).
shifts_hard_lb(11, "S", 3).
shifts_soft_lb(11, "N", 4).
shifts_soft_lb(11, "J", 4).
shifts_soft_lb(11, "S", 4).
shifts_soft_ub(11, "N", 4).
shifts_soft_ub(11, "J", 4).
shifts_soft_ub(11, "S", 4).
shifts_hard_ub(11, "N", 5).
shifts_hard_ub(11, "J", 5).
shifts_hard_ub(11, "S", 5).
% Staff 14:
shifts_soft_ub(14, "日", 20).
shifts_hard_lb(14, "N", 3).
shifts_hard_lb(14, "J", 3).
shifts_hard_lb(14, "S", 3).
shifts_soft_lb(14, "N", 4).
shifts_soft_lb(14, "J", 4).
shifts_soft_lb(14, "S", 4).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_hard_ub(14, "N", 5).
shifts_hard_ub(14, "J", 5).
shifts_hard_ub(14, "S", 5).
% Staff 15:
shifts_soft_ub(15, "日", 20).
shifts_hard_lb(15, "N", 1).
shifts_hard_lb(15, "J", 1).
shifts_hard_lb(15, "S", 1).
shifts_soft_lb(15, "N", 2).
shifts_soft_lb(15, "J", 2).
shifts_soft_lb(15, "S", 2).
shifts_soft_ub(15, "N", 2).
shifts_soft_ub(15, "J", 2).
shifts_soft_ub(15, "S", 2).
shifts_hard_ub(15, "N", 2).
shifts_hard_ub(15, "J", 2).
shifts_hard_ub(15, "S", 2).
% Staff 16:
shifts_soft_ub(16, "日", 20).
shifts_hard_lb(16, "N", 1).
shifts_hard_lb(16, "J", 1).
shifts_hard_lb(16, "S", 1).
shifts_soft_lb(16, "N", 2).
shifts_soft_lb(16, "J", 2).
shifts_soft_lb(16, "S", 2).
shifts_soft_ub(16, "N", 2).
shifts_soft_ub(16, "J", 2).
shifts_soft_ub(16, "S", 2).
shifts_hard_ub(16, "N", 2).
shifts_hard_ub(16, "J", 2).
shifts_hard_ub(16, "S", 2).
% Staff 17:
shifts_soft_ub(17, "日", 20).
shifts_hard_lb(17, "N", 3).
shifts_hard_lb(17, "J", 3).
shifts_hard_lb(17, "S", 3).
shifts_soft_lb(17, "N", 4).
shifts_soft_lb(17, "J", 4).
shifts_soft_lb(17, "S", 4).
shifts_soft_ub(17, "N", 4).
shifts_soft_ub(17, "J", 4).
shifts_soft_ub(17, "S", 4).
shifts_hard_ub(17, "N", 5).
shifts_hard_ub(17, "J", 5).
shifts_hard_ub(17, "S", 5).
% Staff 18:
shifts_soft_ub(18, "日", 20).
shifts_hard_lb(18, "N", 3).
shifts_hard_lb(18, "J", 3).
shifts_hard_lb(18, "S", 3).
shifts_soft_lb(18, "N", 4).
shifts_soft_lb(18, "J", 4).
shifts_soft_lb(18, "S", 4).
shifts_soft_ub(18, "N", 4).
shifts_soft_ub(18, "J", 4).
shifts_soft_ub(18, "S", 4).
shifts_hard_ub(18, "N", 5).
shifts_hard_ub(18, "J", 5).
shifts_hard_ub(18, "S", 5).
% Staff 19:
shifts_soft_ub(19, "日", 20).
shifts_hard_lb(19, "N", 3).
shifts_hard_lb(19, "J", 3).
shifts_hard_lb(19, "S", 3).
shifts_soft_lb(19, "N", 4).
shifts_soft_lb(19, "J", 4).
shifts_soft_lb(19, "S", 4).
shifts_soft_ub(19, "N", 4).
shifts_soft_ub(19, "J", 4).
shifts_soft_ub(19, "S", 4).
shifts_hard_ub(19, "N", 5).
shifts_hard_ub(19, "J", 5).
shifts_hard_ub(19, "S", 5).
% Staff 21:
shifts_soft_ub(21, "日", 20).
shifts_hard_lb(21, "N", 3).
shifts_hard_lb(21, "J", 3).
shifts_hard_lb(21, "S", 3).
shifts_soft_lb(21, "N", 4).
shifts_soft_lb(21, "J", 4).
shifts_soft_lb(21, "S", 4).
shifts_soft_ub(21, "N", 4).
shifts_soft_ub(21, "J", 4).
shifts_soft_ub(21, "S", 4).
shifts_hard_ub(21, "N", 5).
shifts_hard_ub(21, "J", 5).
shifts_hard_ub(21, "S", 5).
% Staff 22:
shifts_soft_ub(22, "日", 20).
shifts_hard_lb(22, "N", 1).
shifts_hard_lb(22, "J", 1).
shifts_hard_lb(22, "S", 1).
shifts_soft_lb(22, "N", 2).
shifts_soft_lb(22, "J", 2).
shifts_soft_lb(22, "S", 2).
shifts_soft_ub(22, "N", 2).
shifts_soft_ub(22, "J", 2).
shifts_soft_ub(22, "S", 2).
shifts_hard_ub(22, "N", 2).
shifts_hard_ub(22, "J", 2).
shifts_hard_ub(22, "S", 2).
% Staff 24:
shifts_soft_ub(24, "日", 20).
shifts_hard_lb(24, "N", 1).
shifts_hard_lb(24, "J", 1).
shifts_hard_lb(24, "S", 1).
shifts_soft_lb(24, "N", 1).
shifts_soft_lb(24, "J", 1).
shifts_soft_lb(24, "S", 1).
shifts_soft_ub(24, "N", 1).
shifts_soft_ub(24, "J", 1).
shifts_soft_ub(24, "S", 1).
shifts_hard_ub(24, "N", 1).
shifts_hard_ub(24, "J", 1).
shifts_hard_ub(24, "S", 1).
% Staff 25:
% Staff 26:
horizontal_constraint_type("日").
horizontal_constraint_type("N").
horizontal_constraint_type("J").
horizontal_constraint_type("S").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
