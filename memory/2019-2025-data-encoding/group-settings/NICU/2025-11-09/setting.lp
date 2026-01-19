% Staffs ---------------------------------------------------
staff(1, "Rachel Hendricks", "Chief Nurse", "46544", 0).
staff(2, "Ernest Oconnell", "Deputy Chief Nurse", "73372", 0).
staff(3, "Linda Baker", "Nurse", "35035", 0).
staff(4, "Sharon Barnes", "Nurse", "03732", 0).
staff(5, "Patrick Wood", "Nurse", "90021", 0).
staff(6, "Barbara Montoya", "Nurse", "71510", 0).
staff(7, "Melissa Bell", "Nurse", "81895", 0).
staff(8, "Cassandra Jackson", "Nurse", "35207", 0).
staff(9, "Jeremy Cunningham", "Nurse", "74427", 0).
staff(10, "Jessica Smith", "Nurse", "49197", 0).
staff(11, "Ezra Nelson", "Nurse", "A102110", 0).
staff(12, "Mason Nelson", "Nurse", "A211613", 0).
staff(13, "Mike Fernandez", "Deputy Chief Nurse", "98109", 0).
staff(14, "Christopher Martin", "Nurse", "39400", 0).
staff(15, "Kelly Massey", "Nurse", "51149", 0).
staff(16, "Alexandra Arnold", "Nurse", "69785", 0).
staff(17, "Laura Chambers MD", "Nurse", "76335", 0).
staff(18, "Alexander Cole", "Nurse", "86779", 0).
staff(19, "April Cook", "Nurse", "62445", 0).
staff(20, "Shannon Soto", "Midwife", "15069", 0).
staff(21, "Roberto Gonzalez", "Nurse", "40286", 0).
staff(22, "Gabriel Davis", "Nurse", "A561083", 0).
staff(23, "Michael Anderson", "Nurse", "A749068", 0).
staff(24, "Owen Mitchell", "補助者", "A197999", 0).
% Groups ---------------------------------------------------
group("All").
group("All", 2).
group("All", 3).
group("All", 4).
group("All", 5).
group("All", 6).
group("All", 7).
group("All", 8).
group("All", 9).
group("All", 10).
group("All", 11).
group("All", 12).
group("All", 13).
group("All", 14).
group("All", 15).
group("All", 16).
group("All", 17).
group("All", 18).
group("All", 19).
group("All", 20).
group("All", 21).
group("All", 22).
group("All", 23).
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 13).
group("代行", 14).
group("代行", 19).
group("代行", 21).
group("Leaders").
group("Leaders", 2).
group("Leaders", 3).
group("Leaders", 4).
group("Leaders", 7).
group("Leaders", 13).
group("Leaders", 14).
group("Leaders", 16).
group("Leaders", 19).
group("Leaders", 21).
group("メンバー").
group("メンバー", 5).
group("メンバー", 6).
group("メンバー", 8).
group("メンバー", 9).
group("メンバー", 10).
group("メンバー", 15).
group("Newcomers").
group("Newcomers", 11).
group("Newcomers", 12).
group("Newcomers", 17).
group("Newcomers", 18).
group("Newcomers", 20).
group("Newcomers", 22).
group("Newcomers", 23).
group("Night").
group("Night", 2).
group("Night", 3).
group("Night", 4).
group("Night", 5).
group("Night", 6).
group("Night", 7).
group("Night", 8).
group("Night", 9).
group("Night", 10).
group("Night", 11).
group("Night", 12).
group("Night", 13).
group("Night", 14).
group("Night", 15).
group("Night", 16).
group("Night", 17).
group("Night", 18).
group("Night", 19).
group("Night", 20).
group("Night", 21).
group("Night", 22).
group("Night", 23).
% Shifts ---------------------------------------------------
shift("D").
shift("LD").
shift("SE").
shift("SN").
shift("E").
shift("N").
shift("EM").
shift("LM").
shift("WR").
shift("PH").
shift("TR").
shift("AL").
shift("HL").
shift("SP").
shift("/").
% Shift constraints ----------------------------------------
% 連続勤務可能日数
consecutive_work_ub(5).
% 禁止パターン
forbidden_pattern("LD-LD").
pattern("LD-LD", 2).
pattern("LD-LD", 0, "LD").
pattern("LD-LD", 1, "LD").
forbidden_pattern("SE-SN-SE-SN").
pattern("SE-SN-SE-SN", 4).
pattern("SE-SN-SE-SN", 0, "SE").
pattern("SE-SN-SE-SN", 1, "SN").
pattern("SE-SN-SE-SN", 2, "SE").
pattern("SE-SN-SE-SN", 3, "SN").
forbidden_pattern("LD-D").
pattern("LD-D", 2).
pattern("LD-D", 0, "LD").
pattern("LD-D", 1, "D").
forbidden_pattern("LM-LD").
pattern("LM-LD", 2).
pattern("LM-LD", 0, "LM").
pattern("LM-LD", 1, "LD").
forbidden_pattern("LM-D").
pattern("LM-D", 2).
pattern("LM-D", 0, "LM").
pattern("LM-D", 1, "D").
forbidden_pattern("D-D-D-D-D-LD").
pattern("D-D-D-D-D-LD", 6).
pattern("D-D-D-D-D-LD", 0, "D").
pattern("D-D-D-D-D-LD", 1, "D").
pattern("D-D-D-D-D-LD", 2, "D").
pattern("D-D-D-D-D-LD", 3, "D").
pattern("D-D-D-D-D-LD", 4, "D").
pattern("D-D-D-D-D-LD", 5, "LD").
forbidden_pattern("D-D-D-D-LD-LM").
pattern("D-D-D-D-LD-LM", 6).
pattern("D-D-D-D-LD-LM", 0, "D").
pattern("D-D-D-D-LD-LM", 1, "D").
pattern("D-D-D-D-LD-LM", 2, "D").
pattern("D-D-D-D-LD-LM", 3, "D").
pattern("D-D-D-D-LD-LM", 4, "LD").
pattern("D-D-D-D-LD-LM", 5, "LM").
forbidden_pattern("D-D-D-D-LD-SE").
pattern("D-D-D-D-LD-SE", 6).
pattern("D-D-D-D-LD-SE", 0, "D").
pattern("D-D-D-D-LD-SE", 1, "D").
pattern("D-D-D-D-LD-SE", 2, "D").
pattern("D-D-D-D-LD-SE", 3, "D").
pattern("D-D-D-D-LD-SE", 4, "LD").
pattern("D-D-D-D-LD-SE", 5, "SE").
forbidden_pattern("D-D-LD-SE-SN-SE-SN").
pattern("D-D-LD-SE-SN-SE-SN", 7).
pattern("D-D-LD-SE-SN-SE-SN", 0, "D").
pattern("D-D-LD-SE-SN-SE-SN", 1, "D").
pattern("D-D-LD-SE-SN-SE-SN", 2, "LD").
pattern("D-D-LD-SE-SN-SE-SN", 3, "SE").
pattern("D-D-LD-SE-SN-SE-SN", 4, "SN").
pattern("D-D-LD-SE-SN-SE-SN", 5, "SE").
pattern("D-D-LD-SE-SN-SE-SN", 6, "SN").
forbidden_pattern("D-D-D-D-D-D").
pattern("D-D-D-D-D-D", 6).
pattern("D-D-D-D-D-D", 0, "D").
pattern("D-D-D-D-D-D", 1, "D").
pattern("D-D-D-D-D-D", 2, "D").
pattern("D-D-D-D-D-D", 3, "D").
pattern("D-D-D-D-D-D", 4, "D").
pattern("D-D-D-D-D-D", 5, "D").
% 前日・翌日可能シフト
pred_shift("SE", "SN").
succ_shift("SN", "SE").
succ_shift("SN", "WR").
succ_shift("SN", "PH").
pred_shift("LD", "SE").
pred_shift("LM", "SE").
pred_shift("D", "SE").
pred_shift("WR", "SE").
pred_shift("PH", "SE").
succ_shift("SE", "SN").
pred_shift("LD", "E").
pred_shift("LM", "E").
pred_shift("D", "E").
pred_shift("WR", "E").
pred_shift("PH", "E").
succ_shift("E", "N").
pred_shift("E", "N").
succ_shift("N", "WR").
succ_shift("N", "PH").
pred_shift("D", "D").
pred_shift("WR", "D").
pred_shift("PH", "D").
succ_shift("D", "LD").
succ_shift("D", "LM").
succ_shift("D", "D").
succ_shift("D", "WR").
succ_shift("D", "PH").
succ_shift("D", "SE").
succ_shift("D", "E").
pred_shift("D", "LD").
pred_shift("WR", "LD").
pred_shift("PH", "LD").
succ_shift("LD", "LM").
succ_shift("LD", "WR").
succ_shift("LD", "PH").
succ_shift("LD", "SE").
succ_shift("LD", "E").
pred_shift("LD", "LM").
pred_shift("D", "LM").
pred_shift("WR", "LM").
pred_shift("PH", "LM").
succ_shift("LM", "WR").
succ_shift("LM", "PH").
succ_shift("LM", "SE").
succ_shift("LM", "E").
% パターン割当数
patterns("WR-WR", "WR-WR").
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("○○", 1).
patterns("LD-D", "LD-D").
patterns_soft_ub("N日", 0).
patterns("LM-D", "LM-D").
patterns_soft_ub("P日", 0).
patterns("LM-LD", "LM-LD").
patterns_soft_ub("PN", 0).
patterns("LM-LM", "LM-LM").
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_soft_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("D").
shifts("D", "D").
staff_dweek_hard_lb("All", "D", "Weekday", 2).
staff_dweek_hard_lb("All", "D", "Weekend", 2).
staff_dweek_hard_lb("All", "D", "Holiday", 2).
staff_dweek_soft_lb("All", "D", "Weekday", 2).
staff_dweek_soft_lb("All", "D", "Weekend", 2).
staff_dweek_soft_lb("All", "D", "Holiday", 2).
staff_dweek_soft_ub("All", "D", "Weekday", 4).
staff_dweek_soft_ub("All", "D", "Weekend", 3).
staff_dweek_soft_ub("All", "D", "Holiday", 3).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 3).
staff_dweek_soft_lb("All", "LD", "Weekend", 3).
staff_dweek_soft_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
staff_dweek_hard_ub("All", "LD", "Weekday", 4).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 3).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 3).
staff_dweek_hard_lb("All", "SE", "Weekend", 3).
staff_dweek_hard_lb("All", "SN", "Weekend", 3).
staff_dweek_hard_lb("All", "SE", "Holiday", 3).
staff_dweek_hard_lb("All", "SN", "Holiday", 3).
staff_dweek_soft_lb("All", "SE", "Weekday", 3).
staff_dweek_soft_lb("All", "SN", "Weekday", 3).
staff_dweek_soft_lb("All", "SE", "Weekend", 3).
staff_dweek_soft_lb("All", "SN", "Weekend", 3).
staff_dweek_soft_lb("All", "SE", "Holiday", 3).
staff_dweek_soft_lb("All", "SN", "Holiday", 3).
staff_dweek_soft_ub("All", "SE", "Weekday", 4).
staff_dweek_soft_ub("All", "SN", "Weekday", 4).
staff_dweek_soft_ub("All", "SE", "Weekend", 4).
staff_dweek_soft_ub("All", "SN", "Weekend", 4).
staff_dweek_soft_ub("All", "SE", "Holiday", 4).
staff_dweek_soft_ub("All", "SN", "Holiday", 4).
shifts("E").
shifts("E", "E").
staff_dweek_hard_lb("All", "E", "Weekday", 0).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("All", "N", "Weekday", 0).
staff_dweek_hard_lb("All", "E", "Weekend", 0).
staff_dweek_hard_lb("All", "N", "Weekend", 0).
staff_dweek_hard_lb("All", "E", "Holiday", 0).
staff_dweek_hard_lb("All", "N", "Holiday", 0).
staff_dweek_soft_lb("All", "E", "Weekday", 0).
staff_dweek_soft_lb("All", "N", "Weekday", 0).
staff_dweek_soft_lb("All", "E", "Weekend", 0).
staff_dweek_soft_lb("All", "N", "Weekend", 0).
staff_dweek_soft_lb("All", "E", "Holiday", 0).
staff_dweek_soft_lb("All", "N", "Holiday", 0).
staff_dweek_soft_ub("All", "E", "Weekday", 0).
staff_dweek_soft_ub("All", "N", "Weekday", 0).
staff_dweek_soft_ub("All", "E", "Weekend", 0).
staff_dweek_soft_ub("All", "N", "Weekend", 0).
staff_dweek_soft_ub("All", "E", "Holiday", 0).
staff_dweek_soft_ub("All", "N", "Holiday", 0).
staff_dweek_hard_ub("All", "E", "Weekday", 0).
staff_dweek_hard_ub("All", "N", "Weekday", 0).
staff_dweek_hard_ub("All", "E", "Weekend", 0).
staff_dweek_hard_ub("All", "N", "Weekend", 0).
staff_dweek_hard_ub("All", "E", "Holiday", 0).
staff_dweek_hard_ub("All", "N", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 0).
staff_dweek_hard_lb("All", "LM", "Weekend", 0).
staff_dweek_hard_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_lb("All", "LM", "Weekday", 0).
staff_dweek_soft_lb("All", "LM", "Weekend", 0).
staff_dweek_soft_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_ub("All", "LM", "Weekday", 0).
staff_dweek_soft_ub("All", "LM", "Weekend", 0).
staff_dweek_soft_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_ub("All", "LM", "Weekday", 0).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_lb("Leaders", "LD", "Weekday", 1).
staff_dweek_hard_lb("Leaders", "SE", "Weekday", 1).
staff_dweek_hard_lb("Leaders", "SN", "Weekday", 1).
staff_dweek_hard_lb("Leaders", "LD", "Weekend", 1).
staff_dweek_hard_lb("Leaders", "SE", "Weekend", 1).
staff_dweek_hard_lb("Leaders", "SN", "Weekend", 1).
staff_dweek_hard_lb("Leaders", "LD", "Holiday", 1).
staff_dweek_hard_lb("Leaders", "SE", "Holiday", 1).
staff_dweek_hard_lb("Leaders", "SN", "Holiday", 1).
staff_dweek_soft_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_lb("代行", "D", "Weekday", 0).
staff_dweek_soft_lb("代行", "D", "Weekend", 1).
staff_dweek_soft_lb("代行", "D", "Holiday", 1).
staff_dweek_soft_ub("代行", "D", "Weekday", 0).
staff_dweek_soft_ub("代行", "D", "Weekend", 1).
staff_dweek_soft_ub("代行", "D", "Holiday", 1).
staff_dweek_hard_lb("メンバー", "LD", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "SE", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "SN", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "LD", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "SE", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "SN", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "LD", "Holiday", 1).
staff_dweek_hard_lb("メンバー", "SE", "Holiday", 1).
staff_dweek_hard_lb("メンバー", "SN", "Holiday", 1).
staff_dweek_soft_ub("メンバー", "LD", "Weekday", 2).
staff_dweek_soft_ub("メンバー", "SE", "Weekday", 2).
staff_dweek_soft_ub("メンバー", "SN", "Weekday", 2).
staff_dweek_soft_ub("メンバー", "LD", "Weekend", 2).
staff_dweek_soft_ub("メンバー", "SE", "Weekend", 2).
staff_dweek_soft_ub("メンバー", "SN", "Weekend", 2).
staff_dweek_soft_ub("メンバー", "LD", "Holiday", 2).
staff_dweek_soft_ub("メンバー", "SE", "Holiday", 2).
staff_dweek_soft_ub("メンバー", "SN", "Holiday", 2).
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Holiday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 0).
staff_dweek_soft_lb("Newcomers", "SE", "Weekday", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Weekday", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Weekend", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Weekend", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Holiday", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Holiday", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Weekday", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Weekday", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Weekend", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Weekend", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Holiday", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Holiday", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Weekday", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Weekday", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Weekend", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Weekend", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Holiday", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Holiday", 1).
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "E", "Staffs").
vertical_constraint_type("All", "N", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("Leaders", "LD", "Staffs").
vertical_constraint_type("Leaders", "SE", "Staffs").
vertical_constraint_type("Leaders", "SN", "Staffs").
vertical_constraint_type("代行", "D", "Staffs").
vertical_constraint_type("メンバー", "LD", "Staffs").
vertical_constraint_type("メンバー", "SE", "Staffs").
vertical_constraint_type("メンバー", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_soft_lb(2, "LD", 1).
shifts_soft_lb(2, "SE", 1).
shifts_soft_lb(2, "SN", 1).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "LM", 0).
shifts("EM").
shifts("EM", "EM").
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 5).
shifts_soft_lb(3, "SE", 5).
shifts_soft_lb(3, "SN", 5).
shifts_soft_ub(3, "LD", 6).
shifts_soft_ub(3, "SE", 6).
shifts_soft_ub(3, "SN", 6).
shifts_hard_lb(3, "LM", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "LM", 0).
shifts_hard_ub(3, "LM", 0).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_soft_lb(4, "LD", 5).
shifts_soft_lb(4, "SE", 5).
shifts_soft_lb(4, "SN", 5).
shifts_soft_ub(4, "LD", 6).
shifts_soft_ub(4, "SE", 6).
shifts_soft_ub(4, "SN", 6).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "LM", 0).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 2).
shifts_soft_ub(5, "SE", 2).
shifts_soft_ub(5, "SN", 2).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 0).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 2).
shifts_soft_lb(6, "SE", 2).
shifts_soft_lb(6, "SN", 2).
shifts_soft_ub(6, "LD", 2).
shifts_soft_ub(6, "SE", 2).
shifts_soft_ub(6, "SN", 2).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "LM", 0).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 5).
shifts_soft_lb(7, "SE", 5).
shifts_soft_lb(7, "SN", 5).
shifts_soft_ub(7, "LD", 6).
shifts_soft_ub(7, "SE", 6).
shifts_soft_ub(7, "SN", 6).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "LM", 0).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 5).
shifts_soft_lb(8, "SE", 5).
shifts_soft_lb(8, "SN", 5).
shifts_soft_ub(8, "LD", 6).
shifts_soft_ub(8, "SE", 6).
shifts_soft_ub(8, "SN", 6).
shifts_hard_lb(8, "LM", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "LM", 0).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 5).
shifts_soft_lb(9, "SE", 5).
shifts_soft_lb(9, "SN", 5).
shifts_soft_ub(9, "LD", 6).
shifts_soft_ub(9, "SE", 6).
shifts_soft_ub(9, "SN", 6).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "LM", 0).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 5).
shifts_soft_lb(10, "SE", 5).
shifts_soft_lb(10, "SN", 5).
shifts_soft_ub(10, "LD", 6).
shifts_soft_ub(10, "SE", 6).
shifts_soft_ub(10, "SN", 6).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "LM", 0).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 4).
shifts_soft_ub(11, "SE", 4).
shifts_soft_ub(11, "SN", 4).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "LM", 0).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "LD", 5).
shifts_soft_lb(13, "SE", 5).
shifts_soft_lb(13, "SN", 5).
shifts_soft_ub(13, "LD", 6).
shifts_soft_ub(13, "SE", 6).
shifts_soft_ub(13, "SN", 6).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "LM", 0).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 3).
shifts_soft_lb(14, "SE", 3).
shifts_soft_lb(14, "SN", 3).
shifts_soft_ub(14, "LD", 4).
shifts_soft_ub(14, "SE", 4).
shifts_soft_ub(14, "SN", 4).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "LM", 0).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 5).
shifts_soft_lb(15, "SE", 5).
shifts_soft_lb(15, "SN", 5).
shifts_soft_ub(15, "LD", 6).
shifts_soft_ub(15, "SE", 6).
shifts_soft_ub(15, "SN", 6).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 2).
shifts_soft_lb(16, "SE", 2).
shifts_soft_lb(16, "SN", 2).
shifts_soft_ub(16, "LD", 3).
shifts_soft_ub(16, "SE", 3).
shifts_soft_ub(16, "SN", 3).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "LM", 0).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 5).
shifts_soft_lb(17, "SE", 5).
shifts_soft_lb(17, "SN", 5).
shifts_soft_ub(17, "LD", 6).
shifts_soft_ub(17, "SE", 6).
shifts_soft_ub(17, "SN", 6).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "LM", 0).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 5).
shifts_soft_lb(18, "SE", 5).
shifts_soft_lb(18, "SN", 5).
shifts_soft_ub(18, "LD", 6).
shifts_soft_ub(18, "SE", 6).
shifts_soft_ub(18, "SN", 6).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "LM", 0).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 5).
shifts_soft_lb(19, "SE", 5).
shifts_soft_lb(19, "SN", 5).
shifts_soft_ub(19, "LD", 6).
shifts_soft_ub(19, "SE", 6).
shifts_soft_ub(19, "SN", 6).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "LM", 0).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 6).
shifts_soft_ub(20, "SE", 6).
shifts_soft_ub(20, "SN", 6).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "LM", 0).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 5).
shifts_soft_lb(21, "SE", 5).
shifts_soft_lb(21, "SN", 5).
shifts_soft_ub(21, "LD", 6).
shifts_soft_ub(21, "SE", 6).
shifts_soft_ub(21, "SN", 6).
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "LM", 0).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 6).
shifts_soft_ub(22, "SE", 6).
shifts_soft_ub(22, "SN", 6).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "LM", 0).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 6).
shifts_soft_ub(23, "SE", 6).
shifts_soft_ub(23, "SN", 6).
shifts_hard_lb(23, "LM", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "LM", 0).
shifts_hard_ub(23, "LM", 0).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
