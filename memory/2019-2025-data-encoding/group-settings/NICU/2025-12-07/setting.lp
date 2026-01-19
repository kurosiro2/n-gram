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
staff_dweek_hard_lb("All", "D", "Mo", 2).
staff_dweek_hard_lb("All", "D", "Tu", 2).
staff_dweek_hard_lb("All", "D", "We", 2).
staff_dweek_hard_lb("All", "D", "Th", 2).
staff_dweek_hard_lb("All", "D", "Fr", 2).
staff_dweek_hard_lb("All", "D", "Sa", 2).
staff_dweek_hard_lb("All", "D", "Su", 2).
staff_dweek_hard_lb("All", "D", "Ho", 2).
staff_dweek_soft_lb("All", "D", "Mo", 2).
staff_dweek_soft_lb("All", "D", "Tu", 2).
staff_dweek_soft_lb("All", "D", "We", 2).
staff_dweek_soft_lb("All", "D", "Th", 2).
staff_dweek_soft_lb("All", "D", "Fr", 2).
staff_dweek_soft_lb("All", "D", "Sa", 2).
staff_dweek_soft_lb("All", "D", "Su", 2).
staff_dweek_soft_lb("All", "D", "Ho", 2).
staff_dweek_soft_ub("All", "D", "Mo", 3).
staff_dweek_soft_ub("All", "D", "Tu", 3).
staff_dweek_soft_ub("All", "D", "We", 3).
staff_dweek_soft_ub("All", "D", "Th", 3).
staff_dweek_soft_ub("All", "D", "Fr", 3).
staff_dweek_soft_ub("All", "D", "Sa", 3).
staff_dweek_soft_ub("All", "D", "Su", 3).
staff_dweek_soft_ub("All", "D", "Ho", 3).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Mo", 3).
staff_dweek_hard_lb("All", "LD", "Tu", 3).
staff_dweek_hard_lb("All", "LD", "We", 3).
staff_dweek_hard_lb("All", "LD", "Th", 3).
staff_dweek_hard_lb("All", "LD", "Fr", 3).
staff_dweek_hard_lb("All", "LD", "Sa", 3).
staff_dweek_hard_lb("All", "LD", "Su", 3).
staff_dweek_hard_lb("All", "LD", "Ho", 3).
staff_dweek_soft_lb("All", "LD", "Mo", 3).
staff_dweek_soft_lb("All", "LD", "Tu", 3).
staff_dweek_soft_lb("All", "LD", "We", 3).
staff_dweek_soft_lb("All", "LD", "Th", 3).
staff_dweek_soft_lb("All", "LD", "Fr", 3).
staff_dweek_soft_lb("All", "LD", "Sa", 3).
staff_dweek_soft_lb("All", "LD", "Su", 3).
staff_dweek_soft_lb("All", "LD", "Ho", 3).
staff_dweek_soft_ub("All", "LD", "Mo", 4).
staff_dweek_soft_ub("All", "LD", "Tu", 4).
staff_dweek_soft_ub("All", "LD", "We", 4).
staff_dweek_soft_ub("All", "LD", "Th", 4).
staff_dweek_soft_ub("All", "LD", "Fr", 4).
staff_dweek_soft_ub("All", "LD", "Sa", 4).
staff_dweek_soft_ub("All", "LD", "Su", 4).
staff_dweek_soft_ub("All", "LD", "Ho", 4).
staff_dweek_hard_ub("All", "LD", "Mo", 4).
staff_dweek_hard_ub("All", "LD", "Tu", 4).
staff_dweek_hard_ub("All", "LD", "We", 4).
staff_dweek_hard_ub("All", "LD", "Th", 4).
staff_dweek_hard_ub("All", "LD", "Fr", 4).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Mo", 3).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Mo", 3).
staff_dweek_hard_lb("All", "SE", "Tu", 3).
staff_dweek_hard_lb("All", "SN", "Tu", 3).
staff_dweek_hard_lb("All", "SE", "We", 3).
staff_dweek_hard_lb("All", "SN", "We", 3).
staff_dweek_hard_lb("All", "SE", "Th", 3).
staff_dweek_hard_lb("All", "SN", "Th", 3).
staff_dweek_hard_lb("All", "SE", "Fr", 3).
staff_dweek_hard_lb("All", "SN", "Fr", 3).
staff_dweek_hard_lb("All", "SE", "Sa", 3).
staff_dweek_hard_lb("All", "SN", "Sa", 3).
staff_dweek_hard_lb("All", "SE", "Su", 3).
staff_dweek_hard_lb("All", "SN", "Su", 3).
staff_dweek_hard_lb("All", "SE", "Ho", 3).
staff_dweek_hard_lb("All", "SN", "Ho", 3).
staff_dweek_soft_lb("All", "SE", "Mo", 3).
staff_dweek_soft_lb("All", "SN", "Mo", 3).
staff_dweek_soft_lb("All", "SE", "Tu", 3).
staff_dweek_soft_lb("All", "SN", "Tu", 3).
staff_dweek_soft_lb("All", "SE", "We", 3).
staff_dweek_soft_lb("All", "SN", "We", 3).
staff_dweek_soft_lb("All", "SE", "Th", 3).
staff_dweek_soft_lb("All", "SN", "Th", 3).
staff_dweek_soft_lb("All", "SE", "Fr", 3).
staff_dweek_soft_lb("All", "SN", "Fr", 3).
staff_dweek_soft_lb("All", "SE", "Sa", 3).
staff_dweek_soft_lb("All", "SN", "Sa", 3).
staff_dweek_soft_lb("All", "SE", "Su", 3).
staff_dweek_soft_lb("All", "SN", "Su", 3).
staff_dweek_soft_lb("All", "SE", "Ho", 3).
staff_dweek_soft_lb("All", "SN", "Ho", 3).
staff_dweek_soft_ub("All", "SE", "Mo", 4).
staff_dweek_soft_ub("All", "SN", "Mo", 4).
staff_dweek_soft_ub("All", "SE", "Tu", 4).
staff_dweek_soft_ub("All", "SN", "Tu", 4).
staff_dweek_soft_ub("All", "SE", "We", 4).
staff_dweek_soft_ub("All", "SN", "We", 4).
staff_dweek_soft_ub("All", "SE", "Th", 4).
staff_dweek_soft_ub("All", "SN", "Th", 4).
staff_dweek_soft_ub("All", "SE", "Fr", 4).
staff_dweek_soft_ub("All", "SN", "Fr", 4).
staff_dweek_soft_ub("All", "SE", "Sa", 4).
staff_dweek_soft_ub("All", "SN", "Sa", 4).
staff_dweek_soft_ub("All", "SE", "Su", 4).
staff_dweek_soft_ub("All", "SN", "Su", 4).
staff_dweek_soft_ub("All", "SE", "Ho", 4).
staff_dweek_soft_ub("All", "SN", "Ho", 4).
shifts("E").
shifts("E", "E").
staff_dweek_hard_lb("All", "E", "Mo", 0).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("All", "N", "Mo", 0).
staff_dweek_hard_lb("All", "E", "Tu", 0).
staff_dweek_hard_lb("All", "N", "Tu", 0).
staff_dweek_hard_lb("All", "E", "We", 0).
staff_dweek_hard_lb("All", "N", "We", 0).
staff_dweek_hard_lb("All", "E", "Th", 0).
staff_dweek_hard_lb("All", "N", "Th", 0).
staff_dweek_hard_lb("All", "E", "Fr", 0).
staff_dweek_hard_lb("All", "N", "Fr", 0).
staff_dweek_hard_lb("All", "E", "Sa", 0).
staff_dweek_hard_lb("All", "N", "Sa", 0).
staff_dweek_hard_lb("All", "E", "Su", 0).
staff_dweek_hard_lb("All", "N", "Su", 0).
staff_dweek_hard_lb("All", "E", "Ho", 0).
staff_dweek_hard_lb("All", "N", "Ho", 0).
staff_dweek_soft_lb("All", "E", "Mo", 0).
staff_dweek_soft_lb("All", "N", "Mo", 0).
staff_dweek_soft_lb("All", "E", "Tu", 0).
staff_dweek_soft_lb("All", "N", "Tu", 0).
staff_dweek_soft_lb("All", "E", "We", 0).
staff_dweek_soft_lb("All", "N", "We", 0).
staff_dweek_soft_lb("All", "E", "Th", 0).
staff_dweek_soft_lb("All", "N", "Th", 0).
staff_dweek_soft_lb("All", "E", "Fr", 0).
staff_dweek_soft_lb("All", "N", "Fr", 0).
staff_dweek_soft_lb("All", "E", "Sa", 0).
staff_dweek_soft_lb("All", "N", "Sa", 0).
staff_dweek_soft_lb("All", "E", "Su", 0).
staff_dweek_soft_lb("All", "N", "Su", 0).
staff_dweek_soft_lb("All", "E", "Ho", 0).
staff_dweek_soft_lb("All", "N", "Ho", 0).
staff_dweek_soft_ub("All", "E", "Mo", 0).
staff_dweek_soft_ub("All", "N", "Mo", 0).
staff_dweek_soft_ub("All", "E", "Tu", 0).
staff_dweek_soft_ub("All", "N", "Tu", 0).
staff_dweek_soft_ub("All", "E", "We", 0).
staff_dweek_soft_ub("All", "N", "We", 0).
staff_dweek_soft_ub("All", "E", "Th", 0).
staff_dweek_soft_ub("All", "N", "Th", 0).
staff_dweek_soft_ub("All", "E", "Fr", 0).
staff_dweek_soft_ub("All", "N", "Fr", 0).
staff_dweek_soft_ub("All", "E", "Sa", 0).
staff_dweek_soft_ub("All", "N", "Sa", 0).
staff_dweek_soft_ub("All", "E", "Su", 0).
staff_dweek_soft_ub("All", "N", "Su", 0).
staff_dweek_soft_ub("All", "E", "Ho", 0).
staff_dweek_soft_ub("All", "N", "Ho", 0).
staff_dweek_hard_ub("All", "E", "Mo", 0).
staff_dweek_hard_ub("All", "N", "Mo", 0).
staff_dweek_hard_ub("All", "E", "Tu", 0).
staff_dweek_hard_ub("All", "N", "Tu", 0).
staff_dweek_hard_ub("All", "E", "We", 0).
staff_dweek_hard_ub("All", "N", "We", 0).
staff_dweek_hard_ub("All", "E", "Th", 0).
staff_dweek_hard_ub("All", "N", "Th", 0).
staff_dweek_hard_ub("All", "E", "Fr", 0).
staff_dweek_hard_ub("All", "N", "Fr", 0).
staff_dweek_hard_ub("All", "E", "Sa", 0).
staff_dweek_hard_ub("All", "N", "Sa", 0).
staff_dweek_hard_ub("All", "E", "Su", 0).
staff_dweek_hard_ub("All", "N", "Su", 0).
staff_dweek_hard_ub("All", "E", "Ho", 0).
staff_dweek_hard_ub("All", "N", "Ho", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Mo", 0).
staff_dweek_hard_lb("All", "LM", "Tu", 0).
staff_dweek_hard_lb("All", "LM", "We", 0).
staff_dweek_hard_lb("All", "LM", "Th", 0).
staff_dweek_hard_lb("All", "LM", "Fr", 0).
staff_dweek_hard_lb("All", "LM", "Sa", 0).
staff_dweek_hard_lb("All", "LM", "Su", 0).
staff_dweek_hard_lb("All", "LM", "Ho", 0).
staff_dweek_soft_lb("All", "LM", "Mo", 0).
staff_dweek_soft_lb("All", "LM", "Tu", 0).
staff_dweek_soft_lb("All", "LM", "We", 0).
staff_dweek_soft_lb("All", "LM", "Th", 0).
staff_dweek_soft_lb("All", "LM", "Fr", 0).
staff_dweek_soft_lb("All", "LM", "Sa", 0).
staff_dweek_soft_lb("All", "LM", "Su", 0).
staff_dweek_soft_lb("All", "LM", "Ho", 0).
staff_dweek_soft_ub("All", "LM", "Mo", 0).
staff_dweek_soft_ub("All", "LM", "Tu", 0).
staff_dweek_soft_ub("All", "LM", "We", 0).
staff_dweek_soft_ub("All", "LM", "Th", 0).
staff_dweek_soft_ub("All", "LM", "Fr", 0).
staff_dweek_soft_ub("All", "LM", "Sa", 0).
staff_dweek_soft_ub("All", "LM", "Su", 0).
staff_dweek_soft_ub("All", "LM", "Ho", 0).
staff_dweek_hard_ub("All", "LM", "Mo", 0).
staff_dweek_hard_ub("All", "LM", "Tu", 0).
staff_dweek_hard_ub("All", "LM", "We", 0).
staff_dweek_hard_ub("All", "LM", "Th", 0).
staff_dweek_hard_ub("All", "LM", "Fr", 0).
staff_dweek_hard_ub("All", "LM", "Sa", 0).
staff_dweek_hard_ub("All", "LM", "Su", 0).
staff_dweek_hard_ub("All", "LM", "Ho", 0).
staff_dweek_hard_lb("Leaders", "LD", "Mo", 1).
staff_dweek_hard_lb("Leaders", "SE", "Mo", 1).
staff_dweek_hard_lb("Leaders", "SN", "Mo", 1).
staff_dweek_hard_lb("Leaders", "LD", "Tu", 1).
staff_dweek_hard_lb("Leaders", "SE", "Tu", 1).
staff_dweek_hard_lb("Leaders", "SN", "Tu", 1).
staff_dweek_hard_lb("Leaders", "LD", "We", 1).
staff_dweek_hard_lb("Leaders", "SE", "We", 1).
staff_dweek_hard_lb("Leaders", "SN", "We", 1).
staff_dweek_hard_lb("Leaders", "LD", "Th", 1).
staff_dweek_hard_lb("Leaders", "SE", "Th", 1).
staff_dweek_hard_lb("Leaders", "SN", "Th", 1).
staff_dweek_hard_lb("Leaders", "LD", "Fr", 1).
staff_dweek_hard_lb("Leaders", "SE", "Fr", 1).
staff_dweek_hard_lb("Leaders", "SN", "Fr", 1).
staff_dweek_hard_lb("Leaders", "LD", "Sa", 1).
staff_dweek_hard_lb("Leaders", "SE", "Sa", 1).
staff_dweek_hard_lb("Leaders", "SN", "Sa", 1).
staff_dweek_hard_lb("Leaders", "LD", "Su", 1).
staff_dweek_hard_lb("Leaders", "SE", "Su", 1).
staff_dweek_hard_lb("Leaders", "SN", "Su", 1).
staff_dweek_hard_lb("Leaders", "LD", "Ho", 1).
staff_dweek_hard_lb("Leaders", "SE", "Ho", 1).
staff_dweek_hard_lb("Leaders", "SN", "Ho", 1).
staff_dweek_soft_lb("Leaders", "LD", "Mo", 2).
staff_dweek_soft_lb("Leaders", "SE", "Mo", 2).
staff_dweek_soft_lb("Leaders", "SN", "Mo", 2).
staff_dweek_soft_lb("Leaders", "LD", "Tu", 2).
staff_dweek_soft_lb("Leaders", "SE", "Tu", 2).
staff_dweek_soft_lb("Leaders", "SN", "Tu", 2).
staff_dweek_soft_lb("Leaders", "LD", "We", 2).
staff_dweek_soft_lb("Leaders", "SE", "We", 2).
staff_dweek_soft_lb("Leaders", "SN", "We", 2).
staff_dweek_soft_lb("Leaders", "LD", "Th", 2).
staff_dweek_soft_lb("Leaders", "SE", "Th", 2).
staff_dweek_soft_lb("Leaders", "SN", "Th", 2).
staff_dweek_soft_lb("Leaders", "LD", "Fr", 2).
staff_dweek_soft_lb("Leaders", "SE", "Fr", 2).
staff_dweek_soft_lb("Leaders", "SN", "Fr", 2).
staff_dweek_soft_lb("Leaders", "LD", "Sa", 2).
staff_dweek_soft_lb("Leaders", "SE", "Sa", 2).
staff_dweek_soft_lb("Leaders", "SN", "Sa", 2).
staff_dweek_soft_lb("Leaders", "LD", "Su", 2).
staff_dweek_soft_lb("Leaders", "SE", "Su", 2).
staff_dweek_soft_lb("Leaders", "SN", "Su", 2).
staff_dweek_soft_lb("Leaders", "LD", "Ho", 2).
staff_dweek_soft_lb("Leaders", "SE", "Ho", 2).
staff_dweek_soft_lb("Leaders", "SN", "Ho", 2).
staff_dweek_soft_ub("Leaders", "LD", "Mo", 2).
staff_dweek_soft_ub("Leaders", "SE", "Mo", 2).
staff_dweek_soft_ub("Leaders", "SN", "Mo", 2).
staff_dweek_soft_ub("Leaders", "LD", "Tu", 2).
staff_dweek_soft_ub("Leaders", "SE", "Tu", 2).
staff_dweek_soft_ub("Leaders", "SN", "Tu", 2).
staff_dweek_soft_ub("Leaders", "LD", "We", 2).
staff_dweek_soft_ub("Leaders", "SE", "We", 2).
staff_dweek_soft_ub("Leaders", "SN", "We", 2).
staff_dweek_soft_ub("Leaders", "LD", "Th", 2).
staff_dweek_soft_ub("Leaders", "SE", "Th", 2).
staff_dweek_soft_ub("Leaders", "SN", "Th", 2).
staff_dweek_soft_ub("Leaders", "LD", "Fr", 2).
staff_dweek_soft_ub("Leaders", "SE", "Fr", 2).
staff_dweek_soft_ub("Leaders", "SN", "Fr", 2).
staff_dweek_soft_ub("Leaders", "LD", "Sa", 2).
staff_dweek_soft_ub("Leaders", "SE", "Sa", 2).
staff_dweek_soft_ub("Leaders", "SN", "Sa", 2).
staff_dweek_soft_ub("Leaders", "LD", "Su", 2).
staff_dweek_soft_ub("Leaders", "SE", "Su", 2).
staff_dweek_soft_ub("Leaders", "SN", "Su", 2).
staff_dweek_soft_ub("Leaders", "LD", "Ho", 2).
staff_dweek_soft_ub("Leaders", "SE", "Ho", 2).
staff_dweek_soft_ub("Leaders", "SN", "Ho", 2).
staff_dweek_soft_lb("代行", "D", "Mo", 0).
staff_dweek_soft_lb("代行", "D", "Tu", 0).
staff_dweek_soft_lb("代行", "D", "We", 0).
staff_dweek_soft_lb("代行", "D", "Th", 0).
staff_dweek_soft_lb("代行", "D", "Fr", 0).
staff_dweek_soft_lb("代行", "D", "Sa", 1).
staff_dweek_soft_lb("代行", "D", "Su", 1).
staff_dweek_soft_lb("代行", "D", "Ho", 1).
staff_dweek_soft_ub("代行", "D", "Mo", 0).
staff_dweek_soft_ub("代行", "D", "Tu", 0).
staff_dweek_soft_ub("代行", "D", "We", 0).
staff_dweek_soft_ub("代行", "D", "Th", 0).
staff_dweek_soft_ub("代行", "D", "Fr", 0).
staff_dweek_soft_ub("代行", "D", "Sa", 1).
staff_dweek_soft_ub("代行", "D", "Su", 1).
staff_dweek_soft_ub("代行", "D", "Ho", 1).
staff_dweek_hard_lb("メンバー", "LD", "Mo", 1).
staff_dweek_hard_lb("メンバー", "SE", "Mo", 1).
staff_dweek_hard_lb("メンバー", "SN", "Mo", 1).
staff_dweek_hard_lb("メンバー", "LD", "Tu", 1).
staff_dweek_hard_lb("メンバー", "SE", "Tu", 1).
staff_dweek_hard_lb("メンバー", "SN", "Tu", 1).
staff_dweek_hard_lb("メンバー", "LD", "We", 1).
staff_dweek_hard_lb("メンバー", "SE", "We", 1).
staff_dweek_hard_lb("メンバー", "SN", "We", 1).
staff_dweek_hard_lb("メンバー", "LD", "Th", 1).
staff_dweek_hard_lb("メンバー", "SE", "Th", 1).
staff_dweek_hard_lb("メンバー", "SN", "Th", 1).
staff_dweek_hard_lb("メンバー", "LD", "Fr", 1).
staff_dweek_hard_lb("メンバー", "SE", "Fr", 1).
staff_dweek_hard_lb("メンバー", "SN", "Fr", 1).
staff_dweek_hard_lb("メンバー", "LD", "Sa", 1).
staff_dweek_hard_lb("メンバー", "SE", "Sa", 1).
staff_dweek_hard_lb("メンバー", "SN", "Sa", 1).
staff_dweek_hard_lb("メンバー", "LD", "Su", 1).
staff_dweek_hard_lb("メンバー", "SE", "Su", 1).
staff_dweek_hard_lb("メンバー", "SN", "Su", 1).
staff_dweek_hard_lb("メンバー", "LD", "Ho", 1).
staff_dweek_hard_lb("メンバー", "SE", "Ho", 1).
staff_dweek_hard_lb("メンバー", "SN", "Ho", 1).
staff_dweek_soft_lb("メンバー", "LD", "Mo", 2).
staff_dweek_soft_lb("メンバー", "SE", "Mo", 2).
staff_dweek_soft_lb("メンバー", "SN", "Mo", 2).
staff_dweek_soft_lb("メンバー", "LD", "Tu", 2).
staff_dweek_soft_lb("メンバー", "SE", "Tu", 2).
staff_dweek_soft_lb("メンバー", "SN", "Tu", 2).
staff_dweek_soft_lb("メンバー", "LD", "We", 2).
staff_dweek_soft_lb("メンバー", "SE", "We", 2).
staff_dweek_soft_lb("メンバー", "SN", "We", 2).
staff_dweek_soft_lb("メンバー", "LD", "Th", 2).
staff_dweek_soft_lb("メンバー", "SE", "Th", 2).
staff_dweek_soft_lb("メンバー", "SN", "Th", 2).
staff_dweek_soft_lb("メンバー", "LD", "Fr", 2).
staff_dweek_soft_lb("メンバー", "SE", "Fr", 2).
staff_dweek_soft_lb("メンバー", "SN", "Fr", 2).
staff_dweek_soft_lb("メンバー", "LD", "Sa", 2).
staff_dweek_soft_lb("メンバー", "SE", "Sa", 2).
staff_dweek_soft_lb("メンバー", "SN", "Sa", 2).
staff_dweek_soft_lb("メンバー", "LD", "Su", 2).
staff_dweek_soft_lb("メンバー", "SE", "Su", 2).
staff_dweek_soft_lb("メンバー", "SN", "Su", 2).
staff_dweek_soft_lb("メンバー", "LD", "Ho", 2).
staff_dweek_soft_lb("メンバー", "SE", "Ho", 2).
staff_dweek_soft_lb("メンバー", "SN", "Ho", 2).
staff_dweek_soft_ub("メンバー", "LD", "Mo", 2).
staff_dweek_soft_ub("メンバー", "SE", "Mo", 2).
staff_dweek_soft_ub("メンバー", "SN", "Mo", 2).
staff_dweek_soft_ub("メンバー", "LD", "Tu", 2).
staff_dweek_soft_ub("メンバー", "SE", "Tu", 2).
staff_dweek_soft_ub("メンバー", "SN", "Tu", 2).
staff_dweek_soft_ub("メンバー", "LD", "We", 2).
staff_dweek_soft_ub("メンバー", "SE", "We", 2).
staff_dweek_soft_ub("メンバー", "SN", "We", 2).
staff_dweek_soft_ub("メンバー", "LD", "Th", 2).
staff_dweek_soft_ub("メンバー", "SE", "Th", 2).
staff_dweek_soft_ub("メンバー", "SN", "Th", 2).
staff_dweek_soft_ub("メンバー", "LD", "Fr", 2).
staff_dweek_soft_ub("メンバー", "SE", "Fr", 2).
staff_dweek_soft_ub("メンバー", "SN", "Fr", 2).
staff_dweek_soft_ub("メンバー", "LD", "Sa", 2).
staff_dweek_soft_ub("メンバー", "SE", "Sa", 2).
staff_dweek_soft_ub("メンバー", "SN", "Sa", 2).
staff_dweek_soft_ub("メンバー", "LD", "Su", 2).
staff_dweek_soft_ub("メンバー", "SE", "Su", 2).
staff_dweek_soft_ub("メンバー", "SN", "Su", 2).
staff_dweek_soft_ub("メンバー", "LD", "Ho", 2).
staff_dweek_soft_ub("メンバー", "SE", "Ho", 2).
staff_dweek_soft_ub("メンバー", "SN", "Ho", 2).
staff_dweek_hard_lb("Newcomers", "SE", "Mo", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Mo", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Tu", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Tu", 0).
staff_dweek_hard_lb("Newcomers", "SE", "We", 0).
staff_dweek_hard_lb("Newcomers", "SN", "We", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Th", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Th", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Fr", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Fr", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Sa", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Sa", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Su", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Su", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Ho", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Ho", 0).
staff_dweek_soft_lb("Newcomers", "SE", "Mo", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Mo", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Tu", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Tu", 1).
staff_dweek_soft_lb("Newcomers", "SE", "We", 1).
staff_dweek_soft_lb("Newcomers", "SN", "We", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Th", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Th", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Fr", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Fr", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Sa", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Sa", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Su", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Su", 1).
staff_dweek_soft_lb("Newcomers", "SE", "Ho", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Ho", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Mo", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Mo", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Tu", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Tu", 1).
staff_dweek_soft_ub("Newcomers", "SE", "We", 1).
staff_dweek_soft_ub("Newcomers", "SN", "We", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Th", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Th", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Fr", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Fr", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Sa", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Sa", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Su", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Su", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Ho", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Ho", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Mo", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Mo", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Tu", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Tu", 1).
staff_dweek_hard_ub("Newcomers", "SE", "We", 1).
staff_dweek_hard_ub("Newcomers", "SN", "We", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Th", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Th", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Fr", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Fr", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Sa", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Sa", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Su", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Su", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Ho", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Ho", 1).
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
shifts_hard_lb(2, "LD", 1).
shifts_hard_lb(2, "SE", 1).
shifts_hard_lb(2, "SN", 1).
shifts_soft_lb(2, "LD", 1).
shifts_soft_lb(2, "SE", 1).
shifts_soft_lb(2, "SN", 1).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
shifts_hard_ub(2, "LD", 2).
shifts_hard_ub(2, "SE", 2).
shifts_hard_ub(2, "SN", 2).
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
shifts_hard_lb(3, "LD", 1).
shifts_hard_lb(3, "SE", 1).
shifts_hard_lb(3, "SN", 1).
shifts_soft_lb(3, "LD", 5).
shifts_soft_lb(3, "SE", 5).
shifts_soft_lb(3, "SN", 5).
shifts_soft_ub(3, "LD", 6).
shifts_soft_ub(3, "SE", 6).
shifts_soft_ub(3, "SN", 6).
shifts_hard_ub(3, "LD", 6).
shifts_hard_ub(3, "SE", 6).
shifts_hard_ub(3, "SN", 6).
shifts_hard_lb(3, "LM", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "LM", 0).
shifts_hard_ub(3, "LM", 0).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_hard_lb(4, "LD", 1).
shifts_hard_lb(4, "SE", 1).
shifts_hard_lb(4, "SN", 1).
shifts_soft_lb(4, "LD", 5).
shifts_soft_lb(4, "SE", 5).
shifts_soft_lb(4, "SN", 5).
shifts_soft_ub(4, "LD", 6).
shifts_soft_ub(4, "SE", 6).
shifts_soft_ub(4, "SN", 6).
shifts_hard_ub(4, "LD", 6).
shifts_hard_ub(4, "SE", 6).
shifts_hard_ub(4, "SN", 6).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "LM", 0).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "LD", 1).
shifts_hard_lb(5, "SE", 1).
shifts_hard_lb(5, "SN", 1).
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 2).
shifts_soft_ub(5, "SE", 2).
shifts_soft_ub(5, "SN", 2).
shifts_hard_ub(5, "LD", 2).
shifts_hard_ub(5, "SE", 2).
shifts_hard_ub(5, "SN", 2).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 0).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_hard_lb(6, "LD", 1).
shifts_hard_lb(6, "SE", 1).
shifts_hard_lb(6, "SN", 1).
shifts_soft_lb(6, "LD", 2).
shifts_soft_lb(6, "SE", 2).
shifts_soft_lb(6, "SN", 2).
shifts_soft_ub(6, "LD", 2).
shifts_soft_ub(6, "SE", 2).
shifts_soft_ub(6, "SN", 2).
shifts_hard_ub(6, "LD", 2).
shifts_hard_ub(6, "SE", 2).
shifts_hard_ub(6, "SN", 2).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "LM", 0).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "LD", 1).
shifts_hard_lb(7, "SE", 1).
shifts_hard_lb(7, "SN", 1).
shifts_soft_lb(7, "LD", 5).
shifts_soft_lb(7, "SE", 5).
shifts_soft_lb(7, "SN", 5).
shifts_soft_ub(7, "LD", 6).
shifts_soft_ub(7, "SE", 6).
shifts_soft_ub(7, "SN", 6).
shifts_hard_ub(7, "LD", 6).
shifts_hard_ub(7, "SE", 6).
shifts_hard_ub(7, "SN", 6).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "LM", 0).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "LD", 1).
shifts_hard_lb(8, "SE", 1).
shifts_hard_lb(8, "SN", 1).
shifts_soft_lb(8, "LD", 5).
shifts_soft_lb(8, "SE", 5).
shifts_soft_lb(8, "SN", 5).
shifts_soft_ub(8, "LD", 6).
shifts_soft_ub(8, "SE", 6).
shifts_soft_ub(8, "SN", 6).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_hard_lb(8, "LM", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "LM", 0).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_hard_lb(9, "LD", 1).
shifts_hard_lb(9, "SE", 1).
shifts_hard_lb(9, "SN", 1).
shifts_soft_lb(9, "LD", 5).
shifts_soft_lb(9, "SE", 5).
shifts_soft_lb(9, "SN", 5).
shifts_soft_ub(9, "LD", 6).
shifts_soft_ub(9, "SE", 6).
shifts_soft_ub(9, "SN", 6).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "LM", 0).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_hard_lb(10, "LD", 1).
shifts_hard_lb(10, "SE", 1).
shifts_hard_lb(10, "SN", 1).
shifts_soft_lb(10, "LD", 5).
shifts_soft_lb(10, "SE", 5).
shifts_soft_lb(10, "SN", 5).
shifts_soft_ub(10, "LD", 6).
shifts_soft_ub(10, "SE", 6).
shifts_soft_ub(10, "SN", 6).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "LM", 0).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_hard_lb(11, "LD", 1).
shifts_hard_lb(11, "SE", 1).
shifts_hard_lb(11, "SN", 1).
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 4).
shifts_soft_ub(11, "SE", 4).
shifts_soft_ub(11, "SN", 4).
shifts_hard_ub(11, "LD", 4).
shifts_hard_ub(11, "SE", 4).
shifts_hard_ub(11, "SN", 4).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "LM", 0).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_hard_lb(12, "LD", 1).
shifts_hard_lb(12, "SE", 1).
shifts_hard_lb(12, "SN", 1).
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_hard_ub(12, "LD", 5).
shifts_hard_ub(12, "SE", 5).
shifts_hard_ub(12, "SN", 5).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_hard_lb(13, "LD", 1).
shifts_hard_lb(13, "SE", 1).
shifts_hard_lb(13, "SN", 1).
shifts_soft_lb(13, "LD", 5).
shifts_soft_lb(13, "SE", 5).
shifts_soft_lb(13, "SN", 5).
shifts_soft_ub(13, "LD", 6).
shifts_soft_ub(13, "SE", 6).
shifts_soft_ub(13, "SN", 6).
shifts_hard_ub(13, "LD", 6).
shifts_hard_ub(13, "SE", 6).
shifts_hard_ub(13, "SN", 6).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "LM", 0).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_hard_lb(14, "LD", 1).
shifts_hard_lb(14, "SE", 1).
shifts_hard_lb(14, "SN", 1).
shifts_soft_lb(14, "LD", 3).
shifts_soft_lb(14, "SE", 3).
shifts_soft_lb(14, "SN", 3).
shifts_soft_ub(14, "LD", 4).
shifts_soft_ub(14, "SE", 4).
shifts_soft_ub(14, "SN", 4).
shifts_hard_ub(14, "LD", 4).
shifts_hard_ub(14, "SE", 4).
shifts_hard_ub(14, "SN", 4).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "LM", 0).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "LD", 1).
shifts_hard_lb(15, "SE", 1).
shifts_hard_lb(15, "SN", 1).
shifts_soft_lb(15, "LD", 5).
shifts_soft_lb(15, "SE", 5).
shifts_soft_lb(15, "SN", 5).
shifts_soft_ub(15, "LD", 6).
shifts_soft_ub(15, "SE", 6).
shifts_soft_ub(15, "SN", 6).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_hard_lb(16, "LD", 1).
shifts_hard_lb(16, "SE", 1).
shifts_hard_lb(16, "SN", 1).
shifts_soft_lb(16, "LD", 2).
shifts_soft_lb(16, "SE", 2).
shifts_soft_lb(16, "SN", 2).
shifts_soft_ub(16, "LD", 3).
shifts_soft_ub(16, "SE", 3).
shifts_soft_ub(16, "SN", 3).
shifts_hard_ub(16, "LD", 3).
shifts_hard_ub(16, "SE", 3).
shifts_hard_ub(16, "SN", 3).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "LM", 0).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_hard_lb(17, "LD", 1).
shifts_hard_lb(17, "SE", 1).
shifts_hard_lb(17, "SN", 1).
shifts_soft_lb(17, "LD", 5).
shifts_soft_lb(17, "SE", 5).
shifts_soft_lb(17, "SN", 5).
shifts_soft_ub(17, "LD", 6).
shifts_soft_ub(17, "SE", 6).
shifts_soft_ub(17, "SN", 6).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "LM", 0).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_hard_lb(18, "LD", 1).
shifts_hard_lb(18, "SE", 1).
shifts_hard_lb(18, "SN", 1).
shifts_soft_lb(18, "LD", 5).
shifts_soft_lb(18, "SE", 5).
shifts_soft_lb(18, "SN", 5).
shifts_soft_ub(18, "LD", 6).
shifts_soft_ub(18, "SE", 6).
shifts_soft_ub(18, "SN", 6).
shifts_hard_ub(18, "LD", 6).
shifts_hard_ub(18, "SE", 6).
shifts_hard_ub(18, "SN", 6).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "LM", 0).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_hard_lb(19, "LD", 1).
shifts_hard_lb(19, "SE", 1).
shifts_hard_lb(19, "SN", 1).
shifts_soft_lb(19, "LD", 5).
shifts_soft_lb(19, "SE", 5).
shifts_soft_lb(19, "SN", 5).
shifts_soft_ub(19, "LD", 6).
shifts_soft_ub(19, "SE", 6).
shifts_soft_ub(19, "SN", 6).
shifts_hard_ub(19, "LD", 6).
shifts_hard_ub(19, "SE", 6).
shifts_hard_ub(19, "SN", 6).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "LM", 0).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "LD", 1).
shifts_hard_lb(20, "SE", 1).
shifts_hard_lb(20, "SN", 1).
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 6).
shifts_soft_ub(20, "SE", 6).
shifts_soft_ub(20, "SN", 6).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "LM", 0).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_hard_lb(21, "LD", 1).
shifts_hard_lb(21, "SE", 1).
shifts_hard_lb(21, "SN", 1).
shifts_soft_lb(21, "LD", 5).
shifts_soft_lb(21, "SE", 5).
shifts_soft_lb(21, "SN", 5).
shifts_soft_ub(21, "LD", 6).
shifts_soft_ub(21, "SE", 6).
shifts_soft_ub(21, "SN", 6).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "LM", 0).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_hard_lb(22, "LD", 1).
shifts_hard_lb(22, "SE", 1).
shifts_hard_lb(22, "SN", 1).
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 6).
shifts_soft_ub(22, "SE", 6).
shifts_soft_ub(22, "SN", 6).
shifts_hard_ub(22, "LD", 6).
shifts_hard_ub(22, "SE", 6).
shifts_hard_ub(22, "SN", 6).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "LM", 0).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_hard_lb(23, "LD", 1).
shifts_hard_lb(23, "SE", 1).
shifts_hard_lb(23, "SN", 1).
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 6).
shifts_soft_ub(23, "SE", 6).
shifts_soft_ub(23, "SN", 6).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
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
