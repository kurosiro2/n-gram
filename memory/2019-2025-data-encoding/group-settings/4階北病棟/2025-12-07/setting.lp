% Staffs ---------------------------------------------------
staff(1, "Jennifer Moran", "Chief Nurse", "29913", 0).
staff(2, "Wanda Williams", "Deputy Chief Nurse", "51294", 0).
staff(3, "Melissa Mckenzie", "Midwife", "56072", 0).
staff(4, "Kevin Clay", "Deputy Chief Nurse", "09040", 0).
staff(5, "Holly Donovan", "Deputy Chief Nurse", "42922", 0).
staff(6, "Martin Stanton", "Midwife", "09568", 0).
staff(7, "Daniel Martin", "Midwife", "57085", 0).
staff(8, "Virginia Davis", "Midwife", "07200", 0).
staff(9, "Kevin Mcclain", "Midwife", "40623", 0).
staff(10, "Tonya Robles", "Midwife", "82441", 0).
staff(11, "David Krueger", "Midwife", "52555", 0).
staff(12, "Alyssa Thomas", "Midwife", "47576", 0).
staff(13, "Christopher Nelson", "Midwife", "46506", 0).
staff(14, "Rebecca Miller", "Midwife", "41952", 0).
staff(15, "Mike Williamson", "Midwife", "84677", 0).
staff(16, "Terri Tyler", "Midwife", "93359", 0).
staff(17, "Amy Simmons", "Midwife", "22090", 0).
staff(18, "John Sherman", "Nurse", "67053", 0).
staff(19, "Cheryl Clark", "Midwife", "47432", 0).
staff(20, "David Castro", "Midwife", "77434", 0).
staff(21, "Sean Sanchez", "Nurse", "10119", 0).
staff(22, "Debra Brown", "Nurse", "15037", 0).
staff(23, "David Stafford", "Midwife", "35066", 0).
staff(24, "James Carter", "Midwife", "A088940", 0).
staff(25, "Easton Miller", "Midwife", "A639676", 0).
staff(26, "Isaiah Martinez", "Midwife", "A132815", 0).
staff(27, "Christopher Miller", "Midwife", "A282601", 0).
staff(28, "Easton Hall", "Midwife", "A677608", 0).
staff(29, "Thomas Ramirez", "Midwife", "A408829", 0).
staff(30, "Adrian Thompson", "Midwife", "A711959", 0).
staff(31, "Jeremy Richmond", "Midwife", "80713", 0).
staff(32, "Richard Rush", "Midwife", "27660", 0).
staff(33, "Matthew Reyes", "Midwife", "36220", 0).
staff(34, "Jonathan Fisher", "パート", "42987", 0).
staff(35, "Krystal Day", "パート", "19821", 0).
staff(36, "Jacqueline Brown", "パート", "90224", 0).
% Groups ---------------------------------------------------
group("All").
group("All", 1).
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
group("All", 24).
group("All", 25).
group("All", 26).
group("All", 27).
group("All", 28).
group("All", 29).
group("All", 30).
group("All", 34).
group("All", 35).
group("All", 36).
group("代行").
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("代行", 6).
group("代行", 7).
group("代行", 8).
group("代行", 9).
group("代行", 10).
group("Leaders").
group("Leaders", 3).
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 6).
group("Leaders", 7).
group("Leaders", 8).
group("Leaders", 9).
group("Leaders", 10).
group("Leaders", 11).
group("Leaders", 12).
group("Leaders", 14).
group("平日代行").
group("平日代行", 3).
group("平日代行", 4).
group("平日代行", 5).
group("平日代行", 6).
group("Newcomers").
group("Newcomers", 18).
group("Newcomers", 21).
group("Newcomers", 22).
group("Newcomers", 25).
group("Newcomers", 26).
group("Newcomers", 27).
group("Newcomers", 28).
group("Newcomers", 29).
group("Newcomers", 30).
group("Night").
group("Night", 3).
group("Night", 4).
group("Night", 5).
group("Night", 6).
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
group("Night", 24).
group("Night", 25).
group("Night", 26).
group("Night", 27).
group("Night", 28).
group("Night", 29).
group("Night", 30).
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
consecutive_work_ub(6).
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
patterns_hard_lb("○○", 2).
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
staff_dweek_hard_lb("All", "D", "Mo", 4).
staff_dweek_hard_lb("All", "D", "Tu", 4).
staff_dweek_hard_lb("All", "D", "We", 4).
staff_dweek_hard_lb("All", "D", "Th", 4).
staff_dweek_hard_lb("All", "D", "Fr", 4).
staff_dweek_hard_lb("All", "D", "Sa", 3).
staff_dweek_hard_lb("All", "D", "Su", 3).
staff_dweek_hard_lb("All", "D", "Ho", 3).
staff_dweek_soft_lb("All", "D", "Mo", 4).
staff_dweek_soft_lb("All", "D", "Tu", 4).
staff_dweek_soft_lb("All", "D", "We", 4).
staff_dweek_soft_lb("All", "D", "Th", 4).
staff_dweek_soft_lb("All", "D", "Fr", 4).
staff_dweek_soft_lb("All", "D", "Sa", 3).
staff_dweek_soft_lb("All", "D", "Su", 3).
staff_dweek_soft_lb("All", "D", "Ho", 3).
staff_dweek_soft_ub("All", "D", "Mo", 9).
staff_dweek_soft_ub("All", "D", "Tu", 9).
staff_dweek_soft_ub("All", "D", "We", 9).
staff_dweek_soft_ub("All", "D", "Th", 9).
staff_dweek_soft_ub("All", "D", "Fr", 9).
staff_dweek_soft_ub("All", "D", "Sa", 5).
staff_dweek_soft_ub("All", "D", "Su", 5).
staff_dweek_soft_ub("All", "D", "Ho", 5).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Mo", 4).
staff_dweek_hard_lb("All", "LD", "Tu", 4).
staff_dweek_hard_lb("All", "LD", "We", 4).
staff_dweek_hard_lb("All", "LD", "Th", 4).
staff_dweek_hard_lb("All", "LD", "Fr", 4).
staff_dweek_hard_lb("All", "LD", "Sa", 4).
staff_dweek_hard_lb("All", "LD", "Su", 4).
staff_dweek_hard_lb("All", "LD", "Ho", 4).
staff_dweek_soft_lb("All", "LD", "Mo", 4).
staff_dweek_soft_lb("All", "LD", "Tu", 4).
staff_dweek_soft_lb("All", "LD", "We", 4).
staff_dweek_soft_lb("All", "LD", "Th", 4).
staff_dweek_soft_lb("All", "LD", "Fr", 4).
staff_dweek_soft_lb("All", "LD", "Sa", 4).
staff_dweek_soft_lb("All", "LD", "Su", 4).
staff_dweek_soft_lb("All", "LD", "Ho", 4).
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
staff_dweek_hard_ub("All", "LD", "Sa", 4).
staff_dweek_hard_ub("All", "LD", "Su", 4).
staff_dweek_hard_ub("All", "LD", "Ho", 4).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Mo", 4).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Mo", 4).
staff_dweek_hard_lb("All", "SE", "Tu", 4).
staff_dweek_hard_lb("All", "SN", "Tu", 4).
staff_dweek_hard_lb("All", "SE", "We", 4).
staff_dweek_hard_lb("All", "SN", "We", 4).
staff_dweek_hard_lb("All", "SE", "Th", 4).
staff_dweek_hard_lb("All", "SN", "Th", 4).
staff_dweek_hard_lb("All", "SE", "Fr", 4).
staff_dweek_hard_lb("All", "SN", "Fr", 4).
staff_dweek_hard_lb("All", "SE", "Sa", 4).
staff_dweek_hard_lb("All", "SN", "Sa", 4).
staff_dweek_hard_lb("All", "SE", "Su", 4).
staff_dweek_hard_lb("All", "SN", "Su", 4).
staff_dweek_hard_lb("All", "SE", "Ho", 4).
staff_dweek_hard_lb("All", "SN", "Ho", 4).
staff_dweek_soft_lb("All", "SE", "Mo", 4).
staff_dweek_soft_lb("All", "SN", "Mo", 4).
staff_dweek_soft_lb("All", "SE", "Tu", 4).
staff_dweek_soft_lb("All", "SN", "Tu", 4).
staff_dweek_soft_lb("All", "SE", "We", 4).
staff_dweek_soft_lb("All", "SN", "We", 4).
staff_dweek_soft_lb("All", "SE", "Th", 4).
staff_dweek_soft_lb("All", "SN", "Th", 4).
staff_dweek_soft_lb("All", "SE", "Fr", 4).
staff_dweek_soft_lb("All", "SN", "Fr", 4).
staff_dweek_soft_lb("All", "SE", "Sa", 4).
staff_dweek_soft_lb("All", "SN", "Sa", 4).
staff_dweek_soft_lb("All", "SE", "Su", 4).
staff_dweek_soft_lb("All", "SN", "Su", 4).
staff_dweek_soft_lb("All", "SE", "Ho", 4).
staff_dweek_soft_lb("All", "SN", "Ho", 4).
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
staff_dweek_hard_ub("All", "SE", "Mo", 4).
staff_dweek_hard_ub("All", "SN", "Mo", 4).
staff_dweek_hard_ub("All", "SE", "Tu", 4).
staff_dweek_hard_ub("All", "SN", "Tu", 4).
staff_dweek_hard_ub("All", "SE", "We", 4).
staff_dweek_hard_ub("All", "SN", "We", 4).
staff_dweek_hard_ub("All", "SE", "Th", 4).
staff_dweek_hard_ub("All", "SN", "Th", 4).
staff_dweek_hard_ub("All", "SE", "Fr", 4).
staff_dweek_hard_ub("All", "SN", "Fr", 4).
staff_dweek_hard_ub("All", "SE", "Sa", 4).
staff_dweek_hard_ub("All", "SN", "Sa", 4).
staff_dweek_hard_ub("All", "SE", "Su", 4).
staff_dweek_hard_ub("All", "SN", "Su", 4).
staff_dweek_hard_ub("All", "SE", "Ho", 4).
staff_dweek_hard_ub("All", "SN", "Ho", 4).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_ub("All", "LM", "Mo", 0).
staff_dweek_hard_ub("All", "LM", "Tu", 0).
staff_dweek_hard_ub("All", "LM", "We", 0).
staff_dweek_hard_ub("All", "LM", "Th", 0).
staff_dweek_hard_ub("All", "LM", "Fr", 0).
staff_dweek_hard_ub("All", "LM", "Sa", 0).
staff_dweek_hard_ub("All", "LM", "Su", 0).
staff_dweek_hard_ub("All", "LM", "Ho", 0).
staff_dweek_hard_lb("代行", "D", "Mo", 1).
staff_dweek_hard_lb("代行", "LD", "Mo", 1).
staff_dweek_hard_lb("代行", "D", "Tu", 1).
staff_dweek_hard_lb("代行", "LD", "Tu", 1).
staff_dweek_hard_lb("代行", "D", "We", 1).
staff_dweek_hard_lb("代行", "LD", "We", 1).
staff_dweek_hard_lb("代行", "D", "Th", 1).
staff_dweek_hard_lb("代行", "LD", "Th", 1).
staff_dweek_hard_lb("代行", "D", "Fr", 1).
staff_dweek_hard_lb("代行", "LD", "Fr", 1).
staff_dweek_hard_lb("代行", "D", "Sa", 1).
staff_dweek_hard_lb("代行", "LD", "Sa", 1).
staff_dweek_hard_lb("代行", "D", "Su", 1).
staff_dweek_hard_lb("代行", "LD", "Su", 1).
staff_dweek_hard_lb("代行", "D", "Ho", 1).
staff_dweek_hard_lb("代行", "LD", "Ho", 1).
staff_dweek_soft_lb("代行", "D", "Mo", 1).
staff_dweek_soft_lb("代行", "LD", "Mo", 1).
staff_dweek_soft_lb("代行", "D", "Tu", 1).
staff_dweek_soft_lb("代行", "LD", "Tu", 1).
staff_dweek_soft_lb("代行", "D", "We", 1).
staff_dweek_soft_lb("代行", "LD", "We", 1).
staff_dweek_soft_lb("代行", "D", "Th", 1).
staff_dweek_soft_lb("代行", "LD", "Th", 1).
staff_dweek_soft_lb("代行", "D", "Fr", 1).
staff_dweek_soft_lb("代行", "LD", "Fr", 1).
staff_dweek_soft_lb("代行", "D", "Sa", 1).
staff_dweek_soft_lb("代行", "LD", "Sa", 1).
staff_dweek_soft_lb("代行", "D", "Su", 1).
staff_dweek_soft_lb("代行", "LD", "Su", 1).
staff_dweek_soft_lb("代行", "D", "Ho", 1).
staff_dweek_soft_lb("代行", "LD", "Ho", 1).
staff_dweek_soft_ub("代行", "D", "Mo", 1).
staff_dweek_soft_ub("代行", "LD", "Mo", 1).
staff_dweek_soft_ub("代行", "D", "Tu", 1).
staff_dweek_soft_ub("代行", "LD", "Tu", 1).
staff_dweek_soft_ub("代行", "D", "We", 1).
staff_dweek_soft_ub("代行", "LD", "We", 1).
staff_dweek_soft_ub("代行", "D", "Th", 1).
staff_dweek_soft_ub("代行", "LD", "Th", 1).
staff_dweek_soft_ub("代行", "D", "Fr", 1).
staff_dweek_soft_ub("代行", "LD", "Fr", 1).
staff_dweek_soft_ub("代行", "D", "Sa", 1).
staff_dweek_soft_ub("代行", "LD", "Sa", 1).
staff_dweek_soft_ub("代行", "D", "Su", 1).
staff_dweek_soft_ub("代行", "LD", "Su", 1).
staff_dweek_soft_ub("代行", "D", "Ho", 1).
staff_dweek_soft_ub("代行", "LD", "Ho", 1).
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
staff_dweek_soft_lb("Leaders", "LD", "Mo", 1).
staff_dweek_soft_lb("Leaders", "SE", "Mo", 1).
staff_dweek_soft_lb("Leaders", "SN", "Mo", 1).
staff_dweek_soft_lb("Leaders", "LD", "Tu", 1).
staff_dweek_soft_lb("Leaders", "SE", "Tu", 1).
staff_dweek_soft_lb("Leaders", "SN", "Tu", 1).
staff_dweek_soft_lb("Leaders", "LD", "We", 1).
staff_dweek_soft_lb("Leaders", "SE", "We", 1).
staff_dweek_soft_lb("Leaders", "SN", "We", 1).
staff_dweek_soft_lb("Leaders", "LD", "Th", 1).
staff_dweek_soft_lb("Leaders", "SE", "Th", 1).
staff_dweek_soft_lb("Leaders", "SN", "Th", 1).
staff_dweek_soft_lb("Leaders", "LD", "Fr", 1).
staff_dweek_soft_lb("Leaders", "SE", "Fr", 1).
staff_dweek_soft_lb("Leaders", "SN", "Fr", 1).
staff_dweek_soft_lb("Leaders", "LD", "Sa", 1).
staff_dweek_soft_lb("Leaders", "SE", "Sa", 1).
staff_dweek_soft_lb("Leaders", "SN", "Sa", 1).
staff_dweek_soft_lb("Leaders", "LD", "Su", 1).
staff_dweek_soft_lb("Leaders", "SE", "Su", 1).
staff_dweek_soft_lb("Leaders", "SN", "Su", 1).
staff_dweek_soft_lb("Leaders", "LD", "Ho", 1).
staff_dweek_soft_lb("Leaders", "SE", "Ho", 1).
staff_dweek_soft_lb("Leaders", "SN", "Ho", 1).
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
staff_dweek_hard_ub("Leaders", "LD", "Mo", 3).
staff_dweek_hard_ub("Leaders", "SE", "Mo", 3).
staff_dweek_hard_ub("Leaders", "SN", "Mo", 3).
staff_dweek_hard_ub("Leaders", "LD", "Tu", 3).
staff_dweek_hard_ub("Leaders", "SE", "Tu", 3).
staff_dweek_hard_ub("Leaders", "SN", "Tu", 3).
staff_dweek_hard_ub("Leaders", "LD", "We", 3).
staff_dweek_hard_ub("Leaders", "SE", "We", 3).
staff_dweek_hard_ub("Leaders", "SN", "We", 3).
staff_dweek_hard_ub("Leaders", "LD", "Th", 3).
staff_dweek_hard_ub("Leaders", "SE", "Th", 3).
staff_dweek_hard_ub("Leaders", "SN", "Th", 3).
staff_dweek_hard_ub("Leaders", "LD", "Fr", 3).
staff_dweek_hard_ub("Leaders", "SE", "Fr", 3).
staff_dweek_hard_ub("Leaders", "SN", "Fr", 3).
staff_dweek_hard_ub("Leaders", "LD", "Sa", 3).
staff_dweek_hard_ub("Leaders", "SE", "Sa", 3).
staff_dweek_hard_ub("Leaders", "SN", "Sa", 3).
staff_dweek_hard_ub("Leaders", "LD", "Su", 3).
staff_dweek_hard_ub("Leaders", "SE", "Su", 3).
staff_dweek_hard_ub("Leaders", "SN", "Su", 3).
staff_dweek_hard_ub("Leaders", "LD", "Ho", 3).
staff_dweek_hard_ub("Leaders", "SE", "Ho", 3).
staff_dweek_hard_ub("Leaders", "SN", "Ho", 3).
staff_dweek_hard_lb("平日代行", "D", "Mo", 1).
staff_dweek_hard_lb("平日代行", "D", "Tu", 1).
staff_dweek_hard_lb("平日代行", "D", "We", 1).
staff_dweek_hard_lb("平日代行", "D", "Th", 1).
staff_dweek_hard_lb("平日代行", "D", "Fr", 1).
staff_dweek_hard_lb("平日代行", "D", "Sa", 0).
staff_dweek_hard_lb("平日代行", "D", "Su", 0).
staff_dweek_hard_lb("平日代行", "D", "Ho", 0).
staff_dweek_soft_lb("平日代行", "D", "Mo", 1).
staff_dweek_soft_lb("平日代行", "D", "Tu", 1).
staff_dweek_soft_lb("平日代行", "D", "We", 1).
staff_dweek_soft_lb("平日代行", "D", "Th", 1).
staff_dweek_soft_lb("平日代行", "D", "Fr", 1).
staff_dweek_soft_lb("平日代行", "D", "Sa", 0).
staff_dweek_soft_lb("平日代行", "D", "Su", 0).
staff_dweek_soft_lb("平日代行", "D", "Ho", 0).
staff_dweek_soft_ub("平日代行", "D", "Mo", 1).
staff_dweek_soft_ub("平日代行", "D", "Tu", 1).
staff_dweek_soft_ub("平日代行", "D", "We", 1).
staff_dweek_soft_ub("平日代行", "D", "Th", 1).
staff_dweek_soft_ub("平日代行", "D", "Fr", 1).
staff_dweek_soft_ub("平日代行", "D", "Sa", 0).
staff_dweek_soft_ub("平日代行", "D", "Su", 0).
staff_dweek_soft_ub("平日代行", "D", "Ho", 0).
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
staff_dweek_hard_lb("Night", "SE", "Mo", 4).
staff_dweek_hard_lb("Night", "SN", "Mo", 4).
staff_dweek_hard_lb("Night", "SE", "Tu", 4).
staff_dweek_hard_lb("Night", "SN", "Tu", 4).
staff_dweek_hard_lb("Night", "SE", "We", 4).
staff_dweek_hard_lb("Night", "SN", "We", 4).
staff_dweek_hard_lb("Night", "SE", "Th", 4).
staff_dweek_hard_lb("Night", "SN", "Th", 4).
staff_dweek_hard_lb("Night", "SE", "Fr", 4).
staff_dweek_hard_lb("Night", "SN", "Fr", 4).
staff_dweek_hard_lb("Night", "SE", "Sa", 4).
staff_dweek_hard_lb("Night", "SN", "Sa", 4).
staff_dweek_hard_lb("Night", "SE", "Su", 4).
staff_dweek_hard_lb("Night", "SN", "Su", 4).
staff_dweek_hard_lb("Night", "SE", "Ho", 4).
staff_dweek_hard_lb("Night", "SN", "Ho", 4).
staff_dweek_soft_lb("Night", "SE", "Mo", 4).
staff_dweek_soft_lb("Night", "SN", "Mo", 4).
staff_dweek_soft_lb("Night", "SE", "Tu", 4).
staff_dweek_soft_lb("Night", "SN", "Tu", 4).
staff_dweek_soft_lb("Night", "SE", "We", 4).
staff_dweek_soft_lb("Night", "SN", "We", 4).
staff_dweek_soft_lb("Night", "SE", "Th", 4).
staff_dweek_soft_lb("Night", "SN", "Th", 4).
staff_dweek_soft_lb("Night", "SE", "Fr", 4).
staff_dweek_soft_lb("Night", "SN", "Fr", 4).
staff_dweek_soft_lb("Night", "SE", "Sa", 4).
staff_dweek_soft_lb("Night", "SN", "Sa", 4).
staff_dweek_soft_lb("Night", "SE", "Su", 4).
staff_dweek_soft_lb("Night", "SN", "Su", 4).
staff_dweek_soft_lb("Night", "SE", "Ho", 4).
staff_dweek_soft_lb("Night", "SN", "Ho", 4).
staff_dweek_soft_ub("Night", "SE", "Mo", 4).
staff_dweek_soft_ub("Night", "SN", "Mo", 4).
staff_dweek_soft_ub("Night", "SE", "Tu", 4).
staff_dweek_soft_ub("Night", "SN", "Tu", 4).
staff_dweek_soft_ub("Night", "SE", "We", 4).
staff_dweek_soft_ub("Night", "SN", "We", 4).
staff_dweek_soft_ub("Night", "SE", "Th", 4).
staff_dweek_soft_ub("Night", "SN", "Th", 4).
staff_dweek_soft_ub("Night", "SE", "Fr", 4).
staff_dweek_soft_ub("Night", "SN", "Fr", 4).
staff_dweek_soft_ub("Night", "SE", "Sa", 4).
staff_dweek_soft_ub("Night", "SN", "Sa", 4).
staff_dweek_soft_ub("Night", "SE", "Su", 4).
staff_dweek_soft_ub("Night", "SN", "Su", 4).
staff_dweek_soft_ub("Night", "SE", "Ho", 4).
staff_dweek_soft_ub("Night", "SN", "Ho", 4).
staff_dweek_hard_ub("Night", "SE", "Mo", 4).
staff_dweek_hard_ub("Night", "SN", "Mo", 4).
staff_dweek_hard_ub("Night", "SE", "Tu", 4).
staff_dweek_hard_ub("Night", "SN", "Tu", 4).
staff_dweek_hard_ub("Night", "SE", "We", 4).
staff_dweek_hard_ub("Night", "SN", "We", 4).
staff_dweek_hard_ub("Night", "SE", "Th", 4).
staff_dweek_hard_ub("Night", "SN", "Th", 4).
staff_dweek_hard_ub("Night", "SE", "Fr", 4).
staff_dweek_hard_ub("Night", "SN", "Fr", 4).
staff_dweek_hard_ub("Night", "SE", "Sa", 4).
staff_dweek_hard_ub("Night", "SN", "Sa", 4).
staff_dweek_hard_ub("Night", "SE", "Su", 4).
staff_dweek_hard_ub("Night", "SN", "Su", 4).
staff_dweek_hard_ub("Night", "SE", "Ho", 4).
staff_dweek_hard_ub("Night", "SN", "Ho", 4).
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("代行", "D", "Staffs").
vertical_constraint_type("代行", "LD", "Staffs").
vertical_constraint_type("Leaders", "LD", "Staffs").
vertical_constraint_type("Leaders", "SE", "Staffs").
vertical_constraint_type("Leaders", "SN", "Staffs").
vertical_constraint_type("平日代行", "D", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
vertical_constraint_type("Night", "SE", "Staffs").
vertical_constraint_type("Night", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_hard_lb(1, "LD", 0).
shifts_hard_lb(1, "SE", 0).
shifts_hard_lb(1, "SN", 0).
shifts_soft_lb(1, "LD", 0).
shifts_soft_lb(1, "SE", 0).
shifts_soft_lb(1, "SN", 0).
shifts_soft_ub(1, "LD", 0).
shifts_soft_ub(1, "SE", 0).
shifts_soft_ub(1, "SN", 0).
shifts_hard_ub(1, "LD", 0).
shifts_hard_ub(1, "SE", 0).
shifts_hard_ub(1, "SN", 0).
shifts("E").
shifts("E", "E").
shifts_hard_lb(1, "E", 0).
shifts("N").
shifts("N", "N").
shifts_hard_lb(1, "N", 0).
shifts_soft_lb(1, "E", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_ub(1, "E", 0).
shifts_soft_ub(1, "N", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_lb(1, "LM", 0).
shifts_soft_lb(1, "LM", 0).
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "LM", 0).
shifts("EM").
shifts("EM", "EM").
shifts_hard_lb(1, "EM", 0).
shifts_soft_lb(1, "EM", 0).
shifts_soft_ub(1, "EM", 0).
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_hard_lb(2, "LD", 0).
shifts_hard_lb(2, "SE", 0).
shifts_hard_lb(2, "SN", 0).
shifts_soft_lb(2, "LD", 0).
shifts_soft_lb(2, "SE", 0).
shifts_soft_lb(2, "SN", 0).
shifts_soft_ub(2, "LD", 0).
shifts_soft_ub(2, "SE", 0).
shifts_soft_ub(2, "SN", 0).
shifts_hard_ub(2, "LD", 0).
shifts_hard_ub(2, "SE", 0).
shifts_hard_ub(2, "SN", 0).
shifts_hard_lb(2, "E", 0).
shifts_hard_lb(2, "N", 0).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "LM", 0).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_hard_lb(3, "LD", 2).
shifts_hard_lb(3, "SE", 2).
shifts_hard_lb(3, "SN", 2).
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 2).
shifts_soft_ub(3, "SE", 2).
shifts_soft_ub(3, "SN", 2).
shifts_hard_ub(3, "LD", 3).
shifts_hard_ub(3, "SE", 3).
shifts_hard_ub(3, "SN", 3).
shifts_hard_lb(3, "E", 0).
shifts_hard_lb(3, "N", 0).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
shifts_hard_ub(3, "E", 0).
shifts_hard_ub(3, "N", 0).
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
shifts_soft_lb(4, "LD", 2).
shifts_soft_lb(4, "SE", 2).
shifts_soft_lb(4, "SN", 2).
shifts_soft_ub(4, "LD", 2).
shifts_soft_ub(4, "SE", 2).
shifts_soft_ub(4, "SN", 2).
shifts_hard_ub(4, "LD", 2).
shifts_hard_ub(4, "SE", 2).
shifts_hard_ub(4, "SN", 2).
shifts_hard_lb(4, "E", 0).
shifts_hard_lb(4, "N", 0).
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
shifts_hard_ub(4, "E", 0).
shifts_hard_ub(4, "N", 0).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "LM", 0).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "LD", 2).
shifts_hard_lb(5, "SE", 2).
shifts_hard_lb(5, "SN", 2).
shifts_soft_lb(5, "LD", 3).
shifts_soft_lb(5, "SE", 3).
shifts_soft_lb(5, "SN", 3).
shifts_soft_ub(5, "LD", 4).
shifts_soft_ub(5, "SE", 4).
shifts_soft_ub(5, "SN", 4).
shifts_hard_ub(5, "LD", 5).
shifts_hard_ub(5, "SE", 5).
shifts_hard_ub(5, "SN", 5).
shifts_hard_lb(5, "E", 0).
shifts_hard_lb(5, "N", 0).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 0).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_hard_lb(6, "LD", 2).
shifts_hard_lb(6, "SE", 2).
shifts_hard_lb(6, "SN", 2).
shifts_soft_lb(6, "LD", 2).
shifts_soft_lb(6, "SE", 2).
shifts_soft_lb(6, "SN", 2).
shifts_soft_ub(6, "LD", 2).
shifts_soft_ub(6, "SE", 2).
shifts_soft_ub(6, "SN", 2).
shifts_hard_ub(6, "LD", 3).
shifts_hard_ub(6, "SE", 3).
shifts_hard_ub(6, "SN", 3).
shifts_hard_lb(6, "E", 0).
shifts_hard_lb(6, "N", 0).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "LM", 0).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "LD", 3).
shifts_hard_lb(7, "SE", 3).
shifts_hard_lb(7, "SN", 3).
shifts_soft_lb(7, "LD", 3).
shifts_soft_lb(7, "SE", 3).
shifts_soft_lb(7, "SN", 3).
shifts_soft_ub(7, "LD", 3).
shifts_soft_ub(7, "SE", 3).
shifts_soft_ub(7, "SN", 3).
shifts_hard_ub(7, "LD", 3).
shifts_hard_ub(7, "SE", 3).
shifts_hard_ub(7, "SN", 3).
shifts_hard_lb(7, "E", 0).
shifts_hard_lb(7, "N", 0).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "LM", 0).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "LD", 3).
shifts_hard_lb(8, "SE", 3).
shifts_hard_lb(8, "SN", 3).
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_hard_lb(8, "E", 0).
shifts_hard_lb(8, "N", 0).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
shifts_hard_lb(8, "LM", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "LM", 0).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_hard_lb(9, "LD", 3).
shifts_hard_lb(9, "SE", 3).
shifts_hard_lb(9, "SN", 3).
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_hard_lb(9, "E", 0).
shifts_hard_lb(9, "N", 0).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "LM", 0).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_hard_lb(10, "LD", 3).
shifts_hard_lb(10, "SE", 3).
shifts_hard_lb(10, "SN", 3).
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_hard_lb(10, "E", 0).
shifts_hard_lb(10, "N", 0).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "LM", 0).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_hard_lb(11, "LD", 3).
shifts_hard_lb(11, "SE", 3).
shifts_hard_lb(11, "SN", 3).
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "LD", 6).
shifts_hard_ub(11, "SE", 6).
shifts_hard_ub(11, "SN", 6).
shifts_hard_lb(11, "E", 0).
shifts_hard_lb(11, "N", 0).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "LM", 0).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_hard_lb(12, "LD", 3).
shifts_hard_lb(12, "SE", 3).
shifts_hard_lb(12, "SN", 3).
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_hard_ub(12, "LD", 6).
shifts_hard_ub(12, "SE", 6).
shifts_hard_ub(12, "SN", 6).
shifts_hard_lb(12, "E", 0).
shifts_hard_lb(12, "N", 0).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_ub(12, "E", 0).
shifts_soft_ub(12, "N", 0).
shifts_hard_ub(12, "E", 0).
shifts_hard_ub(12, "N", 0).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_hard_lb(13, "LD", 2).
shifts_hard_lb(13, "SE", 2).
shifts_hard_lb(13, "SN", 2).
shifts_soft_lb(13, "LD", 3).
shifts_soft_lb(13, "SE", 3).
shifts_soft_lb(13, "SN", 3).
shifts_soft_ub(13, "LD", 3).
shifts_soft_ub(13, "SE", 3).
shifts_soft_ub(13, "SN", 3).
shifts_hard_ub(13, "LD", 3).
shifts_hard_ub(13, "SE", 3).
shifts_hard_ub(13, "SN", 3).
shifts_hard_lb(13, "E", 0).
shifts_hard_lb(13, "N", 0).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "LM", 0).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_hard_lb(14, "LD", 3).
shifts_hard_lb(14, "SE", 3).
shifts_hard_lb(14, "SN", 3).
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_hard_ub(14, "LD", 6).
shifts_hard_ub(14, "SE", 6).
shifts_hard_ub(14, "SN", 6).
shifts_hard_lb(14, "E", 0).
shifts_hard_lb(14, "N", 0).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "LM", 0).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "LD", 3).
shifts_hard_lb(15, "SE", 3).
shifts_hard_lb(15, "SN", 3).
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_hard_lb(15, "E", 0).
shifts_hard_lb(15, "N", 0).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_hard_lb(16, "LD", 3).
shifts_hard_lb(16, "SE", 3).
shifts_hard_lb(16, "SN", 3).
shifts_soft_lb(16, "LD", 4).
shifts_soft_lb(16, "SE", 4).
shifts_soft_lb(16, "SN", 4).
shifts_soft_ub(16, "LD", 5).
shifts_soft_ub(16, "SE", 5).
shifts_soft_ub(16, "SN", 5).
shifts_hard_ub(16, "LD", 6).
shifts_hard_ub(16, "SE", 6).
shifts_hard_ub(16, "SN", 6).
shifts_hard_lb(16, "E", 0).
shifts_hard_lb(16, "N", 0).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "LM", 0).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_hard_lb(17, "LD", 3).
shifts_hard_lb(17, "SE", 3).
shifts_hard_lb(17, "SN", 3).
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 5).
shifts_soft_ub(17, "SE", 5).
shifts_soft_ub(17, "SN", 5).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_hard_lb(17, "E", 0).
shifts_hard_lb(17, "N", 0).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
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
shifts_soft_lb(18, "LD", 2).
shifts_soft_lb(18, "SE", 2).
shifts_soft_lb(18, "SN", 2).
shifts_soft_ub(18, "LD", 2).
shifts_soft_ub(18, "SE", 2).
shifts_soft_ub(18, "SN", 2).
shifts_hard_ub(18, "LD", 2).
shifts_hard_ub(18, "SE", 2).
shifts_hard_ub(18, "SN", 2).
shifts_hard_lb(18, "E", 0).
shifts_hard_lb(18, "N", 0).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
shifts_hard_ub(18, "E", 0).
shifts_hard_ub(18, "N", 0).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "LM", 0).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_hard_lb(19, "LD", 3).
shifts_hard_lb(19, "SE", 3).
shifts_hard_lb(19, "SN", 3).
shifts_soft_lb(19, "LD", 4).
shifts_soft_lb(19, "SE", 4).
shifts_soft_lb(19, "SN", 4).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "LD", 6).
shifts_hard_ub(19, "SE", 6).
shifts_hard_ub(19, "SN", 6).
shifts_hard_lb(19, "E", 0).
shifts_hard_lb(19, "N", 0).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
shifts_hard_ub(19, "E", 0).
shifts_hard_ub(19, "N", 0).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "LM", 0).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "LD", 3).
shifts_hard_lb(20, "SE", 3).
shifts_hard_lb(20, "SN", 3).
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_hard_lb(20, "E", 0).
shifts_hard_lb(20, "N", 0).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
shifts_hard_ub(20, "E", 0).
shifts_hard_ub(20, "N", 0).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "LM", 0).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_hard_lb(21, "LD", 3).
shifts_hard_lb(21, "SE", 3).
shifts_hard_lb(21, "SN", 3).
shifts_soft_lb(21, "LD", 4).
shifts_soft_lb(21, "SE", 4).
shifts_soft_lb(21, "SN", 4).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_hard_lb(21, "E", 0).
shifts_hard_lb(21, "N", 0).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "LM", 0).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_hard_lb(22, "LD", 3).
shifts_hard_lb(22, "SE", 3).
shifts_hard_lb(22, "SN", 3).
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 5).
shifts_soft_ub(22, "SE", 5).
shifts_soft_ub(22, "SN", 5).
shifts_hard_ub(22, "LD", 6).
shifts_hard_ub(22, "SE", 6).
shifts_hard_ub(22, "SN", 6).
shifts_hard_lb(22, "E", 0).
shifts_hard_lb(22, "N", 0).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "LM", 0).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_hard_lb(23, "LD", 3).
shifts_hard_lb(23, "SE", 3).
shifts_hard_lb(23, "SN", 3).
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
shifts_hard_lb(23, "E", 0).
shifts_hard_lb(23, "N", 0).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
shifts_hard_lb(23, "LM", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "LM", 0).
shifts_hard_ub(23, "LM", 0).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
shifts_hard_lb(24, "LD", 1).
shifts_hard_lb(24, "SE", 1).
shifts_hard_lb(24, "SN", 1).
shifts_soft_lb(24, "LD", 2).
shifts_soft_lb(24, "SE", 2).
shifts_soft_lb(24, "SN", 2).
shifts_soft_ub(24, "LD", 3).
shifts_soft_ub(24, "SE", 3).
shifts_soft_ub(24, "SN", 3).
shifts_hard_ub(24, "LD", 3).
shifts_hard_ub(24, "SE", 3).
shifts_hard_ub(24, "SN", 3).
shifts_hard_lb(24, "E", 0).
shifts_hard_lb(24, "N", 0).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_hard_lb(24, "LM", 0).
shifts_soft_lb(24, "LM", 0).
shifts_soft_ub(24, "LM", 0).
shifts_hard_ub(24, "LM", 0).
shifts_hard_lb(24, "EM", 0).
shifts_soft_lb(24, "EM", 0).
shifts_soft_ub(24, "EM", 0).
shifts_hard_ub(24, "EM", 0).
% Staff 25:
shifts_hard_lb(25, "LD", 1).
shifts_hard_lb(25, "SE", 1).
shifts_hard_lb(25, "SN", 1).
shifts_soft_lb(25, "LD", 2).
shifts_soft_lb(25, "SE", 2).
shifts_soft_lb(25, "SN", 2).
shifts_soft_ub(25, "LD", 3).
shifts_soft_ub(25, "SE", 3).
shifts_soft_ub(25, "SN", 3).
shifts_hard_ub(25, "LD", 4).
shifts_hard_ub(25, "SE", 4).
shifts_hard_ub(25, "SN", 4).
shifts_hard_lb(25, "E", 0).
shifts_hard_lb(25, "N", 0).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
shifts_hard_lb(25, "LM", 0).
shifts_soft_lb(25, "LM", 0).
shifts_soft_ub(25, "LM", 0).
shifts_hard_ub(25, "LM", 0).
shifts_hard_lb(25, "EM", 0).
shifts_soft_lb(25, "EM", 0).
shifts_soft_ub(25, "EM", 0).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_hard_lb(26, "LD", 1).
shifts_hard_lb(26, "SE", 1).
shifts_hard_lb(26, "SN", 1).
shifts_soft_lb(26, "LD", 2).
shifts_soft_lb(26, "SE", 2).
shifts_soft_lb(26, "SN", 2).
shifts_soft_ub(26, "LD", 3).
shifts_soft_ub(26, "SE", 3).
shifts_soft_ub(26, "SN", 3).
shifts_hard_ub(26, "LD", 4).
shifts_hard_ub(26, "SE", 4).
shifts_hard_ub(26, "SN", 4).
shifts_hard_lb(26, "E", 0).
shifts_hard_lb(26, "N", 0).
shifts_soft_lb(26, "E", 0).
shifts_soft_lb(26, "N", 0).
shifts_soft_ub(26, "E", 0).
shifts_soft_ub(26, "N", 0).
shifts_hard_ub(26, "E", 0).
shifts_hard_ub(26, "N", 0).
shifts_hard_lb(26, "LM", 0).
shifts_soft_lb(26, "LM", 0).
shifts_soft_ub(26, "LM", 0).
shifts_hard_ub(26, "LM", 0).
shifts_hard_lb(26, "EM", 0).
shifts_soft_lb(26, "EM", 0).
shifts_soft_ub(26, "EM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_hard_lb(27, "LD", 1).
shifts_hard_lb(27, "SE", 1).
shifts_hard_lb(27, "SN", 1).
shifts_soft_lb(27, "LD", 2).
shifts_soft_lb(27, "SE", 2).
shifts_soft_lb(27, "SN", 2).
shifts_soft_ub(27, "LD", 3).
shifts_soft_ub(27, "SE", 3).
shifts_soft_ub(27, "SN", 3).
shifts_hard_ub(27, "LD", 4).
shifts_hard_ub(27, "SE", 4).
shifts_hard_ub(27, "SN", 4).
shifts_hard_lb(27, "E", 0).
shifts_hard_lb(27, "N", 0).
shifts_soft_lb(27, "E", 0).
shifts_soft_lb(27, "N", 0).
shifts_soft_ub(27, "E", 0).
shifts_soft_ub(27, "N", 0).
shifts_hard_ub(27, "E", 0).
shifts_hard_ub(27, "N", 0).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "LM", 0).
shifts_hard_lb(27, "EM", 0).
shifts_soft_lb(27, "EM", 0).
shifts_soft_ub(27, "EM", 0).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_hard_lb(28, "LD", 1).
shifts_hard_lb(28, "SE", 1).
shifts_hard_lb(28, "SN", 1).
shifts_soft_lb(28, "LD", 2).
shifts_soft_lb(28, "SE", 2).
shifts_soft_lb(28, "SN", 2).
shifts_soft_ub(28, "LD", 3).
shifts_soft_ub(28, "SE", 3).
shifts_soft_ub(28, "SN", 3).
shifts_hard_ub(28, "LD", 4).
shifts_hard_ub(28, "SE", 4).
shifts_hard_ub(28, "SN", 4).
shifts_hard_lb(28, "E", 0).
shifts_hard_lb(28, "N", 0).
shifts_soft_lb(28, "E", 0).
shifts_soft_lb(28, "N", 0).
shifts_soft_ub(28, "E", 0).
shifts_soft_ub(28, "N", 0).
shifts_hard_ub(28, "E", 0).
shifts_hard_ub(28, "N", 0).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "LM", 0).
shifts_hard_lb(28, "EM", 0).
shifts_soft_lb(28, "EM", 0).
shifts_soft_ub(28, "EM", 0).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_hard_lb(29, "LD", 1).
shifts_hard_lb(29, "SE", 1).
shifts_hard_lb(29, "SN", 1).
shifts_soft_lb(29, "LD", 2).
shifts_soft_lb(29, "SE", 2).
shifts_soft_lb(29, "SN", 2).
shifts_soft_ub(29, "LD", 3).
shifts_soft_ub(29, "SE", 3).
shifts_soft_ub(29, "SN", 3).
shifts_hard_ub(29, "LD", 4).
shifts_hard_ub(29, "SE", 4).
shifts_hard_ub(29, "SN", 4).
shifts_hard_lb(29, "E", 0).
shifts_hard_lb(29, "N", 0).
shifts_soft_lb(29, "E", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_ub(29, "E", 0).
shifts_soft_ub(29, "N", 0).
shifts_hard_ub(29, "E", 0).
shifts_hard_ub(29, "N", 0).
shifts_hard_lb(29, "LM", 0).
shifts_soft_lb(29, "LM", 0).
shifts_soft_ub(29, "LM", 0).
shifts_hard_ub(29, "LM", 0).
shifts_hard_lb(29, "EM", 0).
shifts_soft_lb(29, "EM", 0).
shifts_soft_ub(29, "EM", 0).
shifts_hard_ub(29, "EM", 0).
% Staff 30:
shifts_hard_lb(30, "LD", 1).
shifts_hard_lb(30, "SE", 1).
shifts_hard_lb(30, "SN", 1).
shifts_soft_lb(30, "LD", 2).
shifts_soft_lb(30, "SE", 2).
shifts_soft_lb(30, "SN", 2).
shifts_soft_ub(30, "LD", 3).
shifts_soft_ub(30, "SE", 3).
shifts_soft_ub(30, "SN", 3).
shifts_hard_ub(30, "LD", 4).
shifts_hard_ub(30, "SE", 4).
shifts_hard_ub(30, "SN", 4).
shifts_hard_lb(30, "E", 0).
shifts_hard_lb(30, "N", 0).
shifts_soft_lb(30, "E", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_ub(30, "E", 0).
shifts_soft_ub(30, "N", 0).
shifts_hard_ub(30, "E", 0).
shifts_hard_ub(30, "N", 0).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "LM", 0).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
% Staff 31:
shifts_hard_lb(31, "LD", 0).
shifts_hard_lb(31, "SE", 0).
shifts_hard_lb(31, "SN", 0).
shifts_soft_lb(31, "LD", 0).
shifts_soft_lb(31, "SE", 0).
shifts_soft_lb(31, "SN", 0).
shifts_soft_ub(31, "LD", 0).
shifts_soft_ub(31, "SE", 0).
shifts_soft_ub(31, "SN", 0).
shifts_hard_ub(31, "LD", 0).
shifts_hard_ub(31, "SE", 0).
shifts_hard_ub(31, "SN", 0).
shifts_hard_lb(31, "E", 0).
shifts_hard_lb(31, "N", 0).
shifts_soft_lb(31, "E", 0).
shifts_soft_lb(31, "N", 0).
shifts_soft_ub(31, "E", 0).
shifts_soft_ub(31, "N", 0).
shifts_hard_ub(31, "E", 0).
shifts_hard_ub(31, "N", 0).
shifts_hard_lb(31, "LM", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "LM", 0).
shifts_hard_lb(31, "EM", 0).
shifts_soft_lb(31, "EM", 0).
shifts_soft_ub(31, "EM", 0).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 0).
shifts_hard_lb(32, "SE", 0).
shifts_hard_lb(32, "SN", 0).
shifts_soft_lb(32, "LD", 0).
shifts_soft_lb(32, "SE", 0).
shifts_soft_lb(32, "SN", 0).
shifts_soft_ub(32, "LD", 0).
shifts_soft_ub(32, "SE", 0).
shifts_soft_ub(32, "SN", 0).
shifts_hard_ub(32, "LD", 0).
shifts_hard_ub(32, "SE", 0).
shifts_hard_ub(32, "SN", 0).
shifts_hard_lb(32, "E", 0).
shifts_hard_lb(32, "N", 0).
shifts_soft_lb(32, "E", 0).
shifts_soft_lb(32, "N", 0).
shifts_soft_ub(32, "E", 0).
shifts_soft_ub(32, "N", 0).
shifts_hard_ub(32, "E", 0).
shifts_hard_ub(32, "N", 0).
shifts_hard_lb(32, "LM", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "LM", 0).
shifts_hard_lb(32, "EM", 0).
shifts_soft_lb(32, "EM", 0).
shifts_soft_ub(32, "EM", 0).
shifts_hard_ub(32, "EM", 0).
% Staff 33:
shifts_hard_lb(33, "LD", 0).
shifts_hard_lb(33, "SE", 0).
shifts_hard_lb(33, "SN", 0).
shifts_soft_lb(33, "LD", 0).
shifts_soft_lb(33, "SE", 0).
shifts_soft_lb(33, "SN", 0).
shifts_soft_ub(33, "LD", 0).
shifts_soft_ub(33, "SE", 0).
shifts_soft_ub(33, "SN", 0).
shifts_hard_ub(33, "LD", 0).
shifts_hard_ub(33, "SE", 0).
shifts_hard_ub(33, "SN", 0).
shifts_hard_lb(33, "E", 0).
shifts_hard_lb(33, "N", 0).
shifts_soft_lb(33, "E", 0).
shifts_soft_lb(33, "N", 0).
shifts_soft_ub(33, "E", 0).
shifts_soft_ub(33, "N", 0).
shifts_hard_ub(33, "E", 0).
shifts_hard_ub(33, "N", 0).
shifts_hard_lb(33, "LM", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "LM", 0).
shifts_hard_lb(33, "EM", 0).
shifts_soft_lb(33, "EM", 0).
shifts_soft_ub(33, "EM", 0).
shifts_hard_ub(33, "EM", 0).
% Staff 34:
shifts_hard_lb(34, "LD", 0).
shifts_hard_lb(34, "SE", 0).
shifts_hard_lb(34, "SN", 0).
shifts_soft_lb(34, "LD", 0).
shifts_soft_lb(34, "SE", 0).
shifts_soft_lb(34, "SN", 0).
shifts_soft_ub(34, "LD", 0).
shifts_soft_ub(34, "SE", 0).
shifts_soft_ub(34, "SN", 0).
shifts_hard_ub(34, "LD", 0).
shifts_hard_ub(34, "SE", 0).
shifts_hard_ub(34, "SN", 0).
shifts_hard_lb(34, "E", 0).
shifts_hard_lb(34, "N", 0).
shifts_soft_lb(34, "E", 0).
shifts_soft_lb(34, "N", 0).
shifts_soft_ub(34, "E", 0).
shifts_soft_ub(34, "N", 0).
shifts_hard_ub(34, "E", 0).
shifts_hard_ub(34, "N", 0).
shifts_hard_lb(34, "LM", 0).
shifts_soft_lb(34, "LM", 0).
shifts_soft_ub(34, "LM", 0).
shifts_hard_ub(34, "LM", 0).
shifts_hard_lb(34, "EM", 0).
shifts_soft_lb(34, "EM", 0).
shifts_soft_ub(34, "EM", 0).
shifts_hard_ub(34, "EM", 0).
% Staff 35:
shifts_hard_lb(35, "LD", 0).
shifts_hard_lb(35, "SE", 0).
shifts_hard_lb(35, "SN", 0).
shifts_soft_lb(35, "LD", 0).
shifts_soft_lb(35, "SE", 0).
shifts_soft_lb(35, "SN", 0).
shifts_soft_ub(35, "LD", 0).
shifts_soft_ub(35, "SE", 0).
shifts_soft_ub(35, "SN", 0).
shifts_hard_ub(35, "LD", 0).
shifts_hard_ub(35, "SE", 0).
shifts_hard_ub(35, "SN", 0).
shifts_hard_lb(35, "E", 0).
shifts_hard_lb(35, "N", 0).
shifts_soft_lb(35, "E", 0).
shifts_soft_lb(35, "N", 0).
shifts_soft_ub(35, "E", 0).
shifts_soft_ub(35, "N", 0).
shifts_hard_ub(35, "E", 0).
shifts_hard_ub(35, "N", 0).
shifts_hard_lb(35, "LM", 0).
shifts_soft_lb(35, "LM", 0).
shifts_soft_ub(35, "LM", 0).
shifts_hard_ub(35, "LM", 0).
shifts_hard_lb(35, "EM", 0).
shifts_soft_lb(35, "EM", 0).
shifts_soft_ub(35, "EM", 0).
shifts_hard_ub(35, "EM", 0).
% Staff 36:
shifts_hard_lb(36, "LD", 0).
shifts_hard_lb(36, "SE", 0).
shifts_hard_lb(36, "SN", 0).
shifts_soft_lb(36, "LD", 0).
shifts_soft_lb(36, "SE", 0).
shifts_soft_lb(36, "SN", 0).
shifts_soft_ub(36, "LD", 0).
shifts_soft_ub(36, "SE", 0).
shifts_soft_ub(36, "SN", 0).
shifts_hard_ub(36, "LD", 0).
shifts_hard_ub(36, "SE", 0).
shifts_hard_ub(36, "SN", 0).
shifts_hard_lb(36, "E", 0).
shifts_hard_lb(36, "N", 0).
shifts_soft_lb(36, "E", 0).
shifts_soft_lb(36, "N", 0).
shifts_soft_ub(36, "E", 0).
shifts_soft_ub(36, "N", 0).
shifts_hard_ub(36, "E", 0).
shifts_hard_ub(36, "N", 0).
shifts_hard_lb(36, "LM", 0).
shifts_soft_lb(36, "LM", 0).
shifts_soft_ub(36, "LM", 0).
shifts_hard_ub(36, "LM", 0).
shifts_hard_lb(36, "EM", 0).
shifts_soft_lb(36, "EM", 0).
shifts_soft_ub(36, "EM", 0).
shifts_hard_ub(36, "EM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
forbidden_pair(4, 7).
forbidden_pair(8, 11).
% Positive shifts ------------------------------------------
% Staff 6:
pos_shift(6, "日", ("N")).
pos_shift(6, "土", ("N")).
pos_shift(6, "祝", ("N")).
% Staff 18:
pos_shift(18, "日", ("S")).
pos_shift(18, "金", ("N")).
pos_shift(18, "土", ("J")).
% Staff 34:
pos_shift(34, "月", ("日")).
pos_shift(34, "火", ("日")).
pos_shift(34, "水", ("日")).
pos_shift(34, "金", ("日")).
% Staff 35:
pos_shift(35, "火", ("日")).
pos_shift(35, "水", ("日")).
pos_shift(35, "木", ("日")).
pos_shift(35, "金", ("日")).
% Staff 36:
pos_shift(36, "月", ("日")).
pos_shift(36, "火", ("日")).
pos_shift(36, "水", ("日")).
pos_shift(36, "木", ("日")).
pos_shift(36, "金", ("日")).
% Negative shifts ------------------------------------------
% Staff 2:
neg_shift(2, "日", ("日";"N";"J";"S")).
neg_shift(2, "土", ("日";"N";"J";"S")).
neg_shift(2, "祝", ("日";"N";"J";"S")).
% Staff 3:
neg_shift(3, "日", ("N")).
neg_shift(3, "土", ("N")).
% Staff 4:
neg_shift(4, "月", ("N")).
neg_shift(4, "火", ("N")).
neg_shift(4, "水", ("N")).
neg_shift(4, "木", ("N")).
neg_shift(4, "金", ("N")).
neg_shift(4, "祝", ("N")).
% Staff 6:
neg_shift(6, "月", ("N")).
neg_shift(6, "火", ("N")).
neg_shift(6, "水", ("N")).
neg_shift(6, "木", ("N")).
neg_shift(6, "金", ("N")).
% Staff 13:
neg_shift(13, "金", ("日";"N")).
% Staff 18:
neg_shift(18, "日", ("日";"N";"J")).
neg_shift(18, "金", ("日";"J";"S")).
neg_shift(18, "土", ("日";"N";"S")).
% Staff 34:
neg_shift(34, "日", ("日";"N";"J";"S";"P")).
neg_shift(34, "土", ("日";"N";"J";"S";"P")).
neg_shift(34, "祝", ("日";"N";"J";"S";"P")).
% Staff 35:
neg_shift(35, "日", ("日";"N";"J";"S";"P")).
neg_shift(35, "土", ("日";"N";"J";"S";"P")).
neg_shift(35, "祝", ("日";"N";"J";"S";"P")).
% Staff 36:
neg_shift(36, "日", ("日";"N";"J";"S";"P")).
neg_shift(36, "土", ("日";"N";"J";"S";"P")).
neg_shift(36, "祝", ("日";"N";"J";"S";"P")).

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
