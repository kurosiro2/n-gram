% Staffs ---------------------------------------------------
staff(1, "Jennifer Moran", "Chief Nurse", "29913", 0).
staff(2, "Wanda Williams", "Deputy Chief Nurse", "51294", 0).
staff(3, "Kevin Clay", "Deputy Chief Nurse", "09040", 0).
staff(4, "Holly Donovan", "Deputy Chief Nurse", "42922", 0).
staff(5, "Daniel Martin", "Midwife", "57085", 0).
staff(6, "Virginia Davis", "Midwife", "07200", 0).
staff(7, "Kevin Mcclain", "Midwife", "40623", 0).
staff(8, "Tonya Robles", "Midwife", "82441", 0).
staff(9, "David Krueger", "Midwife", "52555", 0).
staff(10, "Alyssa Thomas", "Midwife", "47576", 0).
staff(11, "Tammy Stout", "Midwife", "00536", 0).
staff(12, "Matthew Reyes", "Midwife", "36220", 0).
staff(13, "Christopher Nelson", "Midwife", "46506", 0).
staff(14, "Mrs. Laurie Morgan", "Midwife", "41200", 0).
staff(15, "Rebecca Miller", "Midwife", "41952", 0).
staff(16, "Mike Williamson", "Midwife", "84677", 0).
staff(17, "Terri Tyler", "Midwife", "93359", 0).
staff(18, "Amy Simmons", "Midwife", "22090", 0).
staff(19, "John Sherman", "Midwife", "67053", 0).
staff(20, "Richard Rush", "Midwife", "27660", 0).
staff(21, "Cheryl Clark", "Midwife", "47432", 0).
staff(22, "David Castro", "Midwife", "77434", 0).
staff(23, "Sean Sanchez", "Nurse", "10119", 0).
staff(24, "Debra Brown", "Nurse", "15037", 0).
staff(25, "David Stafford", "Midwife", "35066", 0).
staff(26, "James Carter", "Midwife", "A088940", 0).
staff(27, "Jeremy Richmond", "Midwife", "80713", 0).
staff(28, "Easton Miller", "Midwife", "A639676", 0).
staff(29, "Isaiah Martinez", "Midwife", "A132815", 0).
staff(30, "Christopher Miller", "Midwife", "A282601", 0).
staff(31, "Easton Hall", "Midwife", "A677608", 0).
staff(32, "Thomas Ramirez", "Midwife", "A408829", 0).
staff(33, "Adrian Thompson", "Midwife", "A711959", 0).
staff(34, "Martin Stanton", "Midwife", "09568", 0).
staff(35, "Melissa Mckenzie", "Midwife", "56072", 0).
% Groups ---------------------------------------------------
group("All").
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
group("All", 28).
group("All", 29).
group("All", 30).
group("All", 31).
group("All", 32).
group("All", 33).
group("All", 34).
group("All", 35).
group("代行").
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("代行", 6).
group("代行", 7).
group("代行", 8).
group("代行", 34).
group("代行", 35).
group("Leaders").
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 6).
group("Leaders", 7).
group("Leaders", 8).
group("Leaders", 9).
group("Leaders", 10).
group("Leaders", 34).
group("Leaders", 35).
group("メンバー").
group("メンバー", 3).
group("メンバー", 11).
group("メンバー", 12).
group("メンバー", 13).
group("メンバー", 15).
group("メンバー", 16).
group("メンバー", 17).
group("メンバー", 18).
group("メンバー", 19).
group("メンバー", 20).
group("メンバー", 21).
group("メンバー", 22).
group("メンバー", 23).
group("メンバー", 24).
group("メンバー", 25).
group("メンバー", 26).
group("メンバー", 28).
group("メンバー", 29).
group("メンバー", 30).
group("メンバー", 31).
group("メンバー", 32).
group("メンバー", 33).
group("Newcomers").
group("Newcomers", 3).
group("Newcomers", 19).
group("Newcomers", 23).
group("Newcomers", 24).
group("Newcomers", 28).
group("Newcomers", 29).
group("Newcomers", 30).
group("Newcomers", 31).
group("Newcomers", 32).
group("Newcomers", 33).
group("Night").
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
group("Night", 28).
group("Night", 29).
group("Night", 30).
group("Night", 31).
group("Night", 32).
group("Night", 33).
group("Night", 34).
group("Night", 35).
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
patterns_hard_lb("○○", 3).
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
staff_dweek_soft_lb("All", "D", "Weekday", 8).
staff_dweek_soft_lb("All", "D", "Weekend", 3).
staff_dweek_soft_lb("All", "D", "Holiday", 3).
staff_dweek_soft_ub("All", "D", "Weekday", 10).
staff_dweek_soft_ub("All", "D", "Weekend", 4).
staff_dweek_soft_ub("All", "D", "Holiday", 4).
staff_dweek_hard_ub("All", "D", "Weekday", 15).
staff_dweek_hard_ub("All", "D", "Weekend", 5).
staff_dweek_hard_ub("All", "D", "Holiday", 5).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 4).
staff_dweek_hard_lb("All", "LD", "Weekend", 4).
staff_dweek_hard_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_lb("All", "LD", "Weekday", 4).
staff_dweek_soft_lb("All", "LD", "Weekend", 4).
staff_dweek_soft_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
staff_dweek_hard_ub("All", "LD", "Weekday", 4).
staff_dweek_hard_ub("All", "LD", "Weekend", 4).
staff_dweek_hard_ub("All", "LD", "Holiday", 4).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 4).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 4).
staff_dweek_hard_lb("All", "SE", "Weekend", 4).
staff_dweek_hard_lb("All", "SN", "Weekend", 4).
staff_dweek_hard_lb("All", "SE", "Holiday", 4).
staff_dweek_hard_lb("All", "SN", "Holiday", 4).
staff_dweek_soft_lb("All", "SE", "Weekday", 4).
staff_dweek_soft_lb("All", "SN", "Weekday", 4).
staff_dweek_soft_lb("All", "SE", "Weekend", 4).
staff_dweek_soft_lb("All", "SN", "Weekend", 4).
staff_dweek_soft_lb("All", "SE", "Holiday", 4).
staff_dweek_soft_lb("All", "SN", "Holiday", 4).
staff_dweek_soft_ub("All", "SE", "Weekday", 4).
staff_dweek_soft_ub("All", "SN", "Weekday", 4).
staff_dweek_soft_ub("All", "SE", "Weekend", 4).
staff_dweek_soft_ub("All", "SN", "Weekend", 4).
staff_dweek_soft_ub("All", "SE", "Holiday", 4).
staff_dweek_soft_ub("All", "SN", "Holiday", 4).
staff_dweek_hard_ub("All", "SE", "Weekday", 4).
staff_dweek_hard_ub("All", "SN", "Weekday", 4).
staff_dweek_hard_ub("All", "SE", "Weekend", 4).
staff_dweek_hard_ub("All", "SN", "Weekend", 4).
staff_dweek_hard_ub("All", "SE", "Holiday", 4).
staff_dweek_hard_ub("All", "SN", "Holiday", 4).
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
shifts("SE+E").
shifts("SE+E", "SE").
shifts("SE+E", "E").
staff_dweek_hard_lb("All", "SE+E", "Weekday", 4).
staff_dweek_hard_lb("All", "SE+E", "Weekend", 4).
staff_dweek_hard_lb("All", "SE+E", "Holiday", 4).
staff_dweek_soft_lb("All", "SE+E", "Weekday", 4).
staff_dweek_soft_lb("All", "SE+E", "Weekend", 4).
staff_dweek_soft_lb("All", "SE+E", "Holiday", 4).
staff_dweek_soft_ub("All", "SE+E", "Weekday", 4).
staff_dweek_soft_ub("All", "SE+E", "Weekend", 4).
staff_dweek_soft_ub("All", "SE+E", "Holiday", 4).
staff_dweek_hard_ub("All", "SE+E", "Weekday", 4).
staff_dweek_hard_ub("All", "SE+E", "Weekend", 4).
staff_dweek_hard_ub("All", "SE+E", "Holiday", 4).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 0).
staff_dweek_hard_lb("All", "LM", "Weekend", 0).
staff_dweek_hard_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_lb("All", "LM", "Weekday", 0).
staff_dweek_soft_lb("All", "LM", "Weekend", 0).
staff_dweek_soft_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_ub("All", "LM", "Weekday", 1).
staff_dweek_soft_ub("All", "LM", "Weekend", 0).
staff_dweek_soft_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
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
staff_dweek_soft_lb("Leaders", "LD", "Weekday", 1).
staff_dweek_soft_lb("Leaders", "SE", "Weekday", 1).
staff_dweek_soft_lb("Leaders", "SN", "Weekday", 1).
staff_dweek_soft_lb("Leaders", "LD", "Weekend", 1).
staff_dweek_soft_lb("Leaders", "SE", "Weekend", 1).
staff_dweek_soft_lb("Leaders", "SN", "Weekend", 1).
staff_dweek_soft_lb("Leaders", "LD", "Holiday", 1).
staff_dweek_soft_lb("Leaders", "SE", "Holiday", 1).
staff_dweek_soft_lb("Leaders", "SN", "Holiday", 1).
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_lb("代行", "D", "Weekday", 1).
staff_dweek_soft_lb("代行", "D", "Weekend", 0).
staff_dweek_soft_lb("代行", "D", "Holiday", 0).
staff_dweek_soft_ub("代行", "D", "Weekday", 1).
staff_dweek_soft_ub("代行", "D", "Weekend", 0).
staff_dweek_soft_ub("代行", "D", "Holiday", 0).
staff_dweek_hard_lb("メンバー", "LD", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "SE", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "SN", "Weekday", 1).
staff_dweek_hard_lb("メンバー", "LD", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "SE", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "SN", "Weekend", 1).
staff_dweek_hard_lb("メンバー", "LD", "Holiday", 1).
staff_dweek_hard_lb("メンバー", "SE", "Holiday", 1).
staff_dweek_hard_lb("メンバー", "SN", "Holiday", 1).
staff_dweek_soft_lb("メンバー", "LD", "Weekday", 2).
staff_dweek_soft_lb("メンバー", "SE", "Weekday", 2).
staff_dweek_soft_lb("メンバー", "SN", "Weekday", 2).
staff_dweek_soft_lb("メンバー", "LD", "Weekend", 2).
staff_dweek_soft_lb("メンバー", "SE", "Weekend", 2).
staff_dweek_soft_lb("メンバー", "SN", "Weekend", 2).
staff_dweek_soft_lb("メンバー", "LD", "Holiday", 2).
staff_dweek_soft_lb("メンバー", "SE", "Holiday", 2).
staff_dweek_soft_lb("メンバー", "SN", "Holiday", 2).
staff_dweek_soft_ub("メンバー", "LD", "Weekday", 3).
staff_dweek_soft_ub("メンバー", "SE", "Weekday", 3).
staff_dweek_soft_ub("メンバー", "SN", "Weekday", 3).
staff_dweek_soft_ub("メンバー", "LD", "Weekend", 3).
staff_dweek_soft_ub("メンバー", "SE", "Weekend", 3).
staff_dweek_soft_ub("メンバー", "SN", "Weekend", 3).
staff_dweek_soft_ub("メンバー", "LD", "Holiday", 3).
staff_dweek_soft_ub("メンバー", "SE", "Holiday", 3).
staff_dweek_soft_ub("メンバー", "SN", "Holiday", 3).
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
vertical_constraint_type("All", "SE+E", "Staffs").
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
% Staff 3:
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 2).
shifts_soft_ub(3, "SE", 2).
shifts_soft_ub(3, "SN", 2).
% Staff 4:
shifts_soft_lb(4, "LD", 3).
shifts_soft_lb(4, "SE", 3).
shifts_soft_lb(4, "SN", 3).
shifts_soft_ub(4, "LD", 4).
shifts_soft_ub(4, "SE", 4).
shifts_soft_ub(4, "SN", 4).
shifts_soft_lb(4, "LM", 1).
% Staff 5:
shifts_soft_lb(5, "LD", 3).
shifts_soft_lb(5, "SE", 3).
shifts_soft_lb(5, "SN", 3).
shifts_soft_ub(5, "LD", 3).
shifts_soft_ub(5, "SE", 3).
shifts_soft_ub(5, "SN", 3).
% Staff 6:
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 5).
shifts_soft_ub(6, "SE", 5).
shifts_soft_ub(6, "SN", 5).
shifts_hard_ub(6, "LD", 6).
shifts_hard_ub(6, "SE", 6).
shifts_hard_ub(6, "SN", 6).
shifts_soft_lb(6, "LM", 1).
% Staff 7:
shifts_soft_lb(7, "LD", 4).
shifts_soft_lb(7, "SE", 4).
shifts_soft_lb(7, "SN", 4).
shifts_soft_ub(7, "LD", 5).
shifts_soft_ub(7, "SE", 5).
shifts_soft_ub(7, "SN", 5).
shifts_hard_ub(7, "LD", 6).
shifts_hard_ub(7, "SE", 6).
shifts_hard_ub(7, "SN", 6).
shifts_soft_lb(7, "LM", 1).
% Staff 8:
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_soft_lb(8, "LM", 1).
% Staff 9:
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_soft_lb(9, "LM", 1).
% Staff 10:
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_soft_lb(10, "LM", 1).
% Staff 11:
shifts_soft_lb(11, "LD", 1).
shifts_soft_lb(11, "SE", 1).
shifts_soft_lb(11, "SN", 1).
shifts_soft_ub(11, "LD", 3).
shifts_soft_ub(11, "SE", 3).
shifts_soft_ub(11, "SN", 3).
shifts_hard_ub(11, "LD", 3).
shifts_hard_ub(11, "SE", 3).
shifts_hard_ub(11, "SN", 3).
shifts_soft_lb(11, "LM", 1).
% Staff 12:
shifts_soft_lb(12, "LD", 3).
shifts_soft_lb(12, "SE", 3).
shifts_soft_lb(12, "SN", 3).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_hard_ub(12, "LD", 5).
shifts_hard_ub(12, "SE", 5).
shifts_hard_ub(12, "SN", 5).
shifts_soft_lb(12, "LM", 1).
% Staff 13:
shifts_soft_lb(13, "LD", 3).
shifts_soft_lb(13, "SE", 3).
shifts_soft_lb(13, "SN", 3).
shifts_soft_ub(13, "LD", 4).
shifts_soft_ub(13, "SE", 4).
shifts_soft_ub(13, "SN", 4).
shifts_hard_ub(13, "LD", 4).
shifts_hard_ub(13, "SE", 4).
shifts_hard_ub(13, "SN", 4).
% Staff 14:
% Staff 15:
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_soft_lb(15, "LM", 1).
% Staff 16:
shifts_soft_lb(16, "LD", 4).
shifts_soft_lb(16, "SE", 4).
shifts_soft_lb(16, "SN", 4).
shifts_soft_ub(16, "LD", 5).
shifts_soft_ub(16, "SE", 5).
shifts_soft_ub(16, "SN", 5).
shifts_hard_ub(16, "LD", 6).
shifts_hard_ub(16, "SE", 6).
shifts_hard_ub(16, "SN", 6).
shifts_soft_lb(16, "LM", 1).
% Staff 17:
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 5).
shifts_soft_ub(17, "SE", 5).
shifts_soft_ub(17, "SN", 5).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_soft_lb(17, "LM", 1).
% Staff 18:
shifts_soft_lb(18, "LD", 4).
shifts_soft_lb(18, "SE", 4).
shifts_soft_lb(18, "SN", 4).
shifts_soft_ub(18, "LD", 5).
shifts_soft_ub(18, "SE", 5).
shifts_soft_ub(18, "SN", 5).
shifts_hard_ub(18, "LD", 6).
shifts_hard_ub(18, "SE", 6).
shifts_hard_ub(18, "SN", 6).
shifts_soft_lb(18, "LM", 1).
% Staff 19:
shifts_soft_lb(19, "LD", 2).
shifts_soft_lb(19, "SE", 2).
shifts_soft_lb(19, "SN", 2).
shifts_soft_ub(19, "LD", 2).
shifts_soft_ub(19, "SE", 2).
shifts_soft_ub(19, "SN", 2).
shifts_hard_ub(19, "LD", 2).
shifts_hard_ub(19, "SE", 2).
shifts_hard_ub(19, "SN", 2).
% Staff 20:
shifts_soft_lb(20, "LD", 2).
shifts_soft_lb(20, "SE", 2).
shifts_soft_lb(20, "SN", 2).
shifts_soft_ub(20, "LD", 2).
shifts_soft_ub(20, "SE", 2).
shifts_soft_ub(20, "SN", 2).
shifts_hard_ub(20, "LD", 2).
shifts_hard_ub(20, "SE", 2).
shifts_hard_ub(20, "SN", 2).
% Staff 21:
shifts_soft_lb(21, "LD", 4).
shifts_soft_lb(21, "SE", 4).
shifts_soft_lb(21, "SN", 4).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_soft_lb(21, "LM", 1).
% Staff 22:
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 5).
shifts_soft_ub(22, "SE", 5).
shifts_soft_ub(22, "SN", 5).
shifts_hard_ub(22, "LD", 6).
shifts_hard_ub(22, "SE", 6).
shifts_hard_ub(22, "SN", 6).
shifts_soft_lb(22, "LM", 1).
% Staff 23:
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
shifts_soft_lb(23, "LM", 1).
% Staff 24:
shifts_soft_lb(24, "LD", 4).
shifts_soft_lb(24, "SE", 4).
shifts_soft_lb(24, "SN", 4).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_hard_ub(24, "LD", 6).
shifts_hard_ub(24, "SE", 6).
shifts_hard_ub(24, "SN", 6).
shifts_soft_lb(24, "LM", 1).
% Staff 25:
shifts_soft_lb(25, "LD", 4).
shifts_soft_lb(25, "SE", 4).
shifts_soft_lb(25, "SN", 4).
shifts_soft_ub(25, "LD", 5).
shifts_soft_ub(25, "SE", 5).
shifts_soft_ub(25, "SN", 5).
shifts_hard_ub(25, "LD", 6).
shifts_hard_ub(25, "SE", 6).
shifts_hard_ub(25, "SN", 6).
shifts_soft_lb(25, "LM", 1).
% Staff 26:
shifts_soft_lb(26, "LD", 2).
shifts_soft_lb(26, "SE", 2).
shifts_soft_lb(26, "SN", 2).
shifts_soft_ub(26, "LD", 2).
shifts_soft_ub(26, "SE", 2).
shifts_soft_ub(26, "SN", 2).
shifts_hard_ub(26, "LD", 3).
shifts_hard_ub(26, "SE", 3).
shifts_hard_ub(26, "SN", 3).
shifts_soft_lb(26, "LM", 1).
% Staff 27:
shifts_soft_lb(27, "LM", 1).
% Staff 28:
shifts_soft_lb(28, "LD", 2).
shifts_soft_lb(28, "SE", 2).
shifts_soft_lb(28, "SN", 2).
shifts_soft_ub(28, "LD", 3).
shifts_soft_ub(28, "SE", 3).
shifts_soft_ub(28, "SN", 3).
shifts_hard_ub(28, "LD", 4).
shifts_hard_ub(28, "SE", 4).
shifts_hard_ub(28, "SN", 4).
shifts_soft_lb(28, "LM", 1).
% Staff 29:
shifts_soft_lb(29, "LD", 2).
shifts_soft_lb(29, "SE", 2).
shifts_soft_lb(29, "SN", 2).
shifts_soft_ub(29, "LD", 3).
shifts_soft_ub(29, "SE", 3).
shifts_soft_ub(29, "SN", 3).
shifts_hard_ub(29, "LD", 4).
shifts_hard_ub(29, "SE", 4).
shifts_hard_ub(29, "SN", 4).
shifts_soft_lb(29, "LM", 1).
% Staff 30:
shifts_soft_lb(30, "LD", 2).
shifts_soft_lb(30, "SE", 2).
shifts_soft_lb(30, "SN", 2).
shifts_soft_ub(30, "LD", 3).
shifts_soft_ub(30, "SE", 3).
shifts_soft_ub(30, "SN", 3).
shifts_hard_ub(30, "LD", 4).
shifts_hard_ub(30, "SE", 4).
shifts_hard_ub(30, "SN", 4).
shifts_soft_lb(30, "LM", 1).
% Staff 31:
shifts_soft_lb(31, "LD", 2).
shifts_soft_lb(31, "SE", 2).
shifts_soft_lb(31, "SN", 2).
shifts_soft_ub(31, "LD", 3).
shifts_soft_ub(31, "SE", 3).
shifts_soft_ub(31, "SN", 3).
shifts_hard_ub(31, "LD", 4).
shifts_hard_ub(31, "SE", 4).
shifts_hard_ub(31, "SN", 4).
shifts_soft_lb(31, "LM", 1).
% Staff 32:
shifts_soft_lb(32, "LD", 2).
shifts_soft_lb(32, "SE", 2).
shifts_soft_lb(32, "SN", 2).
shifts_soft_ub(32, "LD", 3).
shifts_soft_ub(32, "SE", 3).
shifts_soft_ub(32, "SN", 3).
shifts_hard_ub(32, "LD", 4).
shifts_hard_ub(32, "SE", 4).
shifts_hard_ub(32, "SN", 4).
shifts_soft_lb(32, "LM", 1).
% Staff 33:
shifts_soft_lb(33, "LD", 2).
shifts_soft_lb(33, "SE", 2).
shifts_soft_lb(33, "SN", 2).
shifts_soft_ub(33, "LD", 3).
shifts_soft_ub(33, "SE", 3).
shifts_soft_ub(33, "SN", 3).
shifts_hard_ub(33, "LD", 4).
shifts_hard_ub(33, "SE", 4).
shifts_hard_ub(33, "SN", 4).
shifts_soft_lb(33, "LM", 1).
% Staff 34:
shifts_soft_lb(34, "LD", 2).
shifts_soft_lb(34, "SE", 2).
shifts_soft_lb(34, "SN", 2).
shifts_soft_ub(34, "LD", 2).
shifts_soft_ub(34, "SE", 2).
shifts_soft_ub(34, "SN", 2).
shifts_hard_ub(34, "LD", 2).
shifts_hard_ub(34, "SE", 2).
shifts_hard_ub(34, "SN", 2).
shifts_soft_lb(34, "LM", 0).
% Staff 35:
shifts_soft_lb(35, "LD", 2).
shifts_soft_lb(35, "SE", 2).
shifts_soft_lb(35, "SN", 2).
shifts_soft_ub(35, "LD", 2).
shifts_soft_ub(35, "SE", 2).
shifts_soft_ub(35, "SN", 2).
shifts_hard_ub(35, "LD", 2).
shifts_hard_ub(35, "SE", 2).
shifts_hard_ub(35, "SN", 2).
shifts_soft_lb(35, "LM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
recommended_pair(18, 28).
recommended_pair(17, 29).
recommended_pair(16, 30).
recommended_pair(15, 31).
recommended_pair(11, 32).
recommended_pair(12, 33).
forbidden_pair(3, 5).
forbidden_pair(5, 9).
forbidden_pair(6, 9).
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
