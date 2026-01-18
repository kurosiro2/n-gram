% Staffs ---------------------------------------------------
staff(1, "Miranda Simmons", "Chief Nurse", "22006", 0).
staff(2, "Kevin Clay", "Deputy Chief Nurse", "09040", 5).
staff(3, "Jessica Patel", "Deputy Chief Nurse", "39245", 5).
staff(4, "Mrs. Rachel Price MD", "Nurse", "81119", 5).
staff(5, "Jaclyn Robinson", "Nurse", "99412", 5).
staff(6, "Lauren Salinas", "Nurse", "74398", 5).
staff(7, "David Galloway", "Nurse", "62116", 2).
staff(8, "Jeanne Rodriguez", "Nurse", "58489", 2).
staff(9, "Nathaniel Morse", "Nurse", "28978", 5).
staff(10, "Christina Jacobs", "Nurse", "34921", 2).
staff(11, "Bradley Mccann", "Nurse", "47931", 3).
staff(12, "Joshua Fleming", "Nurse", "25016", 0).
staff(13, "Francisco Escobar", "Nurse", "84362", 3).
staff(14, "William Vasquez", "Nurse", "82336", 3).
staff(15, "Angela Garcia", "Nurse", "20362", 3).
staff(16, "Jason Drake", "Nurse", "46740", 3).
staff(17, "Krista Jones", "Nurse", "52677", 3).
staff(18, "Lauren Wright", "Nurse", "36258", 2).
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
group("All", 13).
group("All", 14).
group("All", 15).
group("All", 16).
group("All", 17).
group("All", 18).
group("Heads").
group("Heads", 1).
group("Seniors").
group("Seniors", 2).
group("Seniors", 3).
group("Seniors", 4).
group("Seniors", 5).
group("Seniors", 6).
group("Seniors", 7).
group("Seniors", 10).
group("Mid-levels").
group("Mid-levels", 11).
group("Mid-levels", 13).
group("Mid-levels", 14).
group("Mid-levels", 15).
group("Newcomers").
group("Newcomers", 9).
group("Newcomers", 16).
group("Newcomers", 17).
group("Newcomers", 18).
group("Night").
group("Night", 2).
group("Night", 3).
group("Night", 4).
group("Night", 5).
group("Night", 6).
group("Night", 7).
group("Night", 9).
group("Night", 10).
group("Night", 11).
group("Night", 13).
group("Night", 14).
group("Night", 15).
group("Night", 16).
group("Night", 17).
group("Night", 18).
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
staff_dweek_hard_lb("All", "D", "Weekday", 0).
staff_dweek_hard_lb("All", "D", "Weekend", 0).
staff_dweek_hard_lb("All", "D", "Holiday", 0).
staff_dweek_soft_lb("All", "D", "Weekday", 3).
staff_dweek_soft_lb("All", "D", "Weekend", 0).
staff_dweek_soft_lb("All", "D", "Holiday", 0).
staff_dweek_soft_ub("All", "D", "Weekday", 7).
staff_dweek_soft_ub("All", "D", "Weekend", 0).
staff_dweek_soft_ub("All", "D", "Holiday", 0).
staff_dweek_hard_ub("All", "D", "Weekday", 7).
staff_dweek_hard_ub("All", "D", "Weekend", 0).
staff_dweek_hard_ub("All", "D", "Holiday", 0).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 2).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 2).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 2).
staff_dweek_hard_lb("All", "LD", "Weekend", 2).
staff_dweek_hard_lb("All", "SE", "Weekend", 2).
staff_dweek_hard_lb("All", "SN", "Weekend", 2).
staff_dweek_hard_lb("All", "LD", "Holiday", 2).
staff_dweek_hard_lb("All", "SE", "Holiday", 2).
staff_dweek_hard_lb("All", "SN", "Holiday", 2).
staff_dweek_soft_lb("All", "LD", "Weekday", 2).
staff_dweek_soft_lb("All", "SE", "Weekday", 2).
staff_dweek_soft_lb("All", "SN", "Weekday", 2).
staff_dweek_soft_lb("All", "LD", "Weekend", 2).
staff_dweek_soft_lb("All", "SE", "Weekend", 2).
staff_dweek_soft_lb("All", "SN", "Weekend", 2).
staff_dweek_soft_lb("All", "LD", "Holiday", 2).
staff_dweek_soft_lb("All", "SE", "Holiday", 2).
staff_dweek_soft_lb("All", "SN", "Holiday", 2).
staff_dweek_soft_ub("All", "LD", "Weekday", 2).
staff_dweek_soft_ub("All", "SE", "Weekday", 2).
staff_dweek_soft_ub("All", "SN", "Weekday", 2).
staff_dweek_soft_ub("All", "LD", "Weekend", 2).
staff_dweek_soft_ub("All", "SE", "Weekend", 2).
staff_dweek_soft_ub("All", "SN", "Weekend", 2).
staff_dweek_soft_ub("All", "LD", "Holiday", 2).
staff_dweek_soft_ub("All", "SE", "Holiday", 2).
staff_dweek_soft_ub("All", "SN", "Holiday", 2).
staff_dweek_hard_ub("All", "LD", "Weekday", 3).
staff_dweek_hard_ub("All", "SE", "Weekday", 3).
staff_dweek_hard_ub("All", "SN", "Weekday", 3).
staff_dweek_hard_ub("All", "LD", "Weekend", 2).
staff_dweek_hard_ub("All", "SE", "Weekend", 2).
staff_dweek_hard_ub("All", "SN", "Weekend", 2).
staff_dweek_hard_ub("All", "LD", "Holiday", 2).
staff_dweek_hard_ub("All", "SE", "Holiday", 2).
staff_dweek_hard_ub("All", "SN", "Holiday", 2).
shifts("E").
shifts("E", "E").
staff_dweek_soft_ub("All", "E", "Weekday", 0).
shifts("N").
shifts("N", "N").
staff_dweek_soft_ub("All", "N", "Weekday", 0).
staff_dweek_soft_ub("All", "E", "Weekend", 0).
staff_dweek_soft_ub("All", "N", "Weekend", 0).
staff_dweek_soft_ub("All", "E", "Holiday", 0).
staff_dweek_soft_ub("All", "N", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_ub("All", "LM", "Weekday", 0).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shifts("LD+LM+E").
shifts("LD+LM+E", "LD").
shifts("LD+LM+E", "LM").
shifts("LD+LM+E", "E").
staff_dweek_soft_lb("All", "LD+LM+E", "Weekday", 2).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekend", 2).
staff_dweek_soft_lb("All", "LD+LM+E", "Holiday", 2).
staff_dweek_hard_lb("Seniors", "LD", "Weekday", 0).
staff_dweek_hard_lb("Seniors", "SE", "Weekday", 0).
staff_dweek_hard_lb("Seniors", "SN", "Weekday", 0).
staff_dweek_hard_lb("Seniors", "LD", "Weekend", 0).
staff_dweek_hard_lb("Seniors", "SE", "Weekend", 0).
staff_dweek_hard_lb("Seniors", "SN", "Weekend", 0).
staff_dweek_hard_lb("Seniors", "LD", "Holiday", 0).
staff_dweek_hard_lb("Seniors", "SE", "Holiday", 0).
staff_dweek_hard_lb("Seniors", "SN", "Holiday", 0).
staff_dweek_soft_lb("Seniors", "LD", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "SE", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "SN", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "LD", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "SE", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "SN", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "LD", "Holiday", 1).
staff_dweek_soft_lb("Seniors", "SE", "Holiday", 1).
staff_dweek_soft_lb("Seniors", "SN", "Holiday", 1).
staff_dweek_soft_ub("Seniors", "LD", "Weekday", 2).
staff_dweek_soft_ub("Seniors", "SE", "Weekday", 2).
staff_dweek_soft_ub("Seniors", "SN", "Weekday", 2).
staff_dweek_soft_ub("Seniors", "LD", "Weekend", 2).
staff_dweek_soft_ub("Seniors", "SE", "Weekend", 2).
staff_dweek_soft_ub("Seniors", "SN", "Weekend", 2).
staff_dweek_soft_ub("Seniors", "LD", "Holiday", 2).
staff_dweek_soft_ub("Seniors", "SE", "Holiday", 2).
staff_dweek_soft_ub("Seniors", "SN", "Holiday", 2).
staff_dweek_hard_ub("Seniors", "LD", "Weekday", 3).
staff_dweek_hard_ub("Seniors", "SE", "Weekday", 3).
staff_dweek_hard_ub("Seniors", "SN", "Weekday", 3).
staff_dweek_hard_ub("Seniors", "LD", "Weekend", 3).
staff_dweek_hard_ub("Seniors", "SE", "Weekend", 3).
staff_dweek_hard_ub("Seniors", "SN", "Weekend", 3).
staff_dweek_hard_ub("Seniors", "LD", "Holiday", 3).
staff_dweek_hard_ub("Seniors", "SE", "Holiday", 3).
staff_dweek_hard_ub("Seniors", "SN", "Holiday", 3).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
staff_dweek_hard_lb("Mid-levels", "LD", "Weekday", 0).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekday", 0).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekday", 0).
staff_dweek_hard_lb("Mid-levels", "LD", "Weekend", 0).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekend", 0).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekend", 0).
staff_dweek_hard_lb("Mid-levels", "LD", "Holiday", 0).
staff_dweek_hard_lb("Mid-levels", "SE", "Holiday", 0).
staff_dweek_hard_lb("Mid-levels", "SN", "Holiday", 0).
staff_dweek_soft_lb("Mid-levels", "LD", "Weekday", 1).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekday", 1).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekday", 1).
staff_dweek_soft_lb("Mid-levels", "LD", "Weekend", 1).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekend", 1).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekend", 1).
staff_dweek_soft_lb("Mid-levels", "LD", "Holiday", 1).
staff_dweek_soft_lb("Mid-levels", "SE", "Holiday", 1).
staff_dweek_soft_lb("Mid-levels", "SN", "Holiday", 1).
staff_dweek_soft_ub("Mid-levels", "LD", "Weekday", 1).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekday", 1).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekday", 1).
staff_dweek_soft_ub("Mid-levels", "LD", "Weekend", 1).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekend", 1).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekend", 1).
staff_dweek_soft_ub("Mid-levels", "LD", "Holiday", 1).
staff_dweek_soft_ub("Mid-levels", "SE", "Holiday", 1).
staff_dweek_soft_ub("Mid-levels", "SN", "Holiday", 1).
staff_dweek_hard_ub("Mid-levels", "LD", "Weekday", 1).
staff_dweek_hard_ub("Mid-levels", "SE", "Weekday", 1).
staff_dweek_hard_ub("Mid-levels", "SN", "Weekday", 1).
staff_dweek_hard_ub("Mid-levels", "LD", "Weekend", 1).
staff_dweek_hard_ub("Mid-levels", "SE", "Weekend", 1).
staff_dweek_hard_ub("Mid-levels", "SN", "Weekend", 1).
staff_dweek_hard_ub("Mid-levels", "LD", "Holiday", 1).
staff_dweek_hard_ub("Mid-levels", "SE", "Holiday", 1).
staff_dweek_hard_ub("Mid-levels", "SN", "Holiday", 1).
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Holiday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 0).
staff_dweek_soft_lb("Newcomers", "SE", "Weekday", 0).
staff_dweek_soft_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_soft_lb("Newcomers", "SE", "Weekend", 0).
staff_dweek_soft_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_soft_lb("Newcomers", "SE", "Holiday", 0).
staff_dweek_soft_lb("Newcomers", "SN", "Holiday", 0).
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
vertical_constraint_type("All", "LD+LM+E", "Staffs").
vertical_constraint_type("Seniors", "LD", "Staffs").
vertical_constraint_type("Seniors", "SE", "Staffs").
vertical_constraint_type("Seniors", "SN", "Staffs").
vertical_constraint_type("Heads", "D", "Staffs").
vertical_constraint_type("Mid-levels", "LD", "Staffs").
vertical_constraint_type("Mid-levels", "SE", "Staffs").
vertical_constraint_type("Mid-levels", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "LD", 0).
shifts_soft_lb(1, "SE", 0).
shifts_soft_lb(1, "SN", 0).
shifts_soft_ub(1, "LD", 0).
shifts_soft_ub(1, "SE", 0).
shifts_soft_ub(1, "SN", 0).
shifts_hard_ub(1, "LD", 0).
shifts_hard_ub(1, "SE", 0).
shifts_hard_ub(1, "SN", 0).
shifts_hard_lb(1, "E", 0).
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
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
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
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 2).
shifts_soft_ub(3, "SE", 2).
shifts_soft_ub(3, "SN", 2).
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
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_soft_lb(4, "LD", 2).
shifts_soft_lb(4, "SE", 2).
shifts_soft_lb(4, "SN", 2).
shifts_soft_ub(4, "LD", 4).
shifts_soft_ub(4, "SE", 4).
shifts_soft_ub(4, "SN", 4).
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
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 0).
shifts_soft_lb(5, "SE", 0).
shifts_soft_lb(5, "SN", 0).
shifts_soft_ub(5, "LD", 0).
shifts_soft_ub(5, "SE", 0).
shifts_soft_ub(5, "SN", 0).
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
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 2).
shifts_soft_lb(6, "SE", 2).
shifts_soft_lb(6, "SN", 2).
shifts_soft_ub(6, "LD", 4).
shifts_soft_ub(6, "SE", 4).
shifts_soft_ub(6, "SN", 4).
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
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 2).
shifts_soft_lb(7, "SE", 2).
shifts_soft_lb(7, "SN", 2).
shifts_soft_ub(7, "LD", 4).
shifts_soft_ub(7, "SE", 4).
shifts_soft_ub(7, "SN", 4).
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
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 0).
shifts_soft_lb(8, "SE", 0).
shifts_soft_lb(8, "SN", 0).
shifts_soft_ub(8, "LD", 0).
shifts_soft_ub(8, "SE", 0).
shifts_soft_ub(8, "SN", 0).
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
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 2).
shifts_soft_lb(9, "SE", 2).
shifts_soft_lb(9, "SN", 2).
shifts_soft_ub(9, "LD", 4).
shifts_soft_ub(9, "SE", 4).
shifts_soft_ub(9, "SN", 4).
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
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 2).
shifts_soft_lb(10, "SE", 2).
shifts_soft_lb(10, "SN", 2).
shifts_soft_ub(10, "LD", 4).
shifts_soft_ub(10, "SE", 4).
shifts_soft_ub(10, "SN", 4).
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
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 2).
shifts_soft_lb(11, "SE", 2).
shifts_soft_lb(11, "SN", 2).
shifts_soft_ub(11, "LD", 4).
shifts_soft_ub(11, "SE", 4).
shifts_soft_ub(11, "SN", 4).
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
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_soft_lb(12, "LD", 0).
shifts_soft_lb(12, "SE", 0).
shifts_soft_lb(12, "SN", 0).
shifts_soft_ub(12, "LD", 0).
shifts_soft_ub(12, "SE", 0).
shifts_soft_ub(12, "SN", 0).
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
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "LD", 2).
shifts_soft_lb(13, "SE", 2).
shifts_soft_lb(13, "SN", 2).
shifts_soft_ub(13, "LD", 4).
shifts_soft_ub(13, "SE", 4).
shifts_soft_ub(13, "SN", 4).
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
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 2).
shifts_soft_lb(14, "SE", 2).
shifts_soft_lb(14, "SN", 2).
shifts_soft_ub(14, "LD", 4).
shifts_soft_ub(14, "SE", 4).
shifts_soft_ub(14, "SN", 4).
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
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 2).
shifts_soft_lb(15, "SE", 2).
shifts_soft_lb(15, "SN", 2).
shifts_soft_ub(15, "LD", 4).
shifts_soft_ub(15, "SE", 4).
shifts_soft_ub(15, "SN", 4).
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
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 2).
shifts_soft_lb(16, "SE", 2).
shifts_soft_lb(16, "SN", 2).
shifts_soft_ub(16, "LD", 3).
shifts_soft_ub(16, "SE", 3).
shifts_soft_ub(16, "SN", 3).
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
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 2).
shifts_soft_lb(17, "SE", 2).
shifts_soft_lb(17, "SN", 2).
shifts_soft_ub(17, "LD", 3).
shifts_soft_ub(17, "SE", 3).
shifts_soft_ub(17, "SN", 3).
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
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 2).
shifts_soft_lb(18, "SE", 2).
shifts_soft_lb(18, "SN", 2).
shifts_soft_ub(18, "LD", 3).
shifts_soft_ub(18, "SE", 3).
shifts_soft_ub(18, "SN", 3).
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
shifts_hard_ub(18, "EM", 0).
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
