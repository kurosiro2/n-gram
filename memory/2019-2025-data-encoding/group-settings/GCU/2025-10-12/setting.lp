% Staffs ---------------------------------------------------
staff(1, "Miranda Simmons", "Chief Nurse", "22006", 0).
staff(2, "Jessica Patel", "Deputy Chief Nurse", "39245", 0).
staff(3, "Jaclyn Robinson", "Nurse", "99412", 0).
staff(4, "Lauren Salinas", "Nurse", "74398", 0).
staff(5, "Nicole Murray", "Nurse", "61627", 0).
staff(6, "Marissa Williams", "Nurse", "86732", 0).
staff(7, "William Vasquez", "Nurse", "82336", 0).
staff(8, "Lauren Wright", "Nurse", "36258", 0).
staff(9, "Kevin Marks", "Nurse", "55540", 0).
staff(10, "Michelle Harrison", "Nurse", "24347", 0).
staff(11, "Barry Matthews", "Deputy Chief Nurse", "98476", 0).
staff(12, "Mrs. Rachel Price MD", "Nurse", "81119", 0).
staff(13, "Jeanne Rodriguez", "Nurse", "58489", 0).
staff(14, "David Galloway", "Nurse", "62116", 0).
staff(15, "Nathaniel Morse", "Nurse", "28978", 0).
staff(16, "Bradley Mccann", "Nurse", "47931", 0).
staff(17, "Krista Jones", "Nurse", "52677", 0).
staff(18, "Santiago King", "Nurse", "A859104", 0).
staff(19, "Chad Davis", "Nurse", "81770", 0).
staff(20, "John Clark", "Nurse", "07754", 0).
% Groups ---------------------------------------------------
group("All").
group("Heads").
group("A").
group("B").
group("Newcomers").
group("Night").
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
staff_dweek_hard_lb("All", "D", "Weekday", 1).
staff_dweek_hard_lb("All", "D", "Weekend", 1).
staff_dweek_hard_lb("All", "D", "Holiday", 1).
staff_dweek_soft_lb("All", "D", "Weekday", 2).
staff_dweek_soft_lb("All", "D", "Weekend", 2).
staff_dweek_soft_lb("All", "D", "Holiday", 2).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 2).
staff_dweek_hard_lb("All", "LD", "Weekend", 2).
staff_dweek_hard_lb("All", "LD", "Holiday", 2).
staff_dweek_soft_lb("All", "LD", "Weekday", 3).
staff_dweek_soft_lb("All", "LD", "Weekend", 3).
staff_dweek_soft_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_ub("All", "LD", "Weekday", 3).
staff_dweek_soft_ub("All", "LD", "Weekend", 3).
staff_dweek_soft_ub("All", "LD", "Holiday", 3).
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
staff_dweek_soft_ub("All", "SE", "Weekday", 3).
staff_dweek_soft_ub("All", "SN", "Weekday", 3).
staff_dweek_soft_ub("All", "SE", "Weekend", 3).
staff_dweek_soft_ub("All", "SN", "Weekend", 3).
staff_dweek_soft_ub("All", "SE", "Holiday", 3).
staff_dweek_soft_ub("All", "SN", "Holiday", 3).
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
staff_dweek_hard_ub("All", "E", "Weekday", 1).
staff_dweek_hard_ub("All", "N", "Weekday", 1).
staff_dweek_hard_ub("All", "E", "Weekend", 0).
staff_dweek_hard_ub("All", "N", "Weekend", 0).
staff_dweek_hard_ub("All", "E", "Holiday", 0).
staff_dweek_hard_ub("All", "N", "Holiday", 0).
shifts("SE+E").
shifts("SE+E", "SE").
shifts("SE+E", "E").
staff_dweek_hard_ub("All", "SE+E", "Weekday", 3).
staff_dweek_hard_ub("All", "SE+E", "Weekend", 0).
staff_dweek_hard_ub("All", "SE+E", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shifts("LD+LM+E").
shifts("LD+LM+E", "LD").
shifts("LD+LM+E", "LM").
shifts("LD+LM+E", "E").
staff_dweek_hard_lb("All", "LD+LM+E", "Weekday", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 5).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
staff_dweek_hard_lb("A", "LD", "Weekday", 1).
staff_dweek_hard_lb("A", "SE", "Weekday", 1).
staff_dweek_hard_lb("A", "SN", "Weekday", 1).
staff_dweek_hard_lb("A", "LD", "Weekend", 1).
staff_dweek_hard_lb("A", "SE", "Weekend", 1).
staff_dweek_hard_lb("A", "SN", "Weekend", 1).
staff_dweek_hard_lb("A", "LD", "Holiday", 1).
staff_dweek_hard_lb("A", "SE", "Holiday", 1).
staff_dweek_hard_lb("A", "SN", "Holiday", 1).
staff_dweek_soft_lb("A", "LD", "Weekday", 2).
staff_dweek_soft_lb("A", "SE", "Weekday", 2).
staff_dweek_soft_lb("A", "SN", "Weekday", 2).
staff_dweek_soft_lb("A", "LD", "Weekend", 2).
staff_dweek_soft_lb("A", "SE", "Weekend", 2).
staff_dweek_soft_lb("A", "SN", "Weekend", 2).
staff_dweek_soft_lb("A", "LD", "Holiday", 2).
staff_dweek_soft_lb("A", "SE", "Holiday", 2).
staff_dweek_soft_lb("A", "SN", "Holiday", 2).
staff_dweek_soft_ub("A", "LD", "Weekday", 5).
staff_dweek_soft_ub("A", "SE", "Weekday", 5).
staff_dweek_soft_ub("A", "SN", "Weekday", 5).
staff_dweek_soft_ub("A", "LD", "Weekend", 5).
staff_dweek_soft_ub("A", "SE", "Weekend", 5).
staff_dweek_soft_ub("A", "SN", "Weekend", 5).
staff_dweek_soft_ub("A", "LD", "Holiday", 5).
staff_dweek_soft_ub("A", "SE", "Holiday", 5).
staff_dweek_soft_ub("A", "SN", "Holiday", 5).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
staff_dweek_hard_lb("B", "LD", "Weekday", 1).
staff_dweek_hard_lb("B", "SE", "Weekday", 1).
staff_dweek_hard_lb("B", "SN", "Weekday", 1).
staff_dweek_hard_lb("B", "LD", "Weekend", 1).
staff_dweek_hard_lb("B", "SE", "Weekend", 1).
staff_dweek_hard_lb("B", "SN", "Weekend", 1).
staff_dweek_hard_lb("B", "LD", "Holiday", 1).
staff_dweek_hard_lb("B", "SE", "Holiday", 1).
staff_dweek_hard_lb("B", "SN", "Holiday", 1).
staff_dweek_soft_lb("B", "LD", "Weekday", 2).
staff_dweek_soft_lb("B", "SE", "Weekday", 2).
staff_dweek_soft_lb("B", "SN", "Weekday", 2).
staff_dweek_soft_lb("B", "LD", "Weekend", 2).
staff_dweek_soft_lb("B", "SE", "Weekend", 2).
staff_dweek_soft_lb("B", "SN", "Weekend", 2).
staff_dweek_soft_lb("B", "LD", "Holiday", 2).
staff_dweek_soft_lb("B", "SE", "Holiday", 2).
staff_dweek_soft_lb("B", "SN", "Holiday", 2).
staff_dweek_soft_ub("B", "LD", "Weekday", 5).
staff_dweek_soft_ub("B", "SE", "Weekday", 5).
staff_dweek_soft_ub("B", "SN", "Weekday", 5).
staff_dweek_soft_ub("B", "LD", "Weekend", 5).
staff_dweek_soft_ub("B", "SE", "Weekend", 5).
staff_dweek_soft_ub("B", "SN", "Weekend", 5).
staff_dweek_soft_ub("B", "LD", "Holiday", 5).
staff_dweek_soft_ub("B", "SE", "Holiday", 5).
staff_dweek_soft_ub("B", "SN", "Holiday", 5).
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
vertical_constraint_type("All", "LD+LM+E", "Staffs").
vertical_constraint_type("A", "LD", "Staffs").
vertical_constraint_type("A", "SE", "Staffs").
vertical_constraint_type("A", "SN", "Staffs").
vertical_constraint_type("Heads", "D", "Staffs").
vertical_constraint_type("B", "LD", "Staffs").
vertical_constraint_type("B", "SE", "Staffs").
vertical_constraint_type("B", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
% Staff 3:
% Staff 4:
% Staff 5:
% Staff 6:
% Staff 7:
% Staff 8:
% Staff 9:
% Staff 10:
% Staff 11:
% Staff 12:
% Staff 13:
% Staff 14:
% Staff 15:
% Staff 16:
% Staff 17:
% Staff 18:
% Staff 19:
% Staff 20:
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
