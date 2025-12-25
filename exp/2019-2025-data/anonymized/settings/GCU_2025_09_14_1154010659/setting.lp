% Staffs ---------------------------------------------------
staff(1, "Debra Ward", "Chief Nurse", "76788", 0).
staff(2, "Wanda Miller", "Deputy Chief Nurse", "44567", 0).
staff(3, "Michael Brown", "Nurse", "67358", 0).
staff(4, "Dr. Amanda Garcia", "Nurse", "76818", 0).
staff(5, "Jacob Griffin", "Nurse", "38487", 0).
staff(6, "Anthony Dillon", "Nurse", "90362", 0).
staff(7, "Angela Murray", "Nurse", "32249", 0).
staff(8, "Janet Carlson", "Nurse", "54502", 0).
staff(9, "Lauren Farley", "Nurse", "08467", 0).
staff(10, "Randall Davis", "Nurse", "88598", 0).
staff(11, "Kevin Thompson", "Deputy Chief Nurse", "82749", 0).
staff(12, "Brittany Matthews", "Nurse", "05653", 0).
staff(13, "Edward Bradley", "Nurse", "70577", 0).
staff(14, "Trevor Pratt", "Nurse", "29176", 0).
staff(15, "Mindy Williams", "Nurse", "67109", 0).
staff(16, "Thomas Atkins", "Nurse", "97409", 0).
staff(17, "Michael Hernandez", "Nurse", "23715", 0).
staff(18, "Steven Peterson", "Nurse", "10142", 0).
staff(19, "Brittany Perry", "Nurse", "69186", 0).
staff(20, "Edward Cooper", "Nurse", "03109", 0).
% Groups ---------------------------------------------------
staff_group("All").
staff_group("Heads").
% Skipped nurse group column "A" (no mapping).
% Skipped nurse group column "B" (no mapping).
staff_group("Newcomers").
staff_group("Night").
% Shift constraints ----------------------------------------
% Maximum consecutive working days
consecutive_work_ub(hard, 6).
% Forbidden patterns
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
% Assignable previous/next shifts
prev_shift(hard, "SN", "SE").
next_shift(hard, "SN", "SE").
next_shift(hard, "SN", "WR").
next_shift(hard, "SN", "PH").
prev_shift(hard, "SE", "LD").
prev_shift(hard, "SE", "LM").
prev_shift(hard, "SE", "D").
prev_shift(hard, "SE", "WR").
prev_shift(hard, "SE", "PH").
next_shift(hard, "SE", "SN").
prev_shift(hard, "E", "LD").
prev_shift(hard, "E", "LM").
prev_shift(hard, "E", "D").
prev_shift(hard, "E", "WR").
prev_shift(hard, "E", "PH").
next_shift(hard, "E", "N").
prev_shift(hard, "N", "E").
next_shift(hard, "N", "WR").
next_shift(hard, "N", "PH").
prev_shift(hard, "D", "D").
prev_shift(hard, "D", "WR").
prev_shift(hard, "D", "PH").
next_shift(hard, "D", "LD").
next_shift(hard, "D", "LM").
next_shift(hard, "D", "D").
next_shift(hard, "D", "WR").
next_shift(hard, "D", "PH").
next_shift(hard, "D", "SE").
next_shift(hard, "D", "E").
prev_shift(hard, "LD", "D").
prev_shift(hard, "LD", "WR").
prev_shift(hard, "LD", "PH").
next_shift(hard, "LD", "LM").
next_shift(hard, "LD", "WR").
next_shift(hard, "LD", "PH").
next_shift(hard, "LD", "SE").
next_shift(hard, "LD", "E").
prev_shift(hard, "LM", "LD").
prev_shift(hard, "LM", "D").
prev_shift(hard, "LM", "WR").
prev_shift(hard, "LM", "PH").
next_shift(hard, "LM", "WR").
next_shift(hard, "LM", "PH").
next_shift(hard, "LM", "SE").
next_shift(hard, "LM", "E").
% Pattern occurrences
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("WR-WR", 3).
patterns_soft_ub("LD-D", 0).
patterns_soft_ub("LM-D", 0).
patterns_soft_ub("LM-LD", 0).
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_soft_ub("LM-LM", 0).
% Vertical constraints -------------------------------------
shift_group("D").
shift_group("D", "D").
staff_dweek_hard_lb("All", "D", "Weekday", 1).
staff_dweek_hard_lb("All", "D", "Weekend", 1).
staff_dweek_hard_lb("All", "D", "Holiday", 1).
staff_dweek_soft_lb("All", "D", "Weekday", 2).
staff_dweek_soft_lb("All", "D", "Weekend", 2).
staff_dweek_soft_lb("All", "D", "Holiday", 2).
shift_group("LD").
shift_group("LD", "LD").
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
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 3).
shift_group("SN").
shift_group("SN", "SN").
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
shift_group("E").
shift_group("E", "E").
staff_dweek_soft_ub("All", "E", "Weekday", 0).
shift_group("N").
shift_group("N", "N").
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
shift_group("SE+E").
shift_group("SE+E", "SE").
shift_group("SE+E", "E").
staff_dweek_hard_ub("All", "SE+E", "Weekday", 3).
staff_dweek_hard_ub("All", "SE+E", "Weekend", 0).
staff_dweek_hard_ub("All", "SE+E", "Holiday", 0).
shift_group("LM").
shift_group("LM", "LM").
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shift_group("LD+LM+E").
shift_group("LD+LM+E", "LD").
shift_group("LD+LM+E", "LM").
shift_group("LD+LM+E", "E").
staff_dweek_hard_lb("All", "LD+LM+E", "Weekday", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 5).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
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
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("All","E","Staffs").
vertical_constraint_type("All","N","Staffs").
vertical_constraint_type("All","SE+E","Staffs").
vertical_constraint_type("All","LM","Staffs").
vertical_constraint_type("All","LD+LM+E","Staffs").
% Skipped vertical_constraint_type for nurse group "A" (no mapping).
vertical_constraint_type("Heads","D","Staffs").
% Skipped vertical_constraint_type for nurse group "B" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
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
