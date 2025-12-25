% Staffs ---------------------------------------------------
staff(1, "Jordan Boyd", "Chief Nurse", "58290", 0).
staff(2, "Bethany Kent", "Deputy Chief Nurse", "51076", 0).
staff(3, "David Clarke", "Deputy Chief Nurse", "66919", 0).
staff(4, "Brian Shaw", "Deputy Chief Nurse", "20779", 0).
staff(5, "Margaret Mckee", "Nurse", "79047", 0).
staff(6, "James Rhodes", "Nurse", "15062", 0).
staff(7, "Sara Mcgee", "Nurse", "11491", 0).
staff(8, "Lisa Cummings", "Nurse", "51801", 0).
staff(9, "Margaret Beck", "Nurse", "96049", 0).
staff(10, "Amy Meyer", "Nurse", "80116", 0).
staff(11, "Jessica Buckley", "Nurse", "28416", 0).
staff(12, "Kathleen Hall", "Nurse", "99860", 0).
staff(13, "Louis Greene", "Nurse", "38526", 0).
staff(14, "Ricky Cortez", "Nurse", "06822", 0).
staff(15, "Judith Wheeler", "Nurse", "80462", 0).
staff(16, "Jessica Torres", "Nurse", "78777", 0).
staff(17, "Billy Flores", "Nurse", "14098", 0).
staff(18, "Haley Sweeney", "Nurse", "76250", 0).
staff(19, "James Smith", "Nurse", "17392", 0).
staff(20, "Yvonne Randall", "Nurse", "63892", 0).
staff(21, "Daniel Young", "Nurse", "37170", 0).
staff(22, "Jennifer Ferguson", "Nurse", "29374", 0).
staff(23, "Scott Thomas", "Nurse", "66132", 0).
staff(24, "Melody Wood", "Nurse", "46555", 0).
staff(25, "Christopher Herring", "Nurse", "59675", 0).
% Skipped staff no=26, name="Robert Morgan", job="補助者" (no mapping).
% Skipped staff no=27, name="Michael Shepard", job="補助者" (no mapping).
% Skipped staff no=28, name="Matthew Hogan", job="補助者" (no mapping).
% Skipped staff no=29, name="Danielle Smith", job="補助者" (no mapping).
% Skipped staff no=30, name="Tammy Ashley", job="補助者" (no mapping).
% Groups ---------------------------------------------------
staff_group("All").
staff_group("All", 2).
staff_group("All", 3).
staff_group("All", 4).
staff_group("All", 5).
staff_group("All", 6).
staff_group("All", 7).
staff_group("All", 8).
staff_group("All", 9).
staff_group("All", 10).
staff_group("All", 11).
staff_group("All", 12).
staff_group("All", 13).
staff_group("All", 14).
staff_group("All", 15).
staff_group("All", 16).
staff_group("All", 17).
staff_group("All", 18).
staff_group("All", 19).
staff_group("All", 20).
staff_group("All", 21).
staff_group("All", 22).
staff_group("All", 23).
staff_group("All", 24).
% Skipped nurse group column "師長代行" (no mapping).
staff_group("Leaders").
staff_group("Leaders", 2).
staff_group("Leaders", 3).
staff_group("Leaders", 5).
staff_group("Leaders", 7).
staff_group("Leaders", 8).
staff_group("Leaders", 9).
staff_group("Leaders", 10).
staff_group("Leaders", 13).
staff_group("Leaders", 14).
staff_group("Leaders", 15).
staff_group("Leaders", 16).
% Skipped nurse group column "メンバー" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 22).
staff_group("Newcomers", 23).
staff_group("Newcomers", 24).
staff_group("Night").
staff_group("Night", 2).
staff_group("Night", 3).
staff_group("Night", 4).
staff_group("Night", 5).
staff_group("Night", 6).
staff_group("Night", 7).
staff_group("Night", 8).
staff_group("Night", 9).
staff_group("Night", 10).
staff_group("Night", 11).
staff_group("Night", 13).
staff_group("Night", 14).
staff_group("Night", 15).
staff_group("Night", 16).
staff_group("Night", 17).
staff_group("Night", 18).
staff_group("Night", 19).
staff_group("Night", 20).
staff_group("Night", 22).
staff_group("Night", 23).
staff_group("Night", 24).
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
prev_shift(hard, "D", "D").
prev_shift(hard, "D", "WR").
prev_shift(hard, "D", "PH").
next_shift(hard, "D", "LD").
next_shift(hard, "D", "LM").
next_shift(hard, "D", "D").
next_shift(hard, "D", "WR").
next_shift(hard, "D", "PH").
next_shift(hard, "D", "SE").
prev_shift(hard, "LD", "D").
prev_shift(hard, "LD", "WR").
prev_shift(hard, "LD", "PH").
next_shift(hard, "LD", "LM").
next_shift(hard, "LD", "WR").
next_shift(hard, "LD", "PH").
next_shift(hard, "LD", "SE").
prev_shift(hard, "LM", "LD").
prev_shift(hard, "LM", "D").
prev_shift(hard, "LM", "WR").
prev_shift(hard, "LM", "PH").
next_shift(hard, "LM", "WR").
next_shift(hard, "LM", "PH").
next_shift(hard, "LM", "SE").
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
staff_dweek_hard_lb("All", "D", "Weekday", 3).
staff_dweek_hard_lb("All", "D", "Weekend", 2).
staff_dweek_hard_lb("All", "D", "Holiday", 2).
staff_dweek_soft_lb("All", "D", "Weekday", 4).
staff_dweek_soft_lb("All", "D", "Weekend", 3).
staff_dweek_soft_lb("All", "D", "Holiday", 3).
staff_dweek_soft_ub("All", "D", "Weekday", 10).
staff_dweek_soft_ub("All", "D", "Weekend", 5).
staff_dweek_soft_ub("All", "D", "Holiday", 5).
staff_dweek_hard_ub("All", "D", "Weekday", 10).
staff_dweek_hard_ub("All", "D", "Weekend", 5).
staff_dweek_hard_ub("All", "D", "Holiday", 5).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 3).
staff_dweek_soft_lb("All", "LD", "Weekend", 3).
staff_dweek_soft_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_ub("All", "LD", "Weekday", 3).
staff_dweek_soft_ub("All", "LD", "Weekend", 3).
staff_dweek_soft_ub("All", "LD", "Holiday", 3).
staff_dweek_hard_ub("All", "LD", "Weekday", 3).
staff_dweek_hard_ub("All", "LD", "Weekend", 3).
staff_dweek_hard_ub("All", "LD", "Holiday", 3).
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
staff_dweek_soft_ub("All", "SE", "Weekday", 4).
staff_dweek_soft_ub("All", "SN", "Weekday", 4).
staff_dweek_soft_ub("All", "SE", "Weekend", 4).
staff_dweek_soft_ub("All", "SN", "Weekend", 4).
staff_dweek_soft_ub("All", "SE", "Holiday", 4).
staff_dweek_soft_ub("All", "SN", "Holiday", 4).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
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
% Skipped vertical constraint for nurse group "メンバー" (no mapping).
% Skipped vertical constraint for nurse group "メンバー" (no mapping).
% Skipped vertical constraint for nurse group "メンバー" (no mapping).
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
staff_dweek_soft_lb("Night", "SE", "Weekday", 3).
staff_dweek_soft_lb("Night", "SE", "Weekend", 3).
staff_dweek_soft_lb("Night", "SE", "Holiday", 3).
staff_dweek_soft_ub("Night", "SE", "Weekday", 4).
staff_dweek_soft_ub("Night", "SE", "Weekend", 4).
staff_dweek_soft_ub("Night", "SE", "Holiday", 4).
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
vertical_constraint_type("Leaders","LD","Staffs").
vertical_constraint_type("Leaders","SE","Staffs").
vertical_constraint_type("Leaders","SN","Staffs").
% Skipped vertical_constraint_type for nurse group "メンバー" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
vertical_constraint_type("Night","SE","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shift_group("E").
shift_group("E", "E").
shifts_hard_lb(1, "E", 0).
shift_group("N").
shift_group("N", "N").
shifts_hard_lb(1, "N", 0).
shifts_soft_lb(1, "E", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_ub(1, "E", 0).
shifts_soft_ub(1, "N", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 3).
shifts_soft_lb(2, "SE", 3).
shifts_soft_lb(2, "SN", 3).
shifts_soft_ub(2, "LD", 3).
shifts_soft_ub(2, "SE", 3).
shifts_soft_ub(2, "SN", 3).
shifts_hard_lb(2, "E", 0).
shifts_hard_lb(2, "N", 0).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 3).
shifts_soft_lb(3, "SE", 3).
shifts_soft_lb(3, "SN", 3).
shifts_soft_ub(3, "LD", 3).
shifts_soft_ub(3, "SE", 3).
shifts_soft_ub(3, "SN", 3).
shifts_hard_lb(3, "E", 0).
shifts_hard_lb(3, "N", 0).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
shifts_hard_ub(3, "E", 0).
shifts_hard_ub(3, "N", 0).
% Staff 4:
shifts_soft_lb(4, "LD", 3).
shifts_soft_lb(4, "SE", 3).
shifts_soft_lb(4, "SN", 3).
shifts_soft_ub(4, "LD", 3).
shifts_soft_ub(4, "SE", 3).
shifts_soft_ub(4, "SN", 3).
shifts_hard_lb(4, "E", 0).
shifts_hard_lb(4, "N", 0).
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
shifts_hard_ub(4, "E", 0).
shifts_hard_ub(4, "N", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 4).
shifts_soft_lb(5, "SE", 4).
shifts_soft_lb(5, "SN", 4).
shifts_soft_ub(5, "LD", 6).
shifts_soft_ub(5, "SE", 6).
shifts_soft_ub(5, "SN", 6).
shifts_hard_lb(5, "E", 0).
shifts_hard_lb(5, "N", 0).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 6).
shifts_soft_ub(6, "SE", 6).
shifts_soft_ub(6, "SN", 6).
shifts_hard_lb(6, "E", 0).
shifts_hard_lb(6, "N", 0).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 2).
shifts_soft_lb(7, "SE", 2).
shifts_soft_lb(7, "SN", 2).
shifts_soft_ub(7, "LD", 2).
shifts_soft_ub(7, "SE", 2).
shifts_soft_ub(7, "SN", 2).
shifts_hard_lb(7, "E", 0).
shifts_hard_lb(7, "N", 0).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 4).
shifts_soft_ub(8, "SE", 4).
shifts_soft_ub(8, "SN", 4).
shifts_hard_lb(8, "E", 0).
shifts_hard_lb(8, "N", 0).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 2).
shifts_soft_lb(9, "SE", 2).
shifts_soft_lb(9, "SN", 2).
shifts_soft_ub(9, "LD", 2).
shifts_soft_ub(9, "SE", 2).
shifts_soft_ub(9, "SN", 2).
shifts_hard_lb(9, "E", 0).
shifts_hard_lb(9, "N", 0).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 6).
shifts_soft_ub(10, "SE", 6).
shifts_soft_ub(10, "SN", 6).
shifts_hard_lb(10, "E", 0).
shifts_hard_lb(10, "N", 0).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 6).
shifts_soft_ub(11, "SE", 6).
shifts_soft_ub(11, "SN", 6).
shifts_hard_lb(11, "E", 0).
shifts_hard_lb(11, "N", 0).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
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
% Staff 13:
shifts_soft_lb(13, "LD", 4).
shifts_soft_lb(13, "SE", 4).
shifts_soft_lb(13, "SN", 4).
shifts_soft_ub(13, "LD", 6).
shifts_soft_ub(13, "SE", 6).
shifts_soft_ub(13, "SN", 6).
shifts_hard_lb(13, "E", 0).
shifts_hard_lb(13, "N", 0).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 6).
shifts_soft_ub(14, "SE", 6).
shifts_soft_ub(14, "SN", 6).
shifts_hard_lb(14, "E", 0).
shifts_hard_lb(14, "N", 0).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 6).
shifts_soft_ub(15, "SE", 6).
shifts_soft_ub(15, "SN", 6).
shifts_hard_lb(15, "E", 0).
shifts_hard_lb(15, "N", 0).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 4).
shifts_soft_lb(16, "SE", 4).
shifts_soft_lb(16, "SN", 4).
shifts_soft_ub(16, "LD", 6).
shifts_soft_ub(16, "SE", 6).
shifts_soft_ub(16, "SN", 6).
shifts_hard_lb(16, "E", 0).
shifts_hard_lb(16, "N", 0).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 6).
shifts_soft_ub(17, "SE", 6).
shifts_soft_ub(17, "SN", 6).
shifts_hard_lb(17, "E", 0).
shifts_hard_lb(17, "N", 0).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 4).
shifts_soft_lb(18, "SE", 4).
shifts_soft_lb(18, "SN", 4).
shifts_soft_ub(18, "LD", 6).
shifts_soft_ub(18, "SE", 6).
shifts_soft_ub(18, "SN", 6).
shifts_hard_lb(18, "E", 0).
shifts_hard_lb(18, "N", 0).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
shifts_hard_ub(18, "E", 0).
shifts_hard_ub(18, "N", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 4).
shifts_soft_lb(19, "SE", 4).
shifts_soft_lb(19, "SN", 4).
shifts_soft_ub(19, "LD", 6).
shifts_soft_ub(19, "SE", 6).
shifts_soft_ub(19, "SN", 6).
shifts_hard_lb(19, "E", 0).
shifts_hard_lb(19, "N", 0).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
shifts_hard_ub(19, "E", 0).
shifts_hard_ub(19, "N", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 6).
shifts_soft_ub(20, "SE", 6).
shifts_soft_ub(20, "SN", 6).
shifts_hard_lb(20, "E", 0).
shifts_hard_lb(20, "N", 0).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
shifts_hard_ub(20, "E", 0).
shifts_hard_ub(20, "N", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 0).
shifts_soft_lb(21, "SE", 0).
shifts_soft_lb(21, "SN", 0).
shifts_soft_ub(21, "LD", 0).
shifts_soft_ub(21, "SE", 0).
shifts_soft_ub(21, "SN", 0).
shifts_hard_lb(21, "E", 0).
shifts_hard_lb(21, "N", 0).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 2).
shifts_soft_lb(22, "SE", 2).
shifts_soft_lb(22, "SN", 2).
shifts_soft_ub(22, "LD", 3).
shifts_soft_ub(22, "SE", 3).
shifts_soft_ub(22, "SN", 3).
shifts_hard_lb(22, "E", 0).
shifts_hard_lb(22, "N", 0).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 2).
shifts_soft_lb(23, "SE", 2).
shifts_soft_lb(23, "SN", 2).
shifts_soft_ub(23, "LD", 3).
shifts_soft_ub(23, "SE", 3).
shifts_soft_ub(23, "SN", 3).
shifts_hard_lb(23, "E", 0).
shifts_hard_lb(23, "N", 0).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
% Staff 24:
shifts_soft_lb(24, "LD", 2).
shifts_soft_lb(24, "SE", 2).
shifts_soft_lb(24, "SN", 2).
shifts_soft_ub(24, "LD", 3).
shifts_soft_ub(24, "SE", 3).
shifts_soft_ub(24, "SN", 3).
shifts_hard_lb(24, "E", 0).
shifts_hard_lb(24, "N", 0).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
% Staff 25:
shifts_hard_lb(25, "E", 0).
shifts_hard_lb(25, "N", 0).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
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
