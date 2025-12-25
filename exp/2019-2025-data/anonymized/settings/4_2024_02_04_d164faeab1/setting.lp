% Staffs ---------------------------------------------------
staff(1, "Margaret Estes", "Chief Nurse", "27341", 0).
staff(2, "Jordan Boyd", "Deputy Chief Nurse", "58290", 5).
staff(3, "Erica Brock", "Deputy Chief Nurse", "38690", 5).
staff(4, "Julie Scott DDS", "Deputy Chief Nurse", "75687", 5).
staff(5, "Angela Graham", "Nurse", "22191", 0).
staff(6, "David Clarke", "Nurse", "66919", 0).
staff(7, "Brian Shaw", "Nurse", "20779", 0).
staff(8, "Margaret Mckee", "Nurse", "79047", 0).
staff(9, "Mary Miller", "Nurse", "63230", 0).
staff(10, "Joseph Smith", "Nurse", "37199", 0).
staff(11, "Sara Mcgee", "Nurse", "11491", 0).
staff(12, "Brandi Hughes", "Nurse", "23271", 0).
staff(13, "Garrett Allen", "Nurse", "84320", 0).
staff(14, "Amy Collins", "Nurse", "95857", 0).
staff(15, "Charles Yates", "Nurse", "00262", 0).
staff(16, "Lisa Cummings", "Nurse", "51801", 0).
staff(17, "Margaret Beck", "Nurse", "96049", 0).
staff(18, "Amy Meyer", "Nurse", "80116", 0).
staff(19, "Kathleen Hall", "Nurse", "99860", 0).
staff(20, "Jessica Buckley", "Nurse", "28416", 0).
staff(21, "Louis Greene", "Nurse", "38526", 0).
staff(22, "Ricky Cortez", "Nurse", "06822", 0).
staff(23, "Marcus Foster", "Nurse", "75235", 0).
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
staff_group("Heads").
staff_group("Heads", 1).
staff_group("Seniors").
staff_group("Seniors1").
staff_group("Seniors1", 3).
staff_group("Seniors1", 4).
staff_group("Seniors1", 5).
staff_group("Seniors1", 6).
staff_group("Seniors1", 8).
staff_group("Seniors1", 9).
staff_group("Seniors2").
staff_group("Mid-levels").
staff_group("Mid-levels", 2).
staff_group("Mid-levels", 7).
staff_group("Mid-levels", 10).
staff_group("Mid-levels", 11).
staff_group("Mid-levels", 12).
staff_group("Mid-levels", 13).
staff_group("Mid-levels", 14).
staff_group("Mid-levels", 15).
staff_group("Mid-levels", 16).
staff_group("Mid-levels", 17).
staff_group("Mid-levels", 18).
staff_group("Mid-levels", 19).
staff_group("Mid-levels", 20).
staff_group("Mid-levels", 21).
staff_group("Newcomers").
staff_group("Newcomers", 22).
staff_group("Newcomers", 23).
staff_group("Newcomers1").
staff_group("Newcomers1", 22).
staff_group("Newcomers1", 23).
staff_group("Newcomers2").
staff_group("Newcomers2", 22).
staff_group("Newcomers2", 23).
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
staff_group("Night", 21).
staff_group("Night", 22).
staff_group("Night", 23).
% Shift constraints ----------------------------------------
% Maximum consecutive working days
consecutive_work_ub(hard, 5).
% Forbidden patterns
forbidden_pattern("LD-LD").
pattern("LD-LD", 2).
pattern("LD-LD", 0, "LD").
pattern("LD-LD", 1, "LD").
forbidden_pattern("LD-D").
pattern("LD-D", 2).
pattern("LD-D", 0, "LD").
pattern("LD-D", 1, "D").
forbidden_pattern("LM-D").
pattern("LM-D", 2).
pattern("LM-D", 0, "LM").
pattern("LM-D", 1, "D").
forbidden_pattern("LM-LD").
pattern("LM-LD", 2).
pattern("LM-LD", 0, "LM").
pattern("LM-LD", 1, "LD").
forbidden_pattern("SE-SN-SE-SN").
pattern("SE-SN-SE-SN", 4).
pattern("SE-SN-SE-SN", 0, "SE").
pattern("SE-SN-SE-SN", 1, "SN").
pattern("SE-SN-SE-SN", 2, "SE").
pattern("SE-SN-SE-SN", 3, "SN").
% Assignable previous/next shifts
prev_shift(hard, "SN", "SE").
next_shift(hard, "SN", "SE").
next_shift(hard, "SN", "WR").
next_shift(hard, "SN", "PH").
prev_shift(hard, "SE", "LD").
prev_shift(hard, "SE", "LM").
next_shift(hard, "SE", "SN").
next_shift(hard, "E", "N").
prev_shift(hard, "N", "E").
next_shift(hard, "N", "WR").
next_shift(hard, "N", "PH").
% Pattern occurrences
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("WR-WR", 2).
patterns_soft_ub("LD-D", 1).
patterns_soft_ub("LM-D", 0).
patterns_soft_ub("LM-LD", 0).
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_soft_ub("LM-LM", 0).
% Vertical constraints -------------------------------------
shift_group("D").
shift_group("D", "D").
staff_dweek_hard_lb("All", "D", "Weekday", 4).
staff_dweek_hard_lb("All", "D", "Weekend", 2).
staff_dweek_hard_lb("All", "D", "Holiday", 2).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 3).
staff_dweek_soft_lb("All", "D", "Holiday", 3).
staff_dweek_soft_ub("All", "D", "Weekday", 13).
staff_dweek_soft_ub("All", "D", "Weekend", 4).
staff_dweek_soft_ub("All", "D", "Holiday", 4).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 3).
shift_group("SN").
shift_group("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "SE", "Weekend", 3).
staff_dweek_hard_lb("All", "SN", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_hard_lb("All", "SE", "Holiday", 3).
staff_dweek_hard_lb("All", "SN", "Holiday", 3).
staff_dweek_hard_ub("All", "LD", "Weekday", 3).
staff_dweek_hard_ub("All", "SE", "Weekday", 3).
staff_dweek_hard_ub("All", "SN", "Weekday", 3).
staff_dweek_hard_ub("All", "LD", "Weekend", 3).
staff_dweek_hard_ub("All", "SE", "Weekend", 3).
staff_dweek_hard_ub("All", "SN", "Weekend", 3).
staff_dweek_hard_ub("All", "LD", "Holiday", 3).
staff_dweek_hard_ub("All", "SE", "Holiday", 3).
staff_dweek_hard_ub("All", "SN", "Holiday", 3).
shift_group("E").
shift_group("E", "E").
staff_dweek_hard_ub("All", "E", "Weekday", 0).
shift_group("N").
shift_group("N", "N").
staff_dweek_hard_ub("All", "N", "Weekday", 0).
staff_dweek_hard_ub("All", "E", "Weekend", 0).
staff_dweek_hard_ub("All", "N", "Weekend", 0).
staff_dweek_hard_ub("All", "E", "Holiday", 0).
staff_dweek_hard_ub("All", "N", "Holiday", 0).
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
staff_dweek_hard_lb("Seniors1", "LD", "Weekday", 0).
staff_dweek_hard_lb("Seniors1", "SE", "Weekday", 0).
staff_dweek_hard_lb("Seniors1", "SN", "Weekday", 0).
staff_dweek_hard_lb("Seniors1", "LD", "Weekend", 0).
staff_dweek_hard_lb("Seniors1", "SE", "Weekend", 0).
staff_dweek_hard_lb("Seniors1", "SN", "Weekend", 0).
staff_dweek_hard_lb("Seniors1", "LD", "Holiday", 0).
staff_dweek_hard_lb("Seniors1", "SE", "Holiday", 0).
staff_dweek_hard_lb("Seniors1", "SN", "Holiday", 0).
staff_dweek_soft_lb("Seniors1", "LD", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "LD", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "LD", "Holiday", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Holiday", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Holiday", 1).
staff_dweek_soft_ub("Seniors1", "LD", "Weekday", 3).
staff_dweek_soft_ub("Seniors1", "SE", "Weekday", 3).
staff_dweek_soft_ub("Seniors1", "SN", "Weekday", 3).
staff_dweek_soft_ub("Seniors1", "LD", "Weekend", 3).
staff_dweek_soft_ub("Seniors1", "SE", "Weekend", 3).
staff_dweek_soft_ub("Seniors1", "SN", "Weekend", 3).
staff_dweek_soft_ub("Seniors1", "LD", "Holiday", 3).
staff_dweek_soft_ub("Seniors1", "SE", "Holiday", 3).
staff_dweek_soft_ub("Seniors1", "SN", "Holiday", 3).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekday", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekday", 1).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekend", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekend", 1).
staff_dweek_hard_lb("Mid-levels", "SE", "Holiday", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Holiday", 1).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekday", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekday", 2).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekend", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekend", 2).
staff_dweek_soft_lb("Mid-levels", "SE", "Holiday", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Holiday", 2).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekday", 3).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekday", 3).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekend", 3).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekend", 3).
staff_dweek_soft_ub("Mid-levels", "SE", "Holiday", 3).
staff_dweek_soft_ub("Mid-levels", "SN", "Holiday", 3).
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
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("All","E","Staffs").
vertical_constraint_type("All","N","Staffs").
vertical_constraint_type("All","LM","Staffs").
vertical_constraint_type("All","LD+LM+E","Staffs").
vertical_constraint_type("Seniors1","LD","Staffs").
vertical_constraint_type("Seniors1","SE","Staffs").
vertical_constraint_type("Seniors1","SN","Staffs").
vertical_constraint_type("Heads","D","Staffs").
vertical_constraint_type("Mid-levels","SE","Staffs").
vertical_constraint_type("Mid-levels","SN","Staffs").
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "LD", 0).
shifts_soft_lb(1, "SE", 0).
shifts_soft_lb(1, "SN", 0).
shifts_soft_ub(1, "LD", 0).
shifts_soft_ub(1, "SE", 0).
shifts_soft_ub(1, "SN", 0).
shifts_soft_lb(1, "E", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_lb(1, "LM", 0).
shifts_soft_ub(1, "E", 0).
shifts_soft_ub(1, "N", 0).
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "LM", 0).
shift_group("EM").
shift_group("EM", "EM").
shifts_hard_lb(1, "EM", 0).
shifts_soft_lb(1, "EM", 0).
shifts_soft_ub(1, "EM", 0).
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 1).
shifts_soft_lb(2, "SE", 1).
shifts_soft_lb(2, "SN", 1).
shifts_soft_ub(2, "LD", 3).
shifts_soft_ub(2, "SE", 3).
shifts_soft_ub(2, "SN", 3).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "E", 1).
shifts_soft_ub(2, "N", 1).
shifts_soft_ub(2, "LM", 1).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 0).
shifts_soft_lb(3, "SE", 0).
shifts_soft_lb(3, "SN", 0).
shifts_soft_ub(3, "LD", 1).
shifts_soft_ub(3, "SE", 1).
shifts_soft_ub(3, "SN", 1).
shifts_hard_ub(3, "LD", 2).
shifts_hard_ub(3, "SE", 2).
shifts_hard_ub(3, "SN", 2).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "E", 1).
shifts_soft_ub(3, "N", 1).
shifts_soft_ub(3, "LM", 1).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_soft_lb(4, "LD", 1).
shifts_soft_lb(4, "SE", 1).
shifts_soft_lb(4, "SN", 1).
shifts_soft_ub(4, "LD", 3).
shifts_soft_ub(4, "SE", 3).
shifts_soft_ub(4, "SN", 3).
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "E", 1).
shifts_soft_ub(4, "N", 1).
shifts_soft_ub(4, "LM", 1).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 5).
shifts_soft_ub(5, "SE", 5).
shifts_soft_ub(5, "SN", 5).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "E", 1).
shifts_soft_ub(5, "N", 1).
shifts_soft_ub(5, "LM", 1).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 1).
shifts_soft_lb(6, "SE", 1).
shifts_soft_lb(6, "SN", 1).
shifts_soft_ub(6, "LD", 4).
shifts_soft_ub(6, "SE", 4).
shifts_soft_ub(6, "SN", 4).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "E", 1).
shifts_soft_ub(6, "N", 1).
shifts_soft_ub(6, "LM", 1).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 0).
shifts_soft_lb(7, "SE", 0).
shifts_soft_lb(7, "SN", 0).
shifts_soft_ub(7, "LD", 1).
shifts_soft_ub(7, "SE", 1).
shifts_soft_ub(7, "SN", 1).
shifts_hard_ub(7, "LD", 2).
shifts_hard_ub(7, "SE", 2).
shifts_hard_ub(7, "SN", 2).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "E", 1).
shifts_soft_ub(7, "N", 1).
shifts_soft_ub(7, "LM", 1).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 1).
shifts_soft_lb(8, "SE", 1).
shifts_soft_lb(8, "SN", 1).
shifts_soft_ub(8, "LD", 4).
shifts_soft_ub(8, "SE", 4).
shifts_soft_ub(8, "SN", 4).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "E", 1).
shifts_soft_ub(8, "N", 1).
shifts_soft_ub(8, "LM", 1).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 1).
shifts_soft_lb(9, "SE", 1).
shifts_soft_lb(9, "SN", 1).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "E", 1).
shifts_soft_ub(9, "N", 1).
shifts_soft_ub(9, "LM", 1).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 1).
shifts_soft_lb(10, "SE", 1).
shifts_soft_lb(10, "SN", 1).
shifts_soft_ub(10, "LD", 4).
shifts_soft_ub(10, "SE", 4).
shifts_soft_ub(10, "SN", 4).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "E", 1).
shifts_soft_ub(10, "N", 1).
shifts_soft_ub(10, "LM", 1).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 1).
shifts_soft_lb(11, "SE", 1).
shifts_soft_lb(11, "SN", 1).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "E", 1).
shifts_soft_ub(11, "N", 1).
shifts_soft_ub(11, "LM", 1).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_soft_lb(12, "LD", 0).
shifts_soft_lb(12, "SE", 0).
shifts_soft_lb(12, "SN", 0).
shifts_soft_ub(12, "LD", 0).
shifts_soft_ub(12, "SE", 0).
shifts_soft_ub(12, "SN", 0).
shifts_hard_ub(12, "LD", 0).
shifts_hard_ub(12, "SE", 0).
shifts_hard_ub(12, "SN", 0).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "E", 1).
shifts_soft_ub(12, "N", 1).
shifts_soft_ub(12, "LM", 1).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "LD", 1).
shifts_soft_lb(13, "SE", 1).
shifts_soft_lb(13, "SN", 1).
shifts_soft_ub(13, "LD", 5).
shifts_soft_ub(13, "SE", 5).
shifts_soft_ub(13, "SN", 5).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "E", 1).
shifts_soft_ub(13, "N", 1).
shifts_soft_ub(13, "LM", 1).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 0).
shifts_soft_lb(14, "SE", 0).
shifts_soft_lb(14, "SN", 0).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "E", 1).
shifts_soft_ub(14, "N", 1).
shifts_soft_ub(14, "LM", 1).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 0).
shifts_soft_lb(15, "SE", 0).
shifts_soft_lb(15, "SN", 0).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "E", 1).
shifts_soft_ub(15, "N", 1).
shifts_soft_ub(15, "LM", 1).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 0).
shifts_soft_lb(16, "SE", 0).
shifts_soft_lb(16, "SN", 0).
shifts_soft_ub(16, "LD", 0).
shifts_soft_ub(16, "SE", 0).
shifts_soft_ub(16, "SN", 0).
shifts_hard_ub(16, "LD", 0).
shifts_hard_ub(16, "SE", 0).
shifts_hard_ub(16, "SN", 0).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "E", 1).
shifts_soft_ub(16, "N", 1).
shifts_soft_ub(16, "LM", 1).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 0).
shifts_soft_lb(17, "SE", 0).
shifts_soft_lb(17, "SN", 0).
shifts_soft_ub(17, "LD", 2).
shifts_soft_ub(17, "SE", 2).
shifts_soft_ub(17, "SN", 2).
shifts_hard_ub(17, "LD", 3).
shifts_hard_ub(17, "SE", 3).
shifts_hard_ub(17, "SN", 3).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "E", 1).
shifts_soft_ub(17, "N", 1).
shifts_soft_ub(17, "LM", 1).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 1).
shifts_soft_lb(18, "SE", 1).
shifts_soft_lb(18, "SN", 1).
shifts_soft_ub(18, "LD", 5).
shifts_soft_ub(18, "SE", 5).
shifts_soft_ub(18, "SN", 5).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "E", 1).
shifts_soft_ub(18, "N", 1).
shifts_soft_ub(18, "LM", 1).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 1).
shifts_soft_lb(19, "SE", 1).
shifts_soft_lb(19, "SN", 1).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "E", 1).
shifts_soft_ub(19, "N", 1).
shifts_soft_ub(19, "LM", 1).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 1).
shifts_soft_lb(20, "SE", 1).
shifts_soft_lb(20, "SN", 1).
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "E", 1).
shifts_soft_ub(20, "N", 1).
shifts_soft_ub(20, "LM", 1).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 1).
shifts_soft_lb(21, "SE", 1).
shifts_soft_lb(21, "SN", 1).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "E", 1).
shifts_soft_ub(21, "N", 1).
shifts_soft_ub(21, "LM", 1).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 1).
shifts_soft_lb(22, "SE", 1).
shifts_soft_lb(22, "SN", 1).
shifts_soft_ub(22, "LD", 3).
shifts_soft_ub(22, "SE", 3).
shifts_soft_ub(22, "SN", 3).
shifts_hard_ub(22, "LD", 4).
shifts_hard_ub(22, "SE", 4).
shifts_hard_ub(22, "SN", 4).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "E", 1).
shifts_soft_ub(22, "N", 1).
shifts_soft_ub(22, "LM", 1).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 1).
shifts_soft_lb(23, "SE", 1).
shifts_soft_lb(23, "SN", 1).
shifts_soft_ub(23, "LD", 3).
shifts_soft_ub(23, "SE", 3).
shifts_soft_ub(23, "SN", 3).
shifts_hard_ub(23, "LD", 4).
shifts_hard_ub(23, "SE", 4).
shifts_hard_ub(23, "SN", 4).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "E", 1).
shifts_soft_ub(23, "N", 1).
shifts_soft_ub(23, "LM", 1).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
forbidden_pair(5,16).
forbidden_pair(5,18).
forbidden_pair(5,21).
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
