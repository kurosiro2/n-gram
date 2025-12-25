% Staffs ---------------------------------------------------
staff(1, "Todd Tyler", "Chief Nurse", "14238", 0).
staff(2, "Scott Johnson", "Deputy Chief Nurse", "50584", 3).
staff(3, "Laura Watson", "Deputy Chief Nurse", "82132", 3).
staff(4, "Michelle Mitchell", "Nurse", "36466", 3).
staff(5, "Christine Taylor", "Nurse", "89033", 3).
staff(6, "Dominic Hernandez", "Nurse", "08779", 0).
staff(7, "Alexander Jackson", "Nurse", "29654", 0).
staff(8, "Larry Porter", "Nurse", "05417", 0).
staff(9, "Tyler Tran", "Nurse", "54907", 0).
staff(10, "Deanna Turner", "Nurse", "42007", 0).
staff(11, "Robert Newman", "Nurse", "52102", 0).
staff(12, "Allison Hull", "Nurse", "69635", 0).
staff(13, "Carolyn Scott", "Nurse", "81764", 0).
staff(14, "Jean Pennington", "Nurse", "28871", 0).
staff(15, "Sandra Joseph", "Nurse", "21510", 0).
staff(16, "Aaron Washington", "Nurse", "62223", 0).
staff(17, "Tracey Johnson", "Deputy Chief Nurse", "15939", 3).
staff(18, "Jimmy Davis", "Nurse", "74243", 3).
staff(19, "April Johnson", "Nurse", "37337", 3).
staff(20, "Richard Stephens", "Nurse", "45891", 3).
staff(21, "Donna Casey", "Nurse", "13905", 3).
staff(22, "Robert Jarvis", "Nurse", "50813", 0).
staff(23, "Destiny Campbell", "Nurse", "21890", 0).
staff(24, "Victor Vargas", "Nurse", "39748", 0).
staff(25, "Amber Thompson", "Nurse", "43374", 0).
staff(26, "Ann Allen DDS", "Nurse", "41192", 0).
staff(27, "Alex Torres", "Nurse", "76023", 0).
staff(28, "Edward Hines", "Nurse", "42223", 0).
staff(29, "Heather Lowe", "Nurse", "16954", 0).
staff(30, "Wesley Rivera", "Nurse", "17683", 0).
staff(31, "Danielle Mcbride", "Nurse", "09002", 0).
staff(32, "Patricia Levy", "Nurse", "41516", 0).
staff(33, "James Evans", "Nurse", "29319", 0).
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
staff_group("All", 25).
staff_group("All", 26).
staff_group("All", 27).
staff_group("All", 28).
staff_group("All", 29).
staff_group("All", 30).
staff_group("All", 31).
staff_group("All", 32).
staff_group("All", 33).
staff_group("Heads").
staff_group("Heads", 1).
% Skipped nurse group column "師長代行" (no mapping).
% Skipped nurse group column "A" (no mapping).
% Skipped nurse group column "B" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 9).
staff_group("Newcomers", 11).
staff_group("Newcomers", 12).
staff_group("Newcomers", 13).
staff_group("Newcomers", 14).
staff_group("Newcomers", 15).
staff_group("Newcomers", 16).
staff_group("Newcomers", 27).
staff_group("Newcomers", 28).
staff_group("Newcomers", 29).
staff_group("Newcomers", 30).
staff_group("Newcomers", 31).
staff_group("Newcomers", 32).
staff_group("Newcomers", 33).
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
staff_group("Night", 12).
staff_group("Night", 13).
staff_group("Night", 14).
staff_group("Night", 15).
staff_group("Night", 16).
staff_group("Night", 17).
staff_group("Night", 18).
staff_group("Night", 20).
staff_group("Night", 21).
staff_group("Night", 22).
staff_group("Night", 23).
staff_group("Night", 24).
staff_group("Night", 25).
staff_group("Night", 26).
staff_group("Night", 27).
staff_group("Night", 28).
staff_group("Night", 29).
staff_group("Night", 30).
staff_group("Night", 31).
staff_group("Night", 32).
staff_group("Night", 33).
% Shift constraints ----------------------------------------
% Maximum consecutive working days
consecutive_work_ub(hard, 6).
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
forbidden_pattern("SE-SN-SE-SN-SE-SN").
pattern("SE-SN-SE-SN-SE-SN", 6).
pattern("SE-SN-SE-SN-SE-SN", 0, "SE").
pattern("SE-SN-SE-SN-SE-SN", 1, "SN").
pattern("SE-SN-SE-SN-SE-SN", 2, "SE").
pattern("SE-SN-SE-SN-SE-SN", 3, "SN").
pattern("SE-SN-SE-SN-SE-SN", 4, "SE").
pattern("SE-SN-SE-SN-SE-SN", 5, "SN").
forbidden_pattern("D-D-D-D-D").
pattern("D-D-D-D-D", 5).
pattern("D-D-D-D-D", 0, "D").
pattern("D-D-D-D-D", 1, "D").
pattern("D-D-D-D-D", 2, "D").
pattern("D-D-D-D-D", 3, "D").
pattern("D-D-D-D-D", 4, "D").
forbidden_pattern("D-D-D-LD-LM-SE-SN").
pattern("D-D-D-LD-LM-SE-SN", 7).
pattern("D-D-D-LD-LM-SE-SN", 0, "D").
pattern("D-D-D-LD-LM-SE-SN", 1, "D").
pattern("D-D-D-LD-LM-SE-SN", 2, "D").
pattern("D-D-D-LD-LM-SE-SN", 3, "LD").
pattern("D-D-D-LD-LM-SE-SN", 4, "LM").
pattern("D-D-D-LD-LM-SE-SN", 5, "SE").
pattern("D-D-D-LD-LM-SE-SN", 6, "SN").
forbidden_pattern("LM-LM").
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
forbidden_pattern("D-D-D-D-LD").
pattern("D-D-D-D-LD", 5).
pattern("D-D-D-D-LD", 0, "D").
pattern("D-D-D-D-LD", 1, "D").
pattern("D-D-D-D-LD", 2, "D").
pattern("D-D-D-D-LD", 3, "D").
pattern("D-D-D-D-LD", 4, "LD").
forbidden_pattern("D-D-D-D-LM").
pattern("D-D-D-D-LM", 5).
pattern("D-D-D-D-LM", 0, "D").
pattern("D-D-D-D-LM", 1, "D").
pattern("D-D-D-D-LM", 2, "D").
pattern("D-D-D-D-LM", 3, "D").
pattern("D-D-D-D-LM", 4, "LM").
forbidden_pattern("D-D-D-D-SE-SN-SE-SN").
pattern("D-D-D-D-SE-SN-SE-SN", 8).
pattern("D-D-D-D-SE-SN-SE-SN", 0, "D").
pattern("D-D-D-D-SE-SN-SE-SN", 1, "D").
pattern("D-D-D-D-SE-SN-SE-SN", 2, "D").
pattern("D-D-D-D-SE-SN-SE-SN", 3, "D").
pattern("D-D-D-D-SE-SN-SE-SN", 4, "SE").
pattern("D-D-D-D-SE-SN-SE-SN", 5, "SN").
pattern("D-D-D-D-SE-SN-SE-SN", 6, "SE").
pattern("D-D-D-D-SE-SN-SE-SN", 7, "SN").
% Assignable previous/next shifts
prev_shift(hard, "SN", "SE").
next_shift(hard, "SN", "SE").
next_shift(hard, "SN", "WR").
next_shift(hard, "SN", "PH").
prev_shift(hard, "SE", "LD").
prev_shift(hard, "SE", "LM").
next_shift(hard, "SE", "SN").
% Pattern occurrences
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_soft_ub("WR-WR", 3).
% Vertical constraints -------------------------------------
shift_group("D").
shift_group("D", "D").
staff_dweek_hard_lb("All", "D", "Weekday", 4).
staff_dweek_hard_lb("All", "D", "Weekend", 1).
staff_dweek_hard_lb("All", "D", "Holiday", 1).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 2).
staff_dweek_soft_lb("All", "D", "Holiday", 2).
staff_dweek_soft_ub("All", "D", "Weekday", 13).
staff_dweek_soft_ub("All", "D", "Weekend", 3).
staff_dweek_soft_ub("All", "D", "Holiday", 3).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 4).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 4).
shift_group("SN").
shift_group("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 4).
staff_dweek_hard_lb("All", "LD", "Weekend", 4).
staff_dweek_hard_lb("All", "SE", "Weekend", 4).
staff_dweek_hard_lb("All", "SN", "Weekend", 4).
staff_dweek_hard_lb("All", "LD", "Holiday", 4).
staff_dweek_hard_lb("All", "SE", "Holiday", 4).
staff_dweek_hard_lb("All", "SN", "Holiday", 4).
staff_dweek_soft_lb("All", "LD", "Weekday", 4).
staff_dweek_soft_lb("All", "SE", "Weekday", 4).
staff_dweek_soft_lb("All", "SN", "Weekday", 4).
staff_dweek_soft_lb("All", "LD", "Weekend", 4).
staff_dweek_soft_lb("All", "SE", "Weekend", 4).
staff_dweek_soft_lb("All", "SN", "Weekend", 4).
staff_dweek_soft_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_lb("All", "SE", "Holiday", 4).
staff_dweek_soft_lb("All", "SN", "Holiday", 4).
staff_dweek_soft_ub("All", "LD", "Weekday", 5).
staff_dweek_soft_ub("All", "SE", "Weekday", 5).
staff_dweek_soft_ub("All", "SN", "Weekday", 5).
staff_dweek_soft_ub("All", "LD", "Weekend", 5).
staff_dweek_soft_ub("All", "SE", "Weekend", 5).
staff_dweek_soft_ub("All", "SN", "Weekend", 5).
staff_dweek_soft_ub("All", "LD", "Holiday", 5).
staff_dweek_soft_ub("All", "SE", "Holiday", 5).
staff_dweek_soft_ub("All", "SN", "Holiday", 5).
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
staff_dweek_soft_lb("All", "LD+LM+E", "Weekday", 4).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_soft_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 5).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
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
shift_group("SE+E").
shift_group("SE+E", "SE").
shift_group("SE+E", "E").
staff_dweek_soft_lb("Night", "SE+E", "Weekday", 4).
shift_group("SN+N").
shift_group("SN+N", "SN").
shift_group("SN+N", "N").
staff_dweek_soft_lb("Night", "SN+N", "Weekday", 4).
staff_dweek_soft_lb("Night", "SE+E", "Weekend", 4).
staff_dweek_soft_lb("Night", "SN+N", "Weekend", 4).
staff_dweek_soft_lb("Night", "SE+E", "Holiday", 4).
staff_dweek_soft_lb("Night", "SN+N", "Holiday", 4).
staff_dweek_soft_ub("Night", "SE+E", "Weekday", 4).
staff_dweek_soft_ub("Night", "SN+N", "Weekday", 4).
staff_dweek_soft_ub("Night", "SE+E", "Weekend", 4).
staff_dweek_soft_ub("Night", "SN+N", "Weekend", 4).
staff_dweek_soft_ub("Night", "SE+E", "Holiday", 4).
staff_dweek_soft_ub("Night", "SN+N", "Holiday", 4).
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("All","E","Staffs").
vertical_constraint_type("All","N","Staffs").
vertical_constraint_type("All","LM","Staffs").
vertical_constraint_type("All","LD+LM+E","Staffs").
vertical_constraint_type("Heads","D","Staffs").
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "A" (no mapping).
% Skipped vertical_constraint_type for nurse group "B" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
vertical_constraint_type("Night","SE+E","Staffs").
vertical_constraint_type("Night","SN+N","Staffs").
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
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 0).
shifts_soft_lb(2, "SE", 0).
shifts_soft_lb(2, "SN", 0).
shifts_soft_ub(2, "LD", 0).
shifts_soft_ub(2, "SE", 0).
shifts_soft_ub(2, "SN", 0).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "E", 1).
shifts_soft_ub(2, "N", 1).
shifts_soft_ub(2, "LM", 1).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 1).
shifts_soft_lb(3, "SE", 1).
shifts_soft_lb(3, "SN", 1).
shifts_soft_ub(3, "LD", 3).
shifts_soft_ub(3, "SE", 3).
shifts_soft_ub(3, "SN", 3).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "E", 1).
shifts_soft_ub(3, "N", 1).
shifts_soft_ub(3, "LM", 1).
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
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 3).
shifts_soft_ub(5, "SE", 3).
shifts_soft_ub(5, "SN", 3).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "E", 1).
shifts_soft_ub(5, "N", 1).
shifts_soft_ub(5, "LM", 1).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 1).
shifts_soft_lb(6, "SE", 1).
shifts_soft_lb(6, "SN", 1).
shifts_soft_ub(6, "LD", 3).
shifts_soft_ub(6, "SE", 3).
shifts_soft_ub(6, "SN", 3).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "E", 1).
shifts_soft_ub(6, "N", 1).
shifts_soft_ub(6, "LM", 1).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 1).
shifts_soft_lb(7, "SE", 1).
shifts_soft_lb(7, "SN", 1).
shifts_soft_ub(7, "LD", 3).
shifts_soft_ub(7, "SE", 3).
shifts_soft_ub(7, "SN", 3).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "E", 1).
shifts_soft_ub(7, "N", 1).
shifts_soft_ub(7, "LM", 1).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 1).
shifts_soft_lb(8, "SE", 1).
shifts_soft_lb(8, "SN", 1).
shifts_soft_ub(8, "LD", 3).
shifts_soft_ub(8, "SE", 3).
shifts_soft_ub(8, "SN", 3).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "E", 1).
shifts_soft_ub(8, "N", 1).
shifts_soft_ub(8, "LM", 1).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 1).
shifts_soft_lb(9, "SE", 1).
shifts_soft_lb(9, "SN", 1).
shifts_soft_ub(9, "LD", 3).
shifts_soft_ub(9, "SE", 3).
shifts_soft_ub(9, "SN", 3).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "E", 1).
shifts_soft_ub(9, "N", 1).
shifts_soft_ub(9, "LM", 1).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 1).
shifts_soft_lb(10, "SE", 1).
shifts_soft_lb(10, "SN", 1).
shifts_soft_ub(10, "LD", 3).
shifts_soft_ub(10, "SE", 3).
shifts_soft_ub(10, "SN", 3).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "E", 1).
shifts_soft_ub(10, "N", 1).
shifts_soft_ub(10, "LM", 1).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 1).
shifts_soft_lb(11, "SE", 1).
shifts_soft_lb(11, "SN", 1).
shifts_soft_ub(11, "LD", 3).
shifts_soft_ub(11, "SE", 3).
shifts_soft_ub(11, "SN", 3).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "E", 1).
shifts_soft_ub(11, "N", 1).
shifts_soft_ub(11, "LM", 1).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_soft_lb(12, "LD", 1).
shifts_soft_lb(12, "SE", 1).
shifts_soft_lb(12, "SN", 1).
shifts_soft_ub(12, "LD", 3).
shifts_soft_ub(12, "SE", 3).
shifts_soft_ub(12, "SN", 3).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "E", 1).
shifts_soft_ub(12, "N", 1).
shifts_soft_ub(12, "LM", 1).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "LD", 1).
shifts_soft_lb(13, "SE", 1).
shifts_soft_lb(13, "SN", 1).
shifts_soft_ub(13, "LD", 3).
shifts_soft_ub(13, "SE", 3).
shifts_soft_ub(13, "SN", 3).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "E", 1).
shifts_soft_ub(13, "N", 1).
shifts_soft_ub(13, "LM", 1).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 1).
shifts_soft_lb(14, "SE", 1).
shifts_soft_lb(14, "SN", 1).
shifts_soft_ub(14, "LD", 3).
shifts_soft_ub(14, "SE", 3).
shifts_soft_ub(14, "SN", 3).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "E", 1).
shifts_soft_ub(14, "N", 1).
shifts_soft_ub(14, "LM", 1).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 1).
shifts_soft_lb(15, "SE", 1).
shifts_soft_lb(15, "SN", 1).
shifts_soft_ub(15, "LD", 3).
shifts_soft_ub(15, "SE", 3).
shifts_soft_ub(15, "SN", 3).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "E", 1).
shifts_soft_ub(15, "N", 1).
shifts_soft_ub(15, "LM", 1).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 1).
shifts_soft_lb(16, "SE", 1).
shifts_soft_lb(16, "SN", 1).
shifts_soft_ub(16, "LD", 3).
shifts_soft_ub(16, "SE", 3).
shifts_soft_ub(16, "SN", 3).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "E", 1).
shifts_soft_ub(16, "N", 1).
shifts_soft_ub(16, "LM", 1).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 1).
shifts_soft_lb(17, "SE", 1).
shifts_soft_lb(17, "SN", 1).
shifts_soft_ub(17, "LD", 2).
shifts_soft_ub(17, "SE", 2).
shifts_soft_ub(17, "SN", 2).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "E", 1).
shifts_soft_ub(17, "N", 1).
shifts_soft_ub(17, "LM", 1).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 1).
shifts_soft_lb(18, "SE", 1).
shifts_soft_lb(18, "SN", 1).
shifts_soft_ub(18, "LD", 3).
shifts_soft_ub(18, "SE", 3).
shifts_soft_ub(18, "SN", 3).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "E", 1).
shifts_soft_ub(18, "N", 1).
shifts_soft_ub(18, "LM", 1).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 0).
shifts_soft_lb(19, "SE", 0).
shifts_soft_lb(19, "SN", 0).
shifts_soft_ub(19, "LD", 0).
shifts_soft_ub(19, "SE", 0).
shifts_soft_ub(19, "SN", 0).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "E", 1).
shifts_soft_ub(19, "N", 1).
shifts_soft_ub(19, "LM", 1).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 1).
shifts_soft_lb(20, "SE", 1).
shifts_soft_lb(20, "SN", 1).
shifts_soft_ub(20, "LD", 3).
shifts_soft_ub(20, "SE", 3).
shifts_soft_ub(20, "SN", 3).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "E", 1).
shifts_soft_ub(20, "N", 1).
shifts_soft_ub(20, "LM", 1).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 1).
shifts_soft_lb(21, "SE", 1).
shifts_soft_lb(21, "SN", 1).
shifts_soft_ub(21, "LD", 3).
shifts_soft_ub(21, "SE", 3).
shifts_soft_ub(21, "SN", 3).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "E", 1).
shifts_soft_ub(21, "N", 1).
shifts_soft_ub(21, "LM", 1).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 1).
shifts_soft_lb(22, "SE", 1).
shifts_soft_lb(22, "SN", 1).
shifts_soft_ub(22, "LD", 3).
shifts_soft_ub(22, "SE", 3).
shifts_soft_ub(22, "SN", 3).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "E", 1).
shifts_soft_ub(22, "N", 1).
shifts_soft_ub(22, "LM", 1).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 1).
shifts_soft_lb(23, "SE", 1).
shifts_soft_lb(23, "SN", 1).
shifts_soft_ub(23, "LD", 3).
shifts_soft_ub(23, "SE", 3).
shifts_soft_ub(23, "SN", 3).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "E", 1).
shifts_soft_ub(23, "N", 1).
shifts_soft_ub(23, "LM", 1).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
shifts_soft_lb(24, "LD", 1).
shifts_soft_lb(24, "SE", 1).
shifts_soft_lb(24, "SN", 1).
shifts_soft_ub(24, "LD", 3).
shifts_soft_ub(24, "SE", 3).
shifts_soft_ub(24, "SN", 3).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_lb(24, "LM", 0).
shifts_soft_ub(24, "E", 1).
shifts_soft_ub(24, "N", 1).
shifts_soft_ub(24, "LM", 1).
shifts_hard_ub(24, "EM", 0).
% Staff 25:
shifts_soft_lb(25, "LD", 1).
shifts_soft_lb(25, "SE", 1).
shifts_soft_lb(25, "SN", 1).
shifts_soft_ub(25, "LD", 3).
shifts_soft_ub(25, "SE", 3).
shifts_soft_ub(25, "SN", 3).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_lb(25, "LM", 0).
shifts_soft_ub(25, "E", 1).
shifts_soft_ub(25, "N", 1).
shifts_soft_ub(25, "LM", 1).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_soft_lb(26, "LD", 1).
shifts_soft_lb(26, "SE", 1).
shifts_soft_lb(26, "SN", 1).
shifts_soft_ub(26, "LD", 3).
shifts_soft_ub(26, "SE", 3).
shifts_soft_ub(26, "SN", 3).
shifts_soft_lb(26, "E", 0).
shifts_soft_lb(26, "N", 0).
shifts_soft_lb(26, "LM", 0).
shifts_soft_ub(26, "E", 1).
shifts_soft_ub(26, "N", 1).
shifts_soft_ub(26, "LM", 1).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_soft_lb(27, "LD", 1).
shifts_soft_lb(27, "SE", 1).
shifts_soft_lb(27, "SN", 1).
shifts_soft_ub(27, "LD", 3).
shifts_soft_ub(27, "SE", 3).
shifts_soft_ub(27, "SN", 3).
shifts_soft_lb(27, "E", 0).
shifts_soft_lb(27, "N", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "E", 1).
shifts_soft_ub(27, "N", 1).
shifts_soft_ub(27, "LM", 1).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_soft_lb(28, "LD", 1).
shifts_soft_lb(28, "SE", 1).
shifts_soft_lb(28, "SN", 1).
shifts_soft_ub(28, "LD", 3).
shifts_soft_ub(28, "SE", 3).
shifts_soft_ub(28, "SN", 3).
shifts_soft_lb(28, "E", 0).
shifts_soft_lb(28, "N", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "E", 1).
shifts_soft_ub(28, "N", 1).
shifts_soft_ub(28, "LM", 1).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_soft_lb(29, "LD", 1).
shifts_soft_lb(29, "SE", 1).
shifts_soft_lb(29, "SN", 1).
shifts_soft_ub(29, "LD", 3).
shifts_soft_ub(29, "SE", 3).
shifts_soft_ub(29, "SN", 3).
shifts_soft_lb(29, "E", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_lb(29, "LM", 0).
shifts_soft_ub(29, "E", 1).
shifts_soft_ub(29, "N", 1).
shifts_soft_ub(29, "LM", 1).
shifts_hard_ub(29, "EM", 0).
% Staff 30:
shifts_soft_lb(30, "LD", 1).
shifts_soft_lb(30, "SE", 1).
shifts_soft_lb(30, "SN", 1).
shifts_soft_ub(30, "LD", 3).
shifts_soft_ub(30, "SE", 3).
shifts_soft_ub(30, "SN", 3).
shifts_soft_lb(30, "E", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "E", 1).
shifts_soft_ub(30, "N", 1).
shifts_soft_ub(30, "LM", 1).
shifts_hard_ub(30, "EM", 0).
% Staff 31:
shifts_soft_lb(31, "LD", 1).
shifts_soft_lb(31, "SE", 1).
shifts_soft_lb(31, "SN", 1).
shifts_soft_ub(31, "LD", 3).
shifts_soft_ub(31, "SE", 3).
shifts_soft_ub(31, "SN", 3).
shifts_soft_lb(31, "E", 0).
shifts_soft_lb(31, "N", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "E", 1).
shifts_soft_ub(31, "N", 1).
shifts_soft_ub(31, "LM", 1).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_soft_lb(32, "LD", 1).
shifts_soft_lb(32, "SE", 1).
shifts_soft_lb(32, "SN", 1).
shifts_soft_ub(32, "LD", 3).
shifts_soft_ub(32, "SE", 3).
shifts_soft_ub(32, "SN", 3).
shifts_soft_lb(32, "E", 0).
shifts_soft_lb(32, "N", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "E", 1).
shifts_soft_ub(32, "N", 1).
shifts_soft_ub(32, "LM", 1).
shifts_hard_ub(32, "EM", 0).
% Staff 33:
shifts_soft_lb(33, "LD", 1).
shifts_soft_lb(33, "SE", 1).
shifts_soft_lb(33, "SN", 1).
shifts_soft_ub(33, "LD", 3).
shifts_soft_ub(33, "SE", 3).
shifts_soft_ub(33, "SN", 3).
shifts_soft_lb(33, "E", 0).
shifts_soft_lb(33, "N", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "E", 1).
shifts_soft_ub(33, "N", 1).
shifts_soft_ub(33, "LM", 1).
shifts_hard_ub(33, "EM", 0).
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
