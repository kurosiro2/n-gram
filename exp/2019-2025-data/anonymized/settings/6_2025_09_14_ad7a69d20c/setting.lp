% Staffs ---------------------------------------------------
staff(1, "John Steele", "Chief Nurse", "11502", 0).
staff(2, "Holly Hale", "Deputy Chief Nurse", "56900", 0).
staff(3, "Michael Blackwell", "Nurse", "76588", 0).
staff(4, "Brandon Dunn", "Nurse", "56391", 0).
staff(5, "Shannon Cobb", "Nurse", "28541", 0).
staff(6, "Sean Orozco", "Nurse", "07086", 0).
staff(7, "Andrew Miller", "Nurse", "47416", 0).
staff(8, "Brent Duran", "Nurse", "74984", 0).
staff(9, "Kathy Wilkins", "Nurse", "39196", 0).
staff(10, "Bruce Hart", "Nurse", "64228", 0).
staff(11, "George Stone", "Nurse", "38948", 0).
staff(12, "Gloria Austin", "Nurse", "99956", 0).
staff(13, "David Blackburn", "Deputy Chief Nurse", "32956", 0).
staff(14, "Wendy Hicks", "Nurse", "79874", 0).
staff(15, "Brittany Merritt", "Nurse", "70461", 0).
staff(16, "Jeremy Smith", "Nurse", "81684", 0).
staff(17, "Michele Mercado", "Nurse", "24924", 0).
staff(18, "Krystal Anderson", "Nurse", "11919", 0).
staff(19, "Elizabeth Johnson", "Nurse", "74513", 0).
staff(20, "Stephen Rodriguez", "Nurse", "65410", 0).
staff(21, "Robert Jacobs", "Nurse", "98236", 0).
staff(22, "David Villanueva", "Nurse", "92799", 0).
staff(23, "Steven Edwards", "Nurse", "60582", 0).
staff(24, "Marcus Gonzalez", "Nurse", "24578", 0).
staff(25, "Lauren Harvey", "Deputy Chief Nurse", "77336", 0).
staff(26, "Sherri Dawson", "Nurse", "15423", 0).
staff(27, "Holly Burns", "Nurse", "36235", 0).
staff(28, "Douglas Tucker", "Nurse", "76784", 0).
staff(29, "Jessica Hernandez", "Nurse", "58951", 0).
staff(30, "Lauren Johnston", "Nurse", "10534", 0).
staff(31, "Pamela Hernandez", "Nurse", "39771", 0).
staff(32, "Sophia Williams", "Nurse", "93704", 0).
staff(33, "Austin Perry", "Nurse", "78847", 0).
staff(34, "Mitchell Guerra", "Nurse", "72902", 0).
staff(35, "Bobby Brennan", "Nurse", "71141", 0).
staff(36, "Adam Hill", "Nurse", "28348", 0).
staff(37, "Jennifer Kramer", "Nurse", "28954", 0).
staff(38, "Aaron Wood", "Nurse", "00526", 0).
staff(39, "Susan Gonzalez", "Nurse", "61976", 0).
staff(40, "Jessica Nash", "Nurse", "89818", 0).
% Skipped staff no=41, name="Joseph Lam", job="補助者" (no mapping).
% Skipped staff no=42, name="Linda Douglas", job="補助者" (no mapping).
% Skipped staff no=43, name="Lisa Yates", job="補助者" (no mapping).
% Skipped staff no=44, name="Robin Vargas", job="補助者" (no mapping).
% Skipped staff no=45, name="Kelly Taylor", job="補助者" (no mapping).
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
staff_group("All", 34).
staff_group("All", 35).
staff_group("All", 36).
staff_group("All", 37).
% Skipped nurse group column "リ－ダ－" (no mapping).
% Skipped nurse group column "A" (no mapping).
% Skipped nurse group column "B" (no mapping).
% Skipped nurse group column "C" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 12).
staff_group("Newcomers", 23).
staff_group("Newcomers", 24).
staff_group("Newcomers", 36).
staff_group("Newcomers", 37).
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
staff_group("Night", 19).
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
staff_group("Night", 34).
staff_group("Night", 35).
staff_group("Night", 36).
staff_group("Night", 37).
staff_group("Night", 39).
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
staff_dweek_hard_lb("All", "D", "Weekday", 7).
staff_dweek_hard_lb("All", "D", "Weekend", 6).
staff_dweek_hard_lb("All", "D", "Holiday", 6).
staff_dweek_soft_lb("All", "D", "Weekday", 8).
staff_dweek_soft_lb("All", "D", "Weekend", 5).
staff_dweek_soft_lb("All", "D", "Holiday", 5).
staff_dweek_soft_ub("All", "D", "Weekday", 14).
staff_dweek_soft_ub("All", "D", "Weekend", 7).
staff_dweek_soft_ub("All", "D", "Holiday", 7).
staff_dweek_hard_ub("All", "D", "Weekday", 18).
staff_dweek_hard_ub("All", "D", "Weekend", 8).
staff_dweek_hard_ub("All", "D", "Holiday", 8).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 3).
staff_dweek_soft_lb("All", "LD", "Weekend", 3).
staff_dweek_soft_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
staff_dweek_hard_ub("All", "LD", "Weekday", 5).
staff_dweek_hard_ub("All", "LD", "Weekend", 5).
staff_dweek_hard_ub("All", "LD", "Holiday", 5).
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
staff_dweek_hard_ub("All", "SE", "Weekday", 5).
staff_dweek_hard_ub("All", "SN", "Weekday", 5).
staff_dweek_hard_ub("All", "SE", "Weekend", 5).
staff_dweek_hard_ub("All", "SN", "Weekend", 5).
staff_dweek_hard_ub("All", "SE", "Holiday", 5).
staff_dweek_hard_ub("All", "SN", "Holiday", 5).
shift_group("LM").
shift_group("LM", "LM").
staff_dweek_soft_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekday", 2).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "リ－ダ－" (no mapping).
% Skipped vertical constraint for nurse group "リ－ダ－" (no mapping).
% Skipped vertical constraint for nurse group "リ－ダ－" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "C" (no mapping).
% Skipped vertical constraint for nurse group "C" (no mapping).
% Skipped vertical constraint for nurse group "C" (no mapping).
% Skipped vertical constraint for nurse group "C" (no mapping).
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
vertical_constraint_type("All","LM","Staffs").
% Skipped vertical_constraint_type for nurse group "A" (no mapping).
% Skipped vertical_constraint_type for nurse group "リ－ダ－" (no mapping).
% Skipped vertical_constraint_type for nurse group "B" (no mapping).
% Skipped vertical_constraint_type for nurse group "C" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_soft_lb(2, "LD", 1).
shifts_soft_lb(2, "SE", 1).
shifts_soft_lb(2, "SN", 1).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
% Staff 3:
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 4).
shifts_soft_ub(3, "SE", 4).
shifts_soft_ub(3, "SN", 4).
% Staff 4:
shifts_soft_lb(4, "LD", 4).
shifts_soft_lb(4, "SE", 4).
shifts_soft_lb(4, "SN", 4).
shifts_soft_ub(4, "LD", 6).
shifts_soft_ub(4, "SE", 6).
shifts_soft_ub(4, "SN", 6).
shifts_soft_lb(4, "LM", 1).
% Staff 5:
shifts_soft_lb(5, "LD", 4).
shifts_soft_lb(5, "SE", 4).
shifts_soft_lb(5, "SN", 4).
shifts_soft_ub(5, "LD", 6).
shifts_soft_ub(5, "SE", 6).
shifts_soft_ub(5, "SN", 6).
shifts_soft_lb(5, "LM", 1).
% Staff 6:
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 6).
shifts_soft_ub(6, "SE", 6).
shifts_soft_ub(6, "SN", 6).
shifts_soft_lb(6, "LM", 1).
% Staff 7:
shifts_soft_lb(7, "LD", 4).
shifts_soft_lb(7, "SE", 4).
shifts_soft_lb(7, "SN", 4).
shifts_soft_ub(7, "LD", 6).
shifts_soft_ub(7, "SE", 6).
shifts_soft_ub(7, "SN", 6).
shifts_soft_lb(7, "LM", 1).
% Staff 8:
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 6).
shifts_soft_ub(8, "SE", 6).
shifts_soft_ub(8, "SN", 6).
shifts_soft_lb(8, "LM", 1).
% Staff 9:
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 6).
shifts_soft_ub(9, "SE", 6).
shifts_soft_ub(9, "SN", 6).
shifts_soft_lb(9, "LM", 1).
% Staff 10:
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 6).
shifts_soft_ub(10, "SE", 6).
shifts_soft_ub(10, "SN", 6).
shifts_soft_lb(10, "LM", 1).
% Staff 11:
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 6).
shifts_soft_ub(11, "SE", 6).
shifts_soft_ub(11, "SN", 6).
shifts_soft_lb(11, "LM", 1).
% Staff 12:
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 6).
shifts_soft_ub(12, "SE", 6).
shifts_soft_ub(12, "SN", 6).
shifts_soft_lb(12, "LM", 1).
% Staff 13:
shifts_soft_lb(13, "LD", 1).
shifts_soft_lb(13, "SE", 1).
shifts_soft_lb(13, "SN", 1).
shifts_soft_ub(13, "LD", 2).
shifts_soft_ub(13, "SE", 2).
shifts_soft_ub(13, "SN", 2).
shifts_soft_lb(13, "LM", 1).
% Staff 14:
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 6).
shifts_soft_ub(14, "SE", 6).
shifts_soft_ub(14, "SN", 6).
shifts_soft_lb(14, "LM", 1).
% Staff 15:
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 6).
shifts_soft_ub(15, "SE", 6).
shifts_soft_ub(15, "SN", 6).
shifts_soft_lb(15, "LM", 1).
% Staff 16:
shifts_soft_lb(16, "LD", 4).
shifts_soft_lb(16, "SE", 4).
shifts_soft_lb(16, "SN", 4).
shifts_soft_ub(16, "LD", 6).
shifts_soft_ub(16, "SE", 6).
shifts_soft_ub(16, "SN", 6).
shifts_soft_lb(16, "LM", 1).
% Staff 17:
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 6).
shifts_soft_ub(17, "SE", 6).
shifts_soft_ub(17, "SN", 6).
shifts_soft_lb(17, "LM", 1).
% Staff 18:
shifts_soft_lb(18, "LD", 4).
shifts_soft_lb(18, "SE", 4).
shifts_soft_lb(18, "SN", 4).
shifts_soft_ub(18, "LD", 6).
shifts_soft_ub(18, "SE", 6).
shifts_soft_ub(18, "SN", 6).
shifts_soft_lb(18, "LM", 1).
% Staff 19:
shifts_soft_lb(19, "LD", 4).
shifts_soft_lb(19, "SE", 4).
shifts_soft_lb(19, "SN", 4).
shifts_soft_ub(19, "LD", 6).
shifts_soft_ub(19, "SE", 6).
shifts_soft_ub(19, "SN", 6).
shifts_soft_lb(19, "LM", 1).
% Staff 20:
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 6).
shifts_soft_ub(20, "SE", 6).
shifts_soft_ub(20, "SN", 6).
shifts_soft_lb(20, "LM", 1).
% Staff 21:
shifts_soft_lb(21, "LD", 4).
shifts_soft_lb(21, "SE", 4).
shifts_soft_lb(21, "SN", 4).
shifts_soft_ub(21, "LD", 6).
shifts_soft_ub(21, "SE", 6).
shifts_soft_ub(21, "SN", 6).
shifts_soft_lb(21, "LM", 1).
% Staff 22:
shifts_soft_lb(22, "LD", 3).
shifts_soft_lb(22, "SE", 3).
shifts_soft_lb(22, "SN", 3).
shifts_soft_ub(22, "LD", 4).
shifts_soft_ub(22, "SE", 4).
shifts_soft_ub(22, "SN", 4).
shifts_soft_lb(22, "LM", 1).
% Staff 23:
shifts_soft_lb(23, "LD", 2).
shifts_soft_lb(23, "SE", 2).
shifts_soft_lb(23, "SN", 2).
shifts_soft_ub(23, "LD", 3).
shifts_soft_ub(23, "SE", 3).
shifts_soft_ub(23, "SN", 3).
shifts_soft_lb(23, "LM", 1).
% Staff 24:
shifts_soft_lb(24, "LD", 2).
shifts_soft_lb(24, "SE", 2).
shifts_soft_lb(24, "SN", 2).
shifts_soft_ub(24, "LD", 3).
shifts_soft_ub(24, "SE", 3).
shifts_soft_ub(24, "SN", 3).
shifts_soft_lb(24, "LM", 1).
% Staff 25:
shifts_soft_lb(25, "LD", 1).
shifts_soft_lb(25, "SE", 1).
shifts_soft_lb(25, "SN", 1).
shifts_soft_ub(25, "LD", 2).
shifts_soft_ub(25, "SE", 2).
shifts_soft_ub(25, "SN", 2).
shifts_soft_lb(25, "LM", 1).
% Staff 26:
shifts_soft_lb(26, "LD", 0).
shifts_soft_lb(26, "SE", 0).
shifts_soft_lb(26, "SN", 0).
shifts_soft_ub(26, "LD", 0).
shifts_soft_ub(26, "SE", 0).
shifts_soft_ub(26, "SN", 0).
shifts_soft_lb(26, "LM", 1).
% Staff 27:
shifts_soft_lb(27, "LD", 4).
shifts_soft_lb(27, "SE", 4).
shifts_soft_lb(27, "SN", 4).
shifts_soft_ub(27, "LD", 6).
shifts_soft_ub(27, "SE", 6).
shifts_soft_ub(27, "SN", 6).
shifts_soft_lb(27, "LM", 1).
% Staff 28:
shifts_soft_lb(28, "LD", 4).
shifts_soft_lb(28, "SE", 4).
shifts_soft_lb(28, "SN", 4).
shifts_soft_ub(28, "LD", 6).
shifts_soft_ub(28, "SE", 6).
shifts_soft_ub(28, "SN", 6).
shifts_soft_lb(28, "LM", 1).
% Staff 29:
shifts_soft_lb(29, "LD", 4).
shifts_soft_lb(29, "SE", 4).
shifts_soft_lb(29, "SN", 4).
shifts_soft_ub(29, "LD", 6).
shifts_soft_ub(29, "SE", 6).
shifts_soft_ub(29, "SN", 6).
shifts_soft_lb(29, "LM", 1).
% Staff 30:
shifts_soft_lb(30, "LD", 4).
shifts_soft_lb(30, "SE", 4).
shifts_soft_lb(30, "SN", 4).
shifts_soft_ub(30, "LD", 6).
shifts_soft_ub(30, "SE", 6).
shifts_soft_ub(30, "SN", 6).
shifts_soft_lb(30, "LM", 1).
% Staff 31:
shifts_soft_lb(31, "LD", 4).
shifts_soft_lb(31, "SE", 4).
shifts_soft_lb(31, "SN", 4).
shifts_soft_ub(31, "LD", 6).
shifts_soft_ub(31, "SE", 6).
shifts_soft_ub(31, "SN", 6).
shifts_soft_lb(31, "LM", 1).
% Staff 32:
shifts_soft_lb(32, "LD", 4).
shifts_soft_lb(32, "SE", 4).
shifts_soft_lb(32, "SN", 4).
shifts_soft_ub(32, "LD", 6).
shifts_soft_ub(32, "SE", 6).
shifts_soft_ub(32, "SN", 6).
shifts_soft_lb(32, "LM", 1).
% Staff 33:
shifts_soft_lb(33, "LD", 4).
shifts_soft_lb(33, "SE", 4).
shifts_soft_lb(33, "SN", 4).
shifts_soft_ub(33, "LD", 6).
shifts_soft_ub(33, "SE", 6).
shifts_soft_ub(33, "SN", 6).
shifts_soft_lb(33, "LM", 1).
% Staff 34:
shifts_soft_lb(34, "LD", 4).
shifts_soft_lb(34, "SE", 4).
shifts_soft_lb(34, "SN", 4).
shifts_soft_ub(34, "LD", 6).
shifts_soft_ub(34, "SE", 6).
shifts_soft_ub(34, "SN", 6).
shifts_soft_lb(34, "LM", 1).
% Staff 35:
shifts_soft_lb(35, "LD", 4).
shifts_soft_lb(35, "SE", 4).
shifts_soft_lb(35, "SN", 4).
shifts_soft_ub(35, "LD", 6).
shifts_soft_ub(35, "SE", 6).
shifts_soft_ub(35, "SN", 6).
shifts_soft_lb(35, "LM", 1).
% Staff 36:
shifts_soft_lb(36, "LD", 2).
shifts_soft_lb(36, "SE", 2).
shifts_soft_lb(36, "SN", 2).
shifts_soft_ub(36, "LD", 3).
shifts_soft_ub(36, "SE", 3).
shifts_soft_ub(36, "SN", 3).
shifts_soft_lb(36, "LM", 1).
% Staff 37:
shifts_soft_lb(37, "LD", 2).
shifts_soft_lb(37, "SE", 2).
shifts_soft_lb(37, "SN", 2).
shifts_soft_ub(37, "LD", 3).
shifts_soft_ub(37, "SE", 3).
shifts_soft_ub(37, "SN", 3).
shifts_soft_lb(37, "LM", 1).
% Staff 38:
% Staff 39:
% Staff 40:
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
