% Staffs ---------------------------------------------------
staff(1, "Andrew Andrews", "Chief Nurse", "47640", 0).
staff(2, "Mary Chen", "Deputy Chief Nurse", "82629", 0).
staff(3, "Lisa Taylor DDS", "Deputy Chief Nurse", "29509", 0).
staff(4, "Heather Clark", "Deputy Chief Nurse", "57900", 0).
staff(5, "Paul Hayden", "Deputy Chief Nurse", "04516", 0).
staff(6, "Audrey Eaton", "Nurse", "57475", 0).
staff(7, "Molly Johnson", "Nurse", "28154", 0).
staff(8, "Angela Smith", "Nurse", "75736", 0).
staff(9, "Amanda Thomas", "Nurse", "95671", 0).
staff(10, "Robert Kelly", "Nurse", "81108", 0).
staff(11, "Kevin Bates", "Nurse", "72430", 0).
staff(12, "Dorothy Watts", "Nurse", "69183", 0).
staff(13, "Raymond Walker", "Nurse", "11547", 0).
staff(14, "Wesley Stark", "Nurse", "86908", 0).
staff(15, "Nathan Foster", "Nurse", "73199", 0).
staff(16, "Tara Mccoy", "Nurse", "48310", 0).
staff(17, "Xavier Bautista", "Nurse", "02671", 0).
staff(18, "Cathy Rogers", "Nurse", "67864", 0).
staff(19, "Melissa Tate", "Nurse", "88512", 0).
staff(20, "Christina Glenn", "Nurse", "27994", 0).
staff(21, "Garrett Allen", "Nurse", "84320", 0).
staff(22, "John Smith", "Nurse", "25017", 0).
staff(23, "Diane Gardner", "Nurse", "80594", 0).
staff(24, "Mariah Davis", "Nurse", "91445", 0).
staff(25, "Tricia Ramirez", "Nurse", "89312", 0).
staff(26, "Isabel Huff", "Nurse", "47149", 0).
staff(27, "Brandy Gomez", "Nurse", "78315", 0).
staff(28, "Timothy Hunter", "Nurse", "16315", 0).
staff(29, "Anthony Thompson", "Nurse", "56086", 0).
staff(30, "Nichole Salas", "Nurse", "12903", 0).
staff(31, "Melissa Thompson", "Nurse", "92332", 0).
staff(32, "William West", "Nurse", "86487", 0).
staff(33, "Curtis Bean", "Nurse", "27751", 0).
staff(34, "William Pierce", "Nurse", "16809", 0).
staff(35, "Kristin Powell", "Nurse", "20946", 0).
staff(36, "Mallory Willis", "Nurse", "12694", 0).
staff(37, "Tanya Shah", "Nurse", "62918", 0).
staff(38, "Lisa Hernandez", "Nurse", "22958", 0).
% Groups ---------------------------------------------------
staff_group("All").
staff_group("All", 1).
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
staff_group("All", 38).
% Skipped nurse group column "師長代行" (no mapping).
staff_group("Leaders").
staff_group("Leaders", 4).
staff_group("Leaders", 5).
staff_group("Leaders", 6).
staff_group("Leaders", 7).
staff_group("Leaders", 8).
staff_group("Leaders", 9).
staff_group("Leaders", 11).
staff_group("Leaders", 12).
staff_group("Leaders", 14).
staff_group("Leaders", 15).
staff_group("Leaders", 17).
staff_group("Leaders", 18).
staff_group("Leaders", 19).
staff_group("Leaders", 20).
staff_group("Leaders", 21).
staff_group("Leaders", 24).
staff_group("Leaders", 25).
% Skipped nurse group column "メンバー" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 35).
staff_group("Newcomers", 36).
staff_group("Newcomers", 37).
staff_group("Newcomers", 38).
% Skipped nurse group column "クリーン" (no mapping).
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
staff_group("Night", 14).
staff_group("Night", 15).
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
staff_group("Night", 38).
% Shift constraints ----------------------------------------
% Maximum consecutive working days
consecutive_work_ub(hard, 5).
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
next_shift(hard, "LD", "SE").
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
patterns_hard_lb("WR-WR", 2).
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
staff_dweek_hard_lb("All", "D", "Weekday", 6).
staff_dweek_hard_lb("All", "D", "Weekend", 6).
staff_dweek_hard_lb("All", "D", "Holiday", 6).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 6).
staff_dweek_soft_lb("All", "D", "Holiday", 6).
staff_dweek_soft_ub("All", "D", "Weekday", 16).
staff_dweek_soft_ub("All", "D", "Weekend", 7).
staff_dweek_soft_ub("All", "D", "Holiday", 7).
staff_dweek_hard_ub("All", "D", "Weekend", 7).
staff_dweek_hard_ub("All", "D", "Holiday", 7).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 5).
staff_dweek_hard_lb("All", "LD", "Weekend", 5).
staff_dweek_hard_lb("All", "LD", "Holiday", 5).
staff_dweek_soft_lb("All", "LD", "Weekday", 5).
staff_dweek_soft_lb("All", "LD", "Weekend", 5).
staff_dweek_soft_lb("All", "LD", "Holiday", 5).
staff_dweek_soft_ub("All", "LD", "Weekday", 5).
staff_dweek_soft_ub("All", "LD", "Weekend", 5).
staff_dweek_soft_ub("All", "LD", "Holiday", 5).
staff_dweek_hard_ub("All", "LD", "Weekday", 5).
staff_dweek_hard_ub("All", "LD", "Weekend", 5).
staff_dweek_hard_ub("All", "LD", "Holiday", 5).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 5).
shift_group("SN").
shift_group("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 5).
staff_dweek_hard_lb("All", "SE", "Weekend", 5).
staff_dweek_hard_lb("All", "SN", "Weekend", 5).
staff_dweek_hard_lb("All", "SE", "Holiday", 5).
staff_dweek_hard_lb("All", "SN", "Holiday", 5).
staff_dweek_soft_lb("All", "SE", "Weekday", 5).
staff_dweek_soft_lb("All", "SN", "Weekday", 5).
staff_dweek_soft_lb("All", "SE", "Weekend", 5).
staff_dweek_soft_lb("All", "SN", "Weekend", 5).
staff_dweek_soft_lb("All", "SE", "Holiday", 5).
staff_dweek_soft_lb("All", "SN", "Holiday", 5).
staff_dweek_soft_ub("All", "SE", "Weekday", 5).
staff_dweek_soft_ub("All", "SN", "Weekday", 5).
staff_dweek_soft_ub("All", "SE", "Weekend", 5).
staff_dweek_soft_ub("All", "SN", "Weekend", 5).
staff_dweek_soft_ub("All", "SE", "Holiday", 5).
staff_dweek_soft_ub("All", "SN", "Holiday", 5).
staff_dweek_hard_ub("All", "SE", "Weekday", 5).
staff_dweek_hard_ub("All", "SN", "Weekday", 5).
staff_dweek_hard_ub("All", "SE", "Weekend", 5).
staff_dweek_hard_ub("All", "SN", "Weekend", 5).
staff_dweek_hard_ub("All", "SE", "Holiday", 5).
staff_dweek_hard_ub("All", "SN", "Holiday", 5).
shift_group("LM").
shift_group("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 0).
staff_dweek_hard_lb("All", "LM", "Weekend", 0).
staff_dweek_hard_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_lb("All", "LM", "Weekday", 0).
staff_dweek_soft_lb("All", "LM", "Weekend", 0).
staff_dweek_soft_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_ub("All", "LM", "Weekday", 0).
staff_dweek_soft_ub("All", "LM", "Weekend", 0).
staff_dweek_soft_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_ub("All", "LM", "Weekday", 0).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
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
staff_dweek_soft_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 3).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 3).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 3).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 3).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 3).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 3).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 3).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 3).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 3).
% Skipped vertical constraint for nurse group "メンバー" (no mapping).
% Skipped vertical constraint for nurse group "メンバー" (no mapping).
% Skipped vertical constraint for nurse group "クリーン" (no mapping).
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Holiday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 0).
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
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
vertical_constraint_type("Leaders","LD","Staffs").
vertical_constraint_type("Leaders","SE","Staffs").
vertical_constraint_type("Leaders","SN","Staffs").
% Skipped vertical_constraint_type for nurse group "メンバー" (no mapping).
% Skipped vertical_constraint_type for nurse group "クリーン" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_hard_lb(2, "LD", 1).
shifts_hard_lb(2, "SE", 1).
shifts_hard_lb(2, "SN", 1).
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
shifts_hard_ub(2, "LD", 3).
shifts_hard_ub(2, "SE", 3).
shifts_hard_ub(2, "SN", 3).
shift_group("E").
shift_group("E", "E").
shifts_soft_lb(2, "E", 0).
shift_group("N").
shift_group("N", "N").
shifts_soft_lb(2, "N", 0).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "LM", 0).
shift_group("EM").
shift_group("EM", "EM").
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_hard_lb(3, "LD", 1).
shifts_hard_lb(3, "SE", 1).
shifts_hard_lb(3, "SN", 1).
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 2).
shifts_soft_ub(3, "SE", 2).
shifts_soft_ub(3, "SN", 2).
shifts_hard_ub(3, "LD", 3).
shifts_hard_ub(3, "SE", 3).
shifts_hard_ub(3, "SN", 3).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
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
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
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
shifts_soft_lb(5, "LD", 2).
shifts_soft_lb(5, "SE", 2).
shifts_soft_lb(5, "SN", 2).
shifts_soft_ub(5, "LD", 2).
shifts_soft_ub(5, "SE", 2).
shifts_soft_ub(5, "SN", 2).
shifts_hard_ub(5, "LD", 2).
shifts_hard_ub(5, "SE", 2).
shifts_hard_ub(5, "SN", 2).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
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
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 4).
shifts_soft_ub(6, "SE", 4).
shifts_soft_ub(6, "SN", 4).
shifts_hard_ub(6, "LD", 6).
shifts_hard_ub(6, "SE", 6).
shifts_hard_ub(6, "SN", 6).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
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
shifts_soft_lb(7, "LD", 4).
shifts_soft_lb(7, "SE", 4).
shifts_soft_lb(7, "SN", 4).
shifts_soft_ub(7, "LD", 5).
shifts_soft_ub(7, "SE", 5).
shifts_soft_ub(7, "SN", 5).
shifts_hard_ub(7, "LD", 6).
shifts_hard_ub(7, "SE", 6).
shifts_hard_ub(7, "SN", 6).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
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
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
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
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
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
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
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
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "LD", 6).
shifts_hard_ub(11, "SE", 6).
shifts_hard_ub(11, "SN", 6).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
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
shifts_hard_ub(12, "LD", 6).
shifts_hard_ub(12, "SE", 6).
shifts_hard_ub(12, "SN", 6).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_ub(12, "E", 0).
shifts_soft_ub(12, "N", 0).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
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
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_hard_ub(14, "LD", 6).
shifts_hard_ub(14, "SE", 6).
shifts_hard_ub(14, "SN", 6).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
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
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
% Staff 17:
shifts_hard_lb(17, "LD", 1).
shifts_hard_lb(17, "SE", 1).
shifts_hard_lb(17, "SN", 1).
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 5).
shifts_soft_ub(17, "SE", 5).
shifts_soft_ub(17, "SN", 5).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
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
shifts_soft_lb(18, "LD", 4).
shifts_soft_lb(18, "SE", 4).
shifts_soft_lb(18, "SN", 4).
shifts_soft_ub(18, "LD", 5).
shifts_soft_ub(18, "SE", 5).
shifts_soft_ub(18, "SN", 5).
shifts_hard_ub(18, "LD", 6).
shifts_hard_ub(18, "SE", 6).
shifts_hard_ub(18, "SN", 6).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
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
shifts_soft_lb(19, "LD", 4).
shifts_soft_lb(19, "SE", 4).
shifts_soft_lb(19, "SN", 4).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "LD", 6).
shifts_hard_ub(19, "SE", 6).
shifts_hard_ub(19, "SN", 6).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
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
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
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
shifts_soft_lb(21, "LD", 4).
shifts_soft_lb(21, "SE", 4).
shifts_soft_lb(21, "SN", 4).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
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
shifts_soft_ub(22, "LD", 5).
shifts_soft_ub(22, "SE", 5).
shifts_soft_ub(22, "SN", 5).
shifts_hard_ub(22, "LD", 6).
shifts_hard_ub(22, "SE", 6).
shifts_hard_ub(22, "SN", 6).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
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
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
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
shifts_soft_lb(24, "LD", 4).
shifts_soft_lb(24, "SE", 4).
shifts_soft_lb(24, "SN", 4).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_hard_ub(24, "LD", 6).
shifts_hard_ub(24, "SE", 6).
shifts_hard_ub(24, "SN", 6).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
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
shifts_soft_lb(25, "LD", 4).
shifts_soft_lb(25, "SE", 4).
shifts_soft_lb(25, "SN", 4).
shifts_soft_ub(25, "LD", 5).
shifts_soft_ub(25, "SE", 5).
shifts_soft_ub(25, "SN", 5).
shifts_hard_ub(25, "LD", 6).
shifts_hard_ub(25, "SE", 6).
shifts_hard_ub(25, "SN", 6).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
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
shifts_soft_ub(26, "LD", 2).
shifts_soft_ub(26, "SE", 2).
shifts_soft_ub(26, "SN", 2).
shifts_hard_ub(26, "LD", 2).
shifts_hard_ub(26, "SE", 2).
shifts_hard_ub(26, "SN", 2).
shifts_soft_lb(26, "E", 0).
shifts_soft_lb(26, "N", 0).
shifts_soft_ub(26, "E", 0).
shifts_soft_ub(26, "N", 0).
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
shifts_soft_lb(27, "LD", 4).
shifts_soft_lb(27, "SE", 4).
shifts_soft_lb(27, "SN", 4).
shifts_soft_ub(27, "LD", 5).
shifts_soft_ub(27, "SE", 5).
shifts_soft_ub(27, "SN", 5).
shifts_hard_ub(27, "LD", 6).
shifts_hard_ub(27, "SE", 6).
shifts_hard_ub(27, "SN", 6).
shifts_soft_lb(27, "E", 0).
shifts_soft_lb(27, "N", 0).
shifts_soft_ub(27, "E", 0).
shifts_soft_ub(27, "N", 0).
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
shifts_soft_lb(28, "LD", 4).
shifts_soft_lb(28, "SE", 4).
shifts_soft_lb(28, "SN", 4).
shifts_soft_ub(28, "LD", 5).
shifts_soft_ub(28, "SE", 5).
shifts_soft_ub(28, "SN", 5).
shifts_hard_ub(28, "LD", 6).
shifts_hard_ub(28, "SE", 6).
shifts_hard_ub(28, "SN", 6).
shifts_soft_lb(28, "E", 0).
shifts_soft_lb(28, "N", 0).
shifts_soft_ub(28, "E", 0).
shifts_soft_ub(28, "N", 0).
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
shifts_soft_lb(29, "LD", 4).
shifts_soft_lb(29, "SE", 4).
shifts_soft_lb(29, "SN", 4).
shifts_soft_ub(29, "LD", 5).
shifts_soft_ub(29, "SE", 5).
shifts_soft_ub(29, "SN", 5).
shifts_hard_ub(29, "LD", 6).
shifts_hard_ub(29, "SE", 6).
shifts_hard_ub(29, "SN", 6).
shifts_soft_lb(29, "E", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_ub(29, "E", 0).
shifts_soft_ub(29, "N", 0).
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
shifts_soft_lb(30, "LD", 4).
shifts_soft_lb(30, "SE", 4).
shifts_soft_lb(30, "SN", 4).
shifts_soft_ub(30, "LD", 5).
shifts_soft_ub(30, "SE", 5).
shifts_soft_ub(30, "SN", 5).
shifts_hard_ub(30, "LD", 6).
shifts_hard_ub(30, "SE", 6).
shifts_hard_ub(30, "SN", 6).
shifts_soft_lb(30, "E", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_ub(30, "E", 0).
shifts_soft_ub(30, "N", 0).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "LM", 0).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
% Staff 31:
shifts_hard_lb(31, "LD", 1).
shifts_hard_lb(31, "SE", 1).
shifts_hard_lb(31, "SN", 1).
shifts_soft_lb(31, "LD", 4).
shifts_soft_lb(31, "SE", 4).
shifts_soft_lb(31, "SN", 4).
shifts_soft_ub(31, "LD", 5).
shifts_soft_ub(31, "SE", 5).
shifts_soft_ub(31, "SN", 5).
shifts_hard_ub(31, "LD", 6).
shifts_hard_ub(31, "SE", 6).
shifts_hard_ub(31, "SN", 6).
shifts_soft_lb(31, "E", 0).
shifts_soft_lb(31, "N", 0).
shifts_soft_ub(31, "E", 0).
shifts_soft_ub(31, "N", 0).
shifts_hard_lb(31, "LM", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "LM", 0).
shifts_hard_lb(31, "EM", 0).
shifts_soft_lb(31, "EM", 0).
shifts_soft_ub(31, "EM", 0).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 1).
shifts_hard_lb(32, "SE", 1).
shifts_hard_lb(32, "SN", 1).
shifts_soft_lb(32, "LD", 4).
shifts_soft_lb(32, "SE", 4).
shifts_soft_lb(32, "SN", 4).
shifts_soft_ub(32, "LD", 5).
shifts_soft_ub(32, "SE", 5).
shifts_soft_ub(32, "SN", 5).
shifts_hard_ub(32, "LD", 6).
shifts_hard_ub(32, "SE", 6).
shifts_hard_ub(32, "SN", 6).
shifts_soft_lb(32, "E", 0).
shifts_soft_lb(32, "N", 0).
shifts_soft_ub(32, "E", 0).
shifts_soft_ub(32, "N", 0).
shifts_hard_lb(32, "LM", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "LM", 0).
shifts_hard_lb(32, "EM", 0).
shifts_soft_lb(32, "EM", 0).
shifts_soft_ub(32, "EM", 0).
shifts_hard_ub(32, "EM", 0).
% Staff 33:
shifts_hard_lb(33, "LD", 1).
shifts_hard_lb(33, "SE", 1).
shifts_hard_lb(33, "SN", 1).
shifts_soft_lb(33, "LD", 4).
shifts_soft_lb(33, "SE", 4).
shifts_soft_lb(33, "SN", 4).
shifts_soft_ub(33, "LD", 5).
shifts_soft_ub(33, "SE", 5).
shifts_soft_ub(33, "SN", 5).
shifts_hard_ub(33, "LD", 6).
shifts_hard_ub(33, "SE", 6).
shifts_hard_ub(33, "SN", 6).
shifts_soft_lb(33, "E", 0).
shifts_soft_lb(33, "N", 0).
shifts_soft_ub(33, "E", 0).
shifts_soft_ub(33, "N", 0).
shifts_hard_lb(33, "LM", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "LM", 0).
shifts_hard_lb(33, "EM", 0).
shifts_soft_lb(33, "EM", 0).
shifts_soft_ub(33, "EM", 0).
shifts_hard_ub(33, "EM", 0).
% Staff 34:
shifts_hard_lb(34, "LD", 1).
shifts_hard_lb(34, "SE", 1).
shifts_hard_lb(34, "SN", 1).
shifts_soft_lb(34, "LD", 4).
shifts_soft_lb(34, "SE", 4).
shifts_soft_lb(34, "SN", 4).
shifts_soft_ub(34, "LD", 5).
shifts_soft_ub(34, "SE", 5).
shifts_soft_ub(34, "SN", 5).
shifts_hard_ub(34, "LD", 6).
shifts_hard_ub(34, "SE", 6).
shifts_hard_ub(34, "SN", 6).
shifts_soft_lb(34, "E", 0).
shifts_soft_lb(34, "N", 0).
shifts_soft_ub(34, "E", 0).
shifts_soft_ub(34, "N", 0).
shifts_hard_lb(34, "LM", 0).
shifts_soft_lb(34, "LM", 0).
shifts_soft_ub(34, "LM", 0).
shifts_hard_ub(34, "LM", 0).
shifts_hard_lb(34, "EM", 0).
shifts_soft_lb(34, "EM", 0).
shifts_soft_ub(34, "EM", 0).
shifts_hard_ub(34, "EM", 0).
% Staff 35:
shifts_hard_lb(35, "LD", 1).
shifts_hard_lb(35, "SE", 1).
shifts_hard_lb(35, "SN", 1).
shifts_soft_lb(35, "LD", 2).
shifts_soft_lb(35, "SE", 2).
shifts_soft_lb(35, "SN", 2).
shifts_soft_ub(35, "LD", 3).
shifts_soft_ub(35, "SE", 3).
shifts_soft_ub(35, "SN", 3).
shifts_hard_ub(35, "LD", 3).
shifts_hard_ub(35, "SE", 3).
shifts_hard_ub(35, "SN", 3).
shifts_soft_lb(35, "E", 0).
shifts_soft_lb(35, "N", 0).
shifts_soft_ub(35, "E", 0).
shifts_soft_ub(35, "N", 0).
shifts_hard_lb(35, "LM", 0).
shifts_soft_lb(35, "LM", 0).
shifts_soft_ub(35, "LM", 0).
shifts_hard_ub(35, "LM", 0).
shifts_hard_lb(35, "EM", 0).
shifts_soft_lb(35, "EM", 0).
shifts_soft_ub(35, "EM", 0).
shifts_hard_ub(35, "EM", 0).
% Staff 36:
shifts_hard_lb(36, "LD", 1).
shifts_hard_lb(36, "SE", 1).
shifts_hard_lb(36, "SN", 1).
shifts_soft_lb(36, "LD", 2).
shifts_soft_lb(36, "SE", 2).
shifts_soft_lb(36, "SN", 2).
shifts_soft_ub(36, "LD", 3).
shifts_soft_ub(36, "SE", 3).
shifts_soft_ub(36, "SN", 3).
shifts_hard_ub(36, "LD", 3).
shifts_hard_ub(36, "SE", 3).
shifts_hard_ub(36, "SN", 3).
shifts_soft_lb(36, "E", 0).
shifts_soft_lb(36, "N", 0).
shifts_soft_ub(36, "E", 0).
shifts_soft_ub(36, "N", 0).
shifts_hard_lb(36, "LM", 0).
shifts_soft_lb(36, "LM", 0).
shifts_soft_ub(36, "LM", 0).
shifts_hard_ub(36, "LM", 0).
shifts_hard_lb(36, "EM", 0).
shifts_soft_lb(36, "EM", 0).
shifts_soft_ub(36, "EM", 0).
shifts_hard_ub(36, "EM", 0).
% Staff 37:
shifts_hard_lb(37, "LD", 1).
shifts_hard_lb(37, "SE", 1).
shifts_hard_lb(37, "SN", 1).
shifts_soft_lb(37, "LD", 2).
shifts_soft_lb(37, "SE", 2).
shifts_soft_lb(37, "SN", 2).
shifts_soft_ub(37, "LD", 3).
shifts_soft_ub(37, "SE", 3).
shifts_soft_ub(37, "SN", 3).
shifts_hard_ub(37, "LD", 3).
shifts_hard_ub(37, "SE", 3).
shifts_hard_ub(37, "SN", 3).
shifts_soft_lb(37, "E", 0).
shifts_soft_lb(37, "N", 0).
shifts_soft_ub(37, "E", 0).
shifts_soft_ub(37, "N", 0).
shifts_hard_lb(37, "LM", 0).
shifts_soft_lb(37, "LM", 0).
shifts_soft_ub(37, "LM", 0).
shifts_hard_ub(37, "LM", 0).
shifts_hard_lb(37, "EM", 0).
shifts_soft_lb(37, "EM", 0).
shifts_soft_ub(37, "EM", 0).
shifts_hard_ub(37, "EM", 0).
% Staff 38:
shifts_hard_lb(38, "LD", 1).
shifts_hard_lb(38, "SE", 1).
shifts_hard_lb(38, "SN", 1).
shifts_soft_lb(38, "LD", 2).
shifts_soft_lb(38, "SE", 2).
shifts_soft_lb(38, "SN", 2).
shifts_soft_ub(38, "LD", 3).
shifts_soft_ub(38, "SE", 3).
shifts_soft_ub(38, "SN", 3).
shifts_hard_ub(38, "LD", 3).
shifts_hard_ub(38, "SE", 3).
shifts_hard_ub(38, "SN", 3).
shifts_soft_lb(38, "E", 0).
shifts_soft_lb(38, "N", 0).
shifts_soft_ub(38, "E", 0).
shifts_soft_ub(38, "N", 0).
shifts_hard_lb(38, "LM", 0).
shifts_soft_lb(38, "LM", 0).
shifts_soft_ub(38, "LM", 0).
shifts_hard_ub(38, "LM", 0).
shifts_hard_lb(38, "EM", 0).
shifts_soft_lb(38, "EM", 0).
shifts_soft_ub(38, "EM", 0).
shifts_hard_ub(38, "EM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
% Positive shifts ------------------------------------------
% Staff 15:
staff_def_pos_request_dweek(15, "Mo", ("D")).
staff_def_pos_request_dweek(15, "Tu", ("D")).
staff_def_pos_request_dweek(15, "We", ("D")).
staff_def_pos_request_dweek(15, "Th", ("D")).
staff_def_pos_request_dweek(15, "Fr", ("D")).
% Negative shifts ------------------------------------------
