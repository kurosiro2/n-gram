% Staffs ---------------------------------------------------
staff(1, "Daniel Brown", "Chief Nurse", "85481", 0).
staff(2, "Tracey Johnson", "Deputy Chief Nurse", "15939", 0).
staff(3, "Andres Andrews", "Nurse", "81231", 0).
staff(4, "Anthony Henderson", "Nurse", "49799", 0).
staff(5, "Tiffany Cobb", "Nurse", "19414", 0).
staff(6, "Rebecca Greene", "Nurse", "07846", 0).
staff(7, "Michael Bridges", "Nurse", "53213", 0).
staff(8, "Susan Ibarra", "Nurse", "28979", 0).
staff(9, "Cody Harrington", "Nurse", "07743", 0).
staff(10, "Anna Estrada", "Nurse", "33050", 0).
staff(11, "Todd Miller", "Nurse", "80307", 0).
staff(12, "Mary Parker", "Nurse", "40036", 0).
staff(13, "Bryan Barr", "Nurse", "18578", 0).
staff(14, "Jamie Ayala", "Nurse", "02339", 0).
staff(15, "Christopher Ray", "Nurse", "33414", 0).
staff(16, "Allison Morgan", "Nurse", "41710", 0).
staff(17, "Dustin Padilla", "Nurse", "73509", 0).
staff(18, "Ellen Thomas", "Deputy Chief Nurse", "47501", 0).
staff(19, "Yesenia Davis", "Deputy Chief Nurse", "63970", 0).
staff(20, "Matthew Hutchinson", "Nurse", "74562", 0).
staff(21, "Tara Doyle", "Nurse", "34166", 0).
staff(22, "Jesse Wagner", "Nurse", "81682", 0).
staff(23, "Anita Lawson", "Nurse", "36009", 0).
staff(24, "Bailey Garcia", "Nurse", "84289", 0).
staff(25, "Christina Austin", "Nurse", "31261", 0).
staff(26, "Rhonda Cannon", "Nurse", "62191", 0).
staff(27, "Maurice Burke", "Nurse", "73445", 0).
staff(28, "Dalton Jones", "Nurse", "87761", 0).
staff(29, "Valerie Watts", "Nurse", "26426", 0).
staff(30, "John Miller", "Nurse", "78452", 0).
staff(31, "Jackie Reynolds", "Nurse", "16073", 0).
staff(32, "John Moyer", "Nurse", "83749", 0).
staff(33, "Heather Anderson", "Nurse", "47548", 0).
staff(34, "Mitchell Wilson", "Nurse", "70621", 0).
staff(35, "Jesus Allen", "Nurse", "51670", 0).
% Skipped staff no=36, name="Tonya Reed", job="補助者" (no mapping).
% Skipped staff no=37, name="Rebecca Kelly", job="補助者" (no mapping).
% Skipped staff no=38, name="Janice Ingram", job="補助者" (no mapping).
% Skipped staff no=39, name="Sabrina Estes", job="補助者" (no mapping).
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
% Skipped nurse group column "代行" (no mapping).
% Skipped nurse group column "A" (no mapping).
% Skipped nurse group column "B" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 16).
staff_group("Newcomers", 17).
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
staff_group("Leaders").
staff_group("Leaders", 2).
staff_group("Leaders", 3).
staff_group("Leaders", 4).
staff_group("Leaders", 6).
staff_group("Leaders", 7).
staff_group("Leaders", 8).
staff_group("Leaders", 9).
staff_group("Leaders", 10).
staff_group("Leaders", 11).
staff_group("Leaders", 12).
staff_group("Leaders", 13).
staff_group("Leaders", 18).
staff_group("Leaders", 20).
staff_group("Leaders", 21).
staff_group("Leaders", 22).
staff_group("Leaders", 23).
staff_group("Leaders", 24).
staff_group("Leaders", 25).
staff_group("Leaders", 26).
staff_group("Leaders", 28).
% Skipped nurse group column "平日代行" (no mapping).
% Skipped nurse group column "メンバー" (no mapping).
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
staff_dweek_hard_lb("All", "D", "Weekday", 6).
staff_dweek_hard_lb("All", "D", "Weekend", 4).
staff_dweek_hard_lb("All", "D", "Holiday", 4).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 4).
staff_dweek_soft_lb("All", "D", "Holiday", 4).
staff_dweek_soft_ub("All", "D", "Weekday", 9).
staff_dweek_soft_ub("All", "D", "Weekend", 5).
staff_dweek_soft_ub("All", "D", "Holiday", 5).
staff_dweek_hard_ub("All", "D", "Weekday", 9).
staff_dweek_hard_ub("All", "D", "Weekend", 5).
staff_dweek_hard_ub("All", "D", "Holiday", 5).
shift_group("LD").
shift_group("LD", "LD").
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
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 4).
shift_group("SN").
shift_group("SN", "SN").
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
shift_group("LM").
shift_group("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 1).
staff_dweek_hard_lb("All", "LM", "Weekend", 0).
staff_dweek_hard_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_lb("All", "LM", "Weekday", 1).
staff_dweek_soft_lb("All", "LM", "Weekend", 0).
staff_dweek_soft_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_ub("All", "LM", "Weekday", 1).
staff_dweek_soft_ub("All", "LM", "Weekend", 0).
staff_dweek_soft_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shift_group("LD+LM+E").
shift_group("LD+LM+E", "LD").
shift_group("LD+LM+E", "LM").
shift_group("LD+LM+E", "E").
staff_dweek_hard_lb("All", "LD+LM+E", "Weekday", 4).
staff_dweek_hard_lb("All", "LD+LM+E", "Weekend", 4).
staff_dweek_hard_lb("All", "LD+LM+E", "Holiday", 4).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekday", 4).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_lb("All", "LD+LM+E", "Holiday", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
staff_dweek_hard_ub("All", "LD+LM+E", "Weekday", 4).
staff_dweek_hard_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_hard_ub("All", "LD+LM+E", "Holiday", 4).
% Skipped vertical constraint for nurse group "代行" (no mapping).
% Skipped vertical constraint for nurse group "代行" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
staff_dweek_hard_lb("Newcomers", "LD", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "LD", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "LD", "Holiday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 0).
staff_dweek_soft_lb("Newcomers", "LD", "Weekday", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Weekday", 1).
staff_dweek_soft_lb("Newcomers", "LD", "Weekend", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Weekend", 1).
staff_dweek_soft_lb("Newcomers", "LD", "Holiday", 1).
staff_dweek_soft_lb("Newcomers", "SN", "Holiday", 1).
staff_dweek_soft_ub("Newcomers", "LD", "Weekday", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Weekday", 1).
staff_dweek_soft_ub("Newcomers", "LD", "Weekend", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Weekend", 1).
staff_dweek_soft_ub("Newcomers", "LD", "Holiday", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Holiday", 1).
staff_dweek_hard_ub("Newcomers", "LD", "Weekday", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Weekday", 1).
staff_dweek_hard_ub("Newcomers", "LD", "Weekend", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Weekend", 1).
staff_dweek_hard_ub("Newcomers", "LD", "Holiday", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Holiday", 1).
staff_dweek_hard_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "LD", "Holiday", 2).
staff_dweek_hard_lb("Leaders", "SE", "Holiday", 2).
staff_dweek_hard_lb("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 2).
staff_dweek_hard_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "LD", "Holiday", 2).
staff_dweek_hard_ub("Leaders", "SE", "Holiday", 2).
staff_dweek_hard_ub("Leaders", "SN", "Holiday", 2).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("All","LM","Staffs").
vertical_constraint_type("All","LD+LM+E","Staffs").
% Skipped vertical_constraint_type for nurse group "代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "A" (no mapping).
% Skipped vertical_constraint_type for nurse group "B" (no mapping).
vertical_constraint_type("Newcomers","LD","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
vertical_constraint_type("Leaders","LD","Staffs").
vertical_constraint_type("Leaders","SE","Staffs").
vertical_constraint_type("Leaders","SN","Staffs").
% Skipped vertical_constraint_type for nurse group "平日代行" (no mapping).
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_hard_lb(2, "LD", 2).
shifts_hard_lb(2, "SE", 2).
shifts_hard_lb(2, "SN", 2).
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 3).
shifts_soft_ub(2, "SE", 3).
shifts_soft_ub(2, "SN", 3).
shifts_hard_ub(2, "LD", 3).
shifts_hard_ub(2, "SE", 3).
shifts_hard_ub(2, "SN", 3).
shifts_hard_lb(2, "LM", 1).
shifts_soft_lb(2, "LM", 1).
shifts_soft_ub(2, "LM", 2).
shifts_hard_ub(2, "LM", 2).
% Staff 3:
shifts_hard_lb(3, "LD", 2).
shifts_hard_lb(3, "SE", 2).
shifts_hard_lb(3, "SN", 2).
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 3).
shifts_soft_ub(3, "SE", 3).
shifts_soft_ub(3, "SN", 3).
shifts_hard_ub(3, "LD", 3).
shifts_hard_ub(3, "SE", 3).
shifts_hard_ub(3, "SN", 3).
shifts_hard_lb(3, "LM", 1).
shifts_soft_lb(3, "LM", 1).
shifts_soft_ub(3, "LM", 2).
shifts_hard_ub(3, "LM", 2).
% Staff 4:
shifts_hard_lb(4, "LD", 2).
shifts_hard_lb(4, "SE", 2).
shifts_hard_lb(4, "SN", 2).
shifts_soft_lb(4, "LD", 3).
shifts_soft_lb(4, "SE", 3).
shifts_soft_lb(4, "SN", 3).
shifts_soft_ub(4, "LD", 4).
shifts_soft_ub(4, "SE", 4).
shifts_soft_ub(4, "SN", 4).
shifts_hard_ub(4, "LD", 5).
shifts_hard_ub(4, "SE", 5).
shifts_hard_ub(4, "SN", 5).
shifts_hard_lb(4, "LM", 1).
shifts_soft_lb(4, "LM", 1).
shifts_soft_ub(4, "LM", 3).
shifts_hard_ub(4, "LM", 3).
% Staff 5:
shifts_hard_lb(5, "LD", 1).
shifts_hard_lb(5, "SE", 1).
shifts_hard_lb(5, "SN", 1).
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 1).
shifts_soft_ub(5, "SE", 1).
shifts_soft_ub(5, "SN", 1).
shifts_hard_ub(5, "LD", 1).
shifts_hard_ub(5, "SE", 1).
shifts_hard_ub(5, "SN", 1).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 0).
% Staff 6:
shifts_hard_lb(6, "LD", 2).
shifts_hard_lb(6, "SE", 2).
shifts_hard_lb(6, "SN", 2).
shifts_soft_lb(6, "LD", 2).
shifts_soft_lb(6, "SE", 2).
shifts_soft_lb(6, "SN", 2).
shifts_soft_ub(6, "LD", 3).
shifts_soft_ub(6, "SE", 3).
shifts_soft_ub(6, "SN", 3).
shifts_hard_ub(6, "LD", 3).
shifts_hard_ub(6, "SE", 3).
shifts_hard_ub(6, "SN", 3).
shifts_hard_lb(6, "LM", 1).
shifts_soft_lb(6, "LM", 1).
shifts_soft_ub(6, "LM", 1).
shifts_hard_ub(6, "LM", 1).
% Staff 7:
shifts_hard_lb(7, "LD", 2).
shifts_hard_lb(7, "SE", 2).
shifts_hard_lb(7, "SN", 2).
shifts_soft_lb(7, "LD", 3).
shifts_soft_lb(7, "SE", 3).
shifts_soft_lb(7, "SN", 3).
shifts_soft_ub(7, "LD", 4).
shifts_soft_ub(7, "SE", 4).
shifts_soft_ub(7, "SN", 4).
shifts_hard_ub(7, "LD", 5).
shifts_hard_ub(7, "SE", 5).
shifts_hard_ub(7, "SN", 5).
shifts_hard_lb(7, "LM", 1).
shifts_soft_lb(7, "LM", 1).
shifts_soft_ub(7, "LM", 3).
shifts_hard_ub(7, "LM", 3).
% Staff 8:
shifts_hard_lb(8, "LD", 2).
shifts_hard_lb(8, "SE", 2).
shifts_hard_lb(8, "SN", 2).
shifts_soft_lb(8, "LD", 3).
shifts_soft_lb(8, "SE", 3).
shifts_soft_lb(8, "SN", 3).
shifts_soft_ub(8, "LD", 4).
shifts_soft_ub(8, "SE", 4).
shifts_soft_ub(8, "SN", 4).
shifts_hard_ub(8, "LD", 5).
shifts_hard_ub(8, "SE", 5).
shifts_hard_ub(8, "SN", 5).
shifts_hard_lb(8, "LM", 1).
shifts_soft_lb(8, "LM", 1).
shifts_soft_ub(8, "LM", 3).
shifts_hard_ub(8, "LM", 3).
% Staff 9:
shifts_hard_lb(9, "LD", 2).
shifts_hard_lb(9, "SE", 2).
shifts_hard_lb(9, "SN", 2).
shifts_soft_lb(9, "LD", 3).
shifts_soft_lb(9, "SE", 3).
shifts_soft_lb(9, "SN", 3).
shifts_soft_ub(9, "LD", 4).
shifts_soft_ub(9, "SE", 4).
shifts_soft_ub(9, "SN", 4).
shifts_hard_ub(9, "LD", 5).
shifts_hard_ub(9, "SE", 5).
shifts_hard_ub(9, "SN", 5).
shifts_hard_lb(9, "LM", 1).
shifts_soft_lb(9, "LM", 1).
shifts_soft_ub(9, "LM", 3).
shifts_hard_ub(9, "LM", 3).
% Staff 10:
shifts_hard_lb(10, "LD", 2).
shifts_hard_lb(10, "SE", 2).
shifts_hard_lb(10, "SN", 2).
shifts_soft_lb(10, "LD", 3).
shifts_soft_lb(10, "SE", 3).
shifts_soft_lb(10, "SN", 3).
shifts_soft_ub(10, "LD", 4).
shifts_soft_ub(10, "SE", 4).
shifts_soft_ub(10, "SN", 4).
shifts_hard_ub(10, "LD", 5).
shifts_hard_ub(10, "SE", 5).
shifts_hard_ub(10, "SN", 5).
shifts_hard_lb(10, "LM", 1).
shifts_soft_lb(10, "LM", 1).
shifts_soft_ub(10, "LM", 3).
shifts_hard_ub(10, "LM", 3).
% Staff 11:
shifts_hard_lb(11, "LD", 2).
shifts_hard_lb(11, "SE", 2).
shifts_hard_lb(11, "SN", 2).
shifts_soft_lb(11, "LD", 3).
shifts_soft_lb(11, "SE", 3).
shifts_soft_lb(11, "SN", 3).
shifts_soft_ub(11, "LD", 4).
shifts_soft_ub(11, "SE", 4).
shifts_soft_ub(11, "SN", 4).
shifts_hard_ub(11, "LD", 5).
shifts_hard_ub(11, "SE", 5).
shifts_hard_ub(11, "SN", 5).
shifts_hard_lb(11, "LM", 1).
shifts_soft_lb(11, "LM", 1).
shifts_soft_ub(11, "LM", 3).
shifts_hard_ub(11, "LM", 3).
% Staff 12:
shifts_hard_lb(12, "LD", 2).
shifts_hard_lb(12, "SE", 2).
shifts_hard_lb(12, "SN", 2).
shifts_soft_lb(12, "LD", 3).
shifts_soft_lb(12, "SE", 3).
shifts_soft_lb(12, "SN", 3).
shifts_soft_ub(12, "LD", 4).
shifts_soft_ub(12, "SE", 4).
shifts_soft_ub(12, "SN", 4).
shifts_hard_ub(12, "LD", 5).
shifts_hard_ub(12, "SE", 5).
shifts_hard_ub(12, "SN", 5).
shifts_hard_lb(12, "LM", 1).
shifts_soft_lb(12, "LM", 1).
shifts_soft_ub(12, "LM", 3).
shifts_hard_ub(12, "LM", 3).
% Staff 13:
shifts_hard_lb(13, "LD", 2).
shifts_hard_lb(13, "SE", 2).
shifts_hard_lb(13, "SN", 2).
shifts_soft_lb(13, "LD", 3).
shifts_soft_lb(13, "SE", 3).
shifts_soft_lb(13, "SN", 3).
shifts_soft_ub(13, "LD", 4).
shifts_soft_ub(13, "SE", 4).
shifts_soft_ub(13, "SN", 4).
shifts_hard_ub(13, "LD", 5).
shifts_hard_ub(13, "SE", 5).
shifts_hard_ub(13, "SN", 5).
shifts_hard_lb(13, "LM", 1).
shifts_soft_lb(13, "LM", 1).
shifts_soft_ub(13, "LM", 3).
shifts_hard_ub(13, "LM", 3).
% Staff 14:
shifts_hard_lb(14, "LD", 2).
shifts_hard_lb(14, "SE", 2).
shifts_hard_lb(14, "SN", 2).
shifts_soft_lb(14, "LD", 3).
shifts_soft_lb(14, "SE", 3).
shifts_soft_lb(14, "SN", 3).
shifts_soft_ub(14, "LD", 4).
shifts_soft_ub(14, "SE", 4).
shifts_soft_ub(14, "SN", 4).
shifts_hard_ub(14, "LD", 5).
shifts_hard_ub(14, "SE", 5).
shifts_hard_ub(14, "SN", 5).
shifts_hard_lb(14, "LM", 1).
shifts_soft_lb(14, "LM", 1).
shifts_soft_ub(14, "LM", 3).
shifts_hard_ub(14, "LM", 3).
% Staff 15:
shifts_hard_lb(15, "LD", 2).
shifts_hard_lb(15, "SE", 2).
shifts_hard_lb(15, "SN", 2).
shifts_soft_lb(15, "LD", 3).
shifts_soft_lb(15, "SE", 3).
shifts_soft_lb(15, "SN", 3).
shifts_soft_ub(15, "LD", 4).
shifts_soft_ub(15, "SE", 4).
shifts_soft_ub(15, "SN", 4).
shifts_hard_ub(15, "LD", 5).
shifts_hard_ub(15, "SE", 5).
shifts_hard_ub(15, "SN", 5).
shifts_hard_lb(15, "LM", 1).
shifts_soft_lb(15, "LM", 1).
shifts_soft_ub(15, "LM", 3).
shifts_hard_ub(15, "LM", 3).
% Staff 16:
shifts_hard_lb(16, "LD", 2).
shifts_hard_lb(16, "SE", 2).
shifts_hard_lb(16, "SN", 2).
shifts_soft_lb(16, "LD", 3).
shifts_soft_lb(16, "SE", 3).
shifts_soft_lb(16, "SN", 3).
shifts_soft_ub(16, "LD", 4).
shifts_soft_ub(16, "SE", 4).
shifts_soft_ub(16, "SN", 4).
shifts_hard_ub(16, "LD", 3).
shifts_hard_ub(16, "SE", 3).
shifts_hard_ub(16, "SN", 3).
shifts_hard_lb(16, "LM", 1).
shifts_soft_lb(16, "LM", 1).
shifts_soft_ub(16, "LM", 2).
shifts_hard_ub(16, "LM", 2).
% Staff 17:
shifts_hard_lb(17, "LD", 2).
shifts_hard_lb(17, "SE", 2).
shifts_hard_lb(17, "SN", 2).
shifts_soft_lb(17, "LD", 2).
shifts_soft_lb(17, "SE", 2).
shifts_soft_lb(17, "SN", 2).
shifts_soft_ub(17, "LD", 3).
shifts_soft_ub(17, "SE", 3).
shifts_soft_ub(17, "SN", 3).
shifts_hard_ub(17, "LD", 3).
shifts_hard_ub(17, "SE", 3).
shifts_hard_ub(17, "SN", 3).
shifts_hard_lb(17, "LM", 1).
shifts_soft_lb(17, "LM", 1).
shifts_soft_ub(17, "LM", 2).
shifts_hard_ub(17, "LM", 2).
% Staff 18:
shifts_hard_lb(18, "LD", 2).
shifts_hard_lb(18, "SE", 2).
shifts_hard_lb(18, "SN", 2).
shifts_soft_lb(18, "LD", 2).
shifts_soft_lb(18, "SE", 2).
shifts_soft_lb(18, "SN", 2).
shifts_soft_ub(18, "LD", 3).
shifts_soft_ub(18, "SE", 3).
shifts_soft_ub(18, "SN", 3).
shifts_hard_ub(18, "LD", 3).
shifts_hard_ub(18, "SE", 3).
shifts_hard_ub(18, "SN", 3).
shifts_hard_lb(18, "LM", 1).
shifts_soft_lb(18, "LM", 1).
shifts_soft_ub(18, "LM", 2).
shifts_hard_ub(18, "LM", 2).
% Staff 19:
shifts_hard_lb(19, "LD", 2).
shifts_hard_lb(19, "SE", 2).
shifts_hard_lb(19, "SN", 2).
shifts_soft_lb(19, "LD", 2).
shifts_soft_lb(19, "SE", 2).
shifts_soft_lb(19, "SN", 2).
shifts_soft_ub(19, "LD", 3).
shifts_soft_ub(19, "SE", 3).
shifts_soft_ub(19, "SN", 3).
shifts_hard_ub(19, "LD", 3).
shifts_hard_ub(19, "SE", 3).
shifts_hard_ub(19, "SN", 3).
shifts_hard_lb(19, "LM", 1).
shifts_soft_lb(19, "LM", 1).
shifts_soft_ub(19, "LM", 2).
shifts_hard_ub(19, "LM", 2).
% Staff 20:
shifts_hard_lb(20, "LD", 2).
shifts_hard_lb(20, "SE", 2).
shifts_hard_lb(20, "SN", 2).
shifts_soft_lb(20, "LD", 3).
shifts_soft_lb(20, "SE", 3).
shifts_soft_lb(20, "SN", 3).
shifts_soft_ub(20, "LD", 4).
shifts_soft_ub(20, "SE", 4).
shifts_soft_ub(20, "SN", 4).
shifts_hard_ub(20, "LD", 5).
shifts_hard_ub(20, "SE", 5).
shifts_hard_ub(20, "SN", 5).
shifts_hard_lb(20, "LM", 1).
shifts_soft_lb(20, "LM", 1).
shifts_soft_ub(20, "LM", 3).
shifts_hard_ub(20, "LM", 3).
% Staff 21:
shifts_hard_lb(21, "LD", 2).
shifts_hard_lb(21, "SE", 2).
shifts_hard_lb(21, "SN", 2).
shifts_soft_lb(21, "LD", 3).
shifts_soft_lb(21, "SE", 3).
shifts_soft_lb(21, "SN", 3).
shifts_soft_ub(21, "LD", 4).
shifts_soft_ub(21, "SE", 4).
shifts_soft_ub(21, "SN", 4).
shifts_hard_ub(21, "LD", 5).
shifts_hard_ub(21, "SE", 5).
shifts_hard_ub(21, "SN", 5).
shifts_hard_lb(21, "LM", 1).
shifts_soft_lb(21, "LM", 1).
shifts_soft_ub(21, "LM", 3).
shifts_hard_ub(21, "LM", 3).
% Staff 22:
shifts_hard_lb(22, "LD", 2).
shifts_hard_lb(22, "SE", 2).
shifts_hard_lb(22, "SN", 2).
shifts_soft_lb(22, "LD", 3).
shifts_soft_lb(22, "SE", 3).
shifts_soft_lb(22, "SN", 3).
shifts_soft_ub(22, "LD", 4).
shifts_soft_ub(22, "SE", 4).
shifts_soft_ub(22, "SN", 4).
shifts_hard_ub(22, "LD", 5).
shifts_hard_ub(22, "SE", 5).
shifts_hard_ub(22, "SN", 5).
shifts_hard_lb(22, "LM", 1).
shifts_soft_lb(22, "LM", 1).
shifts_soft_ub(22, "LM", 3).
shifts_hard_ub(22, "LM", 3).
% Staff 23:
shifts_hard_lb(23, "LD", 2).
shifts_hard_lb(23, "SE", 2).
shifts_hard_lb(23, "SN", 2).
shifts_soft_lb(23, "LD", 3).
shifts_soft_lb(23, "SE", 3).
shifts_soft_lb(23, "SN", 3).
shifts_soft_ub(23, "LD", 4).
shifts_soft_ub(23, "SE", 4).
shifts_soft_ub(23, "SN", 4).
shifts_hard_ub(23, "LD", 5).
shifts_hard_ub(23, "SE", 5).
shifts_hard_ub(23, "SN", 5).
shifts_hard_lb(23, "LM", 1).
shifts_soft_lb(23, "LM", 1).
shifts_soft_ub(23, "LM", 3).
shifts_hard_ub(23, "LM", 3).
% Staff 24:
shifts_hard_lb(24, "LD", 2).
shifts_hard_lb(24, "SE", 2).
shifts_hard_lb(24, "SN", 2).
shifts_soft_lb(24, "LD", 3).
shifts_soft_lb(24, "SE", 3).
shifts_soft_lb(24, "SN", 3).
shifts_soft_ub(24, "LD", 4).
shifts_soft_ub(24, "SE", 4).
shifts_soft_ub(24, "SN", 4).
shifts_hard_ub(24, "LD", 5).
shifts_hard_ub(24, "SE", 5).
shifts_hard_ub(24, "SN", 5).
shifts_hard_lb(24, "LM", 1).
shifts_soft_lb(24, "LM", 1).
shifts_soft_ub(24, "LM", 3).
shifts_hard_ub(24, "LM", 3).
% Staff 25:
shifts_hard_lb(25, "LD", 2).
shifts_hard_lb(25, "SE", 2).
shifts_hard_lb(25, "SN", 2).
shifts_soft_lb(25, "LD", 3).
shifts_soft_lb(25, "SE", 3).
shifts_soft_lb(25, "SN", 3).
shifts_soft_ub(25, "LD", 4).
shifts_soft_ub(25, "SE", 4).
shifts_soft_ub(25, "SN", 4).
shifts_hard_ub(25, "LD", 5).
shifts_hard_ub(25, "SE", 5).
shifts_hard_ub(25, "SN", 5).
shifts_hard_lb(25, "LM", 1).
shifts_soft_lb(25, "LM", 1).
shifts_soft_ub(25, "LM", 3).
shifts_hard_ub(25, "LM", 3).
% Staff 26:
shifts_hard_lb(26, "LD", 2).
shifts_hard_lb(26, "SE", 2).
shifts_hard_lb(26, "SN", 2).
shifts_soft_lb(26, "LD", 3).
shifts_soft_lb(26, "SE", 3).
shifts_soft_lb(26, "SN", 3).
shifts_soft_ub(26, "LD", 4).
shifts_soft_ub(26, "SE", 4).
shifts_soft_ub(26, "SN", 4).
shifts_hard_ub(26, "LD", 5).
shifts_hard_ub(26, "SE", 5).
shifts_hard_ub(26, "SN", 5).
shifts_hard_lb(26, "LM", 1).
shifts_soft_lb(26, "LM", 1).
shifts_soft_ub(26, "LM", 3).
shifts_hard_ub(26, "LM", 3).
% Staff 27:
shifts_hard_lb(27, "LD", 2).
shifts_hard_lb(27, "SE", 2).
shifts_hard_lb(27, "SN", 2).
shifts_soft_lb(27, "LD", 3).
shifts_soft_lb(27, "SE", 3).
shifts_soft_lb(27, "SN", 3).
shifts_soft_ub(27, "LD", 4).
shifts_soft_ub(27, "SE", 4).
shifts_soft_ub(27, "SN", 4).
shifts_hard_ub(27, "LD", 5).
shifts_hard_ub(27, "SE", 5).
shifts_hard_ub(27, "SN", 5).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "LM", 0).
% Staff 28:
shifts_hard_lb(28, "LD", 2).
shifts_hard_lb(28, "SE", 2).
shifts_hard_lb(28, "SN", 2).
shifts_soft_lb(28, "LD", 3).
shifts_soft_lb(28, "SE", 3).
shifts_soft_lb(28, "SN", 3).
shifts_soft_ub(28, "LD", 4).
shifts_soft_ub(28, "SE", 4).
shifts_soft_ub(28, "SN", 4).
shifts_hard_ub(28, "LD", 5).
shifts_hard_ub(28, "SE", 5).
shifts_hard_ub(28, "SN", 5).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "LM", 0).
% Staff 29:
shifts_hard_lb(29, "LD", 2).
shifts_hard_lb(29, "SE", 2).
shifts_hard_lb(29, "SN", 2).
shifts_soft_lb(29, "LD", 3).
shifts_soft_lb(29, "SE", 3).
shifts_soft_lb(29, "SN", 3).
shifts_soft_ub(29, "LD", 4).
shifts_soft_ub(29, "SE", 4).
shifts_soft_ub(29, "SN", 4).
shifts_hard_ub(29, "LD", 5).
shifts_hard_ub(29, "SE", 5).
shifts_hard_ub(29, "SN", 5).
shifts_hard_lb(29, "LM", 0).
shifts_soft_lb(29, "LM", 0).
shifts_soft_ub(29, "LM", 0).
shifts_hard_ub(29, "LM", 0).
% Staff 30:
shifts_hard_lb(30, "LD", 2).
shifts_hard_lb(30, "SE", 2).
shifts_hard_lb(30, "SN", 2).
shifts_soft_lb(30, "LD", 3).
shifts_soft_lb(30, "SE", 3).
shifts_soft_lb(30, "SN", 3).
shifts_soft_ub(30, "LD", 4).
shifts_soft_ub(30, "SE", 4).
shifts_soft_ub(30, "SN", 4).
shifts_hard_ub(30, "LD", 5).
shifts_hard_ub(30, "SE", 5).
shifts_hard_ub(30, "SN", 5).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "LM", 0).
% Staff 31:
shifts_hard_lb(31, "LD", 2).
shifts_hard_lb(31, "SE", 2).
shifts_hard_lb(31, "SN", 2).
shifts_soft_lb(31, "LD", 2).
shifts_soft_lb(31, "SE", 2).
shifts_soft_lb(31, "SN", 2).
shifts_soft_ub(31, "LD", 3).
shifts_soft_ub(31, "SE", 3).
shifts_soft_ub(31, "SN", 3).
shifts_hard_ub(31, "LD", 3).
shifts_hard_ub(31, "SE", 3).
shifts_hard_ub(31, "SN", 3).
shifts_hard_lb(31, "LM", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "LM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 2).
shifts_hard_lb(32, "SE", 2).
shifts_hard_lb(32, "SN", 2).
shifts_soft_lb(32, "LD", 2).
shifts_soft_lb(32, "SE", 2).
shifts_soft_lb(32, "SN", 2).
shifts_soft_ub(32, "LD", 3).
shifts_soft_ub(32, "SE", 3).
shifts_soft_ub(32, "SN", 3).
shifts_hard_ub(32, "LD", 3).
shifts_hard_ub(32, "SE", 3).
shifts_hard_ub(32, "SN", 3).
shifts_hard_lb(32, "LM", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "LM", 0).
% Staff 33:
shifts_hard_lb(33, "LD", 2).
shifts_hard_lb(33, "SE", 2).
shifts_hard_lb(33, "SN", 2).
shifts_soft_lb(33, "LD", 2).
shifts_soft_lb(33, "SE", 2).
shifts_soft_lb(33, "SN", 2).
shifts_soft_ub(33, "LD", 3).
shifts_soft_ub(33, "SE", 3).
shifts_soft_ub(33, "SN", 3).
shifts_hard_ub(33, "LD", 3).
shifts_hard_ub(33, "SE", 3).
shifts_hard_ub(33, "SN", 3).
shifts_hard_lb(33, "LM", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "LM", 0).
% Staff 34:
% Staff 35:
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
