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
staff(26, "Christina Austin", "Nurse", "31261", 0).
staff(27, "Rhonda Cannon", "Nurse", "62191", 0).
staff(29, "Maurice Burke", "Nurse", "73445", 0).
staff(30, "Dalton Jones", "Nurse", "87761", 0).
staff(31, "Valerie Watts", "Nurse", "26426", 0).
staff(32, "John Miller", "Nurse", "78452", 0).
staff(33, "Jackie Reynolds", "Nurse", "16073", 0).
staff(34, "John Moyer", "Nurse", "83749", 0).
staff(35, "Heather Anderson", "Nurse", "47548", 0).
staff(36, "Mitchell Wilson", "Nurse", "70621", 0).
staff(37, "Jesus Allen", "Nurse", "51670", 0).
% Skipped staff no=38, name="Tonya Reed", job="補助者" (no mapping).
% Skipped staff no=39, name="Rebecca Kelly", job="補助者" (no mapping).
% Skipped staff no=40, name="Janice Ingram", job="補助者" (no mapping).
% Skipped staff no=41, name="Sabrina Estes", job="補助者" (no mapping).
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
staff_group("All", 16).
staff_group("All", 17).
staff_group("All", 18).
staff_group("All", 19).
staff_group("All", 20).
staff_group("All", 21).
staff_group("All", 22).
staff_group("All", 23).
staff_group("All", 24).
staff_group("All", 26).
staff_group("All", 27).
staff_group("All", 29).
staff_group("All", 30).
staff_group("All", 31).
staff_group("All", 32).
staff_group("All", 33).
staff_group("All", 34).
staff_group("All", 35).
% Skipped nurse group column "師長代行" (no mapping).
% Skipped nurse group column "平日代行" (no mapping).
% Skipped nurse group column "夜勤リーダー" (no mapping).
% Skipped nurse group column "日勤リーダー" (no mapping).
% Skipped nurse group column "Aチーム" (no mapping).
% Skipped nurse group column "Bチーム" (no mapping).
% Skipped nurse group column "Cチーム" (no mapping).
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
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 4).
staff_dweek_soft_lb("All", "LD", "Weekend", 4).
staff_dweek_soft_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
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
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "平日代行" (no mapping).
% Skipped vertical constraint for nurse group "夜勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "夜勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "夜勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "日勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "日勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "日勤リーダー" (no mapping).
% Skipped vertical constraint for nurse group "Aチーム" (no mapping).
% Skipped vertical constraint for nurse group "Aチーム" (no mapping).
% Skipped vertical constraint for nurse group "Aチーム" (no mapping).
% Skipped vertical constraint for nurse group "Bチーム" (no mapping).
% Skipped vertical constraint for nurse group "Bチーム" (no mapping).
% Skipped vertical constraint for nurse group "Bチーム" (no mapping).
% Skipped vertical constraint for nurse group "Cチーム" (no mapping).
% Skipped vertical constraint for nurse group "Cチーム" (no mapping).
% Skipped vertical constraint for nurse group "Cチーム" (no mapping).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 0).
shift_group("SN").
shift_group("SN", "SN").
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
staff_dweek_hard_lb("Night", "SE", "Weekday", 4).
staff_dweek_hard_lb("Night", "SN", "Weekday", 4).
staff_dweek_hard_lb("Night", "SE", "Weekend", 4).
staff_dweek_hard_lb("Night", "SN", "Weekend", 4).
staff_dweek_hard_lb("Night", "SE", "Holiday", 4).
staff_dweek_hard_lb("Night", "SN", "Holiday", 4).
staff_dweek_soft_lb("Night", "SE", "Weekday", 4).
staff_dweek_soft_lb("Night", "SN", "Weekday", 4).
staff_dweek_soft_lb("Night", "SE", "Weekend", 4).
staff_dweek_soft_lb("Night", "SN", "Weekend", 4).
staff_dweek_soft_lb("Night", "SE", "Holiday", 4).
staff_dweek_soft_lb("Night", "SN", "Holiday", 4).
staff_dweek_soft_ub("Night", "SE", "Weekday", 4).
staff_dweek_soft_ub("Night", "SN", "Weekday", 4).
staff_dweek_soft_ub("Night", "SE", "Weekend", 4).
staff_dweek_soft_ub("Night", "SN", "Weekend", 4).
staff_dweek_soft_ub("Night", "SE", "Holiday", 4).
staff_dweek_soft_ub("Night", "SN", "Holiday", 4).
staff_dweek_hard_ub("Night", "SE", "Weekday", 5).
staff_dweek_hard_ub("Night", "SN", "Weekday", 5).
staff_dweek_hard_ub("Night", "SE", "Weekend", 5).
staff_dweek_hard_ub("Night", "SN", "Weekend", 5).
staff_dweek_hard_ub("Night", "SE", "Holiday", 5).
staff_dweek_hard_ub("Night", "SN", "Holiday", 5).
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","LM","Staffs").
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "平日代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "夜勤リーダー" (no mapping).
% Skipped vertical_constraint_type for nurse group "日勤リーダー" (no mapping).
% Skipped vertical_constraint_type for nurse group "Aチーム" (no mapping).
% Skipped vertical_constraint_type for nurse group "Bチーム" (no mapping).
% Skipped vertical_constraint_type for nurse group "Cチーム" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
vertical_constraint_type("Night","SE","Staffs").
vertical_constraint_type("Night","SN","Staffs").
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
% Staff 21:
% Staff 22:
% Staff 23:
% Staff 24:
% Staff 26:
% Staff 27:
% Staff 29:
% Staff 30:
% Staff 31:
% Staff 32:
% Staff 33:
% Staff 34:
% Staff 35:
% Staff 36:
% Staff 37:
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
