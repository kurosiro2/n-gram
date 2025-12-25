% Staffs ---------------------------------------------------
staff(1, "Franklin Phillips", "Chief Nurse", "52504", 2).
staff(2, "Jacob Anderson", "Chief Nurse", "21672", 2).
staff(3, "Jeremiah Smith", "Chief Nurse", "93159", 2).
staff(4, "Nicole Gonzalez", "Chief Nurse", "53874", 2).
staff(5, "Christopher Ellis", "Chief Nurse", "00659", 2).
staff(6, "Benjamin Knox", "Chief Nurse", "76127", 2).
staff(7, "Ashley Pham", "Chief Nurse", "51916", 2).
staff(8, "Thomas Bruce", "Chief Nurse", "41205", 2).
staff(9, "Jose Jackson", "Chief Nurse", "12991", 2).
staff(10, "Jennifer Collins", "Chief Nurse", "25051", 2).
staff(11, "Caroline Guzman", "Chief Nurse", "60074", 2).
staff(12, "Jonathan Baker", "Chief Nurse", "02200", 2).
staff(13, "Kimberly Moore", "Chief Nurse", "59057", 2).
staff(14, "Austin Jones", "Chief Nurse", "71143", 2).
staff(15, "Stacy Garrett", "Chief Nurse", "75311", 2).
staff(16, "Audrey Solis", "Chief Nurse", "58245", 2).
staff(17, "Terry Woods", "Chief Nurse", "30037", 2).
staff(18, "Bruce Banks", "Chief Nurse", "43703", 2).
staff(19, "Manuel Collins", "Chief Nurse", "57002", 2).
staff(20, "Brenda Ford", "Chief Nurse", "52985", 2).
staff(21, "Julie Lee", "Chief Nurse", "73965", 2).
staff(22, "Ralph Silva", "Chief Nurse", "68750", 2).
staff(23, "Melody Smith", "Chief Nurse", "76825", 2).
staff(24, "Thomas Choi", "Chief Nurse", "11628", 2).
staff(25, "Tony Johnson", "Chief Nurse", "84868", 2).
% Groups ---------------------------------------------------
staff_group("All").
staff_group("All", 1).
staff_group("All", 2).
staff_group("All", 3).
staff_group("All", 5).
staff_group("All", 7).
staff_group("All", 9).
staff_group("All", 10).
staff_group("All", 11).
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
staff_group("Heads").
staff_group("Seniors").
staff_group("Seniors1").
staff_group("Seniors1", 1).
staff_group("Seniors1", 2).
staff_group("Seniors1", 3).
staff_group("Seniors1", 5).
staff_group("Seniors1", 7).
staff_group("Seniors1", 9).
staff_group("Seniors1", 10).
staff_group("Seniors1", 11).
staff_group("Seniors1", 13).
staff_group("Seniors1", 14).
staff_group("Seniors1", 15).
staff_group("Seniors1", 17).
staff_group("Seniors1", 18).
staff_group("Seniors1", 19).
staff_group("Seniors1", 20).
staff_group("Seniors1", 21).
staff_group("Seniors1", 22).
staff_group("Seniors1", 23).
staff_group("Seniors1", 24).
staff_group("Seniors1", 25).
staff_group("Seniors2").
staff_group("Mid-levels").
staff_group("Newcomers").
staff_group("Newcomers1").
staff_group("Newcomers2").
staff_group("Night").
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
patterns_hard_ub("LD-D", 1).
patterns_hard_ub("LM-D", 0).
patterns_hard_ub("LM-LD", 0).
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_hard_ub("LM-LM", 0).
% Vertical constraints -------------------------------------
shift_group("D").
shift_group("D", "D").
staff_dweek_soft_lb("All", "D", "Weekday", 1).
staff_dweek_soft_lb("All", "D", "Weekend", 1).
staff_dweek_soft_lb("All", "D", "Holiday", 1).
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_soft_lb("All", "LD", "Weekday", 1).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_soft_lb("All", "SE", "Weekday", 1).
shift_group("SN").
shift_group("SN", "SN").
staff_dweek_soft_lb("All", "SN", "Weekday", 1).
staff_dweek_soft_lb("All", "LD", "Weekend", 1).
staff_dweek_soft_lb("All", "SE", "Weekend", 1).
staff_dweek_soft_lb("All", "SN", "Weekend", 1).
staff_dweek_soft_lb("All", "LD", "Holiday", 1).
staff_dweek_soft_lb("All", "SE", "Holiday", 1).
staff_dweek_soft_lb("All", "SN", "Holiday", 1).
staff_dweek_soft_ub("All", "LD", "Weekday", 1).
staff_dweek_soft_ub("All", "SE", "Weekday", 1).
staff_dweek_soft_ub("All", "SN", "Weekday", 1).
staff_dweek_soft_ub("All", "LD", "Weekend", 1).
staff_dweek_soft_ub("All", "SE", "Weekend", 1).
staff_dweek_soft_ub("All", "SN", "Weekend", 1).
staff_dweek_soft_ub("All", "LD", "Holiday", 1).
staff_dweek_soft_ub("All", "SE", "Holiday", 1).
staff_dweek_soft_ub("All", "SN", "Holiday", 1).
staff_dweek_soft_lb("Seniors1", "LD", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Weekday", 1).
staff_dweek_soft_lb("Seniors1", "LD", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Weekend", 1).
staff_dweek_soft_lb("Seniors1", "LD", "Holiday", 1).
staff_dweek_soft_lb("Seniors1", "SE", "Holiday", 1).
staff_dweek_soft_lb("Seniors1", "SN", "Holiday", 1).
staff_dweek_soft_ub("Seniors1", "LD", "Weekday", 1).
staff_dweek_soft_ub("Seniors1", "SE", "Weekday", 1).
staff_dweek_soft_ub("Seniors1", "SN", "Weekday", 1).
staff_dweek_soft_ub("Seniors1", "LD", "Weekend", 1).
staff_dweek_soft_ub("Seniors1", "SE", "Weekend", 1).
staff_dweek_soft_ub("Seniors1", "SN", "Weekend", 1).
staff_dweek_soft_ub("Seniors1", "LD", "Holiday", 1).
staff_dweek_soft_ub("Seniors1", "SE", "Holiday", 1).
staff_dweek_soft_ub("Seniors1", "SN", "Holiday", 1).
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("Seniors1","LD","Staffs").
vertical_constraint_type("Seniors1","SE","Staffs").
vertical_constraint_type("Seniors1","SN","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "LD", 1).
shifts_soft_lb(1, "SE", 1).
shifts_soft_lb(1, "SN", 1).
shifts_soft_ub(1, "LD", 2).
shifts_soft_ub(1, "SE", 2).
shifts_soft_ub(1, "SN", 2).
shift_group("E").
shift_group("E", "E").
shifts_soft_ub(1, "E", 0).
shift_group("N").
shift_group("N", "N").
shifts_soft_ub(1, "N", 0).
shift_group("EM").
shift_group("EM", "EM").
shifts_soft_ub(1, "EM", 0).
shift_group("LM").
shift_group("LM", "LM").
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "EM", 0).
shifts_hard_ub(1, "LM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 1).
shifts_soft_lb(2, "SE", 1).
shifts_soft_lb(2, "SN", 1).
shifts_soft_ub(2, "LD", 2).
shifts_soft_ub(2, "SE", 2).
shifts_soft_ub(2, "SN", 2).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_soft_ub(2, "EM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
shifts_hard_ub(2, "EM", 0).
shifts_hard_ub(2, "LM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 1).
shifts_soft_lb(3, "SE", 1).
shifts_soft_lb(3, "SN", 1).
shifts_soft_ub(3, "LD", 2).
shifts_soft_ub(3, "SE", 2).
shifts_soft_ub(3, "SN", 2).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
shifts_soft_ub(3, "EM", 0).
shifts_soft_ub(3, "LM", 0).
shifts_hard_ub(3, "E", 0).
shifts_hard_ub(3, "N", 0).
shifts_hard_ub(3, "EM", 0).
shifts_hard_ub(3, "LM", 0).
% Staff 4:
% Staff 5:
shifts_soft_lb(5, "LD", 1).
shifts_soft_lb(5, "SE", 1).
shifts_soft_lb(5, "SN", 1).
shifts_soft_ub(5, "LD", 2).
shifts_soft_ub(5, "SE", 2).
shifts_soft_ub(5, "SN", 2).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
shifts_soft_ub(5, "EM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
shifts_hard_ub(5, "EM", 0).
shifts_hard_ub(5, "LM", 0).
% Staff 6:
% Staff 7:
shifts_soft_lb(7, "LD", 1).
shifts_soft_lb(7, "SE", 1).
shifts_soft_lb(7, "SN", 1).
shifts_soft_ub(7, "LD", 2).
shifts_soft_ub(7, "SE", 2).
shifts_soft_ub(7, "SN", 2).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
shifts_soft_ub(7, "EM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
shifts_hard_ub(7, "EM", 0).
shifts_hard_ub(7, "LM", 0).
% Staff 8:
% Staff 9:
shifts_soft_lb(9, "LD", 1).
shifts_soft_lb(9, "SE", 1).
shifts_soft_lb(9, "SN", 1).
shifts_soft_ub(9, "LD", 2).
shifts_soft_ub(9, "SE", 2).
shifts_soft_ub(9, "SN", 2).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
shifts_soft_ub(9, "EM", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
shifts_hard_ub(9, "EM", 0).
shifts_hard_ub(9, "LM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 1).
shifts_soft_lb(10, "SE", 1).
shifts_soft_lb(10, "SN", 1).
shifts_soft_ub(10, "LD", 2).
shifts_soft_ub(10, "SE", 2).
shifts_soft_ub(10, "SN", 2).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
shifts_soft_ub(10, "EM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
shifts_hard_ub(10, "EM", 0).
shifts_hard_ub(10, "LM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 1).
shifts_soft_lb(11, "SE", 1).
shifts_soft_lb(11, "SN", 1).
shifts_soft_ub(11, "LD", 2).
shifts_soft_ub(11, "SE", 2).
shifts_soft_ub(11, "SN", 2).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
shifts_soft_ub(11, "EM", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
shifts_hard_ub(11, "EM", 0).
shifts_hard_ub(11, "LM", 0).
% Staff 12:
% Staff 13:
shifts_soft_lb(13, "LD", 1).
shifts_soft_lb(13, "SE", 1).
shifts_soft_lb(13, "SN", 1).
shifts_soft_ub(13, "LD", 2).
shifts_soft_ub(13, "SE", 2).
shifts_soft_ub(13, "SN", 2).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
shifts_soft_ub(13, "EM", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
shifts_hard_ub(13, "EM", 0).
shifts_hard_ub(13, "LM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 1).
shifts_soft_lb(14, "SE", 1).
shifts_soft_lb(14, "SN", 1).
shifts_soft_ub(14, "LD", 2).
shifts_soft_ub(14, "SE", 2).
shifts_soft_ub(14, "SN", 2).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_soft_ub(14, "EM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_hard_ub(14, "EM", 0).
shifts_hard_ub(14, "LM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 1).
shifts_soft_lb(15, "SE", 1).
shifts_soft_lb(15, "SN", 1).
shifts_soft_ub(15, "LD", 2).
shifts_soft_ub(15, "SE", 2).
shifts_soft_ub(15, "SN", 2).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_soft_ub(15, "EM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_ub(15, "EM", 0).
shifts_hard_ub(15, "LM", 0).
% Staff 16:
% Staff 17:
shifts_soft_lb(17, "LD", 1).
shifts_soft_lb(17, "SE", 1).
shifts_soft_lb(17, "SN", 1).
shifts_soft_ub(17, "LD", 2).
shifts_soft_ub(17, "SE", 2).
shifts_soft_ub(17, "SN", 2).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
shifts_soft_ub(17, "EM", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
shifts_hard_ub(17, "EM", 0).
shifts_hard_ub(17, "LM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 1).
shifts_soft_lb(18, "SE", 1).
shifts_soft_lb(18, "SN", 1).
shifts_soft_ub(18, "LD", 2).
shifts_soft_ub(18, "SE", 2).
shifts_soft_ub(18, "SN", 2).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
shifts_soft_ub(18, "EM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "E", 0).
shifts_hard_ub(18, "N", 0).
shifts_hard_ub(18, "EM", 0).
shifts_hard_ub(18, "LM", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 1).
shifts_soft_lb(19, "SE", 1).
shifts_soft_lb(19, "SN", 1).
shifts_soft_ub(19, "LD", 2).
shifts_soft_ub(19, "SE", 2).
shifts_soft_ub(19, "SN", 2).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
shifts_soft_ub(19, "EM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "E", 0).
shifts_hard_ub(19, "N", 0).
shifts_hard_ub(19, "EM", 0).
shifts_hard_ub(19, "LM", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 1).
shifts_soft_lb(20, "SE", 1).
shifts_soft_lb(20, "SN", 1).
shifts_soft_ub(20, "LD", 2).
shifts_soft_ub(20, "SE", 2).
shifts_soft_ub(20, "SN", 2).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
shifts_soft_ub(20, "EM", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "E", 0).
shifts_hard_ub(20, "N", 0).
shifts_hard_ub(20, "EM", 0).
shifts_hard_ub(20, "LM", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 1).
shifts_soft_lb(21, "SE", 1).
shifts_soft_lb(21, "SN", 1).
shifts_soft_ub(21, "LD", 2).
shifts_soft_ub(21, "SE", 2).
shifts_soft_ub(21, "SN", 2).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
shifts_soft_ub(21, "EM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
shifts_hard_ub(21, "EM", 0).
shifts_hard_ub(21, "LM", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 1).
shifts_soft_lb(22, "SE", 1).
shifts_soft_lb(22, "SN", 1).
shifts_soft_ub(22, "LD", 2).
shifts_soft_ub(22, "SE", 2).
shifts_soft_ub(22, "SN", 2).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
shifts_soft_ub(22, "EM", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
shifts_hard_ub(22, "EM", 0).
shifts_hard_ub(22, "LM", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 1).
shifts_soft_lb(23, "SE", 1).
shifts_soft_lb(23, "SN", 1).
shifts_soft_ub(23, "LD", 2).
shifts_soft_ub(23, "SE", 2).
shifts_soft_ub(23, "SN", 2).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
shifts_soft_ub(23, "EM", 0).
shifts_soft_ub(23, "LM", 0).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
shifts_hard_ub(23, "EM", 0).
shifts_hard_ub(23, "LM", 0).
% Staff 24:
shifts_soft_lb(24, "LD", 1).
shifts_soft_lb(24, "SE", 1).
shifts_soft_lb(24, "SN", 1).
shifts_soft_ub(24, "LD", 2).
shifts_soft_ub(24, "SE", 2).
shifts_soft_ub(24, "SN", 2).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_soft_ub(24, "EM", 0).
shifts_soft_ub(24, "LM", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_hard_ub(24, "EM", 0).
shifts_hard_ub(24, "LM", 0).
% Staff 25:
shifts_soft_lb(25, "LD", 1).
shifts_soft_lb(25, "SE", 1).
shifts_soft_lb(25, "SN", 1).
shifts_soft_ub(25, "LD", 2).
shifts_soft_ub(25, "SE", 2).
shifts_soft_ub(25, "SN", 2).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
shifts_soft_ub(25, "EM", 0).
shifts_soft_ub(25, "LM", 0).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
shifts_hard_ub(25, "EM", 0).
shifts_hard_ub(25, "LM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("EM").
horizontal_constraint_type("LM").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
