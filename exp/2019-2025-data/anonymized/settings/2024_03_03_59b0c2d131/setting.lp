% Staffs ---------------------------------------------------
staff(1, "Kimberly Moore", "Chief Nurse", "59057", 2).
staff(2, "Jennifer Collins", "Chief Nurse", "25051", 2).
staff(3, "Caroline Guzman", "Chief Nurse", "60074", 2).
staff(4, "Jacob Anderson", "Chief Nurse", "21672", 2).
staff(5, "Lindsay Johnson", "Chief Nurse", "54248", 2).
staff(6, "Terry Woods", "Chief Nurse", "30037", 2).
staff(7, "Christopher Ellis", "Chief Nurse", "00659", 2).
staff(8, "William Holmes", "Chief Nurse", "39562", 2).
staff(9, "Melody Smith", "Chief Nurse", "76825", 2).
staff(10, "Joseph Wise", "Chief Nurse", "37518", 2).
staff(11, "Julie Lee", "Chief Nurse", "73965", 2).
staff(12, "Thomas Choi", "Chief Nurse", "11628", 2).
staff(13, "Kristy Thomas", "Chief Nurse", "89539", 2).
staff(14, "Alex Le", "Chief Nurse", "45856", 1).
staff(15, "Austin Jones", "Chief Nurse", "71143", 1).
staff(16, "Brandy Pham", "Chief Nurse", "04506", 2).
staff(17, "Ralph Silva", "Chief Nurse", "68750", 2).
staff(18, "Stacy Garrett", "Chief Nurse", "75311", 2).
staff(19, "Bruce Banks", "Chief Nurse", "43703", 2).
staff(20, "Tony Johnson", "Chief Nurse", "84868", 2).
staff(21, "Brenda Ford", "Chief Nurse", "52985", 2).
staff(22, "Ashley Pham", "Chief Nurse", "51916", 5).
staff(23, "Kristi Nguyen", "Chief Nurse", "89861", 5).
staff(24, "Joshua Ewing", "Chief Nurse", "49353", 1).
staff(25, "Daniel Hunter", "Chief Nurse", "51930", 5).
staff(26, "Michael Jones", "Chief Nurse", "19727", 5).
staff(27, "Patrick Baker", "Chief Nurse", "13996", 1).
staff(28, "Franklin Phillips", "Chief Nurse", "52504", 1).
staff(29, "Gina Woods", "Chief Nurse", "27470", 1).
staff(30, "Jordan Weiss", "Chief Nurse", "97276", 1).
staff(31, "Manuel Collins", "Chief Nurse", "57002", 1).
staff(32, "Jose Jackson", "Chief Nurse", "12991", 1).
staff(33, "Catherine Walsh", "Chief Nurse", "39003", 1).
staff(34, "Richard Raymond", "Chief Nurse", "61687", 1).
staff(35, "Juan Cochran", "Chief Nurse", "33241", 1).
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
staff_group("Heads").
staff_group("Seniors").
staff_group("Seniors1").
staff_group("Seniors1", 1).
staff_group("Seniors1", 2).
staff_group("Seniors1", 3).
staff_group("Seniors1", 4).
staff_group("Seniors1", 5).
staff_group("Seniors1", 6).
staff_group("Seniors1", 7).
staff_group("Seniors1", 8).
staff_group("Seniors1", 9).
staff_group("Seniors1", 10).
staff_group("Seniors1", 11).
staff_group("Seniors1", 12).
staff_group("Seniors1", 13).
staff_group("Seniors1", 16).
staff_group("Seniors1", 17).
staff_group("Seniors1", 18).
staff_group("Seniors1", 19).
staff_group("Seniors1", 20).
staff_group("Seniors1", 21).
staff_group("Seniors1", 22).
staff_group("Seniors1", 23).
staff_group("Seniors1", 25).
staff_group("Seniors1", 26).
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
shift_group("LD+LM+E").
shift_group("LD+LM+E", "LD").
shift_group("LD+LM+E", "LM").
shift_group("LD+LM+E", "E").
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 1).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 1).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 1).
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
vertical_constraint_type("All","LD+LM+E","Staffs").
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
shifts_soft_lb(4, "LD", 1).
shifts_soft_lb(4, "SE", 1).
shifts_soft_lb(4, "SN", 1).
shifts_soft_ub(4, "LD", 2).
shifts_soft_ub(4, "SE", 2).
shifts_soft_ub(4, "SN", 2).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
shifts_soft_ub(4, "EM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "E", 0).
shifts_hard_ub(4, "N", 0).
shifts_hard_ub(4, "EM", 0).
shifts_hard_ub(4, "LM", 0).
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
shifts_soft_lb(6, "LD", 1).
shifts_soft_lb(6, "SE", 1).
shifts_soft_lb(6, "SN", 1).
shifts_soft_ub(6, "LD", 2).
shifts_soft_ub(6, "SE", 2).
shifts_soft_ub(6, "SN", 2).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
shifts_soft_ub(6, "EM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
shifts_hard_ub(6, "EM", 0).
shifts_hard_ub(6, "LM", 0).
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
shifts_soft_lb(8, "LD", 1).
shifts_soft_lb(8, "SE", 1).
shifts_soft_lb(8, "SN", 1).
shifts_soft_ub(8, "LD", 2).
shifts_soft_ub(8, "SE", 2).
shifts_soft_ub(8, "SN", 2).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
shifts_soft_ub(8, "EM", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
shifts_hard_ub(8, "EM", 0).
shifts_hard_ub(8, "LM", 0).
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
shifts_soft_lb(12, "LD", 1).
shifts_soft_lb(12, "SE", 1).
shifts_soft_lb(12, "SN", 1).
shifts_soft_ub(12, "LD", 2).
shifts_soft_ub(12, "SE", 2).
shifts_soft_ub(12, "SN", 2).
shifts_soft_ub(12, "E", 0).
shifts_soft_ub(12, "N", 0).
shifts_soft_ub(12, "EM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "E", 0).
shifts_hard_ub(12, "N", 0).
shifts_hard_ub(12, "EM", 0).
shifts_hard_ub(12, "LM", 0).
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
shifts_soft_lb(14, "LD", 0).
shifts_soft_lb(14, "SE", 0).
shifts_soft_lb(14, "SN", 0).
shifts_soft_ub(14, "LD", 0).
shifts_soft_ub(14, "SE", 0).
shifts_soft_ub(14, "SN", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_soft_ub(14, "EM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_hard_ub(14, "EM", 0).
shifts_hard_ub(14, "LM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 0).
shifts_soft_lb(15, "SE", 0).
shifts_soft_lb(15, "SN", 0).
shifts_soft_ub(15, "LD", 0).
shifts_soft_ub(15, "SE", 0).
shifts_soft_ub(15, "SN", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_soft_ub(15, "EM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_ub(15, "EM", 0).
shifts_hard_ub(15, "LM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 1).
shifts_soft_lb(16, "SE", 1).
shifts_soft_lb(16, "SN", 1).
shifts_soft_ub(16, "LD", 2).
shifts_soft_ub(16, "SE", 2).
shifts_soft_ub(16, "SN", 2).
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
shifts_soft_ub(16, "EM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
shifts_hard_ub(16, "EM", 0).
shifts_hard_ub(16, "LM", 0).
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
shifts_soft_lb(24, "LD", 0).
shifts_soft_lb(24, "SE", 0).
shifts_soft_lb(24, "SN", 0).
shifts_soft_ub(24, "LD", 0).
shifts_soft_ub(24, "SE", 0).
shifts_soft_ub(24, "SN", 0).
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
% Staff 26:
shifts_soft_lb(26, "LD", 1).
shifts_soft_lb(26, "SE", 1).
shifts_soft_lb(26, "SN", 1).
shifts_soft_ub(26, "LD", 2).
shifts_soft_ub(26, "SE", 2).
shifts_soft_ub(26, "SN", 2).
shifts_soft_ub(26, "E", 0).
shifts_soft_ub(26, "N", 0).
shifts_soft_ub(26, "EM", 0).
shifts_soft_ub(26, "LM", 0).
shifts_hard_ub(26, "E", 0).
shifts_hard_ub(26, "N", 0).
shifts_hard_ub(26, "EM", 0).
shifts_hard_ub(26, "LM", 0).
% Staff 27:
shifts_soft_lb(27, "LD", 0).
shifts_soft_lb(27, "SE", 0).
shifts_soft_lb(27, "SN", 0).
shifts_soft_ub(27, "LD", 0).
shifts_soft_ub(27, "SE", 0).
shifts_soft_ub(27, "SN", 0).
shifts_soft_ub(27, "E", 0).
shifts_soft_ub(27, "N", 0).
shifts_soft_ub(27, "EM", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "E", 0).
shifts_hard_ub(27, "N", 0).
shifts_hard_ub(27, "EM", 0).
shifts_hard_ub(27, "LM", 0).
% Staff 28:
shifts_soft_lb(28, "LD", 0).
shifts_soft_lb(28, "SE", 0).
shifts_soft_lb(28, "SN", 0).
shifts_soft_ub(28, "LD", 0).
shifts_soft_ub(28, "SE", 0).
shifts_soft_ub(28, "SN", 0).
shifts_soft_ub(28, "E", 0).
shifts_soft_ub(28, "N", 0).
shifts_soft_ub(28, "EM", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "E", 0).
shifts_hard_ub(28, "N", 0).
shifts_hard_ub(28, "EM", 0).
shifts_hard_ub(28, "LM", 0).
% Staff 29:
shifts_soft_lb(29, "LD", 0).
shifts_soft_lb(29, "SE", 0).
shifts_soft_lb(29, "SN", 0).
shifts_soft_ub(29, "LD", 0).
shifts_soft_ub(29, "SE", 0).
shifts_soft_ub(29, "SN", 0).
shifts_soft_ub(29, "E", 0).
shifts_soft_ub(29, "N", 0).
shifts_soft_ub(29, "EM", 0).
shifts_soft_ub(29, "LM", 0).
shifts_hard_ub(29, "E", 0).
shifts_hard_ub(29, "N", 0).
shifts_hard_ub(29, "EM", 0).
shifts_hard_ub(29, "LM", 0).
% Staff 30:
shifts_soft_lb(30, "LD", 0).
shifts_soft_lb(30, "SE", 0).
shifts_soft_lb(30, "SN", 0).
shifts_soft_ub(30, "LD", 0).
shifts_soft_ub(30, "SE", 0).
shifts_soft_ub(30, "SN", 0).
shifts_soft_ub(30, "E", 0).
shifts_soft_ub(30, "N", 0).
shifts_soft_ub(30, "EM", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "E", 0).
shifts_hard_ub(30, "N", 0).
shifts_hard_ub(30, "EM", 0).
shifts_hard_ub(30, "LM", 0).
% Staff 31:
shifts_soft_lb(31, "LD", 0).
shifts_soft_lb(31, "SE", 0).
shifts_soft_lb(31, "SN", 0).
shifts_soft_ub(31, "LD", 0).
shifts_soft_ub(31, "SE", 0).
shifts_soft_ub(31, "SN", 0).
shifts_soft_ub(31, "E", 0).
shifts_soft_ub(31, "N", 0).
shifts_soft_ub(31, "EM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "E", 0).
shifts_hard_ub(31, "N", 0).
shifts_hard_ub(31, "EM", 0).
shifts_hard_ub(31, "LM", 0).
% Staff 32:
shifts_soft_lb(32, "LD", 0).
shifts_soft_lb(32, "SE", 0).
shifts_soft_lb(32, "SN", 0).
shifts_soft_ub(32, "LD", 0).
shifts_soft_ub(32, "SE", 0).
shifts_soft_ub(32, "SN", 0).
shifts_soft_ub(32, "E", 0).
shifts_soft_ub(32, "N", 0).
shifts_soft_ub(32, "EM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "E", 0).
shifts_hard_ub(32, "N", 0).
shifts_hard_ub(32, "EM", 0).
shifts_hard_ub(32, "LM", 0).
% Staff 33:
shifts_soft_lb(33, "LD", 0).
shifts_soft_lb(33, "SE", 0).
shifts_soft_lb(33, "SN", 0).
shifts_soft_ub(33, "LD", 0).
shifts_soft_ub(33, "SE", 0).
shifts_soft_ub(33, "SN", 0).
shifts_soft_ub(33, "E", 0).
shifts_soft_ub(33, "N", 0).
shifts_soft_ub(33, "EM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "E", 0).
shifts_hard_ub(33, "N", 0).
shifts_hard_ub(33, "EM", 0).
shifts_hard_ub(33, "LM", 0).
% Staff 34:
shifts_soft_lb(34, "LD", 0).
shifts_soft_lb(34, "SE", 0).
shifts_soft_lb(34, "SN", 0).
shifts_soft_ub(34, "LD", 0).
shifts_soft_ub(34, "SE", 0).
shifts_soft_ub(34, "SN", 0).
shifts_soft_ub(34, "E", 0).
shifts_soft_ub(34, "N", 0).
shifts_soft_ub(34, "EM", 0).
shifts_soft_ub(34, "LM", 0).
shifts_hard_ub(34, "E", 0).
shifts_hard_ub(34, "N", 0).
shifts_hard_ub(34, "EM", 0).
shifts_hard_ub(34, "LM", 0).
% Staff 35:
shifts_soft_lb(35, "LD", 0).
shifts_soft_lb(35, "SE", 0).
shifts_soft_lb(35, "SN", 0).
shifts_soft_ub(35, "LD", 0).
shifts_soft_ub(35, "SE", 0).
shifts_soft_ub(35, "SN", 0).
shifts_soft_ub(35, "E", 0).
shifts_soft_ub(35, "N", 0).
shifts_soft_ub(35, "EM", 0).
shifts_soft_ub(35, "LM", 0).
shifts_hard_ub(35, "E", 0).
shifts_hard_ub(35, "N", 0).
shifts_hard_ub(35, "EM", 0).
shifts_hard_ub(35, "LM", 0).
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
