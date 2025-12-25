% Staffs ---------------------------------------------------
staff(1, "Allison Nelson", "Chief Nurse", "72463", 0).
staff(2, "Andrew Vasquez", "Deputy Chief Nurse", "48016", 0).
staff(3, "Adam Bright", "Deputy Chief Nurse", "26153", 0).
staff(4, "Erin Huffman", "Deputy Chief Nurse", "68699", 0).
staff(5, "Ashley Gould", "Deputy Chief Nurse", "87735", 0).
staff(6, "Anna Werner", "Nurse", "20374", 0).
staff(7, "Michele Campbell", "Nurse", "23216", 0).
staff(8, "Robert Sanders", "Nurse", "08776", 0).
staff(9, "Christopher Sandoval", "Nurse", "82347", 0).
staff(10, "Alexander Russell", "Nurse", "10099", 0).
staff(11, "Dillon Gentry", "Nurse", "46497", 0).
staff(12, "Brian Mason", "Nurse", "85832", 0).
staff(13, "Jasmine Nash", "Nurse", "56527", 0).
staff(14, "Wesley Chambers", "Nurse", "02249", 0).
staff(15, "Jennifer Guerra", "Nurse", "13063", 0).
staff(16, "Paul Sanders", "Nurse", "07275", 0).
staff(17, "John Griffith", "Nurse", "95331", 0).
staff(18, "Jennifer Ross", "Nurse", "77921", 0).
staff(19, "Jasmine Davis", "Nurse", "06471", 0).
staff(20, "Kathy Garcia", "Nurse", "47075", 0).
staff(21, "Allen Alexander", "Nurse", "87815", 0).
staff(22, "Miss Rachel Caldwell", "Nurse", "36106", 0).
staff(23, "Thomas Miller", "Nurse", "56488", 0).
staff(24, "Jason Leonard", "Nurse", "36912", 0).
staff(25, "Carla Dougherty", "Nurse", "82889", 0).
staff(26, "Heather Phillips", "Nurse", "42456", 0).
staff(27, "Michael Myers", "Nurse", "46087", 0).
staff(28, "Kristin Rodriguez", "Nurse", "48652", 0).
staff(29, "Shannon Gilbert PhD", "Nurse", "76237", 0).
staff(30, "Jordan Crawford", "Nurse", "35412", 0).
staff(31, "Brenda Daniels", "Nurse", "45929", 0).
staff(32, "Annette Wilson", "Nurse", "35206", 0).
staff(33, "Adam Hawkins", "Nurse", "77912", 0).
staff(34, "Lindsay Barrett", "Nurse", "90045", 0).
% Skipped staff no=35, name="Justin Johnson", job="パート" (no mapping).
staff(36, "Jennifer Fisher", "Nurse", "18302", 0).
staff(37, "Maria Patel", "Nurse", "70064", 0).
staff(38, "Ronald Ritter", "Nurse", "06919", 0).
staff(39, "Raymond Marsh", "Nurse", "07244", 0).
staff(40, "Rodney Mendoza", "Nurse", "16434", 0).
staff(41, "James Hudson", "Nurse", "97391", 0).
% Skipped staff no=42, name="Patricia Kelley", job="パート" (no mapping).
staff(43, "Jennifer Thomas", "Nurse", "34002", 0).
staff(44, "Patricia Hughes", "Nurse", "15632", 0).
staff(45, "Brad Hamilton", "Nurse", "71228", 0).
staff(46, "Linda Cooper", "Nurse", "65612", 0).
staff(47, "Louis Washington", "Nurse", "30717", 0).
staff(48, "Jared Williams", "Nurse", "88645", 0).
staff(49, "Melissa Marks", "Nurse", "55834", 0).
staff(50, "Stephanie Powell", "Nurse", "80259", 0).
staff(51, "Sean Rogers", "Nurse", "66542", 0).
staff(52, "Jason Gray", "Nurse", "35235", 0).
% Groups ---------------------------------------------------
% Skipped nurse group column "ER" (no mapping).
staff_group("Newcomers").
staff_group("Newcomers", 29).
staff_group("Newcomers", 30).
staff_group("Newcomers", 31).
staff_group("Newcomers", 32).
staff_group("Newcomers", 33).
staff_group("Newcomers", 34).
staff_group("Night").
staff_group("Night", 2).
staff_group("Night", 3).
staff_group("Night", 4).
staff_group("Night", 5).
staff_group("Night", 6).
staff_group("Night", 7).
staff_group("Night", 9).
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
staff_group("Night", 44).
staff_group("Night", 45).
staff_group("Night", 46).
staff_group("Night", 47).
staff_group("Night", 51).
staff_group("Night", 52).
staff_group("Leaders").
staff_group("Leaders", 2).
staff_group("Leaders", 3).
staff_group("Leaders", 4).
staff_group("Leaders", 5).
staff_group("Leaders", 6).
staff_group("Leaders", 7).
staff_group("Leaders", 9).
staff_group("Leaders", 13).
staff_group("Leaders", 14).
staff_group("Leaders", 16).
staff_group("Leaders", 17).
staff_group("Leaders", 20).
staff_group("Leaders", 21).
staff_group("Leaders", 22).
staff_group("Leaders", 44).
staff_group("Leaders", 45).
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
patterns_hard_lb("WR-WR", 2).
patterns_soft_ub("LD-D", 0).
patterns_soft_ub("LM-D", 0).
patterns_soft_ub("LM-LD", 0).
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_soft_ub("LM-LM", 0).
% Vertical constraints -------------------------------------
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
% Skipped vertical constraint for nurse group "ER" (no mapping).
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
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_hard_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_hard_lb("Leaders", "LD", "Holiday", 3).
staff_dweek_hard_lb("Leaders", "SE", "Holiday", 3).
staff_dweek_hard_lb("Leaders", "SN", "Holiday", 3).
staff_dweek_soft_lb("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_lb("Leaders", "LD", "Holiday", 4).
staff_dweek_soft_lb("Leaders", "SE", "Holiday", 4).
staff_dweek_soft_lb("Leaders", "SN", "Holiday", 4).
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 4).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 4).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 4).
staff_dweek_hard_ub("Leaders", "LD", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "SE", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "SN", "Weekday", 2).
staff_dweek_hard_ub("Leaders", "LD", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "SE", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "SN", "Weekend", 2).
staff_dweek_hard_ub("Leaders", "LD", "Holiday", 4).
staff_dweek_hard_ub("Leaders", "SE", "Holiday", 4).
staff_dweek_hard_ub("Leaders", "SN", "Holiday", 4).
% Skipped vertical_constraint_type for nurse group "ER" (no mapping).
% Skipped vertical_constraint_type for nurse group "ER" (no mapping).
% Skipped vertical_constraint_type for nurse group "ER" (no mapping).
% Skipped vertical_constraint_type for nurse group "ER" (no mapping).
vertical_constraint_type("Newcomers","SE","Staffs").
vertical_constraint_type("Newcomers","SN","Staffs").
vertical_constraint_type("Leaders","LD","Staffs").
vertical_constraint_type("Leaders","SE","Staffs").
vertical_constraint_type("Leaders","SN","Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_hard_lb(2, "LD", 1).
shifts_hard_lb(2, "SE", 1).
shifts_hard_lb(2, "SN", 1).
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 3).
shifts_soft_ub(2, "SE", 3).
shifts_soft_ub(2, "SN", 3).
shifts_hard_ub(2, "LD", 4).
shifts_hard_ub(2, "SE", 4).
shifts_hard_ub(2, "SN", 4).
shift_group("LM").
shift_group("LM", "LM").
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 1).
shifts_soft_ub(2, "LM", 2).
shifts_hard_ub(2, "LM", 2).
% Staff 3:
shifts_hard_lb(3, "LD", 1).
shifts_hard_lb(3, "SE", 1).
shifts_hard_lb(3, "SN", 1).
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 3).
shifts_soft_ub(3, "SE", 3).
shifts_soft_ub(3, "SN", 3).
shifts_hard_ub(3, "LD", 4).
shifts_hard_ub(3, "SE", 4).
shifts_hard_ub(3, "SN", 4).
shifts_hard_lb(3, "LM", 0).
shifts_soft_lb(3, "LM", 1).
shifts_soft_ub(3, "LM", 2).
shifts_hard_ub(3, "LM", 2).
% Staff 4:
shifts_hard_lb(4, "LD", 1).
shifts_hard_lb(4, "SE", 1).
shifts_hard_lb(4, "SN", 1).
shifts_soft_lb(4, "LD", 2).
shifts_soft_lb(4, "SE", 2).
shifts_soft_lb(4, "SN", 2).
shifts_soft_ub(4, "LD", 3).
shifts_soft_ub(4, "SE", 3).
shifts_soft_ub(4, "SN", 3).
shifts_hard_ub(4, "LD", 4).
shifts_hard_ub(4, "SE", 4).
shifts_hard_ub(4, "SN", 4).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 1).
shifts_soft_ub(4, "LM", 2).
shifts_hard_ub(4, "LM", 2).
% Staff 5:
shifts_hard_lb(5, "LD", 1).
shifts_hard_lb(5, "SE", 1).
shifts_hard_lb(5, "SN", 1).
shifts_soft_lb(5, "LD", 2).
shifts_soft_lb(5, "SE", 2).
shifts_soft_lb(5, "SN", 2).
shifts_soft_ub(5, "LD", 3).
shifts_soft_ub(5, "SE", 3).
shifts_soft_ub(5, "SN", 3).
shifts_hard_ub(5, "LD", 4).
shifts_hard_ub(5, "SE", 4).
shifts_hard_ub(5, "SN", 4).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 1).
shifts_soft_ub(5, "LM", 2).
shifts_hard_ub(5, "LM", 2).
% Staff 6:
shifts_hard_lb(6, "LD", 2).
shifts_hard_lb(6, "SE", 2).
shifts_hard_lb(6, "SN", 2).
shifts_soft_lb(6, "LD", 3).
shifts_soft_lb(6, "SE", 3).
shifts_soft_lb(6, "SN", 3).
shifts_soft_ub(6, "LD", 5).
shifts_soft_ub(6, "SE", 5).
shifts_soft_ub(6, "SN", 5).
shifts_hard_ub(6, "LD", 6).
shifts_hard_ub(6, "SE", 6).
shifts_hard_ub(6, "SN", 6).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 1).
shifts_soft_ub(6, "LM", 2).
shifts_hard_ub(6, "LM", 2).
% Staff 7:
shifts_hard_lb(7, "LD", 2).
shifts_hard_lb(7, "SE", 2).
shifts_hard_lb(7, "SN", 2).
shifts_soft_lb(7, "LD", 3).
shifts_soft_lb(7, "SE", 3).
shifts_soft_lb(7, "SN", 3).
shifts_soft_ub(7, "LD", 5).
shifts_soft_ub(7, "SE", 5).
shifts_soft_ub(7, "SN", 5).
shifts_hard_ub(7, "LD", 6).
shifts_hard_ub(7, "SE", 6).
shifts_hard_ub(7, "SN", 6).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 1).
shifts_soft_ub(7, "LM", 2).
shifts_hard_ub(7, "LM", 2).
% Staff 8:
shifts_hard_lb(8, "LD", 0).
shifts_hard_lb(8, "SE", 0).
shifts_hard_lb(8, "SN", 0).
shifts_soft_lb(8, "LD", 0).
shifts_soft_lb(8, "SE", 0).
shifts_soft_lb(8, "SN", 0).
shifts_soft_ub(8, "LD", 0).
shifts_soft_ub(8, "SE", 0).
shifts_soft_ub(8, "SN", 0).
shifts_hard_ub(8, "LD", 0).
shifts_hard_ub(8, "SE", 0).
shifts_hard_ub(8, "SN", 0).
% Staff 9:
shifts_hard_lb(9, "LD", 2).
shifts_hard_lb(9, "SE", 2).
shifts_hard_lb(9, "SN", 2).
shifts_soft_lb(9, "LD", 3).
shifts_soft_lb(9, "SE", 3).
shifts_soft_lb(9, "SN", 3).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "LM", 1).
shifts_soft_ub(9, "LM", 2).
shifts_hard_ub(9, "LM", 2).
% Staff 10:
% Staff 11:
shifts_hard_lb(11, "LD", 2).
shifts_hard_lb(11, "SE", 2).
shifts_hard_lb(11, "SN", 2).
shifts_soft_lb(11, "LD", 3).
shifts_soft_lb(11, "SE", 3).
shifts_soft_lb(11, "SN", 3).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "LD", 6).
shifts_hard_ub(11, "SE", 6).
shifts_hard_ub(11, "SN", 6).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "LM", 1).
shifts_soft_ub(11, "LM", 2).
shifts_hard_ub(11, "LM", 2).
% Staff 12:
shifts_hard_lb(12, "LD", 0).
shifts_hard_lb(12, "SE", 0).
shifts_hard_lb(12, "SN", 0).
shifts_soft_lb(12, "LD", 0).
shifts_soft_lb(12, "SE", 0).
shifts_soft_lb(12, "SN", 0).
shifts_soft_ub(12, "LD", 0).
shifts_soft_ub(12, "SE", 0).
shifts_soft_ub(12, "SN", 0).
shifts_hard_ub(12, "LD", 0).
shifts_hard_ub(12, "SE", 0).
shifts_hard_ub(12, "SN", 0).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 1).
shifts_soft_ub(12, "LM", 2).
shifts_hard_ub(12, "LM", 2).
% Staff 13:
shifts_hard_lb(13, "LD", 2).
shifts_hard_lb(13, "SE", 2).
shifts_hard_lb(13, "SN", 2).
shifts_soft_lb(13, "LD", 3).
shifts_soft_lb(13, "SE", 3).
shifts_soft_lb(13, "SN", 3).
shifts_soft_ub(13, "LD", 5).
shifts_soft_ub(13, "SE", 5).
shifts_soft_ub(13, "SN", 5).
shifts_hard_ub(13, "LD", 6).
shifts_hard_ub(13, "SE", 6).
shifts_hard_ub(13, "SN", 6).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "LM", 1).
shifts_soft_ub(13, "LM", 2).
shifts_hard_ub(13, "LM", 2).
% Staff 14:
shifts_hard_lb(14, "LD", 2).
shifts_hard_lb(14, "SE", 2).
shifts_hard_lb(14, "SN", 2).
shifts_soft_lb(14, "LD", 3).
shifts_soft_lb(14, "SE", 3).
shifts_soft_lb(14, "SN", 3).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_hard_ub(14, "LD", 6).
shifts_hard_ub(14, "SE", 6).
shifts_hard_ub(14, "SN", 6).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 1).
shifts_soft_ub(14, "LM", 2).
shifts_hard_ub(14, "LM", 2).
% Staff 15:
shifts_hard_lb(15, "LD", 2).
shifts_hard_lb(15, "SE", 2).
shifts_hard_lb(15, "SN", 2).
shifts_soft_lb(15, "LD", 3).
shifts_soft_lb(15, "SE", 3).
shifts_soft_lb(15, "SN", 3).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 1).
shifts_soft_ub(15, "LM", 2).
shifts_hard_ub(15, "LM", 2).
% Staff 16:
shifts_hard_lb(16, "LD", 2).
shifts_hard_lb(16, "SE", 2).
shifts_hard_lb(16, "SN", 2).
shifts_soft_lb(16, "LD", 3).
shifts_soft_lb(16, "SE", 3).
shifts_soft_lb(16, "SN", 3).
shifts_soft_ub(16, "LD", 5).
shifts_soft_ub(16, "SE", 5).
shifts_soft_ub(16, "SN", 5).
shifts_hard_ub(16, "LD", 6).
shifts_hard_ub(16, "SE", 6).
shifts_hard_ub(16, "SN", 6).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 1).
shifts_soft_ub(16, "LM", 2).
shifts_hard_ub(16, "LM", 2).
% Staff 17:
shifts_hard_lb(17, "LD", 2).
shifts_hard_lb(17, "SE", 2).
shifts_hard_lb(17, "SN", 2).
shifts_soft_lb(17, "LD", 3).
shifts_soft_lb(17, "SE", 3).
shifts_soft_lb(17, "SN", 3).
shifts_soft_ub(17, "LD", 5).
shifts_soft_ub(17, "SE", 5).
shifts_soft_ub(17, "SN", 5).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "LM", 1).
shifts_soft_ub(17, "LM", 2).
shifts_hard_ub(17, "LM", 2).
% Staff 18:
shifts_hard_lb(18, "LD", 2).
shifts_hard_lb(18, "SE", 2).
shifts_hard_lb(18, "SN", 2).
shifts_soft_lb(18, "LD", 3).
shifts_soft_lb(18, "SE", 3).
shifts_soft_lb(18, "SN", 3).
shifts_soft_ub(18, "LD", 5).
shifts_soft_ub(18, "SE", 5).
shifts_soft_ub(18, "SN", 5).
shifts_hard_ub(18, "LD", 6).
shifts_hard_ub(18, "SE", 6).
shifts_hard_ub(18, "SN", 6).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 1).
shifts_soft_ub(18, "LM", 2).
shifts_hard_ub(18, "LM", 2).
% Staff 19:
shifts_hard_lb(19, "LD", 2).
shifts_hard_lb(19, "SE", 2).
shifts_hard_lb(19, "SN", 2).
shifts_soft_lb(19, "LD", 3).
shifts_soft_lb(19, "SE", 3).
shifts_soft_lb(19, "SN", 3).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "LD", 6).
shifts_hard_ub(19, "SE", 6).
shifts_hard_ub(19, "SN", 6).
shifts_hard_lb(19, "LM", 0).
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
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "LM", 1).
shifts_soft_ub(20, "LM", 2).
shifts_hard_ub(20, "LM", 2).
% Staff 21:
shifts_hard_lb(21, "LD", 2).
shifts_hard_lb(21, "SE", 2).
shifts_hard_lb(21, "SN", 2).
shifts_soft_lb(21, "LD", 3).
shifts_soft_lb(21, "SE", 3).
shifts_soft_lb(21, "SN", 3).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 1).
shifts_soft_ub(21, "LM", 2).
shifts_hard_ub(21, "LM", 2).
% Staff 22:
shifts_hard_lb(22, "LD", 2).
shifts_hard_lb(22, "SE", 2).
shifts_hard_lb(22, "SN", 2).
shifts_soft_lb(22, "LD", 3).
shifts_soft_lb(22, "SE", 3).
shifts_soft_lb(22, "SN", 3).
shifts_soft_ub(22, "LD", 5).
shifts_soft_ub(22, "SE", 5).
shifts_soft_ub(22, "SN", 5).
shifts_hard_ub(22, "LD", 6).
shifts_hard_ub(22, "SE", 6).
shifts_hard_ub(22, "SN", 6).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "LM", 1).
shifts_soft_ub(22, "LM", 2).
shifts_hard_ub(22, "LM", 2).
% Staff 23:
% Staff 24:
shifts_hard_lb(24, "LD", 2).
shifts_hard_lb(24, "SE", 2).
shifts_hard_lb(24, "SN", 2).
shifts_soft_lb(24, "LD", 3).
shifts_soft_lb(24, "SE", 3).
shifts_soft_lb(24, "SN", 3).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_hard_ub(24, "LD", 6).
shifts_hard_ub(24, "SE", 6).
shifts_hard_ub(24, "SN", 6).
shifts_hard_lb(24, "LM", 0).
shifts_soft_lb(24, "LM", 1).
shifts_soft_ub(24, "LM", 2).
shifts_hard_ub(24, "LM", 2).
% Staff 25:
shifts_hard_lb(25, "LD", 2).
shifts_hard_lb(25, "SE", 2).
shifts_hard_lb(25, "SN", 2).
shifts_soft_lb(25, "LD", 3).
shifts_soft_lb(25, "SE", 3).
shifts_soft_lb(25, "SN", 3).
shifts_soft_ub(25, "LD", 5).
shifts_soft_ub(25, "SE", 5).
shifts_soft_ub(25, "SN", 5).
shifts_hard_ub(25, "LD", 6).
shifts_hard_ub(25, "SE", 6).
shifts_hard_ub(25, "SN", 6).
shifts_hard_lb(25, "LM", 0).
shifts_soft_lb(25, "LM", 1).
shifts_soft_ub(25, "LM", 2).
shifts_hard_ub(25, "LM", 2).
% Staff 26:
shifts_hard_lb(26, "LD", 2).
shifts_hard_lb(26, "SE", 2).
shifts_hard_lb(26, "SN", 2).
shifts_soft_lb(26, "LD", 3).
shifts_soft_lb(26, "SE", 3).
shifts_soft_lb(26, "SN", 3).
shifts_soft_ub(26, "LD", 5).
shifts_soft_ub(26, "SE", 5).
shifts_soft_ub(26, "SN", 5).
shifts_hard_ub(26, "LD", 6).
shifts_hard_ub(26, "SE", 6).
shifts_hard_ub(26, "SN", 6).
shifts_hard_lb(26, "LM", 0).
shifts_soft_lb(26, "LM", 1).
shifts_soft_ub(26, "LM", 2).
shifts_hard_ub(26, "LM", 2).
% Staff 27:
shifts_hard_lb(27, "LD", 2).
shifts_hard_lb(27, "SE", 2).
shifts_hard_lb(27, "SN", 2).
shifts_soft_lb(27, "LD", 3).
shifts_soft_lb(27, "SE", 3).
shifts_soft_lb(27, "SN", 3).
shifts_soft_ub(27, "LD", 5).
shifts_soft_ub(27, "SE", 5).
shifts_soft_ub(27, "SN", 5).
shifts_hard_ub(27, "LD", 6).
shifts_hard_ub(27, "SE", 6).
shifts_hard_ub(27, "SN", 6).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "LM", 1).
shifts_soft_ub(27, "LM", 2).
shifts_hard_ub(27, "LM", 2).
% Staff 28:
shifts_hard_lb(28, "LD", 2).
shifts_hard_lb(28, "SE", 2).
shifts_hard_lb(28, "SN", 2).
shifts_soft_lb(28, "LD", 3).
shifts_soft_lb(28, "SE", 3).
shifts_soft_lb(28, "SN", 3).
shifts_soft_ub(28, "LD", 5).
shifts_soft_ub(28, "SE", 5).
shifts_soft_ub(28, "SN", 5).
shifts_hard_ub(28, "LD", 6).
shifts_hard_ub(28, "SE", 6).
shifts_hard_ub(28, "SN", 6).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "LM", 1).
shifts_soft_ub(28, "LM", 2).
shifts_hard_ub(28, "LM", 2).
% Staff 29:
shifts_hard_lb(29, "LD", 1).
shifts_hard_lb(29, "SE", 1).
shifts_hard_lb(29, "SN", 1).
shifts_soft_lb(29, "LD", 2).
shifts_soft_lb(29, "SE", 2).
shifts_soft_lb(29, "SN", 2).
shifts_soft_ub(29, "LD", 3).
shifts_soft_ub(29, "SE", 3).
shifts_soft_ub(29, "SN", 3).
shifts_hard_ub(29, "LD", 3).
shifts_hard_ub(29, "SE", 3).
shifts_hard_ub(29, "SN", 3).
shifts_hard_lb(29, "LM", 0).
shifts_soft_lb(29, "LM", 1).
shifts_soft_ub(29, "LM", 2).
shifts_hard_ub(29, "LM", 2).
% Staff 30:
shifts_hard_lb(30, "LD", 1).
shifts_hard_lb(30, "SE", 1).
shifts_hard_lb(30, "SN", 1).
shifts_soft_lb(30, "LD", 2).
shifts_soft_lb(30, "SE", 2).
shifts_soft_lb(30, "SN", 2).
shifts_soft_ub(30, "LD", 3).
shifts_soft_ub(30, "SE", 3).
shifts_soft_ub(30, "SN", 3).
shifts_hard_ub(30, "LD", 3).
shifts_hard_ub(30, "SE", 3).
shifts_hard_ub(30, "SN", 3).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "LM", 1).
shifts_soft_ub(30, "LM", 2).
shifts_hard_ub(30, "LM", 2).
% Staff 31:
shifts_hard_lb(31, "LD", 1).
shifts_hard_lb(31, "SE", 1).
shifts_hard_lb(31, "SN", 1).
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
shifts_soft_lb(31, "LM", 1).
shifts_soft_ub(31, "LM", 2).
shifts_hard_ub(31, "LM", 2).
% Staff 32:
shifts_hard_lb(32, "LD", 1).
shifts_hard_lb(32, "SE", 1).
shifts_hard_lb(32, "SN", 1).
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
shifts_soft_lb(32, "LM", 1).
shifts_soft_ub(32, "LM", 2).
shifts_hard_ub(32, "LM", 2).
% Staff 33:
shifts_hard_lb(33, "LD", 1).
shifts_hard_lb(33, "SE", 1).
shifts_hard_lb(33, "SN", 1).
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
shifts_soft_lb(33, "LM", 1).
shifts_soft_ub(33, "LM", 2).
shifts_hard_ub(33, "LM", 2).
% Staff 34:
shifts_hard_lb(34, "LD", 1).
shifts_hard_lb(34, "SE", 1).
shifts_hard_lb(34, "SN", 1).
shifts_soft_lb(34, "LD", 2).
shifts_soft_lb(34, "SE", 2).
shifts_soft_lb(34, "SN", 2).
shifts_soft_ub(34, "LD", 3).
shifts_soft_ub(34, "SE", 3).
shifts_soft_ub(34, "SN", 3).
shifts_hard_ub(34, "LD", 3).
shifts_hard_ub(34, "SE", 3).
shifts_hard_ub(34, "SN", 3).
shifts_hard_lb(34, "LM", 0).
shifts_soft_lb(34, "LM", 1).
shifts_soft_ub(34, "LM", 2).
shifts_hard_ub(34, "LM", 2).
% Staff 36:
% Staff 37:
% Staff 38:
% Staff 39:
% Staff 40:
% Staff 41:
shifts_hard_lb(41, "LD", 1).
shifts_hard_lb(41, "SE", 1).
shifts_hard_lb(41, "SN", 1).
shifts_soft_lb(41, "LD", 1).
shifts_soft_lb(41, "SE", 1).
shifts_soft_lb(41, "SN", 1).
shifts_soft_ub(41, "LD", 1).
shifts_soft_ub(41, "SE", 1).
shifts_soft_ub(41, "SN", 1).
shifts_hard_ub(41, "LD", 1).
shifts_hard_ub(41, "SE", 1).
shifts_hard_ub(41, "SN", 1).
shifts_hard_lb(41, "LM", 0).
shifts_soft_lb(41, "LM", 1).
shifts_soft_ub(41, "LM", 2).
shifts_hard_ub(41, "LM", 2).
% Staff 43:
shifts_hard_lb(43, "LM", 0).
shifts_soft_lb(43, "LM", 1).
shifts_soft_ub(43, "LM", 1).
shifts_hard_ub(43, "LM", 1).
% Staff 44:
shifts_hard_lb(44, "LD", 2).
shifts_hard_lb(44, "SE", 2).
shifts_hard_lb(44, "SN", 2).
shifts_soft_lb(44, "LD", 3).
shifts_soft_lb(44, "SE", 3).
shifts_soft_lb(44, "SN", 3).
shifts_soft_ub(44, "LD", 4).
shifts_soft_ub(44, "SE", 4).
shifts_soft_ub(44, "SN", 4).
shifts_hard_ub(44, "LD", 5).
shifts_hard_ub(44, "SE", 5).
shifts_hard_ub(44, "SN", 5).
shifts_hard_lb(44, "LM", 0).
shifts_soft_lb(44, "LM", 1).
shifts_soft_ub(44, "LM", 1).
shifts_hard_ub(44, "LM", 1).
% Staff 45:
shifts_hard_lb(45, "LD", 2).
shifts_hard_lb(45, "SE", 2).
shifts_hard_lb(45, "SN", 2).
shifts_soft_lb(45, "LD", 3).
shifts_soft_lb(45, "SE", 3).
shifts_soft_lb(45, "SN", 3).
shifts_soft_ub(45, "LD", 5).
shifts_soft_ub(45, "SE", 5).
shifts_soft_ub(45, "SN", 5).
shifts_hard_ub(45, "LD", 6).
shifts_hard_ub(45, "SE", 6).
shifts_hard_ub(45, "SN", 6).
shifts_hard_lb(45, "LM", 0).
shifts_soft_lb(45, "LM", 1).
shifts_soft_ub(45, "LM", 1).
shifts_hard_ub(45, "LM", 1).
% Staff 46:
shifts_hard_lb(46, "LD", 2).
shifts_hard_lb(46, "SE", 2).
shifts_hard_lb(46, "SN", 2).
shifts_soft_lb(46, "LD", 3).
shifts_soft_lb(46, "SE", 3).
shifts_soft_lb(46, "SN", 3).
shifts_soft_ub(46, "LD", 4).
shifts_soft_ub(46, "SE", 4).
shifts_soft_ub(46, "SN", 4).
shifts_hard_ub(46, "LD", 5).
shifts_hard_ub(46, "SE", 5).
shifts_hard_ub(46, "SN", 5).
shifts_hard_lb(46, "LM", 0).
shifts_soft_lb(46, "LM", 1).
shifts_soft_ub(46, "LM", 1).
shifts_hard_ub(46, "LM", 1).
% Staff 47:
shifts_hard_lb(47, "LD", 2).
shifts_hard_lb(47, "SE", 2).
shifts_hard_lb(47, "SN", 2).
shifts_soft_lb(47, "LD", 3).
shifts_soft_lb(47, "SE", 3).
shifts_soft_lb(47, "SN", 3).
shifts_soft_ub(47, "LD", 4).
shifts_soft_ub(47, "SE", 4).
shifts_soft_ub(47, "SN", 4).
shifts_hard_ub(47, "LD", 5).
shifts_hard_ub(47, "SE", 5).
shifts_hard_ub(47, "SN", 5).
shifts_hard_lb(47, "LM", 0).
shifts_soft_lb(47, "LM", 1).
shifts_soft_ub(47, "LM", 1).
shifts_hard_ub(47, "LM", 1).
% Staff 48:
% Staff 49:
% Staff 50:
% Staff 51:
shifts_hard_lb(51, "LD", 1).
shifts_hard_lb(51, "SE", 1).
shifts_hard_lb(51, "SN", 1).
shifts_soft_lb(51, "LD", 1).
shifts_soft_lb(51, "SE", 1).
shifts_soft_lb(51, "SN", 1).
shifts_soft_ub(51, "LD", 2).
shifts_soft_ub(51, "SE", 2).
shifts_soft_ub(51, "SN", 2).
shifts_hard_ub(51, "LD", 2).
shifts_hard_ub(51, "SE", 2).
shifts_hard_ub(51, "SN", 2).
% Staff 52:
shifts_hard_lb(52, "LD", 1).
shifts_hard_lb(52, "SE", 1).
shifts_hard_lb(52, "SN", 1).
shifts_soft_lb(52, "LD", 2).
shifts_soft_lb(52, "SE", 2).
shifts_soft_lb(52, "SN", 2).
shifts_soft_ub(52, "LD", 2).
shifts_soft_ub(52, "SE", 2).
shifts_soft_ub(52, "SN", 2).
shifts_hard_ub(52, "LD", 2).
shifts_hard_ub(52, "SE", 2).
shifts_hard_ub(52, "SN", 2).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("LM").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
