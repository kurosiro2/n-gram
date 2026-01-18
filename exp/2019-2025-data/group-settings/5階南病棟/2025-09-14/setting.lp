% Staffs ---------------------------------------------------
staff(1, "Stacey Rhodes", "Chief Nurse", "07622", 0).
staff(2, "Jennifer Maldonado", "Deputy Chief Nurse", "25435", 0).
staff(3, "Christy Collins", "Deputy Chief Nurse", "61026", 0).
staff(4, "Edward Alvarez", "Nurse", "04741", 0).
staff(5, "Natalie Kennedy", "Nurse", "34770", 0).
staff(6, "Jared Howe", "Nurse", "62161", 0).
staff(7, "David Carter", "Nurse", "67610", 0).
staff(8, "Lori Perez", "Nurse", "71316", 0).
staff(9, "Micheal Sexton", "Nurse", "61638", 0).
staff(10, "Stacy Tucker", "Nurse", "94027", 0).
staff(11, "Jennifer Le", "Nurse", "03874", 0).
staff(12, "Lisa Ford", "Nurse", "80153", 0).
staff(13, "Jill Merritt", "Nurse", "03189", 0).
staff(14, "Carla Adams", "Nurse", "60822", 0).
staff(15, "Julian Harris", "Nurse", "A624468", 0).
staff(16, "William Cruz", "Nurse", "A089157", 0).
staff(17, "Levi Wright", "Nurse", "A179247", 0).
staff(18, "Charles Chambers", "Deputy Chief Nurse", "34944", 0).
staff(19, "Michael Garner", "Deputy Chief Nurse", "75479", 0).
staff(20, "Garrett Bean", "Nurse", "67709", 0).
staff(21, "Jaclyn Sanchez", "Nurse", "57921", 0).
staff(22, "Melanie Stephens", "Nurse", "41836", 0).
staff(23, "Brittney Mcdowell", "Nurse", "55629", 0).
staff(24, "James Williams", "Nurse", "21745", 0).
staff(25, "Amy Lester", "Nurse", "94975", 0).
staff(26, "James Jones", "Nurse", "65419", 0).
staff(27, "Brittney Chan", "Nurse", "13320", 0).
staff(28, "Arthur Morrow", "Nurse", "46820", 0).
staff(29, "Dr. Michelle Phillips", "Nurse", "45749", 0).
staff(30, "Jerry Wood", "Nurse", "67943", 0).
staff(31, "Douglas Williams", "Nurse", "95040", 0).
staff(32, "Benjamin Harris", "Nurse", "A697824", 0).
staff(33, "Daniel Scott", "Nurse", "A549561", 0).
staff(34, "Adrian Ortiz", "Nurse", "A157897", 0).
% Groups ---------------------------------------------------
group("All").
group("All", 2).
group("All", 3).
group("All", 4).
group("All", 5).
group("All", 6).
group("All", 7).
group("All", 8).
group("All", 9).
group("All", 10).
group("All", 11).
group("All", 12).
group("All", 13).
group("All", 14).
group("All", 15).
group("All", 16).
group("All", 17).
group("All", 18).
group("All", 19).
group("All", 20).
group("All", 21).
group("All", 22).
group("All", 23).
group("All", 24).
group("All", 25).
group("All", 26).
group("All", 27).
group("All", 28).
group("All", 29).
group("All", 30).
group("All", 31).
group("All", 32).
group("All", 33).
group("All", 34).
group("Leaders").
group("Leaders", 2).
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 6).
group("Leaders", 8).
group("Leaders", 9).
group("Leaders", 10).
group("Leaders", 11).
group("Leaders", 19).
group("Leaders", 20).
group("Leaders", 21).
group("Leaders", 22).
group("Leaders", 23).
group("Leaders", 24).
group("Leaders", 25).
group("Leaders", 26).
group("Leaders", 27).
group("Leaders", 28).
group("A").
group("A", 2).
group("A", 3).
group("A", 4).
group("A", 5).
group("A", 6).
group("A", 7).
group("A", 8).
group("A", 9).
group("A", 10).
group("A", 11).
group("A", 12).
group("A", 13).
group("A", 14).
group("A", 15).
group("A", 16).
group("A", 17).
group("B").
group("B", 18).
group("B", 19).
group("B", 20).
group("B", 21).
group("B", 22).
group("B", 23).
group("B", 24).
group("B", 25).
group("B", 26).
group("B", 27).
group("B", 28).
group("B", 29).
group("B", 30).
group("B", 31).
group("B", 32).
group("B", 33).
group("B", 34).
group("Newcomers").
group("Newcomers", 15).
group("Newcomers", 16).
group("Newcomers", 17).
group("Newcomers", 32).
group("Newcomers", 33).
group("Newcomers", 34).
group("Night").
group("Night", 1).
group("Night", 2).
group("Night", 3).
group("Night", 4).
group("Night", 5).
group("Night", 6).
group("Night", 7).
group("Night", 8).
group("Night", 9).
group("Night", 10).
group("Night", 11).
group("Night", 12).
group("Night", 13).
group("Night", 14).
group("Night", 15).
group("Night", 16).
group("Night", 17).
group("Night", 18).
group("Night", 19).
group("Night", 20).
group("Night", 21).
group("Night", 23).
group("Night", 24).
group("Night", 25).
group("Night", 26).
group("Night", 27).
group("Night", 28).
group("Night", 29).
group("Night", 30).
group("Night", 31).
group("Night", 32).
group("Night", 33).
group("Night", 34).
% Shifts ---------------------------------------------------
shift("D").
shift("LD").
shift("SE").
shift("SN").
shift("E").
shift("N").
shift("EM").
shift("LM").
shift("WR").
shift("PH").
shift("TR").
shift("AL").
shift("HL").
shift("SP").
shift("/").
% Shift constraints ----------------------------------------
% 連続勤務可能日数
consecutive_work_ub(6).
% 禁止パターン
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
% 前日・翌日可能シフト
pred_shift("SE", "SN").
succ_shift("SN", "SE").
succ_shift("SN", "WR").
succ_shift("SN", "PH").
pred_shift("LD", "SE").
pred_shift("LM", "SE").
pred_shift("D", "SE").
pred_shift("WR", "SE").
pred_shift("PH", "SE").
succ_shift("SE", "SN").
pred_shift("LD", "E").
pred_shift("LM", "E").
pred_shift("D", "E").
pred_shift("WR", "E").
pred_shift("PH", "E").
succ_shift("E", "N").
pred_shift("E", "N").
succ_shift("N", "WR").
succ_shift("N", "PH").
pred_shift("D", "D").
pred_shift("WR", "D").
pred_shift("PH", "D").
succ_shift("D", "LD").
succ_shift("D", "LM").
succ_shift("D", "D").
succ_shift("D", "WR").
succ_shift("D", "PH").
succ_shift("D", "SE").
succ_shift("D", "E").
pred_shift("D", "LD").
pred_shift("WR", "LD").
pred_shift("PH", "LD").
succ_shift("LD", "LM").
succ_shift("LD", "WR").
succ_shift("LD", "PH").
succ_shift("LD", "SE").
succ_shift("LD", "E").
pred_shift("LD", "LM").
pred_shift("D", "LM").
pred_shift("WR", "LM").
pred_shift("PH", "LM").
succ_shift("LM", "WR").
succ_shift("LM", "PH").
succ_shift("LM", "SE").
succ_shift("LM", "E").
% パターン割当数
patterns("WR-WR", "WR-WR").
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("○○", 3).
patterns("LD-D", "LD-D").
patterns_soft_ub("N日", 0).
patterns("LM-D", "LM-D").
patterns_soft_ub("P日", 0).
patterns("LM-LD", "LM-LD").
patterns_soft_ub("PN", 0).
patterns("LM-LM", "LM-LM").
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_soft_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("D").
shifts("D", "D").
staff_dweek_soft_lb("All", "D", "Weekday", 5).
staff_dweek_soft_lb("All", "D", "Weekend", 3).
staff_dweek_soft_lb("All", "D", "Holiday", 3).
staff_dweek_soft_ub("All", "D", "Weekday", 8).
staff_dweek_soft_ub("All", "D", "Weekend", 4).
staff_dweek_soft_ub("All", "D", "Holiday", 4).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 4).
staff_dweek_soft_lb("All", "LD", "Weekend", 4).
staff_dweek_soft_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
staff_dweek_hard_ub("All", "LD", "Weekday", 4).
staff_dweek_hard_ub("All", "LD", "Weekend", 4).
staff_dweek_hard_ub("All", "LD", "Holiday", 4).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 4).
shifts("SN").
shifts("SN", "SN").
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
shifts("E").
shifts("E", "E").
staff_dweek_hard_lb("All", "E", "Weekday", 0).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("All", "N", "Weekday", 0).
staff_dweek_hard_lb("All", "E", "Weekend", 0).
staff_dweek_hard_lb("All", "N", "Weekend", 0).
staff_dweek_hard_lb("All", "E", "Holiday", 0).
staff_dweek_hard_lb("All", "N", "Holiday", 0).
staff_dweek_soft_lb("All", "E", "Weekday", 0).
staff_dweek_soft_lb("All", "N", "Weekday", 0).
staff_dweek_soft_lb("All", "E", "Weekend", 0).
staff_dweek_soft_lb("All", "N", "Weekend", 0).
staff_dweek_soft_lb("All", "E", "Holiday", 0).
staff_dweek_soft_lb("All", "N", "Holiday", 0).
staff_dweek_soft_ub("All", "E", "Weekday", 0).
staff_dweek_soft_ub("All", "N", "Weekday", 0).
staff_dweek_soft_ub("All", "E", "Weekend", 0).
staff_dweek_soft_ub("All", "N", "Weekend", 0).
staff_dweek_soft_ub("All", "E", "Holiday", 0).
staff_dweek_soft_ub("All", "N", "Holiday", 0).
staff_dweek_hard_ub("All", "E", "Weekday", 0).
staff_dweek_hard_ub("All", "N", "Weekday", 0).
staff_dweek_hard_ub("All", "E", "Weekend", 0).
staff_dweek_hard_ub("All", "N", "Weekend", 0).
staff_dweek_hard_ub("All", "E", "Holiday", 0).
staff_dweek_hard_ub("All", "N", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_soft_ub("All", "LM", "Weekday", 1).
staff_dweek_soft_ub("All", "LM", "Weekend", 1).
staff_dweek_soft_ub("All", "LM", "Holiday", 1).
staff_dweek_hard_lb("A", "LD", "Weekday", 2).
staff_dweek_hard_lb("A", "SE", "Weekday", 2).
staff_dweek_hard_lb("A", "SN", "Weekday", 2).
staff_dweek_hard_lb("A", "LD", "Weekend", 2).
staff_dweek_hard_lb("A", "SE", "Weekend", 2).
staff_dweek_hard_lb("A", "SN", "Weekend", 2).
staff_dweek_hard_lb("A", "LD", "Holiday", 2).
staff_dweek_hard_lb("A", "SE", "Holiday", 2).
staff_dweek_hard_lb("A", "SN", "Holiday", 2).
staff_dweek_soft_lb("A", "LD", "Weekday", 2).
staff_dweek_soft_lb("A", "SE", "Weekday", 2).
staff_dweek_soft_lb("A", "SN", "Weekday", 2).
staff_dweek_soft_lb("A", "LD", "Weekend", 2).
staff_dweek_soft_lb("A", "SE", "Weekend", 2).
staff_dweek_soft_lb("A", "SN", "Weekend", 2).
staff_dweek_soft_lb("A", "LD", "Holiday", 2).
staff_dweek_soft_lb("A", "SE", "Holiday", 2).
staff_dweek_soft_lb("A", "SN", "Holiday", 2).
staff_dweek_soft_ub("A", "LD", "Weekday", 2).
staff_dweek_soft_ub("A", "SE", "Weekday", 2).
staff_dweek_soft_ub("A", "SN", "Weekday", 2).
staff_dweek_soft_ub("A", "LD", "Weekend", 2).
staff_dweek_soft_ub("A", "SE", "Weekend", 2).
staff_dweek_soft_ub("A", "SN", "Weekend", 2).
staff_dweek_soft_ub("A", "LD", "Holiday", 2).
staff_dweek_soft_ub("A", "SE", "Holiday", 2).
staff_dweek_soft_ub("A", "SN", "Holiday", 2).
staff_dweek_soft_lb("Leaders", "D", "Weekday", 2).
staff_dweek_soft_lb("Leaders", "D", "Weekend", 1).
staff_dweek_soft_lb("Leaders", "D", "Holiday", 1).
staff_dweek_soft_ub("Leaders", "D", "Weekday", 2).
staff_dweek_soft_ub("Leaders", "D", "Weekend", 2).
staff_dweek_soft_ub("Leaders", "D", "Holiday", 2).
staff_dweek_hard_lb("B", "LD", "Weekday", 2).
staff_dweek_hard_lb("B", "SE", "Weekday", 2).
staff_dweek_hard_lb("B", "SN", "Weekday", 2).
staff_dweek_hard_lb("B", "LD", "Weekend", 2).
staff_dweek_hard_lb("B", "SE", "Weekend", 2).
staff_dweek_hard_lb("B", "SN", "Weekend", 2).
staff_dweek_hard_lb("B", "LD", "Holiday", 2).
staff_dweek_hard_lb("B", "SE", "Holiday", 2).
staff_dweek_hard_lb("B", "SN", "Holiday", 2).
staff_dweek_soft_lb("B", "LD", "Weekday", 2).
staff_dweek_soft_lb("B", "SE", "Weekday", 2).
staff_dweek_soft_lb("B", "SN", "Weekday", 2).
staff_dweek_soft_lb("B", "LD", "Weekend", 2).
staff_dweek_soft_lb("B", "SE", "Weekend", 2).
staff_dweek_soft_lb("B", "SN", "Weekend", 2).
staff_dweek_soft_lb("B", "LD", "Holiday", 2).
staff_dweek_soft_lb("B", "SE", "Holiday", 2).
staff_dweek_soft_lb("B", "SN", "Holiday", 2).
staff_dweek_soft_ub("B", "LD", "Weekday", 2).
staff_dweek_soft_ub("B", "SE", "Weekday", 2).
staff_dweek_soft_ub("B", "SN", "Weekday", 2).
staff_dweek_soft_ub("B", "LD", "Weekend", 2).
staff_dweek_soft_ub("B", "SE", "Weekend", 2).
staff_dweek_soft_ub("B", "SN", "Weekend", 2).
staff_dweek_soft_ub("B", "LD", "Holiday", 2).
staff_dweek_soft_ub("B", "SE", "Holiday", 2).
staff_dweek_soft_ub("B", "SN", "Holiday", 2).
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 1).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 1).
staff_dweek_hard_lb("Newcomers", "SE", "Weekend", 1).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 1).
staff_dweek_hard_lb("Newcomers", "SE", "Holiday", 1).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 1).
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
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "E", "Staffs").
vertical_constraint_type("All", "N", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("A", "LD", "Staffs").
vertical_constraint_type("A", "SE", "Staffs").
vertical_constraint_type("A", "SN", "Staffs").
vertical_constraint_type("Leaders", "D", "Staffs").
vertical_constraint_type("B", "LD", "Staffs").
vertical_constraint_type("B", "SE", "Staffs").
vertical_constraint_type("B", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
% Staff 2:
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 5).
shifts_soft_ub(2, "SE", 5).
shifts_soft_ub(2, "SN", 5).
% Staff 3:
shifts_soft_lb(3, "LD", 2).
shifts_soft_lb(3, "SE", 2).
shifts_soft_lb(3, "SN", 2).
shifts_soft_ub(3, "LD", 5).
shifts_soft_ub(3, "SE", 5).
shifts_soft_ub(3, "SN", 5).
% Staff 4:
shifts_soft_lb(4, "LD", 4).
shifts_soft_lb(4, "SE", 4).
shifts_soft_lb(4, "SN", 4).
shifts_soft_ub(4, "LD", 5).
shifts_soft_ub(4, "SE", 5).
shifts_soft_ub(4, "SN", 5).
shifts_soft_ub(4, "LM", 1).
% Staff 5:
shifts_soft_lb(5, "LD", 4).
shifts_soft_lb(5, "SE", 4).
shifts_soft_lb(5, "SN", 4).
shifts_soft_ub(5, "LD", 5).
shifts_soft_ub(5, "SE", 5).
shifts_soft_ub(5, "SN", 5).
shifts_soft_ub(5, "LM", 1).
% Staff 6:
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 5).
shifts_soft_ub(6, "SE", 5).
shifts_soft_ub(6, "SN", 5).
shifts_soft_ub(6, "LM", 1).
% Staff 7:
shifts_soft_lb(7, "LD", 4).
shifts_soft_lb(7, "SE", 4).
shifts_soft_lb(7, "SN", 4).
shifts_soft_ub(7, "LD", 5).
shifts_soft_ub(7, "SE", 5).
shifts_soft_ub(7, "SN", 5).
shifts_soft_ub(7, "LM", 1).
% Staff 8:
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_soft_ub(8, "LM", 1).
% Staff 9:
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_soft_ub(9, "LM", 1).
% Staff 10:
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_soft_ub(10, "LM", 1).
% Staff 11:
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_soft_ub(11, "LM", 1).
% Staff 12:
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_soft_ub(12, "LM", 1).
% Staff 13:
shifts_soft_lb(13, "LD", 4).
shifts_soft_lb(13, "SE", 4).
shifts_soft_lb(13, "SN", 4).
shifts_soft_ub(13, "LD", 5).
shifts_soft_ub(13, "SE", 5).
shifts_soft_ub(13, "SN", 5).
shifts_soft_ub(13, "LM", 1).
% Staff 14:
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_soft_ub(14, "LM", 1).
% Staff 15:
shifts_soft_lb(15, "LD", 2).
shifts_soft_lb(15, "SE", 2).
shifts_soft_lb(15, "SN", 2).
shifts_soft_ub(15, "LD", 3).
shifts_soft_ub(15, "SE", 3).
shifts_soft_ub(15, "SN", 3).
% Staff 16:
shifts_soft_lb(16, "LD", 2).
shifts_soft_lb(16, "SE", 2).
shifts_soft_lb(16, "SN", 2).
shifts_soft_ub(16, "LD", 3).
shifts_soft_ub(16, "SE", 3).
shifts_soft_ub(16, "SN", 3).
% Staff 17:
shifts_soft_lb(17, "LD", 2).
shifts_soft_lb(17, "SE", 2).
shifts_soft_lb(17, "SN", 2).
shifts_soft_ub(17, "LD", 3).
shifts_soft_ub(17, "SE", 3).
shifts_soft_ub(17, "SN", 3).
% Staff 18:
shifts_soft_lb(18, "LD", 2).
shifts_soft_lb(18, "SE", 2).
shifts_soft_lb(18, "SN", 2).
shifts_soft_ub(18, "LD", 4).
shifts_soft_ub(18, "SE", 4).
shifts_soft_ub(18, "SN", 4).
% Staff 19:
shifts_soft_lb(19, "LD", 2).
shifts_soft_lb(19, "SE", 2).
shifts_soft_lb(19, "SN", 2).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
% Staff 20:
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_soft_ub(20, "LM", 1).
% Staff 21:
shifts_soft_lb(21, "LD", 4).
shifts_soft_lb(21, "SE", 4).
shifts_soft_lb(21, "SN", 4).
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_soft_ub(21, "LM", 1).
% Staff 22:
shifts_soft_lb(22, "LD", 0).
shifts_soft_lb(22, "SE", 0).
shifts_soft_lb(22, "SN", 0).
shifts_soft_ub(22, "LD", 0).
shifts_soft_ub(22, "SE", 0).
shifts_soft_ub(22, "SN", 0).
shifts_soft_ub(22, "LM", 1).
% Staff 23:
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_soft_ub(23, "LM", 1).
% Staff 24:
shifts_soft_lb(24, "LD", 4).
shifts_soft_lb(24, "SE", 4).
shifts_soft_lb(24, "SN", 4).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_soft_ub(24, "LM", 1).
% Staff 25:
shifts_soft_lb(25, "LD", 4).
shifts_soft_lb(25, "SE", 4).
shifts_soft_lb(25, "SN", 4).
shifts_soft_ub(25, "LD", 5).
shifts_soft_ub(25, "SE", 5).
shifts_soft_ub(25, "SN", 5).
shifts_soft_ub(25, "LM", 1).
% Staff 26:
shifts_soft_lb(26, "LD", 4).
shifts_soft_lb(26, "SE", 4).
shifts_soft_lb(26, "SN", 4).
shifts_soft_ub(26, "LD", 5).
shifts_soft_ub(26, "SE", 5).
shifts_soft_ub(26, "SN", 5).
shifts_soft_ub(26, "LM", 1).
% Staff 27:
shifts_soft_lb(27, "LD", 4).
shifts_soft_lb(27, "SE", 4).
shifts_soft_lb(27, "SN", 4).
shifts_soft_ub(27, "LD", 5).
shifts_soft_ub(27, "SE", 5).
shifts_soft_ub(27, "SN", 5).
shifts_soft_ub(27, "LM", 1).
% Staff 28:
shifts_soft_lb(28, "LD", 4).
shifts_soft_lb(28, "SE", 4).
shifts_soft_lb(28, "SN", 4).
shifts_soft_ub(28, "LD", 5).
shifts_soft_ub(28, "SE", 5).
shifts_soft_ub(28, "SN", 5).
shifts_soft_ub(28, "LM", 1).
% Staff 29:
shifts_soft_lb(29, "LD", 4).
shifts_soft_lb(29, "SE", 4).
shifts_soft_lb(29, "SN", 4).
shifts_soft_ub(29, "LD", 5).
shifts_soft_ub(29, "SE", 5).
shifts_soft_ub(29, "SN", 5).
shifts_soft_ub(29, "LM", 1).
% Staff 30:
shifts_soft_lb(30, "LD", 4).
shifts_soft_lb(30, "SE", 4).
shifts_soft_lb(30, "SN", 4).
shifts_soft_ub(30, "LD", 5).
shifts_soft_ub(30, "SE", 5).
shifts_soft_ub(30, "SN", 5).
shifts_soft_ub(30, "LM", 1).
% Staff 31:
shifts_soft_lb(31, "LD", 4).
shifts_soft_lb(31, "SE", 4).
shifts_soft_lb(31, "SN", 4).
shifts_soft_ub(31, "LD", 5).
shifts_soft_ub(31, "SE", 5).
shifts_soft_ub(31, "SN", 5).
shifts_soft_ub(31, "LM", 1).
% Staff 32:
shifts_soft_lb(32, "LD", 2).
shifts_soft_lb(32, "SE", 2).
shifts_soft_lb(32, "SN", 2).
shifts_soft_ub(32, "LD", 3).
shifts_soft_ub(32, "SE", 3).
shifts_soft_ub(32, "SN", 3).
% Staff 33:
shifts_soft_lb(33, "LD", 2).
shifts_soft_lb(33, "SE", 2).
shifts_soft_lb(33, "SN", 2).
shifts_soft_ub(33, "LD", 3).
shifts_soft_ub(33, "SE", 3).
shifts_soft_ub(33, "SN", 3).
% Staff 34:
shifts_soft_lb(34, "LD", 2).
shifts_soft_lb(34, "SE", 2).
shifts_soft_lb(34, "SN", 2).
shifts_soft_ub(34, "LD", 3).
shifts_soft_ub(34, "SE", 3).
shifts_soft_ub(34, "SN", 3).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
recommended_pair(9, 17).
recommended_pair(10, 16).
recommended_pair(11, 15).
recommended_pair(18, 20).
recommended_pair(27, 33).
recommended_pair(28, 32).
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
