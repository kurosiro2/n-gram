% Staffs ---------------------------------------------------
staff(1, "Stacey Rhodes", "Chief Nurse", "07622", 0).
staff(2, "Jennifer Maldonado", "Deputy Chief Nurse", "25435", 0).
staff(3, "Edward Alvarez", "Nurse", "04741", 0).
staff(4, "Natalie Kennedy", "Nurse", "34770", 0).
staff(5, "Micheal Sexton", "Nurse", "61638", 0).
staff(6, "Stacy Tucker", "Nurse", "94027", 0).
staff(7, "Carla Adams", "Nurse", "60822", 0).
staff(8, "William Cruz", "Nurse", "A089157", 0).
staff(9, "Levi Wright", "Nurse", "A179247", 0).
staff(10, "Brittney Mcdowell", "Nurse", "55629", 0).
staff(11, "Dr. Michelle Phillips", "Nurse", "45749", 0).
staff(12, "Charles Chambers", "Deputy Chief Nurse", "34944", 0).
staff(13, "Michael Garner", "Deputy Chief Nurse", "75479", 0).
staff(14, "Garrett Bean", "Nurse", "67709", 0).
staff(15, "Amy Lester", "Nurse", "94975", 0).
staff(16, "Brittney Chan", "Nurse", "13320", 0).
staff(17, "Arthur Morrow", "Nurse", "46820", 0).
staff(18, "Jerry Wood", "Nurse", "67943", 0).
staff(19, "Douglas Williams", "Nurse", "95040", 0).
staff(20, "Benjamin Harris", "Nurse", "A697824", 0).
staff(21, "Daniel Scott", "Nurse", "A549561", 0).
staff(22, "Rebecca Mayo", "Nurse", "19364", 0).
staff(23, "Christy Collins", "Deputy Chief Nurse", "61026", 0).
staff(24, "David Carter", "Nurse", "67610", 0).
staff(25, "Lori Perez", "Nurse", "71316", 0).
staff(26, "Jennifer Le", "Nurse", "03874", 0).
staff(27, "Lisa Ford", "Nurse", "80153", 0).
staff(28, "Jill Merritt", "Nurse", "03189", 0).
staff(29, "Julian Harris", "Nurse", "A624468", 0).
staff(30, "Jaclyn Sanchez", "Nurse", "57921", 0).
staff(31, "James Williams", "Nurse", "21745", 0).
staff(32, "James Jones", "Nurse", "65419", 0).
staff(33, "Adrian Ortiz", "Nurse", "A157897", 0).
% Groups ---------------------------------------------------
group("All").
group("All", 1).
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
group("All", 28).
group("All", 29).
group("All", 30).
group("All", 31).
group("All", 32).
group("All", 33).
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("代行", 10).
group("代行", 13).
group("代行", 14).
group("代行", 15).
group("代行", 23).
group("代行", 24).
group("代行", 25).
group("代行", 30).
group("代行", 31).
group("平日代行").
group("平日代行", 2).
group("平日代行", 13).
group("平日代行", 23).
group("Leaders").
group("Leaders", 2).
group("Leaders", 3).
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 6).
group("Leaders", 10).
group("Leaders", 13).
group("Leaders", 14).
group("Leaders", 15).
group("Leaders", 16).
group("Leaders", 17).
group("Leaders", 24).
group("Leaders", 25).
group("Leaders", 26).
group("Leaders", 30).
group("Leaders", 31).
group("Leaders", 32).
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
group("B").
group("B", 12).
group("B", 13).
group("B", 14).
group("B", 15).
group("B", 16).
group("B", 17).
group("B", 18).
group("B", 19).
group("B", 20).
group("B", 21).
group("C").
group("C", 23).
group("C", 24).
group("C", 25).
group("C", 26).
group("C", 28).
group("C", 29).
group("C", 30).
group("C", 31).
group("C", 32).
group("C", 33).
group("Newcomers").
group("Newcomers", 8).
group("Newcomers", 9).
group("Newcomers", 20).
group("Newcomers", 21).
group("Newcomers", 29).
group("Newcomers", 33).
group("Night").
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
group("Night", 28).
group("Night", 29).
group("Night", 30).
group("Night", 31).
group("Night", 32).
group("Night", 33).
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
consecutive_work_ub(5).
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
patterns_hard_lb("○○", 2).
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
staff_dweek_hard_lb("All", "D", "Mo", 8).
staff_dweek_hard_lb("All", "D", "Tu", 8).
staff_dweek_hard_lb("All", "D", "We", 8).
staff_dweek_hard_lb("All", "D", "Th", 8).
staff_dweek_hard_lb("All", "D", "Fr", 8).
staff_dweek_hard_lb("All", "D", "Sa", 4).
staff_dweek_hard_lb("All", "D", "Su", 4).
staff_dweek_hard_lb("All", "D", "Ho", 4).
staff_dweek_soft_lb("All", "D", "Mo", 9).
staff_dweek_soft_lb("All", "D", "Tu", 9).
staff_dweek_soft_lb("All", "D", "We", 9).
staff_dweek_soft_lb("All", "D", "Th", 9).
staff_dweek_soft_lb("All", "D", "Fr", 9).
staff_dweek_soft_lb("All", "D", "Sa", 4).
staff_dweek_soft_lb("All", "D", "Su", 4).
staff_dweek_soft_lb("All", "D", "Ho", 4).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Mo", 4).
staff_dweek_hard_lb("All", "LD", "Tu", 4).
staff_dweek_hard_lb("All", "LD", "We", 4).
staff_dweek_hard_lb("All", "LD", "Th", 4).
staff_dweek_hard_lb("All", "LD", "Fr", 4).
staff_dweek_hard_lb("All", "LD", "Sa", 4).
staff_dweek_hard_lb("All", "LD", "Su", 4).
staff_dweek_hard_lb("All", "LD", "Ho", 4).
staff_dweek_soft_lb("All", "LD", "Mo", 4).
staff_dweek_soft_lb("All", "LD", "Tu", 4).
staff_dweek_soft_lb("All", "LD", "We", 4).
staff_dweek_soft_lb("All", "LD", "Th", 4).
staff_dweek_soft_lb("All", "LD", "Fr", 4).
staff_dweek_soft_lb("All", "LD", "Sa", 4).
staff_dweek_soft_lb("All", "LD", "Su", 4).
staff_dweek_soft_lb("All", "LD", "Ho", 4).
staff_dweek_soft_ub("All", "LD", "Mo", 4).
staff_dweek_soft_ub("All", "LD", "Tu", 4).
staff_dweek_soft_ub("All", "LD", "We", 4).
staff_dweek_soft_ub("All", "LD", "Th", 4).
staff_dweek_soft_ub("All", "LD", "Fr", 4).
staff_dweek_soft_ub("All", "LD", "Sa", 4).
staff_dweek_soft_ub("All", "LD", "Su", 4).
staff_dweek_soft_ub("All", "LD", "Ho", 4).
staff_dweek_hard_ub("All", "LD", "Mo", 4).
staff_dweek_hard_ub("All", "LD", "Tu", 4).
staff_dweek_hard_ub("All", "LD", "We", 4).
staff_dweek_hard_ub("All", "LD", "Th", 4).
staff_dweek_hard_ub("All", "LD", "Fr", 4).
staff_dweek_hard_ub("All", "LD", "Sa", 4).
staff_dweek_hard_ub("All", "LD", "Su", 4).
staff_dweek_hard_ub("All", "LD", "Ho", 4).
shifts("EM").
shifts("EM", "EM").
staff_dweek_hard_lb("All", "EM", "Mo", 0).
staff_dweek_hard_lb("All", "EM", "Tu", 0).
staff_dweek_hard_lb("All", "EM", "We", 0).
staff_dweek_hard_lb("All", "EM", "Th", 0).
staff_dweek_hard_lb("All", "EM", "Fr", 0).
staff_dweek_hard_lb("All", "EM", "Sa", 0).
staff_dweek_hard_lb("All", "EM", "Su", 0).
staff_dweek_hard_lb("All", "EM", "Ho", 0).
staff_dweek_soft_lb("All", "EM", "Mo", 0).
staff_dweek_soft_lb("All", "EM", "Tu", 0).
staff_dweek_soft_lb("All", "EM", "We", 0).
staff_dweek_soft_lb("All", "EM", "Th", 0).
staff_dweek_soft_lb("All", "EM", "Fr", 0).
staff_dweek_soft_lb("All", "EM", "Sa", 0).
staff_dweek_soft_lb("All", "EM", "Su", 0).
staff_dweek_soft_lb("All", "EM", "Ho", 0).
staff_dweek_soft_ub("All", "EM", "Mo", 0).
staff_dweek_soft_ub("All", "EM", "Tu", 0).
staff_dweek_soft_ub("All", "EM", "We", 0).
staff_dweek_soft_ub("All", "EM", "Th", 0).
staff_dweek_soft_ub("All", "EM", "Fr", 0).
staff_dweek_soft_ub("All", "EM", "Sa", 0).
staff_dweek_soft_ub("All", "EM", "Su", 0).
staff_dweek_soft_ub("All", "EM", "Ho", 0).
staff_dweek_hard_ub("All", "EM", "Mo", 0).
staff_dweek_hard_ub("All", "EM", "Tu", 0).
staff_dweek_hard_ub("All", "EM", "We", 0).
staff_dweek_hard_ub("All", "EM", "Th", 0).
staff_dweek_hard_ub("All", "EM", "Fr", 0).
staff_dweek_hard_ub("All", "EM", "Sa", 0).
staff_dweek_hard_ub("All", "EM", "Su", 0).
staff_dweek_hard_ub("All", "EM", "Ho", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Mo", 1).
staff_dweek_hard_lb("All", "LM", "Tu", 1).
staff_dweek_hard_lb("All", "LM", "We", 1).
staff_dweek_hard_lb("All", "LM", "Th", 1).
staff_dweek_hard_lb("All", "LM", "Fr", 1).
staff_dweek_hard_lb("All", "LM", "Sa", 0).
staff_dweek_hard_lb("All", "LM", "Su", 0).
staff_dweek_hard_lb("All", "LM", "Ho", 0).
staff_dweek_soft_lb("All", "LM", "Mo", 1).
staff_dweek_soft_lb("All", "LM", "Tu", 1).
staff_dweek_soft_lb("All", "LM", "We", 1).
staff_dweek_soft_lb("All", "LM", "Th", 1).
staff_dweek_soft_lb("All", "LM", "Fr", 1).
staff_dweek_soft_lb("All", "LM", "Sa", 0).
staff_dweek_soft_lb("All", "LM", "Su", 0).
staff_dweek_soft_lb("All", "LM", "Ho", 0).
staff_dweek_soft_ub("All", "LM", "Mo", 1).
staff_dweek_soft_ub("All", "LM", "Tu", 1).
staff_dweek_soft_ub("All", "LM", "We", 1).
staff_dweek_soft_ub("All", "LM", "Th", 1).
staff_dweek_soft_ub("All", "LM", "Fr", 1).
staff_dweek_soft_ub("All", "LM", "Sa", 0).
staff_dweek_soft_ub("All", "LM", "Su", 0).
staff_dweek_soft_ub("All", "LM", "Ho", 0).
staff_dweek_hard_ub("All", "LM", "Mo", 1).
staff_dweek_hard_ub("All", "LM", "Tu", 1).
staff_dweek_hard_ub("All", "LM", "We", 1).
staff_dweek_hard_ub("All", "LM", "Th", 1).
staff_dweek_hard_ub("All", "LM", "Fr", 1).
staff_dweek_hard_ub("All", "LM", "Sa", 0).
staff_dweek_hard_ub("All", "LM", "Su", 0).
staff_dweek_hard_ub("All", "LM", "Ho", 0).
staff_dweek_soft_lb("代行", "D", "Mo", 0).
staff_dweek_soft_lb("代行", "D", "Tu", 0).
staff_dweek_soft_lb("代行", "D", "We", 0).
staff_dweek_soft_lb("代行", "D", "Th", 0).
staff_dweek_soft_lb("代行", "D", "Fr", 0).
staff_dweek_soft_lb("代行", "D", "Sa", 1).
staff_dweek_soft_lb("代行", "D", "Su", 1).
staff_dweek_soft_lb("代行", "D", "Ho", 1).
staff_dweek_soft_ub("代行", "D", "Mo", 0).
staff_dweek_soft_ub("代行", "D", "Tu", 0).
staff_dweek_soft_ub("代行", "D", "We", 0).
staff_dweek_soft_ub("代行", "D", "Th", 0).
staff_dweek_soft_ub("代行", "D", "Fr", 0).
staff_dweek_soft_ub("代行", "D", "Sa", 1).
staff_dweek_soft_ub("代行", "D", "Su", 1).
staff_dweek_soft_ub("代行", "D", "Ho", 1).
staff_dweek_hard_lb("平日代行", "D", "Mo", 0).
staff_dweek_hard_lb("平日代行", "D", "Tu", 0).
staff_dweek_hard_lb("平日代行", "D", "We", 0).
staff_dweek_hard_lb("平日代行", "D", "Th", 0).
staff_dweek_hard_lb("平日代行", "D", "Fr", 0).
staff_dweek_hard_lb("平日代行", "D", "Sa", 0).
staff_dweek_hard_lb("平日代行", "D", "Su", 0).
staff_dweek_hard_lb("平日代行", "D", "Ho", 0).
staff_dweek_soft_lb("平日代行", "D", "Mo", 1).
staff_dweek_soft_lb("平日代行", "D", "Tu", 1).
staff_dweek_soft_lb("平日代行", "D", "We", 1).
staff_dweek_soft_lb("平日代行", "D", "Th", 1).
staff_dweek_soft_lb("平日代行", "D", "Fr", 1).
staff_dweek_soft_lb("平日代行", "D", "Sa", 0).
staff_dweek_soft_lb("平日代行", "D", "Su", 0).
staff_dweek_soft_lb("平日代行", "D", "Ho", 0).
staff_dweek_soft_ub("平日代行", "D", "Mo", 2).
staff_dweek_soft_ub("平日代行", "D", "Tu", 2).
staff_dweek_soft_ub("平日代行", "D", "We", 2).
staff_dweek_soft_ub("平日代行", "D", "Th", 2).
staff_dweek_soft_ub("平日代行", "D", "Fr", 2).
staff_dweek_soft_ub("平日代行", "D", "Sa", 0).
staff_dweek_soft_ub("平日代行", "D", "Su", 0).
staff_dweek_soft_ub("平日代行", "D", "Ho", 0).
staff_dweek_hard_lb("Leaders", "LD", "Mo", 2).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("Leaders", "SE", "Mo", 2).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("Leaders", "SN", "Mo", 2).
staff_dweek_hard_lb("Leaders", "LD", "Tu", 2).
staff_dweek_hard_lb("Leaders", "SE", "Tu", 2).
staff_dweek_hard_lb("Leaders", "SN", "Tu", 2).
staff_dweek_hard_lb("Leaders", "LD", "We", 2).
staff_dweek_hard_lb("Leaders", "SE", "We", 2).
staff_dweek_hard_lb("Leaders", "SN", "We", 2).
staff_dweek_hard_lb("Leaders", "LD", "Th", 2).
staff_dweek_hard_lb("Leaders", "SE", "Th", 2).
staff_dweek_hard_lb("Leaders", "SN", "Th", 2).
staff_dweek_hard_lb("Leaders", "LD", "Fr", 2).
staff_dweek_hard_lb("Leaders", "SE", "Fr", 2).
staff_dweek_hard_lb("Leaders", "SN", "Fr", 2).
staff_dweek_hard_lb("Leaders", "LD", "Sa", 2).
staff_dweek_hard_lb("Leaders", "SE", "Sa", 2).
staff_dweek_hard_lb("Leaders", "SN", "Sa", 2).
staff_dweek_hard_lb("Leaders", "LD", "Su", 2).
staff_dweek_hard_lb("Leaders", "SE", "Su", 2).
staff_dweek_hard_lb("Leaders", "SN", "Su", 2).
staff_dweek_hard_lb("Leaders", "LD", "Ho", 2).
staff_dweek_hard_lb("Leaders", "SE", "Ho", 2).
staff_dweek_hard_lb("Leaders", "SN", "Ho", 2).
staff_dweek_soft_lb("Leaders", "LD", "Mo", 2).
staff_dweek_soft_lb("Leaders", "SE", "Mo", 2).
staff_dweek_soft_lb("Leaders", "SN", "Mo", 2).
staff_dweek_soft_lb("Leaders", "LD", "Tu", 2).
staff_dweek_soft_lb("Leaders", "SE", "Tu", 2).
staff_dweek_soft_lb("Leaders", "SN", "Tu", 2).
staff_dweek_soft_lb("Leaders", "LD", "We", 2).
staff_dweek_soft_lb("Leaders", "SE", "We", 2).
staff_dweek_soft_lb("Leaders", "SN", "We", 2).
staff_dweek_soft_lb("Leaders", "LD", "Th", 2).
staff_dweek_soft_lb("Leaders", "SE", "Th", 2).
staff_dweek_soft_lb("Leaders", "SN", "Th", 2).
staff_dweek_soft_lb("Leaders", "LD", "Fr", 2).
staff_dweek_soft_lb("Leaders", "SE", "Fr", 2).
staff_dweek_soft_lb("Leaders", "SN", "Fr", 2).
staff_dweek_soft_lb("Leaders", "LD", "Sa", 2).
staff_dweek_soft_lb("Leaders", "SE", "Sa", 2).
staff_dweek_soft_lb("Leaders", "SN", "Sa", 2).
staff_dweek_soft_lb("Leaders", "LD", "Su", 2).
staff_dweek_soft_lb("Leaders", "SE", "Su", 2).
staff_dweek_soft_lb("Leaders", "SN", "Su", 2).
staff_dweek_soft_lb("Leaders", "LD", "Ho", 2).
staff_dweek_soft_lb("Leaders", "SE", "Ho", 2).
staff_dweek_soft_lb("Leaders", "SN", "Ho", 2).
staff_dweek_soft_ub("Leaders", "LD", "Mo", 3).
staff_dweek_soft_ub("Leaders", "SE", "Mo", 3).
staff_dweek_soft_ub("Leaders", "SN", "Mo", 3).
staff_dweek_soft_ub("Leaders", "LD", "Tu", 3).
staff_dweek_soft_ub("Leaders", "SE", "Tu", 3).
staff_dweek_soft_ub("Leaders", "SN", "Tu", 3).
staff_dweek_soft_ub("Leaders", "LD", "We", 3).
staff_dweek_soft_ub("Leaders", "SE", "We", 3).
staff_dweek_soft_ub("Leaders", "SN", "We", 3).
staff_dweek_soft_ub("Leaders", "LD", "Th", 3).
staff_dweek_soft_ub("Leaders", "SE", "Th", 3).
staff_dweek_soft_ub("Leaders", "SN", "Th", 3).
staff_dweek_soft_ub("Leaders", "LD", "Fr", 3).
staff_dweek_soft_ub("Leaders", "SE", "Fr", 3).
staff_dweek_soft_ub("Leaders", "SN", "Fr", 3).
staff_dweek_soft_ub("Leaders", "LD", "Sa", 3).
staff_dweek_soft_ub("Leaders", "SE", "Sa", 3).
staff_dweek_soft_ub("Leaders", "SN", "Sa", 3).
staff_dweek_soft_ub("Leaders", "LD", "Su", 3).
staff_dweek_soft_ub("Leaders", "SE", "Su", 3).
staff_dweek_soft_ub("Leaders", "SN", "Su", 3).
staff_dweek_soft_ub("Leaders", "LD", "Ho", 3).
staff_dweek_soft_ub("Leaders", "SE", "Ho", 3).
staff_dweek_soft_ub("Leaders", "SN", "Ho", 3).
staff_dweek_hard_lb("A", "LD", "Mo", 1).
staff_dweek_hard_lb("A", "SE", "Mo", 1).
staff_dweek_hard_lb("A", "SN", "Mo", 1).
staff_dweek_hard_lb("A", "LD", "Tu", 1).
staff_dweek_hard_lb("A", "SE", "Tu", 1).
staff_dweek_hard_lb("A", "SN", "Tu", 1).
staff_dweek_hard_lb("A", "LD", "We", 1).
staff_dweek_hard_lb("A", "SE", "We", 1).
staff_dweek_hard_lb("A", "SN", "We", 1).
staff_dweek_hard_lb("A", "LD", "Th", 1).
staff_dweek_hard_lb("A", "SE", "Th", 1).
staff_dweek_hard_lb("A", "SN", "Th", 1).
staff_dweek_hard_lb("A", "LD", "Fr", 1).
staff_dweek_hard_lb("A", "SE", "Fr", 1).
staff_dweek_hard_lb("A", "SN", "Fr", 1).
staff_dweek_hard_lb("A", "LD", "Sa", 1).
staff_dweek_hard_lb("A", "SE", "Sa", 1).
staff_dweek_hard_lb("A", "SN", "Sa", 1).
staff_dweek_hard_lb("A", "LD", "Su", 1).
staff_dweek_hard_lb("A", "SE", "Su", 1).
staff_dweek_hard_lb("A", "SN", "Su", 1).
staff_dweek_hard_lb("A", "LD", "Ho", 1).
staff_dweek_hard_lb("A", "SE", "Ho", 1).
staff_dweek_hard_lb("A", "SN", "Ho", 1).
staff_dweek_soft_lb("A", "LD", "Mo", 2).
staff_dweek_soft_lb("A", "SE", "Mo", 2).
staff_dweek_soft_lb("A", "SN", "Mo", 2).
staff_dweek_soft_lb("A", "LD", "Tu", 2).
staff_dweek_soft_lb("A", "SE", "Tu", 2).
staff_dweek_soft_lb("A", "SN", "Tu", 2).
staff_dweek_soft_lb("A", "LD", "We", 2).
staff_dweek_soft_lb("A", "SE", "We", 2).
staff_dweek_soft_lb("A", "SN", "We", 2).
staff_dweek_soft_lb("A", "LD", "Th", 2).
staff_dweek_soft_lb("A", "SE", "Th", 2).
staff_dweek_soft_lb("A", "SN", "Th", 2).
staff_dweek_soft_lb("A", "LD", "Fr", 2).
staff_dweek_soft_lb("A", "SE", "Fr", 2).
staff_dweek_soft_lb("A", "SN", "Fr", 2).
staff_dweek_soft_lb("A", "LD", "Sa", 2).
staff_dweek_soft_lb("A", "SE", "Sa", 2).
staff_dweek_soft_lb("A", "SN", "Sa", 2).
staff_dweek_soft_lb("A", "LD", "Su", 2).
staff_dweek_soft_lb("A", "SE", "Su", 2).
staff_dweek_soft_lb("A", "SN", "Su", 2).
staff_dweek_soft_lb("A", "LD", "Ho", 2).
staff_dweek_soft_lb("A", "SE", "Ho", 2).
staff_dweek_soft_lb("A", "SN", "Ho", 2).
staff_dweek_hard_ub("A", "LD", "Mo", 2).
staff_dweek_hard_ub("A", "SE", "Mo", 2).
staff_dweek_hard_ub("A", "SN", "Mo", 2).
staff_dweek_hard_ub("A", "LD", "Tu", 2).
staff_dweek_hard_ub("A", "SE", "Tu", 2).
staff_dweek_hard_ub("A", "SN", "Tu", 2).
staff_dweek_hard_ub("A", "LD", "We", 2).
staff_dweek_hard_ub("A", "SE", "We", 2).
staff_dweek_hard_ub("A", "SN", "We", 2).
staff_dweek_hard_ub("A", "LD", "Th", 2).
staff_dweek_hard_ub("A", "SE", "Th", 2).
staff_dweek_hard_ub("A", "SN", "Th", 2).
staff_dweek_hard_ub("A", "LD", "Fr", 2).
staff_dweek_hard_ub("A", "SE", "Fr", 2).
staff_dweek_hard_ub("A", "SN", "Fr", 2).
staff_dweek_hard_ub("A", "LD", "Sa", 2).
staff_dweek_hard_ub("A", "SE", "Sa", 2).
staff_dweek_hard_ub("A", "SN", "Sa", 2).
staff_dweek_hard_ub("A", "LD", "Su", 2).
staff_dweek_hard_ub("A", "SE", "Su", 2).
staff_dweek_hard_ub("A", "SN", "Su", 2).
staff_dweek_hard_ub("A", "LD", "Ho", 2).
staff_dweek_hard_ub("A", "SE", "Ho", 2).
staff_dweek_hard_ub("A", "SN", "Ho", 2).
staff_dweek_hard_lb("B", "LD", "Mo", 1).
staff_dweek_hard_lb("B", "SE", "Mo", 1).
staff_dweek_hard_lb("B", "SN", "Mo", 1).
staff_dweek_hard_lb("B", "LD", "Tu", 1).
staff_dweek_hard_lb("B", "SE", "Tu", 1).
staff_dweek_hard_lb("B", "SN", "Tu", 1).
staff_dweek_hard_lb("B", "LD", "We", 1).
staff_dweek_hard_lb("B", "SE", "We", 1).
staff_dweek_hard_lb("B", "SN", "We", 1).
staff_dweek_hard_lb("B", "LD", "Th", 1).
staff_dweek_hard_lb("B", "SE", "Th", 1).
staff_dweek_hard_lb("B", "SN", "Th", 1).
staff_dweek_hard_lb("B", "LD", "Fr", 1).
staff_dweek_hard_lb("B", "SE", "Fr", 1).
staff_dweek_hard_lb("B", "SN", "Fr", 1).
staff_dweek_hard_lb("B", "LD", "Sa", 1).
staff_dweek_hard_lb("B", "SE", "Sa", 1).
staff_dweek_hard_lb("B", "SN", "Sa", 1).
staff_dweek_hard_lb("B", "LD", "Su", 1).
staff_dweek_hard_lb("B", "SE", "Su", 1).
staff_dweek_hard_lb("B", "SN", "Su", 1).
staff_dweek_hard_lb("B", "LD", "Ho", 1).
staff_dweek_hard_lb("B", "SE", "Ho", 1).
staff_dweek_hard_lb("B", "SN", "Ho", 1).
staff_dweek_soft_lb("B", "LD", "Mo", 2).
staff_dweek_soft_lb("B", "SE", "Mo", 2).
staff_dweek_soft_lb("B", "SN", "Mo", 2).
staff_dweek_soft_lb("B", "LD", "Tu", 2).
staff_dweek_soft_lb("B", "SE", "Tu", 2).
staff_dweek_soft_lb("B", "SN", "Tu", 2).
staff_dweek_soft_lb("B", "LD", "We", 2).
staff_dweek_soft_lb("B", "SE", "We", 2).
staff_dweek_soft_lb("B", "SN", "We", 2).
staff_dweek_soft_lb("B", "LD", "Th", 2).
staff_dweek_soft_lb("B", "SE", "Th", 2).
staff_dweek_soft_lb("B", "SN", "Th", 2).
staff_dweek_soft_lb("B", "LD", "Fr", 2).
staff_dweek_soft_lb("B", "SE", "Fr", 2).
staff_dweek_soft_lb("B", "SN", "Fr", 2).
staff_dweek_soft_lb("B", "LD", "Sa", 2).
staff_dweek_soft_lb("B", "SE", "Sa", 2).
staff_dweek_soft_lb("B", "SN", "Sa", 2).
staff_dweek_soft_lb("B", "LD", "Su", 2).
staff_dweek_soft_lb("B", "SE", "Su", 2).
staff_dweek_soft_lb("B", "SN", "Su", 2).
staff_dweek_soft_lb("B", "LD", "Ho", 2).
staff_dweek_soft_lb("B", "SE", "Ho", 2).
staff_dweek_soft_lb("B", "SN", "Ho", 2).
staff_dweek_hard_ub("B", "LD", "Mo", 2).
staff_dweek_hard_ub("B", "SE", "Mo", 2).
staff_dweek_hard_ub("B", "SN", "Mo", 2).
staff_dweek_hard_ub("B", "LD", "Tu", 2).
staff_dweek_hard_ub("B", "SE", "Tu", 2).
staff_dweek_hard_ub("B", "SN", "Tu", 2).
staff_dweek_hard_ub("B", "LD", "We", 2).
staff_dweek_hard_ub("B", "SE", "We", 2).
staff_dweek_hard_ub("B", "SN", "We", 2).
staff_dweek_hard_ub("B", "LD", "Th", 2).
staff_dweek_hard_ub("B", "SE", "Th", 2).
staff_dweek_hard_ub("B", "SN", "Th", 2).
staff_dweek_hard_ub("B", "LD", "Fr", 2).
staff_dweek_hard_ub("B", "SE", "Fr", 2).
staff_dweek_hard_ub("B", "SN", "Fr", 2).
staff_dweek_hard_ub("B", "LD", "Sa", 2).
staff_dweek_hard_ub("B", "SE", "Sa", 2).
staff_dweek_hard_ub("B", "SN", "Sa", 2).
staff_dweek_hard_ub("B", "LD", "Su", 2).
staff_dweek_hard_ub("B", "SE", "Su", 2).
staff_dweek_hard_ub("B", "SN", "Su", 2).
staff_dweek_hard_ub("B", "LD", "Ho", 2).
staff_dweek_hard_ub("B", "SE", "Ho", 2).
staff_dweek_hard_ub("B", "SN", "Ho", 2).
staff_dweek_hard_lb("C", "LD", "Mo", 1).
staff_dweek_hard_lb("C", "SE", "Mo", 1).
staff_dweek_hard_lb("C", "SN", "Mo", 1).
staff_dweek_hard_lb("C", "LD", "Tu", 1).
staff_dweek_hard_lb("C", "SE", "Tu", 1).
staff_dweek_hard_lb("C", "SN", "Tu", 1).
staff_dweek_hard_lb("C", "LD", "We", 1).
staff_dweek_hard_lb("C", "SE", "We", 1).
staff_dweek_hard_lb("C", "SN", "We", 1).
staff_dweek_hard_lb("C", "LD", "Th", 1).
staff_dweek_hard_lb("C", "SE", "Th", 1).
staff_dweek_hard_lb("C", "SN", "Th", 1).
staff_dweek_hard_lb("C", "LD", "Fr", 1).
staff_dweek_hard_lb("C", "SE", "Fr", 1).
staff_dweek_hard_lb("C", "SN", "Fr", 1).
staff_dweek_hard_lb("C", "LD", "Sa", 1).
staff_dweek_hard_lb("C", "SE", "Sa", 1).
staff_dweek_hard_lb("C", "SN", "Sa", 1).
staff_dweek_hard_lb("C", "LD", "Su", 1).
staff_dweek_hard_lb("C", "SE", "Su", 1).
staff_dweek_hard_lb("C", "SN", "Su", 1).
staff_dweek_hard_lb("C", "LD", "Ho", 1).
staff_dweek_hard_lb("C", "SE", "Ho", 1).
staff_dweek_hard_lb("C", "SN", "Ho", 1).
staff_dweek_soft_lb("C", "LD", "Mo", 2).
staff_dweek_soft_lb("C", "SE", "Mo", 2).
staff_dweek_soft_lb("C", "SN", "Mo", 2).
staff_dweek_soft_lb("C", "LD", "Tu", 2).
staff_dweek_soft_lb("C", "SE", "Tu", 2).
staff_dweek_soft_lb("C", "SN", "Tu", 2).
staff_dweek_soft_lb("C", "LD", "We", 2).
staff_dweek_soft_lb("C", "SE", "We", 2).
staff_dweek_soft_lb("C", "SN", "We", 2).
staff_dweek_soft_lb("C", "LD", "Th", 2).
staff_dweek_soft_lb("C", "SE", "Th", 2).
staff_dweek_soft_lb("C", "SN", "Th", 2).
staff_dweek_soft_lb("C", "LD", "Fr", 2).
staff_dweek_soft_lb("C", "SE", "Fr", 2).
staff_dweek_soft_lb("C", "SN", "Fr", 2).
staff_dweek_soft_lb("C", "LD", "Sa", 2).
staff_dweek_soft_lb("C", "SE", "Sa", 2).
staff_dweek_soft_lb("C", "SN", "Sa", 2).
staff_dweek_soft_lb("C", "LD", "Su", 2).
staff_dweek_soft_lb("C", "SE", "Su", 2).
staff_dweek_soft_lb("C", "SN", "Su", 2).
staff_dweek_soft_lb("C", "LD", "Ho", 2).
staff_dweek_soft_lb("C", "SE", "Ho", 2).
staff_dweek_soft_lb("C", "SN", "Ho", 2).
staff_dweek_hard_ub("C", "LD", "Mo", 2).
staff_dweek_hard_ub("C", "SE", "Mo", 2).
staff_dweek_hard_ub("C", "SN", "Mo", 2).
staff_dweek_hard_ub("C", "LD", "Tu", 2).
staff_dweek_hard_ub("C", "SE", "Tu", 2).
staff_dweek_hard_ub("C", "SN", "Tu", 2).
staff_dweek_hard_ub("C", "LD", "We", 2).
staff_dweek_hard_ub("C", "SE", "We", 2).
staff_dweek_hard_ub("C", "SN", "We", 2).
staff_dweek_hard_ub("C", "LD", "Th", 2).
staff_dweek_hard_ub("C", "SE", "Th", 2).
staff_dweek_hard_ub("C", "SN", "Th", 2).
staff_dweek_hard_ub("C", "LD", "Fr", 2).
staff_dweek_hard_ub("C", "SE", "Fr", 2).
staff_dweek_hard_ub("C", "SN", "Fr", 2).
staff_dweek_hard_ub("C", "LD", "Sa", 2).
staff_dweek_hard_ub("C", "SE", "Sa", 2).
staff_dweek_hard_ub("C", "SN", "Sa", 2).
staff_dweek_hard_ub("C", "LD", "Su", 2).
staff_dweek_hard_ub("C", "SE", "Su", 2).
staff_dweek_hard_ub("C", "SN", "Su", 2).
staff_dweek_hard_ub("C", "LD", "Ho", 2).
staff_dweek_hard_ub("C", "SE", "Ho", 2).
staff_dweek_hard_ub("C", "SN", "Ho", 2).
staff_dweek_hard_lb("Newcomers", "SE", "Mo", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Mo", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Tu", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Tu", 0).
staff_dweek_hard_lb("Newcomers", "SE", "We", 0).
staff_dweek_hard_lb("Newcomers", "SN", "We", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Th", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Th", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Fr", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Fr", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Sa", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Sa", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Su", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Su", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Ho", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Ho", 0).
staff_dweek_soft_ub("Newcomers", "SE", "Mo", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Mo", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Tu", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Tu", 1).
staff_dweek_soft_ub("Newcomers", "SE", "We", 1).
staff_dweek_soft_ub("Newcomers", "SN", "We", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Th", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Th", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Fr", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Fr", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Sa", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Sa", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Su", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Su", 1).
staff_dweek_soft_ub("Newcomers", "SE", "Ho", 1).
staff_dweek_soft_ub("Newcomers", "SN", "Ho", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Mo", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Mo", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Tu", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Tu", 1).
staff_dweek_hard_ub("Newcomers", "SE", "We", 1).
staff_dweek_hard_ub("Newcomers", "SN", "We", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Th", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Th", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Fr", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Fr", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Sa", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Sa", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Su", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Su", 1).
staff_dweek_hard_ub("Newcomers", "SE", "Ho", 1).
staff_dweek_hard_ub("Newcomers", "SN", "Ho", 1).
staff_dweek_hard_lb("Night", "SE", "Mo", 4).
staff_dweek_hard_lb("Night", "SN", "Mo", 4).
staff_dweek_hard_lb("Night", "SE", "Tu", 4).
staff_dweek_hard_lb("Night", "SN", "Tu", 4).
staff_dweek_hard_lb("Night", "SE", "We", 4).
staff_dweek_hard_lb("Night", "SN", "We", 4).
staff_dweek_hard_lb("Night", "SE", "Th", 4).
staff_dweek_hard_lb("Night", "SN", "Th", 4).
staff_dweek_hard_lb("Night", "SE", "Fr", 4).
staff_dweek_hard_lb("Night", "SN", "Fr", 4).
staff_dweek_hard_lb("Night", "SE", "Sa", 4).
staff_dweek_hard_lb("Night", "SN", "Sa", 4).
staff_dweek_hard_lb("Night", "SE", "Su", 4).
staff_dweek_hard_lb("Night", "SN", "Su", 4).
staff_dweek_hard_lb("Night", "SE", "Ho", 4).
staff_dweek_hard_lb("Night", "SN", "Ho", 4).
staff_dweek_soft_lb("Night", "SE", "Mo", 4).
staff_dweek_soft_lb("Night", "SN", "Mo", 4).
staff_dweek_soft_lb("Night", "SE", "Tu", 4).
staff_dweek_soft_lb("Night", "SN", "Tu", 4).
staff_dweek_soft_lb("Night", "SE", "We", 4).
staff_dweek_soft_lb("Night", "SN", "We", 4).
staff_dweek_soft_lb("Night", "SE", "Th", 4).
staff_dweek_soft_lb("Night", "SN", "Th", 4).
staff_dweek_soft_lb("Night", "SE", "Fr", 4).
staff_dweek_soft_lb("Night", "SN", "Fr", 4).
staff_dweek_soft_lb("Night", "SE", "Sa", 4).
staff_dweek_soft_lb("Night", "SN", "Sa", 4).
staff_dweek_soft_lb("Night", "SE", "Su", 4).
staff_dweek_soft_lb("Night", "SN", "Su", 4).
staff_dweek_soft_lb("Night", "SE", "Ho", 4).
staff_dweek_soft_lb("Night", "SN", "Ho", 4).
staff_dweek_soft_ub("Night", "SE", "Mo", 4).
staff_dweek_soft_ub("Night", "SN", "Mo", 4).
staff_dweek_soft_ub("Night", "SE", "Tu", 4).
staff_dweek_soft_ub("Night", "SN", "Tu", 4).
staff_dweek_soft_ub("Night", "SE", "We", 4).
staff_dweek_soft_ub("Night", "SN", "We", 4).
staff_dweek_soft_ub("Night", "SE", "Th", 4).
staff_dweek_soft_ub("Night", "SN", "Th", 4).
staff_dweek_soft_ub("Night", "SE", "Fr", 4).
staff_dweek_soft_ub("Night", "SN", "Fr", 4).
staff_dweek_soft_ub("Night", "SE", "Sa", 4).
staff_dweek_soft_ub("Night", "SN", "Sa", 4).
staff_dweek_soft_ub("Night", "SE", "Su", 4).
staff_dweek_soft_ub("Night", "SN", "Su", 4).
staff_dweek_soft_ub("Night", "SE", "Ho", 4).
staff_dweek_soft_ub("Night", "SN", "Ho", 4).
staff_dweek_hard_ub("Night", "SE", "Mo", 5).
staff_dweek_hard_ub("Night", "SN", "Mo", 5).
staff_dweek_hard_ub("Night", "SE", "Tu", 5).
staff_dweek_hard_ub("Night", "SN", "Tu", 5).
staff_dweek_hard_ub("Night", "SE", "We", 5).
staff_dweek_hard_ub("Night", "SN", "We", 5).
staff_dweek_hard_ub("Night", "SE", "Th", 5).
staff_dweek_hard_ub("Night", "SN", "Th", 5).
staff_dweek_hard_ub("Night", "SE", "Fr", 5).
staff_dweek_hard_ub("Night", "SN", "Fr", 5).
staff_dweek_hard_ub("Night", "SE", "Sa", 5).
staff_dweek_hard_ub("Night", "SN", "Sa", 5).
staff_dweek_hard_ub("Night", "SE", "Su", 5).
staff_dweek_hard_ub("Night", "SN", "Su", 5).
staff_dweek_hard_ub("Night", "SE", "Ho", 5).
staff_dweek_hard_ub("Night", "SN", "Ho", 5).
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "EM", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("代行", "D", "Staffs").
vertical_constraint_type("平日代行", "D", "Staffs").
vertical_constraint_type("Leaders", "LD", "Staffs").
vertical_constraint_type("Leaders", "SE", "Staffs").
vertical_constraint_type("Leaders", "SN", "Staffs").
vertical_constraint_type("A", "LD", "Staffs").
vertical_constraint_type("A", "SE", "Staffs").
vertical_constraint_type("A", "SN", "Staffs").
vertical_constraint_type("B", "LD", "Staffs").
vertical_constraint_type("B", "SE", "Staffs").
vertical_constraint_type("B", "SN", "Staffs").
vertical_constraint_type("C", "LD", "Staffs").
vertical_constraint_type("C", "SE", "Staffs").
vertical_constraint_type("C", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
vertical_constraint_type("Night", "SE", "Staffs").
vertical_constraint_type("Night", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_hard_lb(1, "LD", 0).
shifts_hard_lb(1, "SE", 0).
shifts_hard_lb(1, "SN", 0).
shifts_soft_lb(1, "LD", 0).
shifts_soft_lb(1, "SE", 0).
shifts_soft_lb(1, "SN", 0).
shifts_soft_ub(1, "LD", 0).
shifts_soft_ub(1, "SE", 0).
shifts_soft_ub(1, "SN", 0).
shifts_hard_ub(1, "LD", 0).
shifts_hard_ub(1, "SE", 0).
shifts_hard_ub(1, "SN", 0).
shifts_hard_lb(1, "LM", 0).
shifts_soft_lb(1, "LM", 0).
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "LM", 0).
shifts_hard_lb(1, "EM", 0).
shifts_soft_lb(1, "EM", 0).
shifts_soft_ub(1, "EM", 0).
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_hard_lb(2, "LD", 1).
shifts_hard_lb(2, "SE", 1).
shifts_hard_lb(2, "SN", 1).
shifts_soft_lb(2, "LD", 2).
shifts_soft_lb(2, "SE", 2).
shifts_soft_lb(2, "SN", 2).
shifts_soft_ub(2, "LD", 5).
shifts_soft_ub(2, "SE", 5).
shifts_soft_ub(2, "SN", 5).
shifts_hard_ub(2, "LD", 5).
shifts_hard_ub(2, "SE", 5).
shifts_hard_ub(2, "SN", 5).
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "LM", 0).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_hard_lb(3, "LD", 1).
shifts_hard_lb(3, "SE", 1).
shifts_hard_lb(3, "SN", 1).
shifts_soft_lb(3, "LD", 3).
shifts_soft_lb(3, "SE", 3).
shifts_soft_lb(3, "SN", 3).
shifts_soft_ub(3, "LD", 4).
shifts_soft_ub(3, "SE", 4).
shifts_soft_ub(3, "SN", 4).
shifts_hard_ub(3, "LD", 5).
shifts_hard_ub(3, "SE", 5).
shifts_hard_ub(3, "SN", 5).
shifts_hard_lb(3, "LM", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "LM", 0).
shifts_hard_ub(3, "LM", 1).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_hard_lb(4, "LD", 1).
shifts_hard_lb(4, "SE", 1).
shifts_hard_lb(4, "SN", 1).
shifts_soft_lb(4, "LD", 3).
shifts_soft_lb(4, "SE", 3).
shifts_soft_lb(4, "SN", 3).
shifts_soft_ub(4, "LD", 4).
shifts_soft_ub(4, "SE", 4).
shifts_soft_ub(4, "SN", 4).
shifts_hard_ub(4, "LD", 5).
shifts_hard_ub(4, "SE", 5).
shifts_hard_ub(4, "SN", 5).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "LM", 1).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "LD", 1).
shifts_hard_lb(5, "SE", 1).
shifts_hard_lb(5, "SN", 1).
shifts_soft_lb(5, "LD", 3).
shifts_soft_lb(5, "SE", 3).
shifts_soft_lb(5, "SN", 3).
shifts_soft_ub(5, "LD", 4).
shifts_soft_ub(5, "SE", 4).
shifts_soft_ub(5, "SN", 4).
shifts_hard_ub(5, "LD", 5).
shifts_hard_ub(5, "SE", 5).
shifts_hard_ub(5, "SN", 5).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 1).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_hard_lb(6, "LD", 1).
shifts_hard_lb(6, "SE", 1).
shifts_hard_lb(6, "SN", 1).
shifts_soft_lb(6, "LD", 3).
shifts_soft_lb(6, "SE", 3).
shifts_soft_lb(6, "SN", 3).
shifts_soft_ub(6, "LD", 4).
shifts_soft_ub(6, "SE", 4).
shifts_soft_ub(6, "SN", 4).
shifts_hard_ub(6, "LD", 5).
shifts_hard_ub(6, "SE", 5).
shifts_hard_ub(6, "SN", 5).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "LM", 1).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "LD", 1).
shifts_hard_lb(7, "SE", 1).
shifts_hard_lb(7, "SN", 1).
shifts_soft_lb(7, "LD", 3).
shifts_soft_lb(7, "SE", 3).
shifts_soft_lb(7, "SN", 3).
shifts_soft_ub(7, "LD", 4).
shifts_soft_ub(7, "SE", 4).
shifts_soft_ub(7, "SN", 4).
shifts_hard_ub(7, "LD", 5).
shifts_hard_ub(7, "SE", 5).
shifts_hard_ub(7, "SN", 5).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "LM", 1).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "LD", 1).
shifts_hard_lb(8, "SE", 1).
shifts_hard_lb(8, "SN", 1).
shifts_soft_lb(8, "LD", 2).
shifts_soft_lb(8, "SE", 2).
shifts_soft_lb(8, "SN", 2).
shifts_soft_ub(8, "LD", 3).
shifts_soft_ub(8, "SE", 3).
shifts_soft_ub(8, "SN", 3).
shifts_hard_ub(8, "LD", 4).
shifts_hard_ub(8, "SE", 4).
shifts_hard_ub(8, "SN", 4).
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
shifts_soft_lb(9, "LD", 2).
shifts_soft_lb(9, "SE", 2).
shifts_soft_lb(9, "SN", 2).
shifts_soft_ub(9, "LD", 3).
shifts_soft_ub(9, "SE", 3).
shifts_soft_ub(9, "SN", 3).
shifts_hard_ub(9, "LD", 4).
shifts_hard_ub(9, "SE", 4).
shifts_hard_ub(9, "SN", 4).
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
shifts_soft_lb(10, "LD", 3).
shifts_soft_lb(10, "SE", 3).
shifts_soft_lb(10, "SN", 3).
shifts_soft_ub(10, "LD", 4).
shifts_soft_ub(10, "SE", 4).
shifts_soft_ub(10, "SN", 4).
shifts_hard_ub(10, "LD", 5).
shifts_hard_ub(10, "SE", 5).
shifts_hard_ub(10, "SN", 5).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "LM", 1).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_hard_lb(11, "LD", 1).
shifts_hard_lb(11, "SE", 1).
shifts_hard_lb(11, "SN", 1).
shifts_soft_lb(11, "LD", 3).
shifts_soft_lb(11, "SE", 3).
shifts_soft_lb(11, "SN", 3).
shifts_soft_ub(11, "LD", 4).
shifts_soft_ub(11, "SE", 4).
shifts_soft_ub(11, "SN", 4).
shifts_hard_ub(11, "LD", 5).
shifts_hard_ub(11, "SE", 5).
shifts_hard_ub(11, "SN", 5).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "LM", 1).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_hard_lb(12, "LD", 1).
shifts_hard_lb(12, "SE", 1).
shifts_hard_lb(12, "SN", 1).
shifts_soft_lb(12, "LD", 1).
shifts_soft_lb(12, "SE", 1).
shifts_soft_lb(12, "SN", 1).
shifts_soft_ub(12, "LD", 2).
shifts_soft_ub(12, "SE", 2).
shifts_soft_ub(12, "SN", 2).
shifts_hard_ub(12, "LD", 2).
shifts_hard_ub(12, "SE", 2).
shifts_hard_ub(12, "SN", 2).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_hard_lb(13, "LD", 1).
shifts_hard_lb(13, "SE", 1).
shifts_hard_lb(13, "SN", 1).
shifts_soft_lb(13, "LD", 1).
shifts_soft_lb(13, "SE", 1).
shifts_soft_lb(13, "SN", 1).
shifts_soft_ub(13, "LD", 2).
shifts_soft_ub(13, "SE", 2).
shifts_soft_ub(13, "SN", 2).
shifts_hard_ub(13, "LD", 2).
shifts_hard_ub(13, "SE", 2).
shifts_hard_ub(13, "SN", 2).
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
shifts_soft_lb(14, "LD", 3).
shifts_soft_lb(14, "SE", 3).
shifts_soft_lb(14, "SN", 3).
shifts_soft_ub(14, "LD", 4).
shifts_soft_ub(14, "SE", 4).
shifts_soft_ub(14, "SN", 4).
shifts_hard_ub(14, "LD", 5).
shifts_hard_ub(14, "SE", 5).
shifts_hard_ub(14, "SN", 5).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "LM", 1).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "LD", 0).
shifts_hard_lb(15, "SE", 0).
shifts_hard_lb(15, "SN", 0).
shifts_soft_lb(15, "LD", 0).
shifts_soft_lb(15, "SE", 0).
shifts_soft_lb(15, "SN", 0).
shifts_soft_ub(15, "LD", 0).
shifts_soft_ub(15, "SE", 0).
shifts_soft_ub(15, "SN", 0).
shifts_hard_ub(15, "LD", 0).
shifts_hard_ub(15, "SE", 0).
shifts_hard_ub(15, "SN", 0).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_hard_lb(16, "LD", 1).
shifts_hard_lb(16, "SE", 1).
shifts_hard_lb(16, "SN", 1).
shifts_soft_lb(16, "LD", 3).
shifts_soft_lb(16, "SE", 3).
shifts_soft_lb(16, "SN", 3).
shifts_soft_ub(16, "LD", 4).
shifts_soft_ub(16, "SE", 4).
shifts_soft_ub(16, "SN", 4).
shifts_hard_ub(16, "LD", 5).
shifts_hard_ub(16, "SE", 5).
shifts_hard_ub(16, "SN", 5).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "LM", 1).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_hard_lb(17, "LD", 1).
shifts_hard_lb(17, "SE", 1).
shifts_hard_lb(17, "SN", 1).
shifts_soft_lb(17, "LD", 3).
shifts_soft_lb(17, "SE", 3).
shifts_soft_lb(17, "SN", 3).
shifts_soft_ub(17, "LD", 4).
shifts_soft_ub(17, "SE", 4).
shifts_soft_ub(17, "SN", 4).
shifts_hard_ub(17, "LD", 5).
shifts_hard_ub(17, "SE", 5).
shifts_hard_ub(17, "SN", 5).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "LM", 1).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_hard_lb(18, "LD", 1).
shifts_hard_lb(18, "SE", 1).
shifts_hard_lb(18, "SN", 1).
shifts_soft_lb(18, "LD", 3).
shifts_soft_lb(18, "SE", 3).
shifts_soft_lb(18, "SN", 3).
shifts_soft_ub(18, "LD", 4).
shifts_soft_ub(18, "SE", 4).
shifts_soft_ub(18, "SN", 4).
shifts_hard_ub(18, "LD", 5).
shifts_hard_ub(18, "SE", 5).
shifts_hard_ub(18, "SN", 5).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "LM", 1).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_hard_lb(19, "LD", 1).
shifts_hard_lb(19, "SE", 1).
shifts_hard_lb(19, "SN", 1).
shifts_soft_lb(19, "LD", 3).
shifts_soft_lb(19, "SE", 3).
shifts_soft_lb(19, "SN", 3).
shifts_soft_ub(19, "LD", 4).
shifts_soft_ub(19, "SE", 4).
shifts_soft_ub(19, "SN", 4).
shifts_hard_ub(19, "LD", 5).
shifts_hard_ub(19, "SE", 5).
shifts_hard_ub(19, "SN", 5).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "LM", 1).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "LD", 1).
shifts_hard_lb(20, "SE", 1).
shifts_hard_lb(20, "SN", 1).
shifts_soft_lb(20, "LD", 2).
shifts_soft_lb(20, "SE", 2).
shifts_soft_lb(20, "SN", 2).
shifts_soft_ub(20, "LD", 3).
shifts_soft_ub(20, "SE", 3).
shifts_soft_ub(20, "SN", 3).
shifts_hard_ub(20, "LD", 4).
shifts_hard_ub(20, "SE", 4).
shifts_hard_ub(20, "SN", 4).
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
shifts_soft_lb(21, "LD", 2).
shifts_soft_lb(21, "SE", 2).
shifts_soft_lb(21, "SN", 2).
shifts_soft_ub(21, "LD", 3).
shifts_soft_ub(21, "SE", 3).
shifts_soft_ub(21, "SN", 3).
shifts_hard_ub(21, "LD", 4).
shifts_hard_ub(21, "SE", 4).
shifts_hard_ub(21, "SN", 4).
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "LM", 0).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_hard_lb(22, "LD", 0).
shifts_hard_lb(22, "SE", 0).
shifts_hard_lb(22, "SN", 0).
shifts_soft_lb(22, "LD", 0).
shifts_soft_lb(22, "SE", 0).
shifts_soft_lb(22, "SN", 0).
shifts_soft_ub(22, "LD", 0).
shifts_soft_ub(22, "SE", 0).
shifts_soft_ub(22, "SN", 0).
shifts_hard_ub(22, "LD", 0).
shifts_hard_ub(22, "SE", 0).
shifts_hard_ub(22, "SN", 0).
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
shifts_soft_lb(23, "LD", 2).
shifts_soft_lb(23, "SE", 2).
shifts_soft_lb(23, "SN", 2).
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_hard_ub(23, "LD", 5).
shifts_hard_ub(23, "SE", 5).
shifts_hard_ub(23, "SN", 5).
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
shifts_soft_lb(24, "LD", 3).
shifts_soft_lb(24, "SE", 3).
shifts_soft_lb(24, "SN", 3).
shifts_soft_ub(24, "LD", 4).
shifts_soft_ub(24, "SE", 4).
shifts_soft_ub(24, "SN", 4).
shifts_hard_ub(24, "LD", 5).
shifts_hard_ub(24, "SE", 5).
shifts_hard_ub(24, "SN", 5).
shifts_hard_lb(24, "LM", 0).
shifts_soft_lb(24, "LM", 0).
shifts_soft_ub(24, "LM", 0).
shifts_hard_ub(24, "LM", 1).
shifts_hard_lb(24, "EM", 0).
shifts_soft_lb(24, "EM", 0).
shifts_soft_ub(24, "EM", 0).
shifts_hard_ub(24, "EM", 0).
% Staff 25:
shifts_hard_lb(25, "LD", 1).
shifts_hard_lb(25, "SE", 1).
shifts_hard_lb(25, "SN", 1).
shifts_soft_lb(25, "LD", 3).
shifts_soft_lb(25, "SE", 3).
shifts_soft_lb(25, "SN", 3).
shifts_soft_ub(25, "LD", 4).
shifts_soft_ub(25, "SE", 4).
shifts_soft_ub(25, "SN", 4).
shifts_hard_ub(25, "LD", 5).
shifts_hard_ub(25, "SE", 5).
shifts_hard_ub(25, "SN", 5).
shifts_hard_lb(25, "LM", 0).
shifts_soft_lb(25, "LM", 0).
shifts_soft_ub(25, "LM", 0).
shifts_hard_ub(25, "LM", 1).
shifts_hard_lb(25, "EM", 0).
shifts_soft_lb(25, "EM", 0).
shifts_soft_ub(25, "EM", 0).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_hard_lb(26, "LD", 1).
shifts_hard_lb(26, "SE", 1).
shifts_hard_lb(26, "SN", 1).
shifts_soft_lb(26, "LD", 3).
shifts_soft_lb(26, "SE", 3).
shifts_soft_lb(26, "SN", 3).
shifts_soft_ub(26, "LD", 4).
shifts_soft_ub(26, "SE", 4).
shifts_soft_ub(26, "SN", 4).
shifts_hard_ub(26, "LD", 5).
shifts_hard_ub(26, "SE", 5).
shifts_hard_ub(26, "SN", 5).
shifts_hard_lb(26, "LM", 0).
shifts_soft_lb(26, "LM", 0).
shifts_soft_ub(26, "LM", 0).
shifts_hard_ub(26, "LM", 1).
shifts_hard_lb(26, "EM", 0).
shifts_soft_lb(26, "EM", 0).
shifts_soft_ub(26, "EM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_hard_lb(27, "LD", 0).
shifts_hard_lb(27, "SE", 0).
shifts_hard_lb(27, "SN", 0).
shifts_soft_lb(27, "LD", 0).
shifts_soft_lb(27, "SE", 0).
shifts_soft_lb(27, "SN", 0).
shifts_soft_ub(27, "LD", 0).
shifts_soft_ub(27, "SE", 0).
shifts_soft_ub(27, "SN", 0).
shifts_hard_ub(27, "LD", 0).
shifts_hard_ub(27, "SE", 0).
shifts_hard_ub(27, "SN", 0).
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
shifts_hard_ub(28, "LM", 1).
shifts_hard_lb(28, "EM", 0).
shifts_soft_lb(28, "EM", 0).
shifts_soft_ub(28, "EM", 0).
shifts_hard_ub(28, "EM", 0).
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
shifts_hard_ub(29, "LD", 4).
shifts_hard_ub(29, "SE", 4).
shifts_hard_ub(29, "SN", 4).
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
shifts_hard_ub(30, "LM", 1).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
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
shifts_hard_ub(31, "LD", 4).
shifts_hard_ub(31, "SE", 4).
shifts_hard_ub(31, "SN", 4).
shifts_hard_lb(31, "LM", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "LM", 1).
shifts_hard_lb(31, "EM", 0).
shifts_soft_lb(31, "EM", 0).
shifts_soft_ub(31, "EM", 0).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 1).
shifts_hard_lb(32, "SE", 1).
shifts_hard_lb(32, "SN", 1).
shifts_soft_lb(32, "LD", 3).
shifts_soft_lb(32, "SE", 3).
shifts_soft_lb(32, "SN", 3).
shifts_soft_ub(32, "LD", 4).
shifts_soft_ub(32, "SE", 4).
shifts_soft_ub(32, "SN", 4).
shifts_hard_ub(32, "LD", 5).
shifts_hard_ub(32, "SE", 5).
shifts_hard_ub(32, "SN", 5).
shifts_hard_lb(32, "LM", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "LM", 1).
shifts_hard_lb(32, "EM", 0).
shifts_soft_lb(32, "EM", 0).
shifts_soft_ub(32, "EM", 0).
shifts_hard_ub(32, "EM", 0).
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
shifts_hard_ub(33, "LD", 4).
shifts_hard_ub(33, "SE", 4).
shifts_hard_ub(33, "SN", 4).
shifts_hard_lb(33, "LM", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "LM", 0).
shifts_hard_lb(33, "EM", 0).
shifts_soft_lb(33, "EM", 0).
shifts_soft_ub(33, "EM", 0).
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
% Staff 1:
neg_shift(1, "日", ("日";"N";"P";"J";"S")).
neg_shift(1, "土", ("日";"N";"P";"J";"S")).
neg_shift(1, "祝", ("日";"N";"P";"J";"S")).
% Staff 22:
neg_shift(22, "日", ("日";"N";"P";"J";"S")).
neg_shift(22, "土", ("日";"N";"P";"J";"S")).
neg_shift(22, "祝", ("日";"N";"P";"J";"S")).
