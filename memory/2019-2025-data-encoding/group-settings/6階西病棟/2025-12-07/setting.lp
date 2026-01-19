% Staffs ---------------------------------------------------
staff(1, "Brian Sims", "Chief Nurse", "42230", 0).
staff(2, "Natalie Arnold", "Deputy Chief Nurse", "96922", 0).
staff(3, "Maria Green", "Nurse", "01408", 0).
staff(4, "Robert Hernandez", "Nurse", "39976", 0).
staff(5, "Jose Austin", "Nurse", "28679", 0).
staff(6, "Francisco Matthews", "Nurse", "84921", 0).
staff(7, "Taylor Mueller", "Nurse", "43181", 0).
staff(8, "Lori Martin MD", "Nurse", "24451", 0).
staff(9, "Cindy Jacobs", "Nurse", "20359", 0).
staff(10, "Jacqueline Kent", "Nurse", "27517", 0).
staff(11, "Steve Kaiser", "Nurse", "03379", 0).
staff(12, "Henry Roberts", "Nurse", "A934390", 0).
staff(13, "Michelle Morrow", "Deputy Chief Nurse", "29348", 0).
staff(14, "Bradley Morris", "Nurse", "60418", 0).
staff(15, "Shawn Horn", "Nurse", "64568", 0).
staff(16, "Bethany Carson", "Nurse", "80089", 0).
staff(17, "Ashley Webb", "Nurse", "03496", 0).
staff(18, "Crystal Cervantes", "Nurse", "82417", 0).
staff(19, "Randall Cross", "Nurse", "08318", 0).
staff(20, "Kelly Chase", "Nurse", "24465", 0).
staff(21, "Justin Villegas", "Nurse", "00140", 0).
staff(22, "Justin Fields", "Nurse", "00274", 0).
staff(23, "James Ramirez", "Nurse", "A839378", 0).
staff(24, "Michael Ramirez", "Nurse", "A103806", 0).
staff(25, "Kimberly Garcia", "Deputy Chief Nurse", "72886", 0).
staff(26, "Charles James", "Nurse", "72807", 0).
staff(27, "Amy Thompson", "Nurse", "49079", 0).
staff(28, "Jason Garcia", "Nurse", "59373", 0).
staff(29, "Garrett Brown", "Nurse", "91276", 0).
staff(30, "Gabriela Parker", "Nurse", "29761", 0).
staff(31, "Sara Browning", "Nurse", "16221", 0).
staff(32, "Alexander Murphy", "Nurse", "44418", 0).
staff(33, "Richard Hoffman", "Nurse", "81790", 0).
staff(34, "Samuel Turner", "Nurse", "A977812", 0).
staff(35, "Aaron Ortiz", "Nurse", "A258679", 0).
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
group("All", 27).
group("All", 28).
group("All", 29).
group("All", 30).
group("All", 31).
group("All", 32).
group("All", 33).
group("All", 34).
group("All", 35).
group("代行").
group("代行", 2).
group("代行", 13).
group("代行", 25).
group("平日代行").
group("平日代行", 2).
group("平日代行", 13).
group("平日代行", 25).
group("Leaders").
group("Leaders", 2).
group("Leaders", 3).
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 6).
group("Leaders", 13).
group("Leaders", 14).
group("Leaders", 15).
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
group("B").
group("B", 13).
group("B", 14).
group("B", 15).
group("B", 16).
group("B", 17).
group("B", 18).
group("B", 19).
group("B", 20).
group("B", 21).
group("B", 22).
group("B", 23).
group("B", 24).
group("C").
group("C", 25).
group("C", 26).
group("C", 27).
group("C", 28).
group("C", 29).
group("C", 30).
group("C", 31).
group("C", 32).
group("C", 33).
group("C", 34).
group("C", 35).
group("Newcomers").
group("Newcomers", 12).
group("Newcomers", 23).
group("Newcomers", 24).
group("Newcomers", 34).
group("Newcomers", 35).
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
group("Night", 15).
group("Night", 16).
group("Night", 17).
group("Night", 18).
group("Night", 19).
group("Night", 20).
group("Night", 21).
group("Night", 22).
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
group("Night", 35).
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
staff_dweek_hard_lb("All", "D", "Weekday", 8).
staff_dweek_hard_lb("All", "D", "Weekend", 4).
staff_dweek_hard_lb("All", "D", "Holiday", 4).
staff_dweek_soft_lb("All", "D", "Weekday", 9).
staff_dweek_soft_lb("All", "D", "Weekend", 4).
staff_dweek_soft_lb("All", "D", "Holiday", 4).
shifts("LD").
shifts("LD", "LD").
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
shifts("EM").
shifts("EM", "EM").
staff_dweek_hard_lb("All", "EM", "Weekday", 0).
staff_dweek_hard_lb("All", "EM", "Weekend", 0).
staff_dweek_hard_lb("All", "EM", "Holiday", 0).
staff_dweek_soft_lb("All", "EM", "Weekday", 0).
staff_dweek_soft_lb("All", "EM", "Weekend", 0).
staff_dweek_soft_lb("All", "EM", "Holiday", 0).
staff_dweek_soft_ub("All", "EM", "Weekday", 0).
staff_dweek_soft_ub("All", "EM", "Weekend", 0).
staff_dweek_soft_ub("All", "EM", "Holiday", 0).
staff_dweek_hard_ub("All", "EM", "Weekday", 0).
staff_dweek_hard_ub("All", "EM", "Weekend", 0).
staff_dweek_hard_ub("All", "EM", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
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
staff_dweek_soft_lb("代行", "D", "Weekday", 0).
staff_dweek_soft_lb("代行", "D", "Weekend", 1).
staff_dweek_soft_lb("代行", "D", "Holiday", 1).
staff_dweek_soft_ub("代行", "D", "Weekday", 0).
staff_dweek_soft_ub("代行", "D", "Weekend", 1).
staff_dweek_soft_ub("代行", "D", "Holiday", 1).
staff_dweek_hard_lb("平日代行", "D", "Weekday", 0).
staff_dweek_hard_lb("平日代行", "D", "Weekend", 0).
staff_dweek_hard_lb("平日代行", "D", "Holiday", 0).
staff_dweek_soft_lb("平日代行", "D", "Weekday", 1).
staff_dweek_soft_lb("平日代行", "D", "Weekend", 0).
staff_dweek_soft_lb("平日代行", "D", "Holiday", 0).
staff_dweek_soft_ub("平日代行", "D", "Weekday", 2).
staff_dweek_soft_ub("平日代行", "D", "Weekend", 0).
staff_dweek_soft_ub("平日代行", "D", "Holiday", 0).
staff_dweek_hard_lb("Leaders", "LD", "Weekday", 1).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("Leaders", "SE", "Weekday", 1).
shifts("SN").
shifts("SN", "SN").
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
staff_dweek_hard_lb("A", "LD", "Weekday", 1).
staff_dweek_hard_lb("A", "SE", "Weekday", 1).
staff_dweek_hard_lb("A", "SN", "Weekday", 1).
staff_dweek_hard_lb("A", "LD", "Weekend", 1).
staff_dweek_hard_lb("A", "SE", "Weekend", 1).
staff_dweek_hard_lb("A", "SN", "Weekend", 1).
staff_dweek_hard_lb("A", "LD", "Holiday", 1).
staff_dweek_hard_lb("A", "SE", "Holiday", 1).
staff_dweek_hard_lb("A", "SN", "Holiday", 1).
staff_dweek_soft_lb("A", "LD", "Weekday", 2).
staff_dweek_soft_lb("A", "SE", "Weekday", 2).
staff_dweek_soft_lb("A", "SN", "Weekday", 2).
staff_dweek_soft_lb("A", "LD", "Weekend", 2).
staff_dweek_soft_lb("A", "SE", "Weekend", 2).
staff_dweek_soft_lb("A", "SN", "Weekend", 2).
staff_dweek_soft_lb("A", "LD", "Holiday", 2).
staff_dweek_soft_lb("A", "SE", "Holiday", 2).
staff_dweek_soft_lb("A", "SN", "Holiday", 2).
staff_dweek_hard_ub("A", "LD", "Weekday", 2).
staff_dweek_hard_ub("A", "SE", "Weekday", 2).
staff_dweek_hard_ub("A", "SN", "Weekday", 2).
staff_dweek_hard_ub("A", "LD", "Weekend", 2).
staff_dweek_hard_ub("A", "SE", "Weekend", 2).
staff_dweek_hard_ub("A", "SN", "Weekend", 2).
staff_dweek_hard_ub("A", "LD", "Holiday", 2).
staff_dweek_hard_ub("A", "SE", "Holiday", 2).
staff_dweek_hard_ub("A", "SN", "Holiday", 2).
staff_dweek_hard_lb("B", "LD", "Weekday", 1).
staff_dweek_hard_lb("B", "SE", "Weekday", 1).
staff_dweek_hard_lb("B", "SN", "Weekday", 1).
staff_dweek_hard_lb("B", "LD", "Weekend", 1).
staff_dweek_hard_lb("B", "SE", "Weekend", 1).
staff_dweek_hard_lb("B", "SN", "Weekend", 1).
staff_dweek_hard_lb("B", "LD", "Holiday", 1).
staff_dweek_hard_lb("B", "SE", "Holiday", 1).
staff_dweek_hard_lb("B", "SN", "Holiday", 1).
staff_dweek_soft_lb("B", "LD", "Weekday", 2).
staff_dweek_soft_lb("B", "SE", "Weekday", 2).
staff_dweek_soft_lb("B", "SN", "Weekday", 2).
staff_dweek_soft_lb("B", "LD", "Weekend", 2).
staff_dweek_soft_lb("B", "SE", "Weekend", 2).
staff_dweek_soft_lb("B", "SN", "Weekend", 2).
staff_dweek_soft_lb("B", "LD", "Holiday", 2).
staff_dweek_soft_lb("B", "SE", "Holiday", 2).
staff_dweek_soft_lb("B", "SN", "Holiday", 2).
staff_dweek_hard_ub("B", "LD", "Weekday", 2).
staff_dweek_hard_ub("B", "SE", "Weekday", 2).
staff_dweek_hard_ub("B", "SN", "Weekday", 2).
staff_dweek_hard_ub("B", "LD", "Weekend", 2).
staff_dweek_hard_ub("B", "SE", "Weekend", 2).
staff_dweek_hard_ub("B", "SN", "Weekend", 2).
staff_dweek_hard_ub("B", "LD", "Holiday", 2).
staff_dweek_hard_ub("B", "SE", "Holiday", 2).
staff_dweek_hard_ub("B", "SN", "Holiday", 2).
staff_dweek_hard_lb("C", "LD", "Weekday", 1).
staff_dweek_hard_lb("C", "SE", "Weekday", 1).
staff_dweek_hard_lb("C", "SN", "Weekday", 1).
staff_dweek_hard_lb("C", "LD", "Weekend", 1).
staff_dweek_hard_lb("C", "SE", "Weekend", 1).
staff_dweek_hard_lb("C", "SN", "Weekend", 1).
staff_dweek_hard_lb("C", "LD", "Holiday", 1).
staff_dweek_hard_lb("C", "SE", "Holiday", 1).
staff_dweek_hard_lb("C", "SN", "Holiday", 1).
staff_dweek_soft_lb("C", "LD", "Weekday", 2).
staff_dweek_soft_lb("C", "SE", "Weekday", 2).
staff_dweek_soft_lb("C", "SN", "Weekday", 2).
staff_dweek_soft_lb("C", "LD", "Weekend", 2).
staff_dweek_soft_lb("C", "SE", "Weekend", 2).
staff_dweek_soft_lb("C", "SN", "Weekend", 2).
staff_dweek_soft_lb("C", "LD", "Holiday", 2).
staff_dweek_soft_lb("C", "SE", "Holiday", 2).
staff_dweek_soft_lb("C", "SN", "Holiday", 2).
staff_dweek_hard_ub("C", "LD", "Weekday", 2).
staff_dweek_hard_ub("C", "SE", "Weekday", 2).
staff_dweek_hard_ub("C", "SN", "Weekday", 2).
staff_dweek_hard_ub("C", "LD", "Weekend", 2).
staff_dweek_hard_ub("C", "SE", "Weekend", 2).
staff_dweek_hard_ub("C", "SN", "Weekend", 2).
staff_dweek_hard_ub("C", "LD", "Holiday", 2).
staff_dweek_hard_ub("C", "SE", "Holiday", 2).
staff_dweek_hard_ub("C", "SN", "Holiday", 2).
staff_dweek_hard_lb("Newcomers", "SE", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekday", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Weekend", 0).
staff_dweek_hard_lb("Newcomers", "SE", "Holiday", 0).
staff_dweek_hard_lb("Newcomers", "SN", "Holiday", 0).
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
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "LM", 0).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_hard_lb(3, "LD", 2).
shifts_hard_lb(3, "SE", 2).
shifts_hard_lb(3, "SN", 2).
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
shifts_hard_ub(3, "LM", 0).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_hard_lb(4, "LD", 3).
shifts_hard_lb(4, "SE", 3).
shifts_hard_lb(4, "SN", 3).
shifts_soft_lb(4, "LD", 4).
shifts_soft_lb(4, "SE", 4).
shifts_soft_lb(4, "SN", 4).
shifts_soft_ub(4, "LD", 5).
shifts_soft_ub(4, "SE", 5).
shifts_soft_ub(4, "SN", 5).
shifts_hard_ub(4, "LD", 6).
shifts_hard_ub(4, "SE", 6).
shifts_hard_ub(4, "SN", 6).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "LM", 1).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "LD", 3).
shifts_hard_lb(5, "SE", 3).
shifts_hard_lb(5, "SN", 3).
shifts_soft_lb(5, "LD", 4).
shifts_soft_lb(5, "SE", 4).
shifts_soft_lb(5, "SN", 4).
shifts_soft_ub(5, "LD", 5).
shifts_soft_ub(5, "SE", 5).
shifts_soft_ub(5, "SN", 5).
shifts_hard_ub(5, "LD", 6).
shifts_hard_ub(5, "SE", 6).
shifts_hard_ub(5, "SN", 6).
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "LM", 1).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_hard_lb(6, "LD", 3).
shifts_hard_lb(6, "SE", 3).
shifts_hard_lb(6, "SN", 3).
shifts_soft_lb(6, "LD", 4).
shifts_soft_lb(6, "SE", 4).
shifts_soft_lb(6, "SN", 4).
shifts_soft_ub(6, "LD", 5).
shifts_soft_ub(6, "SE", 5).
shifts_soft_ub(6, "SN", 5).
shifts_hard_ub(6, "LD", 6).
shifts_hard_ub(6, "SE", 6).
shifts_hard_ub(6, "SN", 6).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "LM", 1).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "LD", 3).
shifts_hard_lb(7, "SE", 3).
shifts_hard_lb(7, "SN", 3).
shifts_soft_lb(7, "LD", 4).
shifts_soft_lb(7, "SE", 4).
shifts_soft_lb(7, "SN", 4).
shifts_soft_ub(7, "LD", 5).
shifts_soft_ub(7, "SE", 5).
shifts_soft_ub(7, "SN", 5).
shifts_hard_ub(7, "LD", 6).
shifts_hard_ub(7, "SE", 6).
shifts_hard_ub(7, "SN", 6).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "LM", 1).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "LD", 3).
shifts_hard_lb(8, "SE", 3).
shifts_hard_lb(8, "SN", 3).
shifts_soft_lb(8, "LD", 4).
shifts_soft_lb(8, "SE", 4).
shifts_soft_lb(8, "SN", 4).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_hard_lb(8, "LM", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "LM", 1).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_hard_lb(9, "LD", 3).
shifts_hard_lb(9, "SE", 3).
shifts_hard_lb(9, "SN", 3).
shifts_soft_lb(9, "LD", 4).
shifts_soft_lb(9, "SE", 4).
shifts_soft_lb(9, "SN", 4).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 6).
shifts_hard_ub(9, "SE", 6).
shifts_hard_ub(9, "SN", 6).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "LM", 1).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_hard_lb(10, "LD", 3).
shifts_hard_lb(10, "SE", 3).
shifts_hard_lb(10, "SN", 3).
shifts_soft_lb(10, "LD", 4).
shifts_soft_lb(10, "SE", 4).
shifts_soft_lb(10, "SN", 4).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "LM", 1).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_hard_lb(11, "LD", 3).
shifts_hard_lb(11, "SE", 3).
shifts_hard_lb(11, "SN", 3).
shifts_soft_lb(11, "LD", 4).
shifts_soft_lb(11, "SE", 4).
shifts_soft_lb(11, "SN", 4).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "LD", 6).
shifts_hard_ub(11, "SE", 6).
shifts_hard_ub(11, "SN", 6).
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
shifts_soft_lb(12, "LD", 2).
shifts_soft_lb(12, "SE", 2).
shifts_soft_lb(12, "SN", 2).
shifts_soft_ub(12, "LD", 3).
shifts_soft_ub(12, "SE", 3).
shifts_soft_ub(12, "SN", 3).
shifts_hard_ub(12, "LD", 3).
shifts_hard_ub(12, "SE", 3).
shifts_hard_ub(12, "SN", 3).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "LM", 1).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_hard_lb(13, "LD", 1).
shifts_hard_lb(13, "SE", 1).
shifts_hard_lb(13, "SN", 1).
shifts_soft_lb(13, "LD", 2).
shifts_soft_lb(13, "SE", 2).
shifts_soft_lb(13, "SN", 2).
shifts_soft_ub(13, "LD", 2).
shifts_soft_ub(13, "SE", 2).
shifts_soft_ub(13, "SN", 2).
shifts_hard_ub(13, "LD", 3).
shifts_hard_ub(13, "SE", 3).
shifts_hard_ub(13, "SN", 3).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "LM", 1).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_hard_lb(14, "LD", 3).
shifts_hard_lb(14, "SE", 3).
shifts_hard_lb(14, "SN", 3).
shifts_soft_lb(14, "LD", 4).
shifts_soft_lb(14, "SE", 4).
shifts_soft_lb(14, "SN", 4).
shifts_soft_ub(14, "LD", 5).
shifts_soft_ub(14, "SE", 5).
shifts_soft_ub(14, "SN", 5).
shifts_hard_ub(14, "LD", 6).
shifts_hard_ub(14, "SE", 6).
shifts_hard_ub(14, "SN", 6).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "LM", 1).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "LD", 3).
shifts_hard_lb(15, "SE", 3).
shifts_hard_lb(15, "SN", 3).
shifts_soft_lb(15, "LD", 4).
shifts_soft_lb(15, "SE", 4).
shifts_soft_lb(15, "SN", 4).
shifts_soft_ub(15, "LD", 5).
shifts_soft_ub(15, "SE", 5).
shifts_soft_ub(15, "SN", 5).
shifts_hard_ub(15, "LD", 6).
shifts_hard_ub(15, "SE", 6).
shifts_hard_ub(15, "SN", 6).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 1).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_hard_lb(16, "LD", 3).
shifts_hard_lb(16, "SE", 3).
shifts_hard_lb(16, "SN", 3).
shifts_soft_lb(16, "LD", 4).
shifts_soft_lb(16, "SE", 4).
shifts_soft_lb(16, "SN", 4).
shifts_soft_ub(16, "LD", 5).
shifts_soft_ub(16, "SE", 5).
shifts_soft_ub(16, "SN", 5).
shifts_hard_ub(16, "LD", 6).
shifts_hard_ub(16, "SE", 6).
shifts_hard_ub(16, "SN", 6).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "LM", 1).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_hard_lb(17, "LD", 3).
shifts_hard_lb(17, "SE", 3).
shifts_hard_lb(17, "SN", 3).
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 5).
shifts_soft_ub(17, "SE", 5).
shifts_soft_ub(17, "SN", 5).
shifts_hard_ub(17, "LD", 6).
shifts_hard_ub(17, "SE", 6).
shifts_hard_ub(17, "SN", 6).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "LM", 1).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_hard_lb(18, "LD", 3).
shifts_hard_lb(18, "SE", 3).
shifts_hard_lb(18, "SN", 3).
shifts_soft_lb(18, "LD", 4).
shifts_soft_lb(18, "SE", 4).
shifts_soft_lb(18, "SN", 4).
shifts_soft_ub(18, "LD", 5).
shifts_soft_ub(18, "SE", 5).
shifts_soft_ub(18, "SN", 5).
shifts_hard_ub(18, "LD", 6).
shifts_hard_ub(18, "SE", 6).
shifts_hard_ub(18, "SN", 6).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "LM", 1).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_hard_lb(19, "LD", 3).
shifts_hard_lb(19, "SE", 3).
shifts_hard_lb(19, "SN", 3).
shifts_soft_lb(19, "LD", 4).
shifts_soft_lb(19, "SE", 4).
shifts_soft_lb(19, "SN", 4).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "LD", 6).
shifts_hard_ub(19, "SE", 6).
shifts_hard_ub(19, "SN", 6).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "LM", 1).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "LD", 3).
shifts_hard_lb(20, "SE", 3).
shifts_hard_lb(20, "SN", 3).
shifts_soft_lb(20, "LD", 4).
shifts_soft_lb(20, "SE", 4).
shifts_soft_lb(20, "SN", 4).
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "LM", 1).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
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
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "LM", 1).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_hard_lb(22, "LD", 1).
shifts_hard_lb(22, "SE", 1).
shifts_hard_lb(22, "SN", 1).
shifts_soft_lb(22, "LD", 2).
shifts_soft_lb(22, "SE", 2).
shifts_soft_lb(22, "SN", 2).
shifts_soft_ub(22, "LD", 3).
shifts_soft_ub(22, "SE", 3).
shifts_soft_ub(22, "SN", 3).
shifts_hard_ub(22, "LD", 3).
shifts_hard_ub(22, "SE", 3).
shifts_hard_ub(22, "SN", 3).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "LM", 1).
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
shifts_soft_ub(23, "LD", 3).
shifts_soft_ub(23, "SE", 3).
shifts_soft_ub(23, "SN", 3).
shifts_hard_ub(23, "LD", 3).
shifts_hard_ub(23, "SE", 3).
shifts_hard_ub(23, "SN", 3).
shifts_hard_lb(23, "LM", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "LM", 0).
shifts_hard_ub(23, "LM", 1).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
shifts_hard_lb(24, "LD", 1).
shifts_hard_lb(24, "SE", 1).
shifts_hard_lb(24, "SN", 1).
shifts_soft_lb(24, "LD", 2).
shifts_soft_lb(24, "SE", 2).
shifts_soft_lb(24, "SN", 2).
shifts_soft_ub(24, "LD", 3).
shifts_soft_ub(24, "SE", 3).
shifts_soft_ub(24, "SN", 3).
shifts_hard_ub(24, "LD", 3).
shifts_hard_ub(24, "SE", 3).
shifts_hard_ub(24, "SN", 3).
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
shifts_soft_lb(25, "LD", 2).
shifts_soft_lb(25, "SE", 2).
shifts_soft_lb(25, "SN", 2).
shifts_soft_ub(25, "LD", 2).
shifts_soft_ub(25, "SE", 2).
shifts_soft_ub(25, "SN", 2).
shifts_hard_ub(25, "LD", 3).
shifts_hard_ub(25, "SE", 3).
shifts_hard_ub(25, "SN", 3).
shifts_hard_lb(25, "LM", 0).
shifts_soft_lb(25, "LM", 0).
shifts_soft_ub(25, "LM", 0).
shifts_hard_ub(25, "LM", 1).
shifts_hard_lb(25, "EM", 0).
shifts_soft_lb(25, "EM", 0).
shifts_soft_ub(25, "EM", 0).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_hard_lb(26, "LM", 0).
shifts_soft_lb(26, "LM", 0).
shifts_soft_ub(26, "LM", 0).
shifts_hard_ub(26, "LM", 1).
shifts_hard_lb(26, "EM", 0).
shifts_soft_lb(26, "EM", 0).
shifts_soft_ub(26, "EM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_hard_lb(27, "LD", 3).
shifts_hard_lb(27, "SE", 3).
shifts_hard_lb(27, "SN", 3).
shifts_soft_lb(27, "LD", 4).
shifts_soft_lb(27, "SE", 4).
shifts_soft_lb(27, "SN", 4).
shifts_soft_ub(27, "LD", 5).
shifts_soft_ub(27, "SE", 5).
shifts_soft_ub(27, "SN", 5).
shifts_hard_ub(27, "LD", 6).
shifts_hard_ub(27, "SE", 6).
shifts_hard_ub(27, "SN", 6).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "LM", 1).
shifts_hard_lb(27, "EM", 0).
shifts_soft_lb(27, "EM", 0).
shifts_soft_ub(27, "EM", 0).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_hard_lb(28, "LD", 3).
shifts_hard_lb(28, "SE", 3).
shifts_hard_lb(28, "SN", 3).
shifts_soft_lb(28, "LD", 4).
shifts_soft_lb(28, "SE", 4).
shifts_soft_lb(28, "SN", 4).
shifts_soft_ub(28, "LD", 5).
shifts_soft_ub(28, "SE", 5).
shifts_soft_ub(28, "SN", 5).
shifts_hard_ub(28, "LD", 6).
shifts_hard_ub(28, "SE", 6).
shifts_hard_ub(28, "SN", 6).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "LM", 1).
shifts_hard_lb(28, "EM", 0).
shifts_soft_lb(28, "EM", 0).
shifts_soft_ub(28, "EM", 0).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_hard_lb(29, "LD", 3).
shifts_hard_lb(29, "SE", 3).
shifts_hard_lb(29, "SN", 3).
shifts_soft_lb(29, "LD", 4).
shifts_soft_lb(29, "SE", 4).
shifts_soft_lb(29, "SN", 4).
shifts_soft_ub(29, "LD", 5).
shifts_soft_ub(29, "SE", 5).
shifts_soft_ub(29, "SN", 5).
shifts_hard_ub(29, "LD", 6).
shifts_hard_ub(29, "SE", 6).
shifts_hard_ub(29, "SN", 6).
shifts_hard_lb(29, "LM", 0).
shifts_soft_lb(29, "LM", 0).
shifts_soft_ub(29, "LM", 0).
shifts_hard_ub(29, "LM", 1).
shifts_hard_lb(29, "EM", 0).
shifts_soft_lb(29, "EM", 0).
shifts_soft_ub(29, "EM", 0).
shifts_hard_ub(29, "EM", 0).
% Staff 30:
shifts_hard_lb(30, "LD", 3).
shifts_hard_lb(30, "SE", 3).
shifts_hard_lb(30, "SN", 3).
shifts_soft_lb(30, "LD", 4).
shifts_soft_lb(30, "SE", 4).
shifts_soft_lb(30, "SN", 4).
shifts_soft_ub(30, "LD", 5).
shifts_soft_ub(30, "SE", 5).
shifts_soft_ub(30, "SN", 5).
shifts_hard_ub(30, "LD", 6).
shifts_hard_ub(30, "SE", 6).
shifts_hard_ub(30, "SN", 6).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "LM", 1).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
% Staff 31:
shifts_hard_lb(31, "LD", 3).
shifts_hard_lb(31, "SE", 3).
shifts_hard_lb(31, "SN", 3).
shifts_soft_lb(31, "LD", 4).
shifts_soft_lb(31, "SE", 4).
shifts_soft_lb(31, "SN", 4).
shifts_soft_ub(31, "LD", 5).
shifts_soft_ub(31, "SE", 5).
shifts_soft_ub(31, "SN", 5).
shifts_hard_ub(31, "LD", 6).
shifts_hard_ub(31, "SE", 6).
shifts_hard_ub(31, "SN", 6).
shifts_hard_lb(31, "LM", 0).
shifts_soft_lb(31, "LM", 0).
shifts_soft_ub(31, "LM", 0).
shifts_hard_ub(31, "LM", 1).
shifts_hard_lb(31, "EM", 0).
shifts_soft_lb(31, "EM", 0).
shifts_soft_ub(31, "EM", 0).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 3).
shifts_hard_lb(32, "SE", 3).
shifts_hard_lb(32, "SN", 3).
shifts_soft_lb(32, "LD", 4).
shifts_soft_lb(32, "SE", 4).
shifts_soft_lb(32, "SN", 4).
shifts_soft_ub(32, "LD", 5).
shifts_soft_ub(32, "SE", 5).
shifts_soft_ub(32, "SN", 5).
shifts_hard_ub(32, "LD", 6).
shifts_hard_ub(32, "SE", 6).
shifts_hard_ub(32, "SN", 6).
shifts_hard_lb(32, "LM", 0).
shifts_soft_lb(32, "LM", 0).
shifts_soft_ub(32, "LM", 0).
shifts_hard_ub(32, "LM", 1).
shifts_hard_lb(32, "EM", 0).
shifts_soft_lb(32, "EM", 0).
shifts_soft_ub(32, "EM", 0).
shifts_hard_ub(32, "EM", 0).
% Staff 33:
shifts_hard_lb(33, "LD", 3).
shifts_hard_lb(33, "SE", 3).
shifts_hard_lb(33, "SN", 3).
shifts_soft_lb(33, "LD", 4).
shifts_soft_lb(33, "SE", 4).
shifts_soft_lb(33, "SN", 4).
shifts_soft_ub(33, "LD", 5).
shifts_soft_ub(33, "SE", 5).
shifts_soft_ub(33, "SN", 5).
shifts_hard_ub(33, "LD", 6).
shifts_hard_ub(33, "SE", 6).
shifts_hard_ub(33, "SN", 6).
shifts_hard_lb(33, "LM", 0).
shifts_soft_lb(33, "LM", 0).
shifts_soft_ub(33, "LM", 0).
shifts_hard_ub(33, "LM", 1).
shifts_hard_lb(33, "EM", 0).
shifts_soft_lb(33, "EM", 0).
shifts_soft_ub(33, "EM", 0).
shifts_hard_ub(33, "EM", 0).
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
shifts_soft_lb(34, "LM", 0).
shifts_soft_ub(34, "LM", 0).
shifts_hard_ub(34, "LM", 1).
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
shifts_hard_lb(35, "LM", 0).
shifts_soft_lb(35, "LM", 0).
shifts_soft_ub(35, "LM", 0).
shifts_hard_ub(35, "LM", 1).
shifts_hard_lb(35, "EM", 0).
shifts_soft_lb(35, "EM", 0).
shifts_soft_ub(35, "EM", 0).
shifts_hard_ub(35, "EM", 0).
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
forbidden_pair(7, 31).
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
