% Staffs ---------------------------------------------------
staff(1, "相川　真弓", "Chief Nurse", "0058327", 0).
staff(2, "Anthony Willis", "Deputy Chief Nurse", "12017", 0).
staff(3, "Cassandra Hickman", "Deputy Chief Nurse", "39037", 0).
staff(4, "Jorge Bryant", "Deputy Chief Nurse", "54438", 0).
staff(5, "Dawn Taylor", "Nurse", "02455", 0).
staff(6, "Phillip Wiley", "Nurse", "44311", 0).
staff(7, "Adriana Roberts", "Nurse", "95912", 0).
staff(8, "John Davis", "Nurse", "52033", 0).
staff(9, "Benjamin White", "Nurse", "89354", 0).
staff(10, "Michael Clark", "Nurse", "01751", 0).
staff(11, "Lisa Haney", "Nurse", "43702", 0).
staff(12, "Sarah Matthews", "Nurse", "62925", 0).
staff(13, "Michelle Davenport", "Nurse", "15401", 0).
staff(14, "Jesse Richards", "Nurse", "83274", 0).
staff(15, "Kathy Mccall", "Nurse", "59203", 0).
staff(16, "Shannon Mason", "Nurse", "37296", 0).
staff(17, "Nicole Fowler", "Nurse", "28066", 0).
staff(18, "Mariah Cole", "Nurse", "85877", 0).
staff(19, "Sharon Thomas", "Nurse", "44602", 0).
staff(20, "Isaiah Henry", "Nurse", "51566", 0).
staff(21, "Danielle Thompson", "Nurse", "32132", 0).
staff(22, "Donald Knapp", "Nurse", "84641", 0).
staff(23, "Dana Cruz", "Nurse", "43126", 0).
staff(24, "Sandra Frost", "Nurse", "33872", 0).
staff(25, "Andre Thompson", "Nurse", "26818", 0).
staff(26, "Hunter Martinez DVM", "Nurse", "89817", 0).
staff(27, "Natalie Wise", "Nurse", "27882", 0).
staff(28, "Brian Sullivan", "Nurse", "39721", 0).
staff(29, "Aiden Moore", "Nurse", "A353091", 0).
staff(30, "Hudson Clark", "Nurse", "A261217", 0).
staff(31, "Adrian Hernandez", "Nurse", "A130583", 0).
staff(32, "Thomas Allen", "Nurse", "A240983", 0).
staff(33, "Ethan Ortiz", "Nurse", "A585369", 0).
staff(34, "Jonathan Hall", "Nurse", "A512425", 0).
staff(35, "Abigail Wise", "Nurse", "44884", 0).
staff(36, "Nicole Alexander", "Nurse", "79904", 0).
staff(37, "Paula Forbes", "Nurse", "97693", 0).
staff(38, "Stephen Ryan", "Nurse", "35174", 0).
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
group("代行").
group("代行", 2).
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("代行", 8).
group("代行", 11).
group("代行", 12).
group("Leaders").
group("Leaders", 2).
group("Leaders", 3).
group("Leaders", 4).
group("Leaders", 5).
group("Leaders", 8).
group("Leaders", 9).
group("Leaders", 11).
group("Leaders", 12).
group("Leaders", 15).
group("Leaders", 17).
group("Leaders", 18).
group("Leaders", 20).
group("平日代行").
group("平日代行", 2).
group("平日代行", 3).
group("平日代行", 4).
group("Newcomers").
group("Newcomers", 29).
group("Newcomers", 30).
group("Newcomers", 31).
group("Newcomers", 32).
group("Newcomers", 33).
group("Newcomers", 34).
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
staff_dweek_hard_lb("All", "D", "Weekday", 4).
staff_dweek_hard_lb("All", "D", "Weekend", 4).
staff_dweek_hard_lb("All", "D", "Holiday", 4).
staff_dweek_soft_lb("All", "D", "Weekday", 4).
staff_dweek_soft_lb("All", "D", "Weekend", 4).
staff_dweek_soft_lb("All", "D", "Holiday", 4).
staff_dweek_soft_ub("All", "D", "Weekday", 16).
staff_dweek_soft_ub("All", "D", "Weekend", 4).
staff_dweek_soft_ub("All", "D", "Holiday", 4).
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
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 1).
staff_dweek_soft_lb("All", "LM", "Weekday", 1).
staff_dweek_soft_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
staff_dweek_soft_lb("代行", "D", "Weekday", 1).
staff_dweek_soft_lb("代行", "LD", "Weekday", 1).
staff_dweek_soft_lb("代行", "D", "Weekend", 0).
staff_dweek_soft_lb("代行", "LD", "Weekend", 0).
staff_dweek_soft_lb("代行", "D", "Holiday", 0).
staff_dweek_soft_lb("代行", "LD", "Holiday", 0).
staff_dweek_soft_ub("代行", "D", "Weekday", 1).
staff_dweek_soft_ub("代行", "LD", "Weekday", 1).
staff_dweek_soft_ub("代行", "D", "Weekend", 0).
staff_dweek_soft_ub("代行", "LD", "Weekend", 0).
staff_dweek_soft_ub("代行", "D", "Holiday", 0).
staff_dweek_soft_ub("代行", "LD", "Holiday", 0).
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
staff_dweek_soft_ub("Leaders", "LD", "Weekday", 5).
staff_dweek_soft_ub("Leaders", "SE", "Weekday", 5).
staff_dweek_soft_ub("Leaders", "SN", "Weekday", 5).
staff_dweek_soft_ub("Leaders", "LD", "Weekend", 5).
staff_dweek_soft_ub("Leaders", "SE", "Weekend", 5).
staff_dweek_soft_ub("Leaders", "SN", "Weekend", 5).
staff_dweek_soft_ub("Leaders", "LD", "Holiday", 5).
staff_dweek_soft_ub("Leaders", "SE", "Holiday", 5).
staff_dweek_soft_ub("Leaders", "SN", "Holiday", 5).
staff_dweek_hard_lb("平日代行", "D", "Weekday", 1).
staff_dweek_hard_lb("平日代行", "D", "Weekend", 1).
staff_dweek_hard_lb("平日代行", "D", "Holiday", 1).
staff_dweek_soft_lb("平日代行", "D", "Weekday", 1).
staff_dweek_soft_lb("平日代行", "D", "Weekend", 1).
staff_dweek_soft_lb("平日代行", "D", "Holiday", 1).
staff_dweek_soft_ub("平日代行", "D", "Weekday", 2).
staff_dweek_soft_ub("平日代行", "D", "Weekend", 2).
staff_dweek_soft_ub("平日代行", "D", "Holiday", 2).
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
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("代行", "D", "Staffs").
vertical_constraint_type("代行", "LD", "Staffs").
vertical_constraint_type("Leaders", "LD", "Staffs").
vertical_constraint_type("Leaders", "SE", "Staffs").
vertical_constraint_type("Leaders", "SN", "Staffs").
vertical_constraint_type("平日代行", "D", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts("E").
shifts("E", "E").
shifts_hard_lb(1, "E", 0).
shifts("N").
shifts("N", "N").
shifts_hard_lb(1, "N", 0).
shifts_soft_lb(1, "E", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_ub(1, "E", 0).
shifts_soft_ub(1, "N", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_lb(1, "LM", 0).
shifts_soft_lb(1, "LM", 0).
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "LM", 0).
shifts("EM").
shifts("EM", "EM").
shifts_hard_lb(1, "EM", 0).
shifts_soft_lb(1, "EM", 0).
shifts_soft_ub(1, "EM", 0).
shifts_hard_ub(1, "EM", 0).
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
shifts_hard_lb(2, "E", 0).
shifts_hard_lb(2, "N", 0).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
shifts_hard_lb(2, "LM", 1).
shifts_soft_lb(2, "LM", 1).
shifts_soft_ub(2, "LM", 1).
shifts_hard_ub(2, "LM", 1).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
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
shifts_hard_lb(3, "E", 0).
shifts_hard_lb(3, "N", 0).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
shifts_hard_ub(3, "E", 0).
shifts_hard_ub(3, "N", 0).
shifts_hard_lb(3, "LM", 1).
shifts_soft_lb(3, "LM", 1).
shifts_soft_ub(3, "LM", 1).
shifts_hard_ub(3, "LM", 1).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_hard_lb(4, "LD", 2).
shifts_hard_lb(4, "SE", 2).
shifts_hard_lb(4, "SN", 2).
shifts_soft_lb(4, "LD", 2).
shifts_soft_lb(4, "SE", 2).
shifts_soft_lb(4, "SN", 2).
shifts_soft_ub(4, "LD", 3).
shifts_soft_ub(4, "SE", 3).
shifts_soft_ub(4, "SN", 3).
shifts_hard_ub(4, "LD", 3).
shifts_hard_ub(4, "SE", 3).
shifts_hard_ub(4, "SN", 3).
shifts_hard_lb(4, "E", 0).
shifts_hard_lb(4, "N", 0).
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
shifts_hard_ub(4, "E", 0).
shifts_hard_ub(4, "N", 0).
shifts_hard_lb(4, "LM", 1).
shifts_soft_lb(4, "LM", 1).
shifts_soft_ub(4, "LM", 1).
shifts_hard_ub(4, "LM", 1).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "LD", 2).
shifts_hard_lb(5, "SE", 2).
shifts_hard_lb(5, "SN", 2).
shifts_soft_lb(5, "LD", 3).
shifts_soft_lb(5, "SE", 3).
shifts_soft_lb(5, "SN", 3).
shifts_soft_ub(5, "LD", 5).
shifts_soft_ub(5, "SE", 5).
shifts_soft_ub(5, "SN", 5).
shifts_hard_ub(5, "LD", 6).
shifts_hard_ub(5, "SE", 6).
shifts_hard_ub(5, "SN", 6).
shifts_hard_lb(5, "E", 0).
shifts_hard_lb(5, "N", 0).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
shifts_hard_lb(5, "LM", 1).
shifts_soft_lb(5, "LM", 1).
shifts_soft_ub(5, "LM", 3).
shifts_hard_ub(5, "LM", 3).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
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
shifts_hard_lb(6, "E", 0).
shifts_hard_lb(6, "N", 0).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
shifts_hard_lb(6, "LM", 1).
shifts_soft_lb(6, "LM", 1).
shifts_soft_ub(6, "LM", 3).
shifts_hard_ub(6, "LM", 3).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "LD", 2).
shifts_hard_lb(7, "SE", 2).
shifts_hard_lb(7, "SN", 2).
shifts_soft_lb(7, "LD", 2).
shifts_soft_lb(7, "SE", 2).
shifts_soft_lb(7, "SN", 2).
shifts_soft_ub(7, "LD", 2).
shifts_soft_ub(7, "SE", 2).
shifts_soft_ub(7, "SN", 2).
shifts_hard_ub(7, "LD", 2).
shifts_hard_ub(7, "SE", 2).
shifts_hard_ub(7, "SN", 2).
shifts_hard_lb(7, "E", 0).
shifts_hard_lb(7, "N", 0).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
shifts_hard_lb(7, "LM", 1).
shifts_soft_lb(7, "LM", 1).
shifts_soft_ub(7, "LM", 3).
shifts_hard_ub(7, "LM", 3).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "LD", 2).
shifts_hard_lb(8, "SE", 2).
shifts_hard_lb(8, "SN", 2).
shifts_soft_lb(8, "LD", 3).
shifts_soft_lb(8, "SE", 3).
shifts_soft_lb(8, "SN", 3).
shifts_soft_ub(8, "LD", 5).
shifts_soft_ub(8, "SE", 5).
shifts_soft_ub(8, "SN", 5).
shifts_hard_ub(8, "LD", 6).
shifts_hard_ub(8, "SE", 6).
shifts_hard_ub(8, "SN", 6).
shifts_hard_lb(8, "E", 0).
shifts_hard_lb(8, "N", 0).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
shifts_hard_lb(8, "LM", 1).
shifts_soft_lb(8, "LM", 1).
shifts_soft_ub(8, "LM", 3).
shifts_hard_ub(8, "LM", 3).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
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
shifts_hard_lb(9, "E", 0).
shifts_hard_lb(9, "N", 0).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
shifts_hard_lb(9, "LM", 1).
shifts_soft_lb(9, "LM", 1).
shifts_soft_ub(9, "LM", 3).
shifts_hard_ub(9, "LM", 3).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_hard_lb(10, "LD", 2).
shifts_hard_lb(10, "SE", 2).
shifts_hard_lb(10, "SN", 2).
shifts_soft_lb(10, "LD", 2).
shifts_soft_lb(10, "SE", 2).
shifts_soft_lb(10, "SN", 2).
shifts_soft_ub(10, "LD", 2).
shifts_soft_ub(10, "SE", 2).
shifts_soft_ub(10, "SN", 2).
shifts_hard_ub(10, "LD", 2).
shifts_hard_ub(10, "SE", 2).
shifts_hard_ub(10, "SN", 2).
shifts_hard_lb(10, "E", 0).
shifts_hard_lb(10, "N", 0).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
shifts_hard_lb(10, "LM", 1).
shifts_soft_lb(10, "LM", 1).
shifts_soft_ub(10, "LM", 3).
shifts_hard_ub(10, "LM", 3).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
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
shifts_hard_lb(11, "E", 0).
shifts_hard_lb(11, "N", 0).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
shifts_hard_lb(11, "LM", 1).
shifts_soft_lb(11, "LM", 1).
shifts_soft_ub(11, "LM", 3).
shifts_hard_ub(11, "LM", 3).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_hard_lb(12, "LD", 2).
shifts_hard_lb(12, "SE", 2).
shifts_hard_lb(12, "SN", 2).
shifts_soft_lb(12, "LD", 3).
shifts_soft_lb(12, "SE", 3).
shifts_soft_lb(12, "SN", 3).
shifts_soft_ub(12, "LD", 5).
shifts_soft_ub(12, "SE", 5).
shifts_soft_ub(12, "SN", 5).
shifts_hard_ub(12, "LD", 6).
shifts_hard_ub(12, "SE", 6).
shifts_hard_ub(12, "SN", 6).
shifts_hard_lb(12, "E", 0).
shifts_hard_lb(12, "N", 0).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_ub(12, "E", 0).
shifts_soft_ub(12, "N", 0).
shifts_hard_ub(12, "E", 0).
shifts_hard_ub(12, "N", 0).
shifts_hard_lb(12, "LM", 1).
shifts_soft_lb(12, "LM", 1).
shifts_soft_ub(12, "LM", 3).
shifts_hard_ub(12, "LM", 3).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
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
shifts_hard_lb(13, "E", 0).
shifts_hard_lb(13, "N", 0).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
shifts_hard_lb(13, "LM", 1).
shifts_soft_lb(13, "LM", 1).
shifts_soft_ub(13, "LM", 3).
shifts_hard_ub(13, "LM", 3).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
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
shifts_hard_lb(14, "E", 0).
shifts_hard_lb(14, "N", 0).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_hard_lb(14, "LM", 1).
shifts_soft_lb(14, "LM", 1).
shifts_soft_ub(14, "LM", 3).
shifts_hard_ub(14, "LM", 3).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "LD", 1).
shifts_hard_lb(15, "SE", 1).
shifts_hard_lb(15, "SN", 1).
shifts_soft_lb(15, "LD", 1).
shifts_soft_lb(15, "SE", 1).
shifts_soft_lb(15, "SN", 1).
shifts_soft_ub(15, "LD", 1).
shifts_soft_ub(15, "SE", 1).
shifts_soft_ub(15, "SN", 1).
shifts_hard_ub(15, "LD", 1).
shifts_hard_ub(15, "SE", 1).
shifts_hard_ub(15, "SN", 1).
shifts_hard_lb(15, "E", 0).
shifts_hard_lb(15, "N", 0).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
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
shifts_hard_lb(16, "E", 0).
shifts_hard_lb(16, "N", 0).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
shifts_hard_lb(16, "LM", 1).
shifts_soft_lb(16, "LM", 1).
shifts_soft_ub(16, "LM", 3).
shifts_hard_ub(16, "LM", 3).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
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
shifts_hard_lb(17, "E", 0).
shifts_hard_lb(17, "N", 0).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
shifts_hard_lb(17, "LM", 1).
shifts_soft_lb(17, "LM", 1).
shifts_soft_ub(17, "LM", 3).
shifts_hard_ub(17, "LM", 3).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
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
shifts_hard_lb(18, "E", 0).
shifts_hard_lb(18, "N", 0).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
shifts_hard_ub(18, "E", 0).
shifts_hard_ub(18, "N", 0).
shifts_hard_lb(18, "LM", 1).
shifts_soft_lb(18, "LM", 1).
shifts_soft_ub(18, "LM", 3).
shifts_hard_ub(18, "LM", 3).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
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
shifts_hard_lb(19, "E", 0).
shifts_hard_lb(19, "N", 0).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
shifts_hard_ub(19, "E", 0).
shifts_hard_ub(19, "N", 0).
shifts_hard_lb(19, "LM", 1).
shifts_soft_lb(19, "LM", 1).
shifts_soft_ub(19, "LM", 3).
shifts_hard_ub(19, "LM", 3).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "LD", 2).
shifts_hard_lb(20, "SE", 2).
shifts_hard_lb(20, "SN", 2).
shifts_soft_lb(20, "LD", 2).
shifts_soft_lb(20, "SE", 2).
shifts_soft_lb(20, "SN", 2).
shifts_soft_ub(20, "LD", 4).
shifts_soft_ub(20, "SE", 4).
shifts_soft_ub(20, "SN", 4).
shifts_hard_ub(20, "LD", 4).
shifts_hard_ub(20, "SE", 4).
shifts_hard_ub(20, "SN", 4).
shifts_hard_lb(20, "E", 0).
shifts_hard_lb(20, "N", 0).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
shifts_hard_ub(20, "E", 0).
shifts_hard_ub(20, "N", 0).
shifts_hard_lb(20, "LM", 1).
shifts_soft_lb(20, "LM", 1).
shifts_soft_ub(20, "LM", 3).
shifts_hard_ub(20, "LM", 3).
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
shifts_soft_ub(21, "LD", 5).
shifts_soft_ub(21, "SE", 5).
shifts_soft_ub(21, "SN", 5).
shifts_hard_ub(21, "LD", 6).
shifts_hard_ub(21, "SE", 6).
shifts_hard_ub(21, "SN", 6).
shifts_hard_lb(21, "E", 0).
shifts_hard_lb(21, "N", 0).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
shifts_hard_lb(21, "LM", 1).
shifts_soft_lb(21, "LM", 1).
shifts_soft_ub(21, "LM", 3).
shifts_hard_ub(21, "LM", 3).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
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
shifts_hard_lb(22, "E", 0).
shifts_hard_lb(22, "N", 0).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
shifts_hard_lb(22, "LM", 1).
shifts_soft_lb(22, "LM", 1).
shifts_soft_ub(22, "LM", 3).
shifts_hard_ub(22, "LM", 3).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_hard_lb(23, "LD", 2).
shifts_hard_lb(23, "SE", 2).
shifts_hard_lb(23, "SN", 2).
shifts_soft_lb(23, "LD", 3).
shifts_soft_lb(23, "SE", 3).
shifts_soft_lb(23, "SN", 3).
shifts_soft_ub(23, "LD", 5).
shifts_soft_ub(23, "SE", 5).
shifts_soft_ub(23, "SN", 5).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
shifts_hard_lb(23, "E", 0).
shifts_hard_lb(23, "N", 0).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
shifts_hard_lb(23, "LM", 1).
shifts_soft_lb(23, "LM", 1).
shifts_soft_ub(23, "LM", 3).
shifts_hard_ub(23, "LM", 3).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
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
shifts_hard_lb(24, "E", 0).
shifts_hard_lb(24, "N", 0).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_hard_lb(24, "LM", 1).
shifts_soft_lb(24, "LM", 1).
shifts_soft_ub(24, "LM", 3).
shifts_hard_ub(24, "LM", 3).
shifts_hard_lb(24, "EM", 0).
shifts_soft_lb(24, "EM", 0).
shifts_soft_ub(24, "EM", 0).
shifts_hard_ub(24, "EM", 0).
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
shifts_hard_lb(25, "E", 0).
shifts_hard_lb(25, "N", 0).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
shifts_hard_lb(25, "LM", 1).
shifts_soft_lb(25, "LM", 1).
shifts_soft_ub(25, "LM", 3).
shifts_hard_ub(25, "LM", 3).
shifts_hard_lb(25, "EM", 0).
shifts_soft_lb(25, "EM", 0).
shifts_soft_ub(25, "EM", 0).
shifts_hard_ub(25, "EM", 0).
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
shifts_hard_lb(26, "E", 0).
shifts_hard_lb(26, "N", 0).
shifts_soft_lb(26, "E", 0).
shifts_soft_lb(26, "N", 0).
shifts_soft_ub(26, "E", 0).
shifts_soft_ub(26, "N", 0).
shifts_hard_ub(26, "E", 0).
shifts_hard_ub(26, "N", 0).
shifts_hard_lb(26, "LM", 1).
shifts_soft_lb(26, "LM", 1).
shifts_soft_ub(26, "LM", 3).
shifts_hard_ub(26, "LM", 3).
shifts_hard_lb(26, "EM", 0).
shifts_soft_lb(26, "EM", 0).
shifts_soft_ub(26, "EM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_hard_lb(27, "LD", 1).
shifts_hard_lb(27, "SE", 1).
shifts_hard_lb(27, "SN", 1).
shifts_soft_lb(27, "LD", 1).
shifts_soft_lb(27, "SE", 1).
shifts_soft_lb(27, "SN", 1).
shifts_soft_ub(27, "LD", 1).
shifts_soft_ub(27, "SE", 1).
shifts_soft_ub(27, "SN", 1).
shifts_hard_ub(27, "LD", 1).
shifts_hard_ub(27, "SE", 1).
shifts_hard_ub(27, "SN", 1).
shifts_hard_lb(27, "E", 0).
shifts_hard_lb(27, "N", 0).
shifts_soft_lb(27, "E", 0).
shifts_soft_lb(27, "N", 0).
shifts_soft_ub(27, "E", 0).
shifts_soft_ub(27, "N", 0).
shifts_hard_ub(27, "E", 0).
shifts_hard_ub(27, "N", 0).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "LM", 0).
shifts_hard_lb(27, "EM", 0).
shifts_soft_lb(27, "EM", 0).
shifts_soft_ub(27, "EM", 0).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_hard_lb(28, "LD", 0).
shifts_hard_lb(28, "SE", 0).
shifts_hard_lb(28, "SN", 0).
shifts_soft_lb(28, "LD", 0).
shifts_soft_lb(28, "SE", 0).
shifts_soft_lb(28, "SN", 0).
shifts_soft_ub(28, "LD", 0).
shifts_soft_ub(28, "SE", 0).
shifts_soft_ub(28, "SN", 0).
shifts_hard_ub(28, "LD", 0).
shifts_hard_ub(28, "SE", 0).
shifts_hard_ub(28, "SN", 0).
shifts_hard_lb(28, "E", 0).
shifts_hard_lb(28, "N", 0).
shifts_soft_lb(28, "E", 0).
shifts_soft_lb(28, "N", 0).
shifts_soft_ub(28, "E", 0).
shifts_soft_ub(28, "N", 0).
shifts_hard_ub(28, "E", 0).
shifts_hard_ub(28, "N", 0).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "LM", 0).
shifts_hard_lb(28, "EM", 0).
shifts_soft_lb(28, "EM", 0).
shifts_soft_ub(28, "EM", 0).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_hard_lb(29, "LD", 2).
shifts_hard_lb(29, "SE", 2).
shifts_hard_lb(29, "SN", 2).
shifts_soft_lb(29, "LD", 2).
shifts_soft_lb(29, "SE", 2).
shifts_soft_lb(29, "SN", 2).
shifts_soft_ub(29, "LD", 3).
shifts_soft_ub(29, "SE", 3).
shifts_soft_ub(29, "SN", 3).
shifts_hard_ub(29, "LD", 3).
shifts_hard_ub(29, "SE", 3).
shifts_hard_ub(29, "SN", 3).
shifts_hard_lb(29, "E", 0).
shifts_hard_lb(29, "N", 0).
shifts_soft_lb(29, "E", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_ub(29, "E", 0).
shifts_soft_ub(29, "N", 0).
shifts_hard_ub(29, "E", 0).
shifts_hard_ub(29, "N", 0).
shifts_hard_lb(29, "LM", 1).
shifts_soft_lb(29, "LM", 1).
shifts_soft_ub(29, "LM", 1).
shifts_hard_ub(29, "LM", 1).
shifts_hard_lb(29, "EM", 0).
shifts_soft_lb(29, "EM", 0).
shifts_soft_ub(29, "EM", 0).
shifts_hard_ub(29, "EM", 0).
% Staff 30:
shifts_hard_lb(30, "LD", 2).
shifts_hard_lb(30, "SE", 2).
shifts_hard_lb(30, "SN", 2).
shifts_soft_lb(30, "LD", 2).
shifts_soft_lb(30, "SE", 2).
shifts_soft_lb(30, "SN", 2).
shifts_soft_ub(30, "LD", 2).
shifts_soft_ub(30, "SE", 2).
shifts_soft_ub(30, "SN", 2).
shifts_hard_ub(30, "LD", 3).
shifts_hard_ub(30, "SE", 3).
shifts_hard_ub(30, "SN", 3).
shifts_hard_lb(30, "E", 0).
shifts_hard_lb(30, "N", 0).
shifts_soft_lb(30, "E", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_ub(30, "E", 0).
shifts_soft_ub(30, "N", 0).
shifts_hard_ub(30, "E", 0).
shifts_hard_ub(30, "N", 0).
shifts_hard_lb(30, "LM", 1).
shifts_soft_lb(30, "LM", 1).
shifts_soft_ub(30, "LM", 1).
shifts_hard_ub(30, "LM", 1).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
% Staff 31:
shifts_hard_lb(31, "LD", 2).
shifts_hard_lb(31, "SE", 2).
shifts_hard_lb(31, "SN", 2).
shifts_soft_lb(31, "LD", 2).
shifts_soft_lb(31, "SE", 2).
shifts_soft_lb(31, "SN", 2).
shifts_soft_ub(31, "LD", 2).
shifts_soft_ub(31, "SE", 2).
shifts_soft_ub(31, "SN", 2).
shifts_hard_ub(31, "LD", 3).
shifts_hard_ub(31, "SE", 3).
shifts_hard_ub(31, "SN", 3).
shifts_hard_lb(31, "E", 0).
shifts_hard_lb(31, "N", 0).
shifts_soft_lb(31, "E", 0).
shifts_soft_lb(31, "N", 0).
shifts_soft_ub(31, "E", 0).
shifts_soft_ub(31, "N", 0).
shifts_hard_ub(31, "E", 0).
shifts_hard_ub(31, "N", 0).
shifts_hard_lb(31, "LM", 1).
shifts_soft_lb(31, "LM", 1).
shifts_soft_ub(31, "LM", 1).
shifts_hard_ub(31, "LM", 1).
shifts_hard_lb(31, "EM", 0).
shifts_soft_lb(31, "EM", 0).
shifts_soft_ub(31, "EM", 0).
shifts_hard_ub(31, "EM", 0).
% Staff 32:
shifts_hard_lb(32, "LD", 2).
shifts_hard_lb(32, "SE", 2).
shifts_hard_lb(32, "SN", 2).
shifts_soft_lb(32, "LD", 2).
shifts_soft_lb(32, "SE", 2).
shifts_soft_lb(32, "SN", 2).
shifts_soft_ub(32, "LD", 2).
shifts_soft_ub(32, "SE", 2).
shifts_soft_ub(32, "SN", 2).
shifts_hard_ub(32, "LD", 3).
shifts_hard_ub(32, "SE", 3).
shifts_hard_ub(32, "SN", 3).
shifts_hard_lb(32, "E", 0).
shifts_hard_lb(32, "N", 0).
shifts_soft_lb(32, "E", 0).
shifts_soft_lb(32, "N", 0).
shifts_soft_ub(32, "E", 0).
shifts_soft_ub(32, "N", 0).
shifts_hard_ub(32, "E", 0).
shifts_hard_ub(32, "N", 0).
shifts_hard_lb(32, "LM", 1).
shifts_soft_lb(32, "LM", 1).
shifts_soft_ub(32, "LM", 1).
shifts_hard_ub(32, "LM", 1).
shifts_hard_lb(32, "EM", 0).
shifts_soft_lb(32, "EM", 0).
shifts_soft_ub(32, "EM", 0).
shifts_hard_ub(32, "EM", 0).
% Staff 33:
shifts_hard_lb(33, "LD", 2).
shifts_hard_lb(33, "SE", 2).
shifts_hard_lb(33, "SN", 2).
shifts_soft_lb(33, "LD", 2).
shifts_soft_lb(33, "SE", 2).
shifts_soft_lb(33, "SN", 2).
shifts_soft_ub(33, "LD", 2).
shifts_soft_ub(33, "SE", 2).
shifts_soft_ub(33, "SN", 2).
shifts_hard_ub(33, "LD", 3).
shifts_hard_ub(33, "SE", 3).
shifts_hard_ub(33, "SN", 3).
shifts_hard_lb(33, "E", 0).
shifts_hard_lb(33, "N", 0).
shifts_soft_lb(33, "E", 0).
shifts_soft_lb(33, "N", 0).
shifts_soft_ub(33, "E", 0).
shifts_soft_ub(33, "N", 0).
shifts_hard_ub(33, "E", 0).
shifts_hard_ub(33, "N", 0).
shifts_hard_lb(33, "LM", 1).
shifts_soft_lb(33, "LM", 1).
shifts_soft_ub(33, "LM", 1).
shifts_hard_ub(33, "LM", 1).
shifts_hard_lb(33, "EM", 0).
shifts_soft_lb(33, "EM", 0).
shifts_soft_ub(33, "EM", 0).
shifts_hard_ub(33, "EM", 0).
% Staff 34:
shifts_hard_lb(34, "LD", 2).
shifts_hard_lb(34, "SE", 2).
shifts_hard_lb(34, "SN", 2).
shifts_soft_lb(34, "LD", 2).
shifts_soft_lb(34, "SE", 2).
shifts_soft_lb(34, "SN", 2).
shifts_soft_ub(34, "LD", 2).
shifts_soft_ub(34, "SE", 2).
shifts_soft_ub(34, "SN", 2).
shifts_hard_ub(34, "LD", 3).
shifts_hard_ub(34, "SE", 3).
shifts_hard_ub(34, "SN", 3).
shifts_hard_lb(34, "E", 0).
shifts_hard_lb(34, "N", 0).
shifts_soft_lb(34, "E", 0).
shifts_soft_lb(34, "N", 0).
shifts_soft_ub(34, "E", 0).
shifts_soft_ub(34, "N", 0).
shifts_hard_ub(34, "E", 0).
shifts_hard_ub(34, "N", 0).
shifts_hard_lb(34, "LM", 1).
shifts_soft_lb(34, "LM", 1).
shifts_soft_ub(34, "LM", 1).
shifts_hard_ub(34, "LM", 1).
shifts_hard_lb(34, "EM", 0).
shifts_soft_lb(34, "EM", 0).
shifts_soft_ub(34, "EM", 0).
shifts_hard_ub(34, "EM", 0).
% Staff 35:
% Staff 36:
% Staff 37:
% Staff 38:
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
forbidden_pair(15, 18).
forbidden_pair(19, 23).
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
% Staff 7:
neg_shift(7, "日", ("日";"N";"J";"S")).
neg_shift(7, "土", ("日";"N";"J";"S")).
neg_shift(7, "祝", ("日";"N";"J";"S")).
% Staff 8:
neg_shift(8, "月", ("日";"N";"J";"S")).
neg_shift(8, "木", ("日";"N";"J";"S")).
% Staff 10:
neg_shift(10, "日", ("日";"N";"J";"S")).
neg_shift(10, "土", ("日";"N";"J";"S")).
neg_shift(10, "祝", ("日";"N";"J";"S")).
% Staff 27:
neg_shift(27, "日", ("日";"N";"J";"S")).
neg_shift(27, "土", ("日";"N";"J";"S")).
neg_shift(27, "祝", ("日";"N";"J";"S")).
% Staff 28:
neg_shift(28, "日", ("日";"N";"J";"S")).
neg_shift(28, "土", ("日";"N";"J";"S")).
neg_shift(28, "祝", ("日";"N";"J";"S")).

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
