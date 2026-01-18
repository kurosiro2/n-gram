% Staffs ---------------------------------------------------
staff(1, "Nicole Herrera", "Deputy Chief Nurse", "88731", 0).
staff(2, "Anthony Willis", "Deputy Chief Nurse", "12017", 0).
staff(3, "Jorge Bryant", "Deputy Chief Nurse", "54438", 0).
staff(4, "Cassandra Hickman", "Nurse", "39037", 0).
staff(5, "Benjamin White", "Chief Nurse", "89354", 0).
staff(6, "Kaitlyn Jordan", "Deputy Chief Nurse", "57573", 0).
staff(7, "Nancy Dixon", "Nurse", "93877", 0).
staff(8, "Heather Wilson", "Chief Nurse", "51449", 0).
staff(9, "Lisa Haney", "Deputy Chief Nurse", "43702", 0).
staff(10, "Jesse Richards", "Nurse", "83274", 0).
staff(11, "Sarah Matthews", "Chief Nurse", "62925", 0).
staff(12, "Nathan Jordan", "Deputy Chief Nurse", "72849", 0).
staff(13, "Billy Soto", "Nurse", "63006", 0).
staff(14, "Danielle Thompson", "Chief Nurse", "32132", 0).
staff(15, "William Martinez", "Deputy Chief Nurse", "61237", 0).
staff(16, "Emily Bowman", "Nurse", "98051", 0).
staff(17, "Nicole Fowler", "Chief Nurse", "28066", 0).
staff(18, "Kathy Mccall", "Deputy Chief Nurse", "59203", 0).
staff(19, "Shannon Mason", "Nurse", "37296", 0).
staff(20, "Elizabeth Weeks", "Chief Nurse", "70269", 0).
staff(21, "Lori Copeland", "Deputy Chief Nurse", "98569", 0).
staff(22, "Sharon Thomas", "Nurse", "44602", 0).
staff(23, "Isaiah Henry", "Chief Nurse", "51566", 0).
staff(24, "Dana Cruz", "Deputy Chief Nurse", "43126", 0).
staff(25, "Donald Knapp", "Nurse", "84641", 0).
staff(26, "Michelle Davenport", "Chief Nurse", "15401", 0).
staff(27, "Sandra Frost", "Deputy Chief Nurse", "33872", 0).
staff(28, "Andre Thompson", "Nurse", "26818", 0).
staff(29, "Hunter Martinez DVM", "Chief Nurse", "89817", 0).
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
group("Heads").
group("Seniors").
group("Seniors", 1).
group("Seniors", 2).
group("Seniors", 3).
group("Seniors", 4).
group("Seniors", 5).
group("Seniors", 6).
group("Mid-levels").
group("Mid-levels", 7).
group("Mid-levels", 8).
group("Mid-levels", 9).
group("Mid-levels", 10).
group("Mid-levels", 11).
group("Mid-levels", 12).
group("Mid-levels", 13).
group("Mid-levels", 14).
group("Mid-levels", 15).
group("Mid-levels", 16).
group("Mid-levels", 17).
group("Mid-levels", 18).
group("Mid-levels", 19).
group("Mid-levels", 20).
group("Mid-levels", 21).
group("Mid-levels", 22).
group("Mid-levels", 23).
group("Mid-levels", 24).
group("Mid-levels", 25).
group("Mid-levels", 26).
group("Newcomers").
group("Newcomers", 27).
group("Newcomers", 28).
group("Newcomers", 29).
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
group("Night", 22).
group("Night", 23).
group("Night", 24).
group("Night", 25).
group("Night", 26).
group("Night", 27).
group("Night", 28).
group("Night", 29).
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
succ_shift("SE", "SN").
succ_shift("E", "N").
pred_shift("E", "N").
succ_shift("N", "WR").
succ_shift("N", "PH").
% パターン割当数
patterns("WR-WR", "WR-WR").
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("○○", 2).
patterns("LD-D", "LD-D").
patterns_soft_ub("N日", 1).
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
staff_dweek_hard_lb("All", "D", "Weekend", 1).
staff_dweek_hard_lb("All", "D", "Holiday", 1).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 2).
staff_dweek_soft_lb("All", "D", "Holiday", 2).
staff_dweek_soft_ub("All", "D", "Weekday", 13).
staff_dweek_soft_ub("All", "D", "Weekend", 3).
staff_dweek_soft_ub("All", "D", "Holiday", 3).
shifts("LD").
shifts("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
shifts("SE").
shifts("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 3).
shifts("SN").
shifts("SN", "SN").
staff_dweek_hard_lb("All", "SN", "Weekday", 3).
staff_dweek_hard_lb("All", "LD", "Weekend", 3).
staff_dweek_hard_lb("All", "SE", "Weekend", 3).
staff_dweek_hard_lb("All", "SN", "Weekend", 3).
staff_dweek_hard_lb("All", "LD", "Holiday", 3).
staff_dweek_hard_lb("All", "SE", "Holiday", 3).
staff_dweek_hard_lb("All", "SN", "Holiday", 3).
staff_dweek_soft_lb("All", "LD", "Weekday", 4).
staff_dweek_soft_lb("All", "SE", "Weekday", 4).
staff_dweek_soft_lb("All", "SN", "Weekday", 4).
staff_dweek_soft_lb("All", "LD", "Weekend", 4).
staff_dweek_soft_lb("All", "SE", "Weekend", 4).
staff_dweek_soft_lb("All", "SN", "Weekend", 4).
staff_dweek_soft_lb("All", "LD", "Holiday", 4).
staff_dweek_soft_lb("All", "SE", "Holiday", 4).
staff_dweek_soft_lb("All", "SN", "Holiday", 4).
staff_dweek_soft_ub("All", "LD", "Weekday", 5).
staff_dweek_soft_ub("All", "SE", "Weekday", 5).
staff_dweek_soft_ub("All", "SN", "Weekday", 5).
staff_dweek_soft_ub("All", "LD", "Weekend", 5).
staff_dweek_soft_ub("All", "SE", "Weekend", 5).
staff_dweek_soft_ub("All", "SN", "Weekend", 5).
staff_dweek_soft_ub("All", "LD", "Holiday", 5).
staff_dweek_soft_ub("All", "SE", "Holiday", 5).
staff_dweek_soft_ub("All", "SN", "Holiday", 5).
shifts("E").
shifts("E", "E").
staff_dweek_soft_ub("All", "E", "Weekday", 0).
shifts("N").
shifts("N", "N").
staff_dweek_soft_ub("All", "N", "Weekday", 0).
staff_dweek_soft_ub("All", "E", "Weekend", 0).
staff_dweek_soft_ub("All", "N", "Weekend", 0).
staff_dweek_soft_ub("All", "E", "Holiday", 0).
staff_dweek_soft_ub("All", "N", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_ub("All", "LM", "Weekday", 1).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shifts("LD+LM+E").
shifts("LD+LM+E", "LD").
shifts("LD+LM+E", "LM").
shifts("LD+LM+E", "E").
staff_dweek_hard_lb("All", "LD+LM+E", "Weekday", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 5).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
staff_dweek_soft_lb("Heads", "D", "Weekday", 1).
staff_dweek_soft_lb("Heads", "D", "Weekend", 0).
staff_dweek_soft_lb("Heads", "D", "Holiday", 0).
staff_dweek_soft_ub("Heads", "D", "Weekday", 1).
staff_dweek_soft_ub("Heads", "D", "Weekend", 0).
staff_dweek_soft_ub("Heads", "D", "Holiday", 0).
staff_dweek_hard_lb("Seniors", "LD", "Weekday", 1).
staff_dweek_hard_lb("Seniors", "SE", "Weekday", 1).
staff_dweek_hard_lb("Seniors", "SN", "Weekday", 1).
staff_dweek_hard_lb("Seniors", "LD", "Weekend", 1).
staff_dweek_hard_lb("Seniors", "SE", "Weekend", 1).
staff_dweek_hard_lb("Seniors", "SN", "Weekend", 1).
staff_dweek_hard_lb("Seniors", "LD", "Holiday", 1).
staff_dweek_hard_lb("Seniors", "SE", "Holiday", 1).
staff_dweek_hard_lb("Seniors", "SN", "Holiday", 1).
staff_dweek_soft_lb("Seniors", "LD", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "SE", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "SN", "Weekday", 1).
staff_dweek_soft_lb("Seniors", "LD", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "SE", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "SN", "Weekend", 1).
staff_dweek_soft_lb("Seniors", "LD", "Holiday", 1).
staff_dweek_soft_lb("Seniors", "SE", "Holiday", 1).
staff_dweek_soft_lb("Seniors", "SN", "Holiday", 1).
staff_dweek_soft_ub("Seniors", "LD", "Weekday", 4).
staff_dweek_soft_ub("Seniors", "SE", "Weekday", 4).
staff_dweek_soft_ub("Seniors", "SN", "Weekday", 4).
staff_dweek_soft_ub("Seniors", "LD", "Weekend", 4).
staff_dweek_soft_ub("Seniors", "SE", "Weekend", 4).
staff_dweek_soft_ub("Seniors", "SN", "Weekend", 4).
staff_dweek_soft_ub("Seniors", "LD", "Holiday", 4).
staff_dweek_soft_ub("Seniors", "SE", "Holiday", 4).
staff_dweek_soft_ub("Seniors", "SN", "Holiday", 4).
staff_dweek_hard_lb("Mid-levels", "LD", "Weekday", 1).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekday", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekday", 1).
staff_dweek_hard_lb("Mid-levels", "LD", "Weekend", 1).
staff_dweek_hard_lb("Mid-levels", "SE", "Weekend", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Weekend", 1).
staff_dweek_hard_lb("Mid-levels", "LD", "Holiday", 1).
staff_dweek_hard_lb("Mid-levels", "SE", "Holiday", 1).
staff_dweek_hard_lb("Mid-levels", "SN", "Holiday", 1).
staff_dweek_soft_lb("Mid-levels", "LD", "Weekday", 2).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekday", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekday", 2).
staff_dweek_soft_lb("Mid-levels", "LD", "Weekend", 2).
staff_dweek_soft_lb("Mid-levels", "SE", "Weekend", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Weekend", 2).
staff_dweek_soft_lb("Mid-levels", "LD", "Holiday", 2).
staff_dweek_soft_lb("Mid-levels", "SE", "Holiday", 2).
staff_dweek_soft_lb("Mid-levels", "SN", "Holiday", 2).
staff_dweek_soft_ub("Mid-levels", "LD", "Weekday", 4).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekday", 4).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekday", 4).
staff_dweek_soft_ub("Mid-levels", "LD", "Weekend", 4).
staff_dweek_soft_ub("Mid-levels", "SE", "Weekend", 4).
staff_dweek_soft_ub("Mid-levels", "SN", "Weekend", 4).
staff_dweek_soft_ub("Mid-levels", "LD", "Holiday", 4).
staff_dweek_soft_ub("Mid-levels", "SE", "Holiday", 4).
staff_dweek_soft_ub("Mid-levels", "SN", "Holiday", 4).
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
vertical_constraint_type("All", "E", "Staffs").
vertical_constraint_type("All", "N", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("All", "LD+LM+E", "Staffs").
vertical_constraint_type("Heads", "D", "Staffs").
vertical_constraint_type("Seniors", "LD", "Staffs").
vertical_constraint_type("Seniors", "SE", "Staffs").
vertical_constraint_type("Seniors", "SN", "Staffs").
vertical_constraint_type("Mid-levels", "LD", "Staffs").
vertical_constraint_type("Mid-levels", "SE", "Staffs").
vertical_constraint_type("Mid-levels", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "LD", 4).
shifts_soft_lb(1, "SE", 4).
shifts_soft_lb(1, "SN", 4).
shifts_soft_ub(1, "LD", 4).
shifts_soft_ub(1, "SE", 4).
shifts_soft_ub(1, "SN", 4).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_soft_lb(1, "LM", 0).
shifts("EM").
shifts("EM", "EM").
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 3).
shifts_soft_lb(2, "SE", 3).
shifts_soft_lb(2, "SN", 3).
shifts_soft_ub(2, "LD", 3).
shifts_soft_ub(2, "SE", 3).
shifts_soft_ub(2, "SN", 3).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
shifts_soft_lb(2, "LM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_soft_lb(3, "LD", 3).
shifts_soft_lb(3, "SE", 3).
shifts_soft_lb(3, "SN", 3).
shifts_soft_ub(3, "LD", 3).
shifts_soft_ub(3, "SE", 3).
shifts_soft_ub(3, "SN", 3).
shifts_hard_ub(3, "E", 0).
shifts_hard_ub(3, "N", 0).
shifts_soft_lb(3, "LM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_soft_lb(4, "LD", 4).
shifts_soft_lb(4, "SE", 4).
shifts_soft_lb(4, "SN", 4).
shifts_soft_ub(4, "LD", 4).
shifts_soft_ub(4, "SE", 4).
shifts_soft_ub(4, "SN", 4).
shifts_hard_ub(4, "E", 0).
shifts_hard_ub(4, "N", 0).
shifts_soft_lb(4, "LM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_soft_lb(5, "LD", 5).
shifts_soft_lb(5, "SE", 5).
shifts_soft_lb(5, "SN", 5).
shifts_soft_ub(5, "LD", 5).
shifts_soft_ub(5, "SE", 5).
shifts_soft_ub(5, "SN", 5).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
shifts_soft_lb(5, "LM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_soft_lb(6, "LD", 6).
shifts_soft_lb(6, "SE", 6).
shifts_soft_lb(6, "SN", 6).
shifts_soft_ub(6, "LD", 6).
shifts_soft_ub(6, "SE", 6).
shifts_soft_ub(6, "SN", 6).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
shifts_soft_lb(6, "LM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_soft_lb(7, "LD", 3).
shifts_soft_lb(7, "SE", 3).
shifts_soft_lb(7, "SN", 3).
shifts_soft_ub(7, "LD", 3).
shifts_soft_ub(7, "SE", 3).
shifts_soft_ub(7, "SN", 3).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
shifts_soft_lb(7, "LM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_soft_lb(8, "LD", 7).
shifts_soft_lb(8, "SE", 7).
shifts_soft_lb(8, "SN", 7).
shifts_soft_ub(8, "LD", 7).
shifts_soft_ub(8, "SE", 7).
shifts_soft_ub(8, "SN", 7).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
shifts_soft_lb(8, "LM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_soft_lb(9, "LD", 5).
shifts_soft_lb(9, "SE", 5).
shifts_soft_lb(9, "SN", 5).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
shifts_soft_lb(9, "LM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_soft_lb(10, "LD", 6).
shifts_soft_lb(10, "SE", 6).
shifts_soft_lb(10, "SN", 6).
shifts_soft_ub(10, "LD", 6).
shifts_soft_ub(10, "SE", 6).
shifts_soft_ub(10, "SN", 6).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
shifts_soft_lb(10, "LM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_soft_lb(11, "LD", 5).
shifts_soft_lb(11, "SE", 5).
shifts_soft_lb(11, "SN", 5).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
shifts_soft_lb(11, "LM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_soft_lb(12, "LD", 4).
shifts_soft_lb(12, "SE", 4).
shifts_soft_lb(12, "SN", 4).
shifts_soft_ub(12, "LD", 4).
shifts_soft_ub(12, "SE", 4).
shifts_soft_ub(12, "SN", 4).
shifts_hard_ub(12, "E", 0).
shifts_hard_ub(12, "N", 0).
shifts_soft_lb(12, "LM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_soft_lb(13, "LD", 5).
shifts_soft_lb(13, "SE", 5).
shifts_soft_lb(13, "SN", 5).
shifts_soft_ub(13, "LD", 5).
shifts_soft_ub(13, "SE", 5).
shifts_soft_ub(13, "SN", 5).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
shifts_soft_lb(13, "LM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_soft_lb(14, "LD", 6).
shifts_soft_lb(14, "SE", 6).
shifts_soft_lb(14, "SN", 6).
shifts_soft_ub(14, "LD", 6).
shifts_soft_ub(14, "SE", 6).
shifts_soft_ub(14, "SN", 6).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_soft_lb(14, "LM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_soft_lb(15, "LD", 3).
shifts_soft_lb(15, "SE", 3).
shifts_soft_lb(15, "SN", 3).
shifts_soft_ub(15, "LD", 3).
shifts_soft_ub(15, "SE", 3).
shifts_soft_ub(15, "SN", 3).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_soft_lb(15, "LM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_soft_lb(16, "LD", 6).
shifts_soft_lb(16, "SE", 6).
shifts_soft_lb(16, "SN", 6).
shifts_soft_ub(16, "LD", 6).
shifts_soft_ub(16, "SE", 6).
shifts_soft_ub(16, "SN", 6).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
shifts_soft_lb(16, "LM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_soft_lb(17, "LD", 4).
shifts_soft_lb(17, "SE", 4).
shifts_soft_lb(17, "SN", 4).
shifts_soft_ub(17, "LD", 4).
shifts_soft_ub(17, "SE", 4).
shifts_soft_ub(17, "SN", 4).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
shifts_soft_lb(17, "LM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_soft_lb(18, "LD", 1).
shifts_soft_lb(18, "SE", 1).
shifts_soft_lb(18, "SN", 1).
shifts_soft_ub(18, "LD", 1).
shifts_soft_ub(18, "SE", 1).
shifts_soft_ub(18, "SN", 1).
shifts_hard_ub(18, "E", 3).
shifts_hard_ub(18, "N", 3).
shifts_soft_lb(18, "LM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_soft_lb(19, "LD", 5).
shifts_soft_lb(19, "SE", 5).
shifts_soft_lb(19, "SN", 5).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "E", 0).
shifts_hard_ub(19, "N", 0).
shifts_soft_lb(19, "LM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_soft_lb(20, "LD", 6).
shifts_soft_lb(20, "SE", 6).
shifts_soft_lb(20, "SN", 6).
shifts_soft_ub(20, "LD", 6).
shifts_soft_ub(20, "SE", 6).
shifts_soft_ub(20, "SN", 6).
shifts_hard_ub(20, "E", 0).
shifts_hard_ub(20, "N", 0).
shifts_soft_lb(20, "LM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_soft_lb(21, "LD", 7).
shifts_soft_lb(21, "SE", 7).
shifts_soft_lb(21, "SN", 7).
shifts_soft_ub(21, "LD", 7).
shifts_soft_ub(21, "SE", 7).
shifts_soft_ub(21, "SN", 7).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
shifts_soft_lb(21, "LM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 4).
shifts_soft_ub(22, "SE", 4).
shifts_soft_ub(22, "SN", 4).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
shifts_soft_lb(22, "LM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 4).
shifts_soft_ub(23, "SE", 4).
shifts_soft_ub(23, "SN", 4).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
shifts_soft_lb(23, "LM", 0).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
shifts_soft_lb(24, "LD", 5).
shifts_soft_lb(24, "SE", 5).
shifts_soft_lb(24, "SN", 5).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_soft_lb(24, "LM", 0).
shifts_hard_ub(24, "EM", 0).
% Staff 25:
shifts_soft_lb(25, "LD", 5).
shifts_soft_lb(25, "SE", 5).
shifts_soft_lb(25, "SN", 5).
shifts_soft_ub(25, "LD", 5).
shifts_soft_ub(25, "SE", 5).
shifts_soft_ub(25, "SN", 5).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
shifts_soft_lb(25, "LM", 0).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_soft_lb(26, "LD", 6).
shifts_soft_lb(26, "SE", 6).
shifts_soft_lb(26, "SN", 6).
shifts_soft_ub(26, "LD", 6).
shifts_soft_ub(26, "SE", 6).
shifts_soft_ub(26, "SN", 6).
shifts_hard_ub(26, "E", 0).
shifts_hard_ub(26, "N", 0).
shifts_soft_lb(26, "LM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_soft_lb(27, "LD", 1).
shifts_soft_lb(27, "SE", 1).
shifts_soft_lb(27, "SN", 1).
shifts_soft_ub(27, "LD", 1).
shifts_soft_ub(27, "SE", 1).
shifts_soft_ub(27, "SN", 1).
shifts_hard_ub(27, "E", 0).
shifts_hard_ub(27, "N", 0).
shifts_soft_lb(27, "LM", 0).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_soft_lb(28, "LD", 1).
shifts_soft_lb(28, "SE", 1).
shifts_soft_lb(28, "SN", 1).
shifts_soft_ub(28, "LD", 1).
shifts_soft_ub(28, "SE", 1).
shifts_soft_ub(28, "SN", 1).
shifts_hard_ub(28, "E", 0).
shifts_hard_ub(28, "N", 0).
shifts_soft_lb(28, "LM", 0).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_soft_lb(29, "LD", 1).
shifts_soft_lb(29, "SE", 1).
shifts_soft_lb(29, "SN", 1).
shifts_soft_ub(29, "LD", 1).
shifts_soft_ub(29, "SE", 1).
shifts_soft_ub(29, "SN", 1).
shifts_hard_ub(29, "E", 0).
shifts_hard_ub(29, "N", 0).
shifts_soft_lb(29, "LM", 0).
shifts_hard_ub(29, "EM", 0).
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
