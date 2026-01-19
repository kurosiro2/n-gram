% Staffs ---------------------------------------------------
staff(1, "Elizabeth Hatfield", "Chief Nurse", "77171", 2).
staff(2, "John Wood", "Chief Nurse", "00518", 2).
staff(3, "Dawn Smith", "Chief Nurse", "57379", 2).
staff(4, "Annette Quinn", "Chief Nurse", "73875", 2).
staff(5, "Phillip Solis", "Chief Nurse", "61382", 2).
staff(6, "Warren Crawford", "Chief Nurse", "54660", 2).
staff(7, "Tina Fernandez", "Chief Nurse", "10632", 2).
staff(8, "Gloria Davis", "Chief Nurse", "24088", 2).
staff(9, "Brandy Cole", "Chief Nurse", "93209", 2).
staff(10, "Stuart Mullins", "Chief Nurse", "24510", 2).
staff(11, "Amanda Jones", "Chief Nurse", "42302", 2).
staff(12, "Jeanne Harvey", "Chief Nurse", "53956", 2).
staff(13, "John Rodriguez", "Chief Nurse", "25329", 2).
staff(14, "James Lawson", "Chief Nurse", "34621", 2).
staff(15, "Angela Barton", "Chief Nurse", "59752", 2).
staff(16, "Kurt Barrett", "Chief Nurse", "38689", 2).
staff(17, "Roger Martin", "Chief Nurse", "34733", 2).
staff(18, "Deanna Sparks", "Chief Nurse", "04005", 2).
staff(19, "Jared Hanson", "Chief Nurse", "11832", 2).
staff(20, "David Smith", "Chief Nurse", "47528", 2).
staff(21, "Stacey Bryant", "Chief Nurse", "83726", 2).
staff(22, "Alyssa Hill", "Chief Nurse", "77268", 2).
staff(23, "Paula Harding", "Chief Nurse", "36681", 2).
staff(24, "George Ross", "Chief Nurse", "23547", 2).
staff(25, "Jerry Vargas", "Chief Nurse", "58628", 0).
staff(26, "Mrs. Maria Khan MD", "Chief Nurse", "03019", 0).
staff(27, "Kristin Thompson", "Chief Nurse", "68090", 0).
staff(28, "Kimberly Ross", "Chief Nurse", "26523", 0).
staff(29, "Daniel Johnson", "Chief Nurse", "02291", 0).
staff(30, "Traci Jackson", "Chief Nurse", "97016", 0).
staff(31, "Connor Wood", "Chief Nurse", "67806", 0).
staff(32, "山本　ゆかり", "Chief Nurse", "55555580", 0).
staff(33, "Christine Bell", "Chief Nurse", "28661", 0).
staff(34, "Elizabeth Carroll", "Chief Nurse", "94279", 0).
staff(35, "Anna Cummings", "Chief Nurse", "39271", 0).
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
group("Heads").
group("Seniors").
group("Seniors1").
group("Seniors1", 1).
group("Seniors1", 3).
group("Seniors1", 4).
group("Seniors1", 5).
group("Seniors1", 6).
group("Seniors1", 7).
group("Seniors1", 8).
group("Seniors1", 9).
group("Seniors1", 10).
group("Seniors1", 11).
group("Seniors1", 12).
group("Seniors1", 13).
group("Seniors1", 14).
group("Seniors1", 15).
group("Seniors1", 16).
group("Seniors1", 17).
group("Seniors1", 18).
group("Seniors1", 19).
group("Seniors1", 20).
group("Seniors1", 21).
group("Seniors1", 22).
group("Seniors1", 23).
group("Seniors1", 24).
group("Seniors2").
group("Mid-levels").
group("Newcomers").
group("Newcomers1").
group("Newcomers2").
group("Night").
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
patterns_hard_ub("N日", 1).
patterns("LM-D", "LM-D").
patterns_hard_ub("P日", 0).
patterns("LM-LD", "LM-LD").
patterns_hard_ub("PN", 0).
patterns("LM-LM", "LM-LM").
pattern("LM-LM", 2).
pattern("LM-LM", 0, "LM").
pattern("LM-LM", 1, "LM").
patterns_hard_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("D").
shifts("D", "D").
staff_dweek_soft_lb("All", "D", "Weekday", 1).
staff_dweek_soft_lb("All", "D", "Weekend", 1).
staff_dweek_soft_lb("All", "D", "Holiday", 1).
shifts("LD").
shifts("LD", "LD").
staff_dweek_soft_lb("All", "LD", "Weekday", 1).
shifts("SE").
shifts("SE", "SE").
staff_dweek_soft_lb("All", "SE", "Weekday", 1).
shifts("SN").
shifts("SN", "SN").
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
shifts("LD+LM+E").
shifts("LD+LM+E", "LD").
shifts("LD+LM+E", "LM").
shifts("LD+LM+E", "E").
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
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "LD+LM+E", "Staffs").
vertical_constraint_type("Seniors1", "LD", "Staffs").
vertical_constraint_type("Seniors1", "SE", "Staffs").
vertical_constraint_type("Seniors1", "SN", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "LD", 1).
shifts_soft_lb(1, "SE", 1).
shifts_soft_lb(1, "SN", 1).
shifts_soft_ub(1, "LD", 1).
shifts_soft_ub(1, "SE", 1).
shifts_soft_ub(1, "SN", 1).
shifts("E").
shifts("E", "E").
shifts_soft_ub(1, "E", 0).
shifts("N").
shifts("N", "N").
shifts_soft_ub(1, "N", 0).
shifts("EM").
shifts("EM", "EM").
shifts_soft_ub(1, "EM", 0).
shifts("LM").
shifts("LM", "LM").
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "EM", 0).
shifts_hard_ub(1, "LM", 0).
% Staff 2:
shifts_soft_lb(2, "LD", 0).
shifts_soft_lb(2, "SE", 0).
shifts_soft_lb(2, "SN", 0).
shifts_soft_ub(2, "LD", 0).
shifts_soft_ub(2, "SE", 0).
shifts_soft_ub(2, "SN", 0).
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
shifts_soft_ub(3, "LD", 1).
shifts_soft_ub(3, "SE", 1).
shifts_soft_ub(3, "SN", 1).
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
shifts_soft_ub(4, "LD", 1).
shifts_soft_ub(4, "SE", 1).
shifts_soft_ub(4, "SN", 1).
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
shifts_soft_ub(5, "LD", 1).
shifts_soft_ub(5, "SE", 1).
shifts_soft_ub(5, "SN", 1).
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
shifts_soft_ub(6, "LD", 1).
shifts_soft_ub(6, "SE", 1).
shifts_soft_ub(6, "SN", 1).
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
shifts_soft_ub(7, "LD", 1).
shifts_soft_ub(7, "SE", 1).
shifts_soft_ub(7, "SN", 1).
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
shifts_soft_ub(8, "LD", 1).
shifts_soft_ub(8, "SE", 1).
shifts_soft_ub(8, "SN", 1).
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
shifts_soft_ub(9, "LD", 1).
shifts_soft_ub(9, "SE", 1).
shifts_soft_ub(9, "SN", 1).
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
shifts_soft_ub(10, "LD", 1).
shifts_soft_ub(10, "SE", 1).
shifts_soft_ub(10, "SN", 1).
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
shifts_soft_ub(11, "LD", 1).
shifts_soft_ub(11, "SE", 1).
shifts_soft_ub(11, "SN", 1).
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
shifts_soft_ub(12, "LD", 1).
shifts_soft_ub(12, "SE", 1).
shifts_soft_ub(12, "SN", 1).
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
shifts_soft_ub(13, "LD", 1).
shifts_soft_ub(13, "SE", 1).
shifts_soft_ub(13, "SN", 1).
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
shifts_soft_ub(14, "LD", 1).
shifts_soft_ub(14, "SE", 1).
shifts_soft_ub(14, "SN", 1).
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
shifts_soft_ub(15, "LD", 1).
shifts_soft_ub(15, "SE", 1).
shifts_soft_ub(15, "SN", 1).
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
shifts_soft_ub(16, "LD", 1).
shifts_soft_ub(16, "SE", 1).
shifts_soft_ub(16, "SN", 1).
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
shifts_soft_ub(17, "LD", 1).
shifts_soft_ub(17, "SE", 1).
shifts_soft_ub(17, "SN", 1).
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
shifts_soft_ub(18, "LD", 1).
shifts_soft_ub(18, "SE", 1).
shifts_soft_ub(18, "SN", 1).
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
shifts_soft_ub(19, "LD", 1).
shifts_soft_ub(19, "SE", 1).
shifts_soft_ub(19, "SN", 1).
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
shifts_soft_ub(20, "LD", 1).
shifts_soft_ub(20, "SE", 1).
shifts_soft_ub(20, "SN", 1).
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
shifts_soft_ub(21, "LD", 1).
shifts_soft_ub(21, "SE", 1).
shifts_soft_ub(21, "SN", 1).
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
shifts_soft_ub(22, "LD", 1).
shifts_soft_ub(22, "SE", 1).
shifts_soft_ub(22, "SN", 1).
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
shifts_soft_ub(23, "LD", 1).
shifts_soft_ub(23, "SE", 1).
shifts_soft_ub(23, "SN", 1).
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
shifts_soft_ub(24, "LD", 1).
shifts_soft_ub(24, "SE", 1).
shifts_soft_ub(24, "SN", 1).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_soft_ub(24, "EM", 0).
shifts_soft_ub(24, "LM", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_hard_ub(24, "EM", 0).
shifts_hard_ub(24, "LM", 0).
% Staff 25:
% Staff 26:
% Staff 27:
% Staff 28:
% Staff 29:
% Staff 30:
% Staff 31:
% Staff 32:
% Staff 33:
% Staff 34:
% Staff 35:
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

% group/2 -> staff_group/2
staff_group(G, N) :- group(G, N).
