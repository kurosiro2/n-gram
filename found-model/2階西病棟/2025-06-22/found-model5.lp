header("Cost: 2 1 2 80 -134, Elapsed: 486s").
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
out_stats(num_national_holiday_offs(1, 0, 0, 0)).
out_stats(num_national_holiday_offs(2, 0, 0, 0)).
out_stats(num_national_holiday_offs(3, 0, 0, 0)).
out_stats(num_national_holiday_offs(4, 0, 0, 0)).
out_stats(num_national_holiday_offs(5, 0, 0, 0)).
out_stats(num_national_holiday_offs(6, 0, 0, 0)).
out_stats(num_national_holiday_offs(7, 0, 0, 0)).
out_stats(num_national_holiday_offs(8, 0, 0, 0)).
out_stats(num_national_holiday_offs(9, 0, 0, 0)).
out_stats(num_national_holiday_offs(10, 0, 0, 0)).
out_stats(num_national_holiday_offs(11, 0, 0, 0)).
out_stats(num_national_holiday_offs(12, 0, 0, 0)).
out_stats(num_national_holiday_offs(13, 0, 0, 0)).
out_stats(num_national_holiday_offs(14, 0, 0, 0)).
out_stats(num_national_holiday_offs(15, 0, 0, 0)).
out_stats(num_national_holiday_offs(16, 0, 0, 0)).
out_stats(num_national_holiday_offs(17, 0, 0, 0)).
out_stats(num_national_holiday_offs(18, 0, 0, 0)).
out_stats(num_national_holiday_offs(19, 0, 0, 0)).
out_stats(num_national_holiday_offs(20, 0, 0, 0)).
out_stats(num_national_holiday_offs(21, 0, 0, 0)).
out_stats(num_national_holiday_offs(22, 0, 0, 0)).
out_stats(num_national_holiday_offs(23, 0, 0, 0)).
out_stats(num_national_holiday_offs(24, 0, 0, 0)).
out_stats(num_national_holiday_offs(25, 0, 0, 0)).
out_stats(num_national_holiday_offs(26, 0, 0, 0)).
out_stats(num_national_holiday_offs(27, 0, 0, 0)).
out_stats(num_national_holiday_offs(28, 0, 0, 0)).
out_stats(num_national_holiday_offs(29, 0, 0, 0)).
out_stats(num_national_holiday_offs(30, 0, 0, 0)).
out_stats(num_national_holiday_offs(31, 0, 0, 0)).
out_stats(num_national_holiday_offs(32, 0, 0, 0)).
section("2階西病棟").
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
group("Seniors", 1).
group("Seniors", 2).
group("Seniors", 3).
group("Seniors", 5).
group("Seniors", 17).
group("Seniors", 19).
group("Seniors", 20).
group("Mid-levels", 6).
group("Mid-levels", 7).
group("Mid-levels", 8).
group("Mid-levels", 9).
group("Mid-levels", 10).
group("Mid-levels", 11).
group("Mid-levels", 12).
group("Mid-levels", 21).
group("Mid-levels", 22).
group("Mid-levels", 23).
group("Mid-levels", 24).
group("Mid-levels", 25).
group("Mid-levels", 26).
group("Mid-levels", 27).
group("Mid-levels", 28).
group("Mid-levels", 29).
group("Newcomers", 13).
group("Newcomers", 14).
group("Newcomers", 15).
group("Newcomers", 16).
group("Newcomers", 18).
group("Newcomers", 30).
group("Newcomers", 31).
group("Newcomers", 32).
group("Night", 1).
group("Night", 2).
group("Night", 3).
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
group("Night", 17).
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
group_order(1, "All").
group_order(2, "Heads").
group_order(3, "Seniors").
group_order(4, "Mid-levels").
group_order(5, "Newcomers").
group_order(6, "Night").
group_rel("深夜勤", "Heads").
group_rel("深夜勤", "Seniors").
group_rel("深夜勤", "Mid-levels").
group_rel("深夜勤", "Newcomers").
group_rel("長日勤", "Heads").
group_rel("長日勤", "Seniors").
group_rel("長日勤", "Mid-levels").
group_rel("長日勤", "Newcomers").
group_rel("日勤", "Heads").
group_rel("日勤", "Seniors").
group_rel("日勤", "Mid-levels").
group_rel("日勤", "Newcomers").
group_rel("早出", "All").
group_rel("早出", "Heads").
group_rel("遅出", "All").
group_rel("遅出", "Heads").
group_rel("準夜勤", "Heads").
group_rel("準夜勤", "Seniors").
group_rel("準夜勤", "Mid-levels").
group_rel("準夜勤", "Newcomers").
out_assigned(1, 20250720).
out_assigned(2, 20250720).
out_assigned(3, 20250720).
out_assigned(4, 20250720).
out_assigned(5, 20250720).
out_assigned(7, 20250720).
out_assigned(8, 20250720).
out_assigned(9, 20250720).
out_assigned(10, 20250720).
out_assigned(11, 20250720).
out_assigned(12, 20250720).
out_assigned(13, 20250720).
out_assigned(14, 20250720).
out_assigned(15, 20250720).
out_assigned(16, 20250720).
out_assigned(17, 20250720).
out_assigned(18, 20250720).
out_assigned(20, 20250720).
out_assigned(21, 20250720).
out_assigned(22, 20250720).
out_assigned(23, 20250720).
out_assigned(24, 20250720).
out_assigned(25, 20250720).
out_assigned(27, 20250720).
out_assigned(28, 20250720).
out_assigned(29, 20250720).
out_assigned(30, 20250720).
out_assigned(31, 20250720).
out_assigned(32, 20250720).
out_assigned(1, 20250721).
out_assigned(2, 20250721).
out_assigned(3, 20250721).
out_assigned(4, 20250721).
out_assigned(5, 20250721).
out_assigned(7, 20250721).
out_assigned(8, 20250721).
out_assigned(9, 20250721).
out_assigned(10, 20250721).
out_assigned(11, 20250721).
out_assigned(12, 20250721).
out_assigned(13, 20250721).
out_assigned(14, 20250721).
out_assigned(15, 20250721).
out_assigned(16, 20250721).
out_assigned(17, 20250721).
out_assigned(18, 20250721).
out_assigned(20, 20250721).
out_assigned(21, 20250721).
out_assigned(22, 20250721).
out_assigned(23, 20250721).
out_assigned(24, 20250721).
out_assigned(25, 20250721).
out_assigned(26, 20250721).
out_assigned(27, 20250721).
out_assigned(28, 20250721).
out_assigned(29, 20250721).
out_assigned(30, 20250721).
out_assigned(31, 20250721).
out_assigned(32, 20250721).
out_assigned(1, 20250722).
out_assigned(2, 20250722).
out_assigned(3, 20250722).
out_assigned(4, 20250722).
out_assigned(5, 20250722).
out_assigned(6, 20250722).
out_assigned(7, 20250722).
out_assigned(8, 20250722).
out_assigned(9, 20250722).
out_assigned(10, 20250722).
out_assigned(11, 20250722).
out_assigned(12, 20250722).
out_assigned(13, 20250722).
out_assigned(14, 20250722).
out_assigned(15, 20250722).
out_assigned(16, 20250722).
out_assigned(17, 20250722).
out_assigned(18, 20250722).
out_assigned(20, 20250722).
out_assigned(21, 20250722).
out_assigned(22, 20250722).
out_assigned(23, 20250722).
out_assigned(24, 20250722).
out_assigned(25, 20250722).
out_assigned(26, 20250722).
out_assigned(27, 20250722).
out_assigned(28, 20250722).
out_assigned(29, 20250722).
out_assigned(30, 20250722).
out_assigned(31, 20250722).
out_assigned(32, 20250722).
out_assigned(1, 20250723).
out_assigned(2, 20250723).
out_assigned(3, 20250723).
out_assigned(4, 20250723).
out_assigned(5, 20250723).
out_assigned(6, 20250723).
out_assigned(7, 20250723).
out_assigned(8, 20250723).
out_assigned(9, 20250723).
out_assigned(11, 20250723).
out_assigned(12, 20250723).
out_assigned(14, 20250723).
out_assigned(15, 20250723).
out_assigned(16, 20250723).
out_assigned(18, 20250723).
out_assigned(19, 20250723).
out_assigned(20, 20250723).
out_assigned(21, 20250723).
out_assigned(22, 20250723).
out_assigned(23, 20250723).
out_assigned(24, 20250723).
out_assigned(25, 20250723).
out_assigned(26, 20250723).
out_assigned(27, 20250723).
out_assigned(28, 20250723).
out_assigned(29, 20250723).
out_assigned(30, 20250723).
out_assigned(31, 20250723).
out_assigned(32, 20250723).
out_assigned(1, 20250724).
out_assigned(2, 20250724).
out_assigned(3, 20250724).
out_assigned(4, 20250724).
out_assigned(5, 20250724).
out_assigned(6, 20250724).
out_assigned(7, 20250724).
out_assigned(8, 20250724).
out_assigned(9, 20250724).
out_assigned(11, 20250724).
out_assigned(12, 20250724).
out_assigned(15, 20250724).
out_assigned(16, 20250724).
out_assigned(17, 20250724).
out_assigned(18, 20250724).
out_assigned(19, 20250724).
out_assigned(20, 20250724).
out_assigned(21, 20250724).
out_assigned(22, 20250724).
out_assigned(23, 20250724).
out_assigned(24, 20250724).
out_assigned(25, 20250724).
out_assigned(27, 20250724).
out_assigned(28, 20250724).
out_assigned(29, 20250724).
out_assigned(30, 20250724).
out_assigned(31, 20250724).
out_assigned(32, 20250724).
out_assigned(1, 20250725).
out_assigned(2, 20250725).
out_assigned(3, 20250725).
out_assigned(4, 20250725).
out_assigned(5, 20250725).
out_assigned(6, 20250725).
out_assigned(7, 20250725).
out_assigned(8, 20250725).
out_assigned(9, 20250725).
out_assigned(11, 20250725).
out_assigned(12, 20250725).
out_assigned(14, 20250725).
out_assigned(15, 20250725).
out_assigned(16, 20250725).
out_assigned(18, 20250725).
out_assigned(19, 20250725).
out_assigned(20, 20250725).
out_assigned(21, 20250725).
out_assigned(22, 20250725).
out_assigned(23, 20250725).
out_assigned(24, 20250725).
out_assigned(25, 20250725).
out_assigned(26, 20250725).
out_assigned(27, 20250725).
out_assigned(28, 20250725).
out_assigned(29, 20250725).
out_assigned(30, 20250725).
out_assigned(31, 20250725).
out_assigned(32, 20250725).
out_assigned(1, 20250726).
out_assigned(2, 20250726).
out_assigned(3, 20250726).
out_assigned(4, 20250726).
out_assigned(5, 20250726).
out_assigned(6, 20250726).
out_assigned(7, 20250726).
out_assigned(8, 20250726).
out_assigned(9, 20250726).
out_assigned(11, 20250726).
out_assigned(12, 20250726).
out_assigned(14, 20250726).
out_assigned(15, 20250726).
out_assigned(16, 20250726).
out_assigned(18, 20250726).
out_assigned(19, 20250726).
out_assigned(20, 20250726).
out_assigned(21, 20250726).
out_assigned(22, 20250726).
out_assigned(23, 20250726).
out_assigned(24, 20250726).
out_assigned(25, 20250726).
out_assigned(26, 20250726).
out_assigned(27, 20250726).
out_assigned(28, 20250726).
out_assigned(29, 20250726).
out_assigned(30, 20250726).
out_assigned(31, 20250726).
out_assigned(32, 20250726).
out_assigned(2, 20250615).
out_assigned(3, 20250615).
out_assigned(5, 20250615).
out_assigned(6, 20250615).
out_assigned(8, 20250615).
out_assigned(9, 20250615).
out_assigned(11, 20250615).
out_assigned(12, 20250615).
out_assigned(13, 20250615).
out_assigned(14, 20250615).
out_assigned(15, 20250615).
out_assigned(16, 20250615).
out_assigned(30, 20250615).
out_assigned(31, 20250615).
out_assigned(32, 20250615).
out_assigned(2, 20250616).
out_assigned(3, 20250616).
out_assigned(4, 20250616).
out_assigned(5, 20250616).
out_assigned(6, 20250616).
out_assigned(7, 20250616).
out_assigned(8, 20250616).
out_assigned(9, 20250616).
out_assigned(10, 20250616).
out_assigned(11, 20250616).
out_assigned(12, 20250616).
out_assigned(13, 20250616).
out_assigned(15, 20250616).
out_assigned(16, 20250616).
out_assigned(30, 20250616).
out_assigned(31, 20250616).
out_assigned(32, 20250616).
out_assigned(2, 20250617).
out_assigned(3, 20250617).
out_assigned(4, 20250617).
out_assigned(5, 20250617).
out_assigned(6, 20250617).
out_assigned(7, 20250617).
out_assigned(8, 20250617).
out_assigned(9, 20250617).
out_assigned(10, 20250617).
out_assigned(11, 20250617).
out_assigned(12, 20250617).
out_assigned(13, 20250617).
out_assigned(14, 20250617).
out_assigned(15, 20250617).
out_assigned(16, 20250617).
out_assigned(30, 20250617).
out_assigned(31, 20250617).
out_assigned(32, 20250617).
out_assigned(2, 20250618).
out_assigned(3, 20250618).
out_assigned(4, 20250618).
out_assigned(5, 20250618).
out_assigned(6, 20250618).
out_assigned(7, 20250618).
out_assigned(8, 20250618).
out_assigned(9, 20250618).
out_assigned(10, 20250618).
out_assigned(11, 20250618).
out_assigned(12, 20250618).
out_assigned(13, 20250618).
out_assigned(14, 20250618).
out_assigned(2, 20250619).
out_assigned(3, 20250619).
out_assigned(4, 20250619).
out_assigned(5, 20250619).
out_assigned(6, 20250619).
out_assigned(7, 20250619).
out_assigned(8, 20250619).
out_assigned(9, 20250619).
out_assigned(10, 20250619).
out_assigned(11, 20250619).
out_assigned(12, 20250619).
out_assigned(13, 20250619).
out_assigned(14, 20250619).
out_assigned(15, 20250619).
out_assigned(16, 20250619).
out_assigned(30, 20250619).
out_assigned(31, 20250619).
out_assigned(32, 20250619).
out_assigned(2, 20250620).
out_assigned(3, 20250620).
out_assigned(4, 20250620).
out_assigned(8, 20250620).
out_assigned(9, 20250620).
out_assigned(10, 20250620).
out_assigned(11, 20250620).
out_assigned(12, 20250620).
out_assigned(13, 20250620).
out_assigned(14, 20250620).
out_assigned(15, 20250620).
out_assigned(16, 20250620).
out_assigned(30, 20250620).
out_assigned(31, 20250620).
out_assigned(32, 20250620).
out_assigned(2, 20250621).
out_assigned(3, 20250621).
out_assigned(4, 20250621).
out_assigned(5, 20250621).
out_assigned(7, 20250621).
out_assigned(8, 20250621).
out_assigned(9, 20250621).
out_assigned(10, 20250621).
out_assigned(11, 20250621).
out_assigned(12, 20250621).
out_assigned(13, 20250621).
out_assigned(14, 20250621).
out_assigned(15, 20250621).
out_assigned(16, 20250621).
out_assigned(30, 20250621).
out_assigned(31, 20250621).
out_assigned(32, 20250621).
out_date(20250721, "Ho").
out_date(20250622, "Su").
out_date(20250623, "Mo").
out_date(20250624, "Tu").
out_date(20250625, "We").
out_date(20250626, "Th").
out_date(20250627, "Fr").
out_date(20250628, "Sa").
out_date(20250629, "Su").
out_date(20250630, "Mo").
out_date(20250701, "Tu").
out_date(20250702, "We").
out_date(20250703, "Th").
out_date(20250704, "Fr").
out_date(20250705, "Sa").
out_date(20250706, "Su").
out_date(20250707, "Mo").
out_date(20250708, "Tu").
out_date(20250709, "We").
out_date(20250710, "Th").
out_date(20250711, "Fr").
out_date(20250712, "Sa").
out_date(20250713, "Su").
out_date(20250714, "Mo").
out_date(20250715, "Tu").
out_date(20250716, "We").
out_date(20250717, "Th").
out_date(20250718, "Fr").
out_date(20250719, "Sa").
out_date(20250720, "Su").
out_date(20250722, "Tu").
out_date(20250723, "We").
out_date(20250724, "Th").
out_date(20250725, "Fr").
out_date(20250726, "Sa").
out_date(20250615, "Su").
out_date(20250616, "Mo").
out_date(20250617, "Tu").
out_date(20250618, "We").
out_date(20250619, "Th").
out_date(20250620, "Fr").
out_date(20250621, "Sa").
shift_order(1, "深夜勤").
shift_order(2, "長日勤").
shift_order(3, "日勤").
shift_order(4, "早出").
shift_order(5, "遅出").
shift_order(6, "準夜勤").
shift_rel("長日勤", "LD").
shift_rel("日勤", "D").
shift_rel("早出", "EM").
shift_rel("遅出", "LM").
shift_rel("深夜勤", "SN").
shift_rel("深夜勤", "N").
shift_rel("準夜勤", "SE").
shift_rel("準夜勤", "E").
out_assigned(6, 20250720, "WR").
out_assigned(19, 20250720, "WR").
out_assigned(26, 20250720, "WR").
out_assigned(6, 20250721, "WR").
out_assigned(19, 20250721, "PH").
out_assigned(19, 20250722, "D").
out_assigned(10, 20250723, "HL").
out_assigned(13, 20250723, "AL").
out_assigned(17, 20250723, "D").
out_assigned(10, 20250724, "AL").
out_assigned(13, 20250724, "WR").
out_assigned(14, 20250724, "D").
out_assigned(26, 20250724, "D").
out_assigned(10, 20250725, "AL").
out_assigned(13, 20250725, "WR").
out_assigned(17, 20250725, "WR").
out_assigned(10, 20250726, "WR").
out_assigned(13, 20250726, "AL").
out_assigned(17, 20250726, "AL").
out_assigned(1, 20250615, "WR").
out_assigned(4, 20250615, "WR").
out_assigned(7, 20250615, "WR").
out_assigned(10, 20250615, "WR").
out_assigned(17, 20250615, "SN").
out_assigned(18, 20250615, "SE").
out_assigned(19, 20250615, "SE").
out_assigned(20, 20250615, "D").
out_assigned(21, 20250615, "WR").
out_assigned(22, 20250615, "D").
out_assigned(23, 20250615, "LD").
out_assigned(24, 20250615, "WR").
out_assigned(25, 20250615, "D").
out_assigned(26, 20250615, "SN").
out_assigned(27, 20250615, "D").
out_assigned(28, 20250615, "WR").
out_assigned(29, 20250615, "LD").
out_assigned(1, 20250616, "D").
out_assigned(14, 20250616, "D").
out_assigned(17, 20250616, "WR").
out_assigned(18, 20250616, "SN").
out_assigned(19, 20250616, "SN").
out_assigned(20, 20250616, "LD").
out_assigned(21, 20250616, "D").
out_assigned(22, 20250616, "D").
out_assigned(23, 20250616, "SE").
out_assigned(24, 20250616, "D").
out_assigned(25, 20250616, "D").
out_assigned(26, 20250616, "WR").
out_assigned(27, 20250616, "LD").
out_assigned(28, 20250616, "D").
out_assigned(29, 20250616, "SE").
out_assigned(1, 20250617, "D").
out_assigned(17, 20250617, "WR").
out_assigned(18, 20250617, "AL").
out_assigned(19, 20250617, "WR").
out_assigned(20, 20250617, "SE").
out_assigned(21, 20250617, "D").
out_assigned(22, 20250617, "D").
out_assigned(23, 20250617, "SN").
out_assigned(24, 20250617, "D").
out_assigned(25, 20250617, "LD").
out_assigned(26, 20250617, "LD").
out_assigned(27, 20250617, "SE").
out_assigned(28, 20250617, "D").
out_assigned(29, 20250617, "SN").
out_assigned(1, 20250618, "D").
out_assigned(15, 20250618, "D").
out_assigned(16, 20250618, "D").
out_assigned(17, 20250618, "D").
out_assigned(18, 20250618, "WR").
out_assigned(19, 20250618, "LD").
out_assigned(20, 20250618, "SN").
out_assigned(21, 20250618, "D").
out_assigned(22, 20250618, "AL").
out_assigned(23, 20250618, "WR").
out_assigned(24, 20250618, "LD").
out_assigned(25, 20250618, "SE").
out_assigned(26, 20250618, "SE").
out_assigned(27, 20250618, "SN").
out_assigned(28, 20250618, "WR").
out_assigned(29, 20250618, "WR").
out_assigned(30, 20250618, "D").
out_assigned(31, 20250618, "D").
out_assigned(32, 20250618, "D").
out_assigned(1, 20250619, "D").
out_assigned(17, 20250619, "D").
out_assigned(18, 20250619, "D").
out_assigned(19, 20250619, "SE").
out_assigned(20, 20250619, "WR").
out_assigned(21, 20250619, "D").
out_assigned(22, 20250619, "WR").
out_assigned(23, 20250619, "LD").
out_assigned(24, 20250619, "SE").
out_assigned(25, 20250619, "SN").
out_assigned(26, 20250619, "SN").
out_assigned(27, 20250619, "WR").
out_assigned(28, 20250619, "D").
out_assigned(29, 20250619, "LD").
out_assigned(1, 20250620, "D").
out_assigned(5, 20250620, "D").
out_assigned(6, 20250620, "WR").
out_assigned(7, 20250620, "D").
out_assigned(17, 20250620, "D").
out_assigned(18, 20250620, "D").
out_assigned(19, 20250620, "SN").
out_assigned(20, 20250620, "D").
out_assigned(21, 20250620, "WR").
out_assigned(22, 20250620, "LD").
out_assigned(23, 20250620, "SE").
out_assigned(24, 20250620, "SN").
out_assigned(25, 20250620, "WR").
out_assigned(26, 20250620, "WR").
out_assigned(27, 20250620, "WR").
out_assigned(28, 20250620, "LD").
out_assigned(29, 20250620, "SE").
out_assigned(1, 20250621, "WR").
out_assigned(6, 20250621, "AL").
out_assigned(17, 20250621, "D").
out_assigned(18, 20250621, "LD").
out_assigned(19, 20250621, "WR").
out_assigned(20, 20250621, "WR").
out_assigned(21, 20250621, "D").
out_assigned(22, 20250621, "SE").
out_assigned(23, 20250621, "SN").
out_assigned(24, 20250621, "WR").
out_assigned(25, 20250621, "LD").
out_assigned(26, 20250621, "WR").
out_assigned(27, 20250621, "WR").
out_assigned(28, 20250621, "SE").
out_assigned(29, 20250621, "SN").
out_pos_request(1, 20250708, "D").
out_pos_request(1, 20250716, "D").
out_pos_request(2, 20250701, "D").
out_pos_request(2, 20250708, "D").
out_pos_request(3, 20250623, "D").
out_pos_request(4, 20250708, "WR").
out_pos_request(4, 20250709, "AL").
out_pos_request(5, 20250701, "HL").
out_pos_request(5, 20250702, "HL").
out_pos_request(5, 20250703, "AL").
out_pos_request(5, 20250704, "AL").
out_pos_request(5, 20250717, "D").
out_pos_request(6, 20250703, "D").
out_pos_request(6, 20250716, "WR").
out_pos_request(6, 20250717, "WR").
out_pos_request(6, 20250720, "WR").
out_pos_request(6, 20250721, "WR").
out_pos_request(7, 20250628, "WR").
out_pos_request(7, 20250629, "WR").
out_pos_request(7, 20250705, "HL").
out_pos_request(7, 20250706, "AL").
out_pos_request(7, 20250711, "D").
out_pos_request(7, 20250719, "WR").
out_pos_request(8, 20250623, "D").
out_pos_request(8, 20250711, "D").
out_pos_request(10, 20250628, "WR").
out_pos_request(10, 20250629, "HL").
out_pos_request(10, 20250716, "WR").
out_pos_request(10, 20250717, "WR").
out_pos_request(10, 20250723, "HL").
out_pos_request(10, 20250724, "AL").
out_pos_request(10, 20250725, "AL").
out_pos_request(10, 20250726, "WR").
out_pos_request(11, 20250625, "D").
out_pos_request(11, 20250626, "WR").
out_pos_request(11, 20250702, "WR").
out_pos_request(12, 20250702, "WR").
out_pos_request(12, 20250707, "D").
out_pos_request(12, 20250708, "WR").
out_pos_request(12, 20250709, "HL").
out_pos_request(12, 20250710, "HL").
out_pos_request(12, 20250712, "WR").
out_pos_request(13, 20250709, "D").
out_pos_request(13, 20250715, "WR").
out_pos_request(13, 20250716, "WR").
out_pos_request(13, 20250723, "AL").
out_pos_request(13, 20250724, "WR").
out_pos_request(13, 20250725, "WR").
out_pos_request(13, 20250726, "AL").
out_pos_request(14, 20250626, "D").
out_pos_request(14, 20250724, "D").
out_pos_request(17, 20250623, "D").
out_pos_request(17, 20250625, "D").
out_pos_request(17, 20250708, "D").
out_pos_request(17, 20250709, "D").
out_pos_request(17, 20250711, "D").
out_pos_request(17, 20250723, "D").
out_pos_request(17, 20250725, "WR").
out_pos_request(17, 20250726, "AL").
out_pos_request(19, 20250624, "D").
out_pos_request(19, 20250717, "D").
out_pos_request(19, 20250719, "WR").
out_pos_request(19, 20250720, "WR").
out_pos_request(19, 20250721, "PH").
out_pos_request(19, 20250722, "D").
out_pos_request(20, 20250623, "D").
out_pos_request(21, 20250703, "WR").
out_pos_request(21, 20250708, "WR").
out_pos_request(21, 20250709, "WR").
out_pos_request(22, 20250623, "D").
out_pos_request(22, 20250711, "D").
out_pos_request(22, 20250716, "HL").
out_pos_request(24, 20250704, "D").
out_pos_request(25, 20250627, "D").
out_pos_request(25, 20250628, "WR").
out_pos_request(25, 20250629, "WR").
out_pos_request(25, 20250701, "WR").
out_pos_request(26, 20250626, "D").
out_pos_request(26, 20250711, "D").
out_pos_request(26, 20250717, "WR").
out_pos_request(26, 20250718, "AL").
out_pos_request(26, 20250719, "WR").
out_pos_request(26, 20250720, "WR").
out_pos_request(26, 20250724, "D").
out_pos_request(27, 20250622, "WR").
out_pos_request(27, 20250704, "D").
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
staff(1, "深沢泉", "Nurse", "0192253", 0).
staff(2, "長沼みづき", "Nurse", "0055030", 0).
staff(3, "中込結", "Nurse", "0312931", 0).
staff(4, "丹澤いず美", "Nurse", "0237017", 0).
staff(5, "田口沙紀", "Nurse", "0301733", 0).
staff(6, "長田七彩", "Nurse", "0370203", 0).
staff(7, "竹内七菜", "Nurse", "0384528", 0).
staff(8, "小原莉央", "Nurse", "0384368", 0).
staff(9, "杉山ちはる", "Nurse", "0400254", 0).
staff(10, "今沢優希", "Nurse", "0400221", 0).
staff(11, "中嶋千智", "Nurse", "0418973", 0).
staff(12, "北村莉々", "Nurse", "0418995", 0).
staff(13, "奥脇香子", "Nurse", "0372342", 0).
staff(14, "駒井渚", "Nurse", "0469508", 0).
staff(15, "遠藤遥香", "Nurse", "0606151", 0).
staff(16, "岡本莉里衣", "Nurse", "0606173", 0).
staff(17, "三枝栄江", "Nurse", "0019822", 0).
staff(18, "渡邉裕美", "Nurse", "0150172", 0).
staff(19, "山寺智美", "Nurse", "0312760", 0).
staff(20, "岡本こずえ", "Nurse", "0355317", 0).
staff(21, "小林利里愛", "Nurse", "0384448", 0).
staff(22, "長谷川千華", "Nurse", "0329441", 0).
staff(23, "初鹿春花", "Nurse", "0384596", 0).
staff(24, "福田優花", "Nurse", "0400152", 0).
staff(25, "長嶺世奈", "Nurse", "0399696", 0).
staff(26, "名取遥", "Nurse", "0418962", 0).
staff(27, "望月夢音香", "Nurse", "0418940", 0).
staff(28, "田中理子", "Nurse", "0469576", 0).
staff(29, "麻川のん子", "Nurse", "0469372", 0).
staff(30, "太田真叶", "Nurse", "0606162", 0).
staff(31, "安藤優李", "Nurse", "0606116", 0).
staff(32, "平山綾乃", "Nurse", "0606548", 0).
out_assigned(2, 20250622, "LD").
out_assigned(5, 20250622, "SE").
out_assigned(6, 20250622, "SN").
out_assigned(9, 20250622, "SE").
out_assigned(10, 20250622, "LD").
out_assigned(12, 20250622, "D").
out_assigned(14, 20250622, "LD").
out_assigned(15, 20250622, "D").
out_assigned(18, 20250622, "SE").
out_assigned(19, 20250622, "SN").
out_assigned(20, 20250622, "D").
out_assigned(22, 20250622, "SN").
out_assigned(24, 20250622, "D").
out_assigned(25, 20250622, "SE").
out_assigned(26, 20250622, "LD").
out_assigned(28, 20250622, "SN").
out_assigned(27, 20250622, "WR").
out_assigned(1, 20250622, "WR").
out_assigned(3, 20250622, "WR").
out_assigned(4, 20250622, "WR").
out_assigned(7, 20250622, "WR").
out_assigned(8, 20250622, "WR").
out_assigned(11, 20250622, "WR").
out_assigned(13, 20250622, "WR").
out_assigned(16, 20250622, "WR").
out_assigned(17, 20250622, "WR").
out_assigned(21, 20250622, "WR").
out_assigned(23, 20250622, "WR").
out_assigned(29, 20250622, "WR").
out_assigned(30, 20250622, "WR").
out_assigned(31, 20250622, "WR").
out_assigned(32, 20250622, "WR").
out_assigned(1, 20250623, "LD").
out_assigned(2, 20250623, "SE").
out_assigned(3, 20250623, "D").
out_assigned(5, 20250623, "SN").
out_assigned(7, 20250623, "LD").
out_assigned(8, 20250623, "D").
out_assigned(9, 20250623, "SN").
out_assigned(10, 20250623, "SE").
out_assigned(12, 20250623, "D").
out_assigned(13, 20250623, "LD").
out_assigned(14, 20250623, "SE").
out_assigned(15, 20250623, "D").
out_assigned(17, 20250623, "D").
out_assigned(18, 20250623, "SN").
out_assigned(20, 20250623, "D").
out_assigned(21, 20250623, "LD").
out_assigned(24, 20250623, "D").
out_assigned(25, 20250623, "SN").
out_assigned(26, 20250623, "SE").
out_assigned(29, 20250623, "D").
out_assigned(4, 20250623, "WR").
out_assigned(6, 20250623, "WR").
out_assigned(11, 20250623, "WR").
out_assigned(16, 20250623, "WR").
out_assigned(19, 20250623, "WR").
out_assigned(22, 20250623, "WR").
out_assigned(23, 20250623, "WR").
out_assigned(27, 20250623, "WR").
out_assigned(28, 20250623, "WR").
out_assigned(30, 20250623, "WR").
out_assigned(31, 20250623, "WR").
out_assigned(32, 20250623, "WR").
out_assigned(1, 20250624, "SE").
out_assigned(2, 20250624, "SN").
out_assigned(3, 20250624, "LD").
out_assigned(4, 20250624, "D").
out_assigned(7, 20250624, "SE").
out_assigned(8, 20250624, "LD").
out_assigned(10, 20250624, "SN").
out_assigned(13, 20250624, "SE").
out_assigned(14, 20250624, "SN").
out_assigned(15, 20250624, "D").
out_assigned(16, 20250624, "D").
out_assigned(19, 20250624, "D").
out_assigned(20, 20250624, "LD").
out_assigned(21, 20250624, "SE").
out_assigned(24, 20250624, "LD").
out_assigned(26, 20250624, "SN").
out_assigned(27, 20250624, "D").
out_assigned(29, 20250624, "D").
out_assigned(30, 20250624, "D").
out_assigned(31, 20250624, "D").
out_assigned(32, 20250624, "D").
out_assigned(5, 20250624, "WR").
out_assigned(6, 20250624, "WR").
out_assigned(9, 20250624, "WR").
out_assigned(11, 20250624, "WR").
out_assigned(12, 20250624, "WR").
out_assigned(17, 20250624, "WR").
out_assigned(18, 20250624, "WR").
out_assigned(22, 20250624, "WR").
out_assigned(23, 20250624, "WR").
out_assigned(25, 20250624, "WR").
out_assigned(28, 20250624, "WR").
out_assigned(1, 20250625, "SN").
out_assigned(3, 20250625, "SE").
out_assigned(6, 20250625, "D").
out_assigned(7, 20250625, "SN").
out_assigned(8, 20250625, "SE").
out_assigned(11, 20250625, "D").
out_assigned(12, 20250625, "LD").
out_assigned(13, 20250625, "SN").
out_assigned(17, 20250625, "D").
out_assigned(18, 20250625, "D").
out_assigned(19, 20250625, "LD").
out_assigned(20, 20250625, "SE").
out_assigned(21, 20250625, "SN").
out_assigned(23, 20250625, "D").
out_assigned(24, 20250625, "SE").
out_assigned(25, 20250625, "D").
out_assigned(27, 20250625, "D").
out_assigned(28, 20250625, "D").
out_assigned(29, 20250625, "LD").
out_assigned(30, 20250625, "LD").
out_assigned(31, 20250625, "D").
out_assigned(32, 20250625, "D").
out_assigned(2, 20250625, "WR").
out_assigned(4, 20250625, "WR").
out_assigned(5, 20250625, "WR").
out_assigned(9, 20250625, "WR").
out_assigned(10, 20250625, "WR").
out_assigned(14, 20250625, "WR").
out_assigned(15, 20250625, "WR").
out_assigned(16, 20250625, "WR").
out_assigned(22, 20250625, "WR").
out_assigned(26, 20250625, "WR").
out_assigned(2, 20250626, "LD").
out_assigned(3, 20250626, "SN").
out_assigned(6, 20250626, "LD").
out_assigned(8, 20250626, "SN").
out_assigned(10, 20250626, "D").
out_assigned(12, 20250626, "SE").
out_assigned(14, 20250626, "D").
out_assigned(15, 20250626, "D").
out_assigned(16, 20250626, "D").
out_assigned(17, 20250626, "D").
out_assigned(18, 20250626, "D").
out_assigned(19, 20250626, "SE").
out_assigned(20, 20250626, "SN").
out_assigned(22, 20250626, "LD").
out_assigned(24, 20250626, "SN").
out_assigned(25, 20250626, "D").
out_assigned(26, 20250626, "D").
out_assigned(27, 20250626, "D").
out_assigned(28, 20250626, "D").
out_assigned(29, 20250626, "SE").
out_assigned(30, 20250626, "SE").
out_assigned(32, 20250626, "LD").
out_assigned(11, 20250626, "WR").
out_assigned(1, 20250626, "WR").
out_assigned(4, 20250626, "WR").
out_assigned(5, 20250626, "WR").
out_assigned(7, 20250626, "WR").
out_assigned(9, 20250626, "WR").
out_assigned(13, 20250626, "WR").
out_assigned(21, 20250626, "WR").
out_assigned(23, 20250626, "WR").
out_assigned(31, 20250626, "WR").
out_assigned(1, 20250627, "LD").
out_assigned(2, 20250627, "SE").
out_assigned(4, 20250627, "D").
out_assigned(6, 20250627, "SE").
out_assigned(7, 20250627, "D").
out_assigned(9, 20250627, "LD").
out_assigned(10, 20250627, "D").
out_assigned(11, 20250627, "D").
out_assigned(12, 20250627, "SN").
out_assigned(15, 20250627, "D").
out_assigned(16, 20250627, "D").
out_assigned(17, 20250627, "LD").
out_assigned(18, 20250627, "D").
out_assigned(19, 20250627, "SN").
out_assigned(21, 20250627, "LD").
out_assigned(22, 20250627, "SE").
out_assigned(23, 20250627, "D").
out_assigned(25, 20250627, "D").
out_assigned(28, 20250627, "D").
out_assigned(29, 20250627, "SN").
out_assigned(30, 20250627, "SN").
out_assigned(31, 20250627, "D").
out_assigned(32, 20250627, "SE").
out_assigned(3, 20250627, "WR").
out_assigned(5, 20250627, "WR").
out_assigned(8, 20250627, "WR").
out_assigned(13, 20250627, "WR").
out_assigned(14, 20250627, "WR").
out_assigned(20, 20250627, "WR").
out_assigned(24, 20250627, "WR").
out_assigned(26, 20250627, "WR").
out_assigned(27, 20250627, "WR").
out_assigned(1, 20250628, "SE").
out_assigned(2, 20250628, "SN").
out_assigned(4, 20250628, "D").
out_assigned(5, 20250628, "LD").
out_assigned(6, 20250628, "SN").
out_assigned(9, 20250628, "SE").
out_assigned(13, 20250628, "LD").
out_assigned(15, 20250628, "D").
out_assigned(16, 20250628, "D").
out_assigned(17, 20250628, "SE").
out_assigned(20, 20250628, "D").
out_assigned(21, 20250628, "SE").
out_assigned(22, 20250628, "SN").
out_assigned(27, 20250628, "LD").
out_assigned(28, 20250628, "LD").
out_assigned(32, 20250628, "SN").
out_assigned(7, 20250628, "WR").
out_assigned(10, 20250628, "WR").
out_assigned(25, 20250628, "WR").
out_assigned(3, 20250628, "WR").
out_assigned(8, 20250628, "WR").
out_assigned(11, 20250628, "WR").
out_assigned(12, 20250628, "WR").
out_assigned(14, 20250628, "WR").
out_assigned(18, 20250628, "WR").
out_assigned(19, 20250628, "WR").
out_assigned(23, 20250628, "WR").
out_assigned(24, 20250628, "WR").
out_assigned(26, 20250628, "WR").
out_assigned(29, 20250628, "WR").
out_assigned(30, 20250628, "WR").
out_assigned(31, 20250628, "WR").
out_assigned(1, 20250629, "SN").
out_assigned(5, 20250629, "SE").
out_assigned(9, 20250629, "SN").
out_assigned(11, 20250629, "LD").
out_assigned(12, 20250629, "LD").
out_assigned(13, 20250629, "SE").
out_assigned(14, 20250629, "LD").
out_assigned(17, 20250629, "SN").
out_assigned(19, 20250629, "LD").
out_assigned(21, 20250629, "SN").
out_assigned(23, 20250629, "D").
out_assigned(26, 20250629, "D").
out_assigned(27, 20250629, "SE").
out_assigned(28, 20250629, "SE").
out_assigned(29, 20250629, "D").
out_assigned(30, 20250629, "D").
out_assigned(7, 20250629, "WR").
out_assigned(10, 20250629, "HL").
out_assigned(25, 20250629, "WR").
out_assigned(2, 20250629, "WR").
out_assigned(3, 20250629, "WR").
out_assigned(4, 20250629, "WR").
out_assigned(6, 20250629, "WR").
out_assigned(8, 20250629, "WR").
out_assigned(15, 20250629, "WR").
out_assigned(16, 20250629, "WR").
out_assigned(18, 20250629, "WR").
out_assigned(20, 20250629, "WR").
out_assigned(22, 20250629, "WR").
out_assigned(24, 20250629, "WR").
out_assigned(31, 20250629, "WR").
out_assigned(32, 20250629, "WR").
out_assigned(3, 20250630, "D").
out_assigned(5, 20250630, "SN").
out_assigned(6, 20250630, "D").
out_assigned(8, 20250630, "LD").
out_assigned(10, 20250630, "LD").
out_assigned(11, 20250630, "SE").
out_assigned(12, 20250630, "SE").
out_assigned(13, 20250630, "SN").
out_assigned(14, 20250630, "SE").
out_assigned(18, 20250630, "LD").
out_assigned(19, 20250630, "SE").
out_assigned(20, 20250630, "LD").
out_assigned(22, 20250630, "D").
out_assigned(23, 20250630, "D").
out_assigned(24, 20250630, "LD").
out_assigned(25, 20250630, "D").
out_assigned(26, 20250630, "D").
out_assigned(27, 20250630, "SN").
out_assigned(28, 20250630, "SN").
out_assigned(31, 20250630, "D").
out_assigned(32, 20250630, "D").
out_assigned(1, 20250630, "WR").
out_assigned(2, 20250630, "WR").
out_assigned(4, 20250630, "WR").
out_assigned(7, 20250630, "WR").
out_assigned(9, 20250630, "WR").
out_assigned(15, 20250630, "WR").
out_assigned(16, 20250630, "WR").
out_assigned(17, 20250630, "WR").
out_assigned(21, 20250630, "WR").
out_assigned(29, 20250630, "WR").
out_assigned(30, 20250630, "WR").
out_assigned(1, 20250701, "LD").
out_assigned(2, 20250701, "D").
out_assigned(3, 20250701, "D").
out_assigned(4, 20250701, "D").
out_assigned(7, 20250701, "D").
out_assigned(8, 20250701, "SE").
out_assigned(9, 20250701, "LD").
out_assigned(10, 20250701, "SE").
out_assigned(11, 20250701, "SN").
out_assigned(12, 20250701, "SN").
out_assigned(14, 20250701, "SN").
out_assigned(15, 20250701, "LD").
out_assigned(16, 20250701, "D").
out_assigned(17, 20250701, "D").
out_assigned(18, 20250701, "SE").
out_assigned(19, 20250701, "SN").
out_assigned(20, 20250701, "SE").
out_assigned(23, 20250701, "D").
out_assigned(24, 20250701, "SE").
out_assigned(26, 20250701, "LD").
out_assigned(30, 20250701, "D").
out_assigned(5, 20250701, "HL").
out_assigned(25, 20250701, "WR").
out_assigned(6, 20250701, "WR").
out_assigned(13, 20250701, "WR").
out_assigned(21, 20250701, "WR").
out_assigned(22, 20250701, "WR").
out_assigned(27, 20250701, "WR").
out_assigned(28, 20250701, "WR").
out_assigned(29, 20250701, "WR").
out_assigned(31, 20250701, "WR").
out_assigned(32, 20250701, "WR").
out_assigned(1, 20250702, "SE").
out_assigned(2, 20250702, "LD").
out_assigned(3, 20250702, "D").
out_assigned(4, 20250702, "D").
out_assigned(6, 20250702, "D").
out_assigned(7, 20250702, "D").
out_assigned(8, 20250702, "SN").
out_assigned(9, 20250702, "SE").
out_assigned(10, 20250702, "SN").
out_assigned(13, 20250702, "D").
out_assigned(15, 20250702, "SE").
out_assigned(18, 20250702, "SN").
out_assigned(20, 20250702, "SN").
out_assigned(21, 20250702, "D").
out_assigned(22, 20250702, "LD").
out_assigned(24, 20250702, "SN").
out_assigned(26, 20250702, "SE").
out_assigned(27, 20250702, "D").
out_assigned(28, 20250702, "LD").
out_assigned(29, 20250702, "D").
out_assigned(31, 20250702, "LD").
out_assigned(32, 20250702, "D").
out_assigned(5, 20250702, "HL").
out_assigned(11, 20250702, "WR").
out_assigned(12, 20250702, "WR").
out_assigned(14, 20250702, "WR").
out_assigned(16, 20250702, "WR").
out_assigned(17, 20250702, "WR").
out_assigned(19, 20250702, "WR").
out_assigned(23, 20250702, "WR").
out_assigned(25, 20250702, "WR").
out_assigned(30, 20250702, "WR").
out_assigned(1, 20250703, "SN").
out_assigned(2, 20250703, "SE").
out_assigned(3, 20250703, "D").
out_assigned(6, 20250703, "D").
out_assigned(9, 20250703, "SN").
out_assigned(11, 20250703, "D").
out_assigned(12, 20250703, "D").
out_assigned(13, 20250703, "LD").
out_assigned(14, 20250703, "D").
out_assigned(15, 20250703, "SN").
out_assigned(16, 20250703, "D").
out_assigned(19, 20250703, "LD").
out_assigned(22, 20250703, "SE").
out_assigned(23, 20250703, "D").
out_assigned(25, 20250703, "LD").
out_assigned(26, 20250703, "SN").
out_assigned(27, 20250703, "D").
out_assigned(28, 20250703, "SE").
out_assigned(29, 20250703, "LD").
out_assigned(31, 20250703, "SE").
out_assigned(5, 20250703, "AL").
out_assigned(21, 20250703, "WR").
out_assigned(4, 20250703, "WR").
out_assigned(7, 20250703, "WR").
out_assigned(8, 20250703, "WR").
out_assigned(10, 20250703, "WR").
out_assigned(17, 20250703, "WR").
out_assigned(18, 20250703, "WR").
out_assigned(20, 20250703, "WR").
out_assigned(24, 20250703, "WR").
out_assigned(30, 20250703, "WR").
out_assigned(32, 20250703, "WR").
out_assigned(2, 20250704, "SN").
out_assigned(6, 20250704, "LD").
out_assigned(7, 20250704, "D").
out_assigned(8, 20250704, "D").
out_assigned(11, 20250704, "LD").
out_assigned(12, 20250704, "D").
out_assigned(13, 20250704, "SE").
out_assigned(14, 20250704, "D").
out_assigned(16, 20250704, "LD").
out_assigned(17, 20250704, "LD").
out_assigned(18, 20250704, "D").
out_assigned(19, 20250704, "SE").
out_assigned(20, 20250704, "D").
out_assigned(21, 20250704, "D").
out_assigned(22, 20250704, "SN").
out_assigned(24, 20250704, "D").
out_assigned(25, 20250704, "SE").
out_assigned(27, 20250704, "D").
out_assigned(28, 20250704, "SN").
out_assigned(29, 20250704, "SE").
out_assigned(30, 20250704, "D").
out_assigned(31, 20250704, "SN").
out_assigned(5, 20250704, "AL").
out_assigned(1, 20250704, "WR").
out_assigned(3, 20250704, "WR").
out_assigned(4, 20250704, "WR").
out_assigned(9, 20250704, "WR").
out_assigned(10, 20250704, "WR").
out_assigned(15, 20250704, "WR").
out_assigned(23, 20250704, "WR").
out_assigned(26, 20250704, "WR").
out_assigned(32, 20250704, "WR").
out_assigned(1, 20250705, "D").
out_assigned(3, 20250705, "LD").
out_assigned(4, 20250705, "LD").
out_assigned(6, 20250705, "SE").
out_assigned(9, 20250705, "D").
out_assigned(10, 20250705, "D").
out_assigned(11, 20250705, "SE").
out_assigned(13, 20250705, "SN").
out_assigned(16, 20250705, "SE").
out_assigned(17, 20250705, "SE").
out_assigned(18, 20250705, "LD").
out_assigned(19, 20250705, "SN").
out_assigned(20, 20250705, "D").
out_assigned(23, 20250705, "LD").
out_assigned(25, 20250705, "SN").
out_assigned(27, 20250705, "LD").
out_assigned(29, 20250705, "SN").
out_assigned(7, 20250705, "HL").
out_assigned(2, 20250705, "WR").
out_assigned(5, 20250705, "WR").
out_assigned(8, 20250705, "WR").
out_assigned(12, 20250705, "WR").
out_assigned(14, 20250705, "WR").
out_assigned(15, 20250705, "WR").
out_assigned(21, 20250705, "WR").
out_assigned(22, 20250705, "WR").
out_assigned(24, 20250705, "WR").
out_assigned(26, 20250705, "WR").
out_assigned(28, 20250705, "WR").
out_assigned(30, 20250705, "WR").
out_assigned(31, 20250705, "WR").
out_assigned(32, 20250705, "WR").
out_assigned(2, 20250706, "D").
out_assigned(3, 20250706, "SE").
out_assigned(4, 20250706, "SE").
out_assigned(5, 20250706, "LD").
out_assigned(6, 20250706, "SN").
out_assigned(11, 20250706, "SN").
out_assigned(14, 20250706, "LD").
out_assigned(16, 20250706, "SN").
out_assigned(17, 20250706, "SN").
out_assigned(18, 20250706, "SE").
out_assigned(21, 20250706, "D").
out_assigned(22, 20250706, "LD").
out_assigned(23, 20250706, "SE").
out_assigned(24, 20250706, "LD").
out_assigned(26, 20250706, "D").
out_assigned(27, 20250706, "SE").
out_assigned(30, 20250706, "D").
out_assigned(7, 20250706, "AL").
out_assigned(1, 20250706, "WR").
out_assigned(8, 20250706, "WR").
out_assigned(9, 20250706, "WR").
out_assigned(10, 20250706, "WR").
out_assigned(12, 20250706, "WR").
out_assigned(13, 20250706, "WR").
out_assigned(15, 20250706, "WR").
out_assigned(19, 20250706, "WR").
out_assigned(20, 20250706, "WR").
out_assigned(25, 20250706, "WR").
out_assigned(28, 20250706, "WR").
out_assigned(29, 20250706, "WR").
out_assigned(31, 20250706, "WR").
out_assigned(32, 20250706, "WR").
out_assigned(3, 20250707, "SN").
out_assigned(4, 20250707, "SN").
out_assigned(5, 20250707, "SE").
out_assigned(7, 20250707, "LD").
out_assigned(8, 20250707, "D").
out_assigned(10, 20250707, "D").
out_assigned(12, 20250707, "D").
out_assigned(13, 20250707, "D").
out_assigned(14, 20250707, "SE").
out_assigned(15, 20250707, "D").
out_assigned(18, 20250707, "SN").
out_assigned(19, 20250707, "LD").
out_assigned(21, 20250707, "D").
out_assigned(22, 20250707, "SE").
out_assigned(23, 20250707, "SN").
out_assigned(24, 20250707, "SE").
out_assigned(25, 20250707, "D").
out_assigned(26, 20250707, "LD").
out_assigned(27, 20250707, "SN").
out_assigned(28, 20250707, "D").
out_assigned(29, 20250707, "LD").
out_assigned(30, 20250707, "D").
out_assigned(31, 20250707, "D").
out_assigned(1, 20250707, "WR").
out_assigned(2, 20250707, "WR").
out_assigned(6, 20250707, "WR").
out_assigned(9, 20250707, "WR").
out_assigned(11, 20250707, "WR").
out_assigned(16, 20250707, "WR").
out_assigned(17, 20250707, "WR").
out_assigned(20, 20250707, "WR").
out_assigned(32, 20250707, "WR").
out_assigned(1, 20250708, "D").
out_assigned(2, 20250708, "D").
out_assigned(5, 20250708, "SN").
out_assigned(6, 20250708, "LD").
out_assigned(7, 20250708, "SE").
out_assigned(8, 20250708, "D").
out_assigned(10, 20250708, "D").
out_assigned(11, 20250708, "LD").
out_assigned(14, 20250708, "SN").
out_assigned(15, 20250708, "D").
out_assigned(16, 20250708, "D").
out_assigned(17, 20250708, "D").
out_assigned(19, 20250708, "SE").
out_assigned(20, 20250708, "LD").
out_assigned(22, 20250708, "SN").
out_assigned(24, 20250708, "SN").
out_assigned(25, 20250708, "D").
out_assigned(26, 20250708, "SE").
out_assigned(29, 20250708, "SE").
out_assigned(30, 20250708, "D").
out_assigned(31, 20250708, "LD").
out_assigned(4, 20250708, "WR").
out_assigned(12, 20250708, "WR").
out_assigned(21, 20250708, "WR").
out_assigned(3, 20250708, "WR").
out_assigned(9, 20250708, "WR").
out_assigned(13, 20250708, "WR").
out_assigned(18, 20250708, "WR").
out_assigned(23, 20250708, "WR").
out_assigned(27, 20250708, "WR").
out_assigned(28, 20250708, "WR").
out_assigned(32, 20250708, "WR").
out_assigned(1, 20250709, "D").
out_assigned(2, 20250709, "D").
out_assigned(3, 20250709, "LD").
out_assigned(6, 20250709, "SE").
out_assigned(7, 20250709, "SN").
out_assigned(11, 20250709, "SE").
out_assigned(13, 20250709, "D").
out_assigned(15, 20250709, "D").
out_assigned(17, 20250709, "D").
out_assigned(18, 20250709, "LD").
out_assigned(19, 20250709, "SN").
out_assigned(20, 20250709, "SE").
out_assigned(23, 20250709, "D").
out_assigned(25, 20250709, "LD").
out_assigned(26, 20250709, "SN").
out_assigned(27, 20250709, "LD").
out_assigned(28, 20250709, "D").
out_assigned(29, 20250709, "SN").
out_assigned(30, 20250709, "D").
out_assigned(31, 20250709, "SE").
out_assigned(32, 20250709, "D").
out_assigned(4, 20250709, "AL").
out_assigned(12, 20250709, "HL").
out_assigned(21, 20250709, "WR").
out_assigned(5, 20250709, "WR").
out_assigned(8, 20250709, "WR").
out_assigned(9, 20250709, "WR").
out_assigned(10, 20250709, "WR").
out_assigned(14, 20250709, "WR").
out_assigned(16, 20250709, "WR").
out_assigned(22, 20250709, "WR").
out_assigned(24, 20250709, "WR").
out_assigned(1, 20250710, "LD").
out_assigned(3, 20250710, "SE").
out_assigned(4, 20250710, "D").
out_assigned(5, 20250710, "D").
out_assigned(6, 20250710, "SN").
out_assigned(8, 20250710, "D").
out_assigned(9, 20250710, "LD").
out_assigned(10, 20250710, "D").
out_assigned(11, 20250710, "SN").
out_assigned(16, 20250710, "D").
out_assigned(18, 20250710, "SE").
out_assigned(20, 20250710, "SN").
out_assigned(21, 20250710, "D").
out_assigned(22, 20250710, "D").
out_assigned(23, 20250710, "LD").
out_assigned(24, 20250710, "LD").
out_assigned(25, 20250710, "SE").
out_assigned(27, 20250710, "SE").
out_assigned(28, 20250710, "LD").
out_assigned(31, 20250710, "SN").
out_assigned(32, 20250710, "D").
out_assigned(12, 20250710, "HL").
out_assigned(2, 20250710, "WR").
out_assigned(7, 20250710, "WR").
out_assigned(13, 20250710, "WR").
out_assigned(14, 20250710, "WR").
out_assigned(15, 20250710, "WR").
out_assigned(17, 20250710, "WR").
out_assigned(19, 20250710, "WR").
out_assigned(26, 20250710, "WR").
out_assigned(29, 20250710, "WR").
out_assigned(30, 20250710, "WR").
out_assigned(1, 20250711, "SE").
out_assigned(2, 20250711, "D").
out_assigned(3, 20250711, "SN").
out_assigned(4, 20250711, "D").
out_assigned(7, 20250711, "D").
out_assigned(8, 20250711, "D").
out_assigned(9, 20250711, "SE").
out_assigned(14, 20250711, "D").
out_assigned(15, 20250711, "D").
out_assigned(17, 20250711, "D").
out_assigned(18, 20250711, "SN").
out_assigned(19, 20250711, "LD").
out_assigned(21, 20250711, "LD").
out_assigned(22, 20250711, "D").
out_assigned(23, 20250711, "SE").
out_assigned(24, 20250711, "SE").
out_assigned(25, 20250711, "SN").
out_assigned(26, 20250711, "D").
out_assigned(27, 20250711, "SN").
out_assigned(28, 20250711, "SE").
out_assigned(29, 20250711, "LD").
out_assigned(30, 20250711, "LD").
out_assigned(32, 20250711, "D").
out_assigned(5, 20250711, "WR").
out_assigned(6, 20250711, "WR").
out_assigned(10, 20250711, "WR").
out_assigned(11, 20250711, "WR").
out_assigned(12, 20250711, "WR").
out_assigned(13, 20250711, "WR").
out_assigned(16, 20250711, "WR").
out_assigned(20, 20250711, "WR").
out_assigned(31, 20250711, "WR").
out_assigned(1, 20250712, "SN").
out_assigned(2, 20250712, "D").
out_assigned(5, 20250712, "D").
out_assigned(6, 20250712, "LD").
out_assigned(7, 20250712, "LD").
out_assigned(8, 20250712, "LD").
out_assigned(9, 20250712, "SN").
out_assigned(13, 20250712, "LD").
out_assigned(15, 20250712, "D").
out_assigned(17, 20250712, "LD").
out_assigned(19, 20250712, "SE").
out_assigned(21, 20250712, "SE").
out_assigned(23, 20250712, "SN").
out_assigned(24, 20250712, "SN").
out_assigned(28, 20250712, "SN").
out_assigned(29, 20250712, "SE").
out_assigned(30, 20250712, "SE").
out_assigned(12, 20250712, "WR").
out_assigned(3, 20250712, "WR").
out_assigned(4, 20250712, "WR").
out_assigned(10, 20250712, "WR").
out_assigned(11, 20250712, "WR").
out_assigned(14, 20250712, "WR").
out_assigned(16, 20250712, "WR").
out_assigned(18, 20250712, "WR").
out_assigned(20, 20250712, "WR").
out_assigned(22, 20250712, "WR").
out_assigned(25, 20250712, "WR").
out_assigned(26, 20250712, "WR").
out_assigned(27, 20250712, "WR").
out_assigned(31, 20250712, "WR").
out_assigned(32, 20250712, "WR").
out_assigned(2, 20250713, "D").
out_assigned(5, 20250713, "LD").
out_assigned(6, 20250713, "SE").
out_assigned(7, 20250713, "SE").
out_assigned(8, 20250713, "SE").
out_assigned(10, 20250713, "LD").
out_assigned(11, 20250713, "LD").
out_assigned(13, 20250713, "SE").
out_assigned(16, 20250713, "D").
out_assigned(17, 20250713, "SE").
out_assigned(19, 20250713, "SN").
out_assigned(20, 20250713, "D").
out_assigned(21, 20250713, "SN").
out_assigned(26, 20250713, "D").
out_assigned(27, 20250713, "LD").
out_assigned(29, 20250713, "SN").
out_assigned(30, 20250713, "SN").
out_assigned(1, 20250713, "WR").
out_assigned(3, 20250713, "WR").
out_assigned(4, 20250713, "WR").
out_assigned(9, 20250713, "WR").
out_assigned(12, 20250713, "WR").
out_assigned(14, 20250713, "WR").
out_assigned(15, 20250713, "WR").
out_assigned(18, 20250713, "WR").
out_assigned(22, 20250713, "WR").
out_assigned(23, 20250713, "WR").
out_assigned(24, 20250713, "WR").
out_assigned(25, 20250713, "WR").
out_assigned(28, 20250713, "WR").
out_assigned(31, 20250713, "WR").
out_assigned(32, 20250713, "WR").
out_assigned(3, 20250714, "LD").
out_assigned(5, 20250714, "SE").
out_assigned(6, 20250714, "SN").
out_assigned(7, 20250714, "SN").
out_assigned(8, 20250714, "SN").
out_assigned(9, 20250714, "D").
out_assigned(10, 20250714, "SE").
out_assigned(11, 20250714, "SE").
out_assigned(12, 20250714, "D").
out_assigned(13, 20250714, "SN").
out_assigned(16, 20250714, "LD").
out_assigned(17, 20250714, "SN").
out_assigned(18, 20250714, "D").
out_assigned(20, 20250714, "D").
out_assigned(22, 20250714, "D").
out_assigned(23, 20250714, "D").
out_assigned(25, 20250714, "LD").
out_assigned(26, 20250714, "LD").
out_assigned(27, 20250714, "SE").
out_assigned(28, 20250714, "D").
out_assigned(31, 20250714, "D").
out_assigned(32, 20250714, "D").
out_assigned(1, 20250714, "WR").
out_assigned(2, 20250714, "WR").
out_assigned(4, 20250714, "WR").
out_assigned(14, 20250714, "WR").
out_assigned(15, 20250714, "WR").
out_assigned(19, 20250714, "WR").
out_assigned(21, 20250714, "WR").
out_assigned(24, 20250714, "WR").
out_assigned(29, 20250714, "WR").
out_assigned(30, 20250714, "WR").
out_assigned(1, 20250715, "D").
out_assigned(3, 20250715, "SE").
out_assigned(4, 20250715, "D").
out_assigned(5, 20250715, "SN").
out_assigned(9, 20250715, "D").
out_assigned(10, 20250715, "SN").
out_assigned(11, 20250715, "SN").
out_assigned(12, 20250715, "LD").
out_assigned(14, 20250715, "LD").
out_assigned(15, 20250715, "D").
out_assigned(16, 20250715, "SE").
out_assigned(18, 20250715, "D").
out_assigned(20, 20250715, "LD").
out_assigned(21, 20250715, "D").
out_assigned(23, 20250715, "LD").
out_assigned(24, 20250715, "LD").
out_assigned(25, 20250715, "SE").
out_assigned(26, 20250715, "SE").
out_assigned(27, 20250715, "SN").
out_assigned(28, 20250715, "D").
out_assigned(32, 20250715, "D").
out_assigned(13, 20250715, "WR").
out_assigned(2, 20250715, "WR").
out_assigned(6, 20250715, "WR").
out_assigned(7, 20250715, "WR").
out_assigned(8, 20250715, "WR").
out_assigned(17, 20250715, "WR").
out_assigned(19, 20250715, "WR").
out_assigned(22, 20250715, "WR").
out_assigned(29, 20250715, "WR").
out_assigned(30, 20250715, "WR").
out_assigned(31, 20250715, "WR").
out_assigned(1, 20250716, "D").
out_assigned(2, 20250716, "LD").
out_assigned(3, 20250716, "SN").
out_assigned(4, 20250716, "D").
out_assigned(7, 20250716, "LD").
out_assigned(8, 20250716, "LD").
out_assigned(9, 20250716, "D").
out_assigned(12, 20250716, "SE").
out_assigned(14, 20250716, "SE").
out_assigned(15, 20250716, "LD").
out_assigned(16, 20250716, "SN").
out_assigned(18, 20250716, "D").
out_assigned(20, 20250716, "SE").
out_assigned(21, 20250716, "D").
out_assigned(23, 20250716, "SE").
out_assigned(24, 20250716, "SE").
out_assigned(25, 20250716, "SN").
out_assigned(26, 20250716, "SN").
out_assigned(28, 20250716, "D").
out_assigned(29, 20250716, "D").
out_assigned(30, 20250716, "D").
out_assigned(31, 20250716, "D").
out_assigned(6, 20250716, "WR").
out_assigned(10, 20250716, "WR").
out_assigned(13, 20250716, "WR").
out_assigned(22, 20250716, "HL").
out_assigned(5, 20250716, "WR").
out_assigned(11, 20250716, "WR").
out_assigned(17, 20250716, "WR").
out_assigned(19, 20250716, "WR").
out_assigned(27, 20250716, "WR").
out_assigned(32, 20250716, "WR").
out_assigned(1, 20250717, "LD").
out_assigned(2, 20250717, "SE").
out_assigned(4, 20250717, "D").
out_assigned(5, 20250717, "D").
out_assigned(7, 20250717, "SE").
out_assigned(8, 20250717, "SE").
out_assigned(9, 20250717, "D").
out_assigned(11, 20250717, "D").
out_assigned(12, 20250717, "SN").
out_assigned(13, 20250717, "LD").
out_assigned(14, 20250717, "SN").
out_assigned(15, 20250717, "SE").
out_assigned(17, 20250717, "LD").
out_assigned(18, 20250717, "D").
out_assigned(19, 20250717, "D").
out_assigned(20, 20250717, "SN").
out_assigned(21, 20250717, "D").
out_assigned(22, 20250717, "LD").
out_assigned(23, 20250717, "SN").
out_assigned(24, 20250717, "SN").
out_assigned(27, 20250717, "LD").
out_assigned(30, 20250717, "D").
out_assigned(31, 20250717, "D").
out_assigned(6, 20250717, "WR").
out_assigned(10, 20250717, "WR").
out_assigned(26, 20250717, "WR").
out_assigned(3, 20250717, "WR").
out_assigned(16, 20250717, "WR").
out_assigned(25, 20250717, "WR").
out_assigned(28, 20250717, "WR").
out_assigned(29, 20250717, "WR").
out_assigned(32, 20250717, "WR").
out_assigned(1, 20250718, "SE").
out_assigned(2, 20250718, "SN").
out_assigned(3, 20250718, "LD").
out_assigned(4, 20250718, "D").
out_assigned(5, 20250718, "D").
out_assigned(7, 20250718, "SN").
out_assigned(8, 20250718, "SN").
out_assigned(10, 20250718, "LD").
out_assigned(11, 20250718, "D").
out_assigned(13, 20250718, "SE").
out_assigned(15, 20250718, "SN").
out_assigned(16, 20250718, "D").
out_assigned(17, 20250718, "SE").
out_assigned(19, 20250718, "D").
out_assigned(21, 20250718, "LD").
out_assigned(22, 20250718, "SE").
out_assigned(25, 20250718, "D").
out_assigned(27, 20250718, "SE").
out_assigned(28, 20250718, "D").
out_assigned(29, 20250718, "LD").
out_assigned(31, 20250718, "D").
out_assigned(32, 20250718, "D").
out_assigned(26, 20250718, "AL").
out_assigned(6, 20250718, "WR").
out_assigned(9, 20250718, "WR").
out_assigned(12, 20250718, "WR").
out_assigned(14, 20250718, "WR").
out_assigned(18, 20250718, "WR").
out_assigned(20, 20250718, "WR").
out_assigned(23, 20250718, "WR").
out_assigned(24, 20250718, "WR").
out_assigned(30, 20250718, "WR").
out_assigned(1, 20250719, "SN").
out_assigned(3, 20250719, "SE").
out_assigned(4, 20250719, "D").
out_assigned(5, 20250719, "LD").
out_assigned(9, 20250719, "LD").
out_assigned(10, 20250719, "SE").
out_assigned(12, 20250719, "D").
out_assigned(13, 20250719, "SN").
out_assigned(14, 20250719, "D").
out_assigned(17, 20250719, "SN").
out_assigned(18, 20250719, "LD").
out_assigned(21, 20250719, "SE").
out_assigned(22, 20250719, "SN").
out_assigned(25, 20250719, "LD").
out_assigned(27, 20250719, "SN").
out_assigned(29, 20250719, "SE").
out_assigned(7, 20250719, "WR").
out_assigned(19, 20250719, "WR").
out_assigned(26, 20250719, "WR").
out_assigned(2, 20250719, "WR").
out_assigned(6, 20250719, "WR").
out_assigned(8, 20250719, "WR").
out_assigned(11, 20250719, "WR").
out_assigned(15, 20250719, "WR").
out_assigned(16, 20250719, "WR").
out_assigned(20, 20250719, "WR").
out_assigned(23, 20250719, "WR").
out_assigned(24, 20250719, "WR").
out_assigned(28, 20250719, "WR").
out_assigned(30, 20250719, "WR").
out_assigned(31, 20250719, "WR").
out_assigned(32, 20250719, "WR").
penalty(pred_shift(6, 20250622, "S"), table(6, 0), 1, 10).
penalty(pred_shift(19, 20250622, "S"), table(19, 0), 1, 10).
out_stats(num_consecutive_holidays(1, 0, 3, 3)).
out_stats(num_consecutive_holidays(2, 0, 2, 2)).
out_stats(num_consecutive_holidays(3, 0, 2, 2)).
out_stats(num_consecutive_holidays(4, 0, 4, 4)).
out_stats(num_consecutive_holidays(5, 0, 1, 1)).
out_stats(num_consecutive_holidays(6, 0, 2, 2)).
out_stats(num_consecutive_holidays(7, 0, 1, 1)).
out_stats(num_consecutive_holidays(8, 0, 2, 2)).
out_stats(num_consecutive_holidays(9, 0, 2, 2)).
out_stats(num_consecutive_holidays(10, 0, 3, 3)).
out_stats(num_consecutive_holidays(11, 0, 1, 1)).
out_stats(num_consecutive_holidays(12, 0, 1, 1)).
out_stats(num_consecutive_holidays(13, 0, 3, 3)).
out_stats(num_consecutive_holidays(14, 0, 3, 3)).
out_stats(num_consecutive_holidays(15, 0, 3, 3)).
out_stats(num_consecutive_holidays(16, 0, 2, 2)).
out_stats(num_consecutive_holidays(17, 0, 2, 2)).
out_stats(num_consecutive_holidays(18, 0, 2, 2)).
out_stats(num_consecutive_holidays(19, 0, 1, 1)).
out_stats(num_consecutive_holidays(20, 0, 3, 3)).
out_stats(num_consecutive_holidays(21, 0, 2, 2)).
out_stats(num_consecutive_holidays(22, 0, 2, 2)).
out_stats(num_consecutive_holidays(23, 0, 2, 2)).
out_stats(num_consecutive_holidays(24, 0, 3, 3)).
out_stats(num_consecutive_holidays(25, 0, 3, 3)).
out_stats(num_consecutive_holidays(26, 0, 2, 2)).
out_stats(num_consecutive_holidays(27, 0, 0, 0)).
out_stats(num_consecutive_holidays(28, 0, 2, 2)).
out_stats(num_consecutive_holidays(29, 0, 2, 2)).
out_stats(num_consecutive_holidays(30, 0, 3, 3)).
out_stats(num_consecutive_holidays(31, 0, 3, 3)).
out_stats(num_consecutive_holidays(32, 0, 3, 3)).
out_stats(num_weekend_offs(1, 0, 1, 1)).
out_stats(num_weekend_offs(2, 0, 0, 0)).
out_stats(num_weekend_offs(3, 0, 2, 2)).
out_stats(num_weekend_offs(4, 0, 1, 1)).
out_stats(num_weekend_offs(5, 0, 0, 0)).
out_stats(num_weekend_offs(6, 0, 0, 0)).
out_stats(num_weekend_offs(7, 0, 1, 1)).
out_stats(num_weekend_offs(8, 0, 2, 2)).
out_stats(num_weekend_offs(9, 0, 0, 0)).
out_stats(num_weekend_offs(10, 0, 0, 0)).
out_stats(num_weekend_offs(11, 0, 0, 0)).
out_stats(num_weekend_offs(12, 0, 2, 2)).
out_stats(num_weekend_offs(13, 0, 0, 0)).
out_stats(num_weekend_offs(14, 0, 1, 1)).
out_stats(num_weekend_offs(15, 0, 1, 1)).
out_stats(num_weekend_offs(16, 0, 0, 0)).
out_stats(num_weekend_offs(17, 0, 0, 0)).
out_stats(num_weekend_offs(18, 0, 2, 2)).
out_stats(num_weekend_offs(19, 0, 0, 0)).
out_stats(num_weekend_offs(20, 0, 0, 0)).
out_stats(num_weekend_offs(21, 0, 0, 0)).
out_stats(num_weekend_offs(22, 0, 1, 1)).
out_stats(num_weekend_offs(23, 0, 0, 0)).
out_stats(num_weekend_offs(24, 0, 1, 1)).
out_stats(num_weekend_offs(25, 0, 2, 2)).
out_stats(num_weekend_offs(26, 0, 0, 0)).
out_stats(num_weekend_offs(27, 0, 1, 1)).
out_stats(num_weekend_offs(28, 0, 1, 1)).
out_stats(num_weekend_offs(29, 0, 0, 0)).
out_stats(num_weekend_offs(30, 0, 0, 0)).
out_stats(num_weekend_offs(31, 0, 3, 3)).
out_stats(num_weekend_offs(32, 0, 2, 2)).
reward(weekend_offs(1, 0), staff(1), 4, 1).
reward(weekend_offs(3, 0), staff(3), 4, 1).
reward(weekend_offs(4, 0), staff(4), 4, 1).
reward(weekend_offs(7, 0), staff(7), 4, 1).
reward(weekend_offs(8, 0), staff(8), 4, 1).
reward(weekend_offs(12, 0), staff(12), 4, 1).
reward(weekend_offs(14, 0), staff(14), 4, 1).
reward(weekend_offs(15, 0), staff(15), 4, 1).
reward(weekend_offs(18, 0), staff(18), 4, 1).
reward(weekend_offs(22, 0), staff(22), 4, 1).
reward(weekend_offs(24, 0), staff(24), 4, 1).
reward(weekend_offs(25, 0), staff(25), 4, 1).
reward(weekend_offs(27, 0), staff(27), 4, 1).
reward(weekend_offs(28, 0), staff(28), 4, 1).
reward(weekend_offs(31, 0), staff(31), 4, 1).
reward(weekend_offs(32, 0), staff(32), 4, 1).
reward(consecutive_holidays(1, 0), staff(1), 4, 1).
reward(consecutive_holidays(2, 0), staff(2), 4, 1).
reward(consecutive_holidays(3, 0), staff(3), 4, 1).
reward(consecutive_holidays(4, 0), staff(4), 4, 1).
reward(consecutive_holidays(5, 0), staff(5), 4, 1).
reward(consecutive_holidays(6, 0), staff(6), 4, 1).
reward(consecutive_holidays(7, 0), staff(7), 4, 1).
reward(consecutive_holidays(8, 0), staff(8), 4, 1).
reward(consecutive_holidays(9, 0), staff(9), 4, 1).
reward(consecutive_holidays(10, 0), staff(10), 4, 1).
reward(consecutive_holidays(11, 0), staff(11), 4, 1).
reward(consecutive_holidays(12, 0), staff(12), 4, 1).
reward(consecutive_holidays(13, 0), staff(13), 4, 1).
reward(consecutive_holidays(14, 0), staff(14), 4, 1).
reward(consecutive_holidays(15, 0), staff(15), 4, 1).
reward(consecutive_holidays(16, 0), staff(16), 4, 1).
reward(consecutive_holidays(17, 0), staff(17), 4, 1).
reward(consecutive_holidays(18, 0), staff(18), 4, 1).
reward(consecutive_holidays(19, 0), staff(19), 4, 1).
reward(consecutive_holidays(20, 0), staff(20), 4, 1).
reward(consecutive_holidays(21, 0), staff(21), 4, 1).
reward(consecutive_holidays(22, 0), staff(22), 4, 1).
reward(consecutive_holidays(23, 0), staff(23), 4, 1).
reward(consecutive_holidays(24, 0), staff(24), 4, 1).
reward(consecutive_holidays(25, 0), staff(25), 4, 1).
reward(consecutive_holidays(26, 0), staff(26), 4, 1).
reward(consecutive_holidays(28, 0), staff(28), 4, 1).
reward(consecutive_holidays(29, 0), staff(29), 4, 1).
reward(consecutive_holidays(30, 0), staff(30), 4, 1).
reward(consecutive_holidays(31, 0), staff(31), 4, 1).
reward(consecutive_holidays(32, 0), staff(32), 4, 1).
penalty(staff_soft_lb(20250714, "SE", "Newcomers", 1, 0), num_staffs(22, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250708, "SE", "Newcomers", 1, 0), num_staffs(16, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250625, "SE", "Newcomers", 1, 0), num_staffs(3, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250711, "SE", "Newcomers", 1, 0), num_staffs(19, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250628, "SE", "Newcomers", 1, 0), num_staffs(6, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250719, "SE", "Newcomers", 1, 0), num_staffs(27, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20250715, "SN", "Newcomers", 1, 0), num_staffs(23, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20250709, "SN", "Newcomers", 1, 0), num_staffs(17, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20250626, "SN", "Newcomers", 1, 0), num_staffs(4, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20250712, "SN", "Newcomers", 1, 0), num_staffs(20, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20250622, "SN", "Newcomers", 1, 0), num_staffs(0, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20250629, "SN", "Newcomers", 1, 0), num_staffs(7, "SN", "新人"), 1, 2).
penalty(staff_soft_ub(20250630, "LD", "All", 4, 5), num_staffs(8, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250715, "LD", "All", 4, 5), num_staffs(23, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250710, "LD", "All", 4, 5), num_staffs(18, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250717, "LD", "All", 4, 5), num_staffs(25, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250705, "LD", "All", 4, 5), num_staffs(13, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250712, "LD", "All", 4, 5), num_staffs(20, "LD", "全員"), 1, 2).
penalty(staff_soft_ub(20250701, "SE", "All", 4, 5), num_staffs(9, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250716, "SE", "All", 4, 5), num_staffs(24, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250711, "SE", "All", 4, 5), num_staffs(19, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250718, "SE", "All", 4, 5), num_staffs(26, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250706, "SE", "All", 4, 5), num_staffs(14, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250713, "SE", "All", 4, 5), num_staffs(21, "SE", "全員"), 1, 2).
penalty(staff_soft_ub(20250707, "SN", "All", 4, 5), num_staffs(15, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250714, "SN", "All", 4, 5), num_staffs(22, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250702, "SN", "All", 4, 5), num_staffs(10, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250717, "SN", "All", 4, 5), num_staffs(25, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250712, "SN", "All", 4, 5), num_staffs(20, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250719, "SN", "All", 4, 5), num_staffs(27, "SN", "全員"), 1, 2).
penalty(staff_soft_ub(20250630, "LD+LM+E", "All", 4, 5), num_staffs(8, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250715, "LD+LM+E", "All", 4, 5), num_staffs(23, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250710, "LD+LM+E", "All", 4, 5), num_staffs(18, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250717, "LD+LM+E", "All", 4, 5), num_staffs(25, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250705, "LD+LM+E", "All", 4, 5), num_staffs(13, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250712, "LD+LM+E", "All", 4, 5), num_staffs(20, "LD+LM+E", "全員"), 1, 2).
penalty(staff_soft_ub(20250710, "LD", "Mid-levels", 3, 4), num_staffs(18, "LD", "中堅"), 1, 2).
penalty(staff_soft_ub(20250711, "SE", "Mid-levels", 3, 4), num_staffs(19, "SE", "中堅"), 1, 2).
penalty(staff_soft_ub(20250712, "SN", "Mid-levels", 3, 4), num_staffs(20, "SN", "中堅"), 1, 2).
penalty(shifts_soft_lb(10, "SN", 4, 3), num_shifts(10, "S"), 1, 2).
penalty(shifts_soft_lb(12, "LD", 4, 3), num_shifts(12, "N"), 1, 2).
penalty(shifts_soft_lb(12, "SE", 4, 3), num_shifts(12, "J"), 1, 2).
penalty(shifts_soft_lb(12, "SN", 4, 3), num_shifts(12, "S"), 1, 2).
penalty(shifts_soft_lb(21, "SN", 4, 3), num_shifts(21, "S"), 1, 2).
penalty(shifts_soft_lb(23, "LD", 4, 3), num_shifts(23, "N"), 1, 2).
penalty(shifts_soft_lb(23, "SE", 4, 3), num_shifts(23, "J"), 1, 2).
penalty(shifts_soft_lb(23, "SN", 4, 3), num_shifts(23, "S"), 1, 2).
penalty(shifts_soft_ub(1, "LD", 3, 5), num_shifts(1, "N"), 4, 2).
penalty(shifts_soft_ub(1, "SE", 3, 5), num_shifts(1, "J"), 4, 2).
penalty(shifts_soft_ub(1, "SN", 3, 5), num_shifts(1, "S"), 4, 2).
penalty(shifts_soft_ub(2, "LD", 3, 4), num_shifts(2, "N"), 1, 2).
penalty(shifts_soft_ub(2, "SE", 3, 4), num_shifts(2, "J"), 1, 2).
penalty(shifts_soft_ub(2, "SN", 3, 4), num_shifts(2, "S"), 1, 2).
penalty(shifts_soft_ub(3, "LD", 3, 5), num_shifts(3, "N"), 4, 2).
penalty(shifts_soft_ub(3, "SE", 3, 5), num_shifts(3, "J"), 4, 2).
penalty(shifts_soft_ub(3, "SN", 3, 4), num_shifts(3, "S"), 1, 2).
penalty(shifts_soft_ub(5, "LD", 3, 4), num_shifts(5, "N"), 1, 2).
penalty(shifts_soft_ub(5, "SE", 3, 4), num_shifts(5, "J"), 1, 2).
penalty(shifts_soft_ub(5, "SN", 3, 4), num_shifts(5, "S"), 1, 2).
penalty(shifts_soft_ub(18, "LD", 3, 4), num_shifts(18, "N"), 1, 2).
penalty(shifts_soft_ub(18, "SE", 3, 4), num_shifts(18, "J"), 1, 2).
penalty(shifts_soft_ub(18, "SN", 3, 4), num_shifts(18, "S"), 1, 2).
penalty(shifts_soft_ub(19, "SN", 5, 6), num_shifts(19, "S"), 1, 2).
penalty(shifts_soft_ub(29, "LD", 4, 5), num_shifts(29, "N"), 1, 2).
penalty(shifts_soft_ub(29, "SE", 4, 5), num_shifts(29, "J"), 1, 2).
penalty(isolated_working_day(19, 20250622), table(19, 0), 1, 1).
penalty(isolated_working_day(6, 20250622), table(6, 0), 1, 1).
penalty(isolated_working_day(17, 20250623), table(17, 1), 1, 1).
penalty(isolated_working_day(4, 20250624), table(4, 2), 1, 1).
penalty(isolated_working_day(16, 20250624), table(16, 2), 1, 1).
penalty(isolated_working_day(11, 20250625), table(11, 3), 1, 1).
penalty(isolated_working_day(23, 20250625), table(23, 3), 1, 1).
penalty(isolated_working_day(14, 20250626), table(14, 4), 1, 1).
penalty(isolated_working_day(26, 20250626), table(26, 4), 1, 1).
penalty(isolated_working_day(7, 20250627), table(7, 5), 1, 1).
penalty(isolated_working_day(23, 20250627), table(23, 5), 1, 1).
penalty(isolated_working_day(31, 20250627), table(31, 5), 1, 1).
penalty(isolated_working_day(11, 20250627), table(11, 5), 1, 1).
penalty(isolated_working_day(20, 20250628), table(20, 6), 1, 1).
penalty(isolated_working_day(29, 20250629), table(29, 7), 1, 1).
penalty(isolated_working_day(30, 20250629), table(30, 7), 1, 1).
penalty(isolated_working_day(6, 20250630), table(6, 8), 1, 1).
penalty(isolated_working_day(22, 20250630), table(22, 8), 1, 1).
penalty(isolated_working_day(31, 20250630), table(31, 8), 1, 1).
penalty(isolated_working_day(32, 20250630), table(32, 8), 1, 1).
penalty(isolated_working_day(25, 20250630), table(25, 8), 1, 1).
penalty(isolated_working_day(16, 20250701), table(16, 9), 1, 1).
penalty(isolated_working_day(17, 20250701), table(17, 9), 1, 1).
penalty(isolated_working_day(30, 20250701), table(30, 9), 1, 1).
penalty(isolated_working_day(21, 20250702), table(21, 10), 1, 1).
penalty(isolated_working_day(32, 20250702), table(32, 10), 1, 1).
penalty(isolated_working_day(23, 20250703), table(23, 11), 1, 1).
penalty(isolated_working_day(7, 20250704), table(7, 12), 1, 1).
penalty(isolated_working_day(8, 20250704), table(8, 12), 1, 1).
penalty(isolated_working_day(24, 20250704), table(24, 12), 1, 1).
penalty(isolated_working_day(30, 20250704), table(30, 12), 1, 1).
penalty(isolated_working_day(21, 20250704), table(21, 12), 1, 1).
penalty(isolated_working_day(1, 20250705), table(1, 13), 1, 1).
penalty(isolated_working_day(9, 20250705), table(9, 13), 1, 1).
penalty(isolated_working_day(10, 20250705), table(10, 13), 1, 1).
penalty(isolated_working_day(2, 20250706), table(2, 14), 1, 1).
penalty(isolated_working_day(12, 20250707), table(12, 15), 1, 1).
penalty(isolated_working_day(13, 20250707), table(13, 15), 1, 1).
penalty(isolated_working_day(28, 20250707), table(28, 15), 1, 1).
penalty(isolated_working_day(16, 20250708), table(16, 16), 1, 1).
penalty(isolated_working_day(13, 20250709), table(13, 17), 1, 1).
penalty(isolated_working_day(5, 20250710), table(5, 18), 1, 1).
penalty(isolated_working_day(10, 20250710), table(10, 18), 1, 1).
penalty(isolated_working_day(16, 20250710), table(16, 18), 1, 1).
penalty(isolated_working_day(14, 20250711), table(14, 19), 1, 1).
penalty(isolated_working_day(26, 20250711), table(26, 19), 1, 1).
penalty(isolated_working_day(22, 20250714), table(22, 22), 1, 1).
penalty(isolated_working_day(31, 20250714), table(31, 22), 1, 1).
penalty(isolated_working_day(29, 20250716), table(29, 24), 1, 1).
penalty(isolated_working_day(16, 20250718), table(16, 26), 1, 1).
penalty(isolated_working_day(28, 20250718), table(28, 26), 1, 1).
penalty(isolated_working_day(32, 20250718), table(32, 26), 1, 1).
penalty(isolated_working_day(21, 20250621), table(21, -1), 1, 1).
penalty(pred_shift(5, 20250622, "J"), table(5, 0), 1, 4).
penalty(pred_shift(9, 20250622, "J"), table(9, 0), 1, 4).
penalty(adjacent_holiday(7, 20250705, 20250706), table(7, 13), 1, 1).
penalty(staff_request(22, 20250623, "D", "○"), table(22, 1), 1, 5).
