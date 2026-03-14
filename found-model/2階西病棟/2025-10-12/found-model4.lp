header("Cost: 0 0 0 0 2 7 93 -217, Elapsed: 1577s").
horizontal_constraint_type("D").
horizontal_constraint_type("LD").
horizontal_constraint_type("SE").
horizontal_constraint_type("SN").
horizontal_constraint_type("E").
horizontal_constraint_type("N").
horizontal_constraint_type("LM").
horizontal_constraint_type("EM").
out_stats(num_consecutive_holidays(1, 0, 2, 2)).
out_stats(num_consecutive_holidays(34, 0, 0, 0)).
out_stats(num_consecutive_holidays(35, 0, 0, 0)).
out_stats(num_consecutive_holidays(36, 0, 0, 0)).
out_stats(num_consecutive_holidays(37, 0, 0, 0)).
out_stats(num_consecutive_holidays(38, 0, 0, 0)).
out_stats(num_consecutive_holidays(39, 0, 0, 0)).
out_stats(num_national_holiday_offs(1, 0, 2, 2)).
out_stats(num_national_holiday_offs(34, 0, 2, 2)).
out_stats(num_national_holiday_offs(35, 0, 2, 2)).
out_stats(num_national_holiday_offs(36, 0, 2, 2)).
out_stats(num_national_holiday_offs(37, 0, 2, 2)).
out_stats(num_national_holiday_offs(38, 0, 2, 2)).
out_stats(num_national_holiday_offs(39, 0, 2, 2)).
out_stats(num_weekend_offs(1, 0, 3, 3)).
out_stats(num_weekend_offs(34, 0, 3, 3)).
out_stats(num_weekend_offs(35, 0, 3, 3)).
out_stats(num_weekend_offs(36, 0, 3, 3)).
out_stats(num_weekend_offs(37, 0, 3, 3)).
out_stats(num_weekend_offs(38, 0, 3, 3)).
out_stats(num_weekend_offs(39, 0, 3, 3)).
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
group("リ－ダ－", 2).
group("リ－ダ－", 3).
group("リ－ダ－", 4).
group("リ－ダ－", 6).
group("リ－ダ－", 7).
group("リ－ダ－", 8).
group("リ－ダ－", 9).
group("リ－ダ－", 10).
group("リ－ダ－", 11).
group("リ－ダ－", 12).
group("リ－ダ－", 13).
group("リ－ダ－", 18).
group("リ－ダ－", 20).
group("リ－ダ－", 21).
group("リ－ダ－", 22).
group("リ－ダ－", 23).
group("リ－ダ－", 24).
group("リ－ダ－", 25).
group("リ－ダ－", 26).
group("リ－ダ－", 28).
group("EM", 2).
group("EM", 3).
group("EM", 4).
group("EM", 5).
group("EM", 6).
group("EM", 7).
group("EM", 8).
group("EM", 9).
group("EM", 10).
group("EM", 11).
group("EM", 12).
group("EM", 13).
group("EM", 14).
group("EM", 15).
group("EM", 16).
group("EM", 17).
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
group("Newcomers", 16).
group("Newcomers", 17).
group("Newcomers", 31).
group("Newcomers", 32).
group("Newcomers", 33).
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
out_assigned(1, 20251109).
out_assigned(2, 20251109).
out_assigned(3, 20251109).
out_assigned(4, 20251109).
out_assigned(5, 20251109).
out_assigned(6, 20251109).
out_assigned(7, 20251109).
out_assigned(9, 20251109).
out_assigned(10, 20251109).
out_assigned(11, 20251109).
out_assigned(12, 20251109).
out_assigned(13, 20251109).
out_assigned(14, 20251109).
out_assigned(16, 20251109).
out_assigned(17, 20251109).
out_assigned(18, 20251109).
out_assigned(19, 20251109).
out_assigned(20, 20251109).
out_assigned(21, 20251109).
out_assigned(22, 20251109).
out_assigned(23, 20251109).
out_assigned(24, 20251109).
out_assigned(25, 20251109).
out_assigned(26, 20251109).
out_assigned(27, 20251109).
out_assigned(28, 20251109).
out_assigned(29, 20251109).
out_assigned(30, 20251109).
out_assigned(32, 20251109).
out_assigned(33, 20251109).
out_assigned(34, 20251109).
out_assigned(35, 20251109).
out_assigned(36, 20251109).
out_assigned(37, 20251109).
out_assigned(38, 20251109).
out_assigned(39, 20251109).
out_assigned(1, 20251110).
out_assigned(2, 20251110).
out_assigned(3, 20251110).
out_assigned(4, 20251110).
out_assigned(5, 20251110).
out_assigned(6, 20251110).
out_assigned(7, 20251110).
out_assigned(8, 20251110).
out_assigned(9, 20251110).
out_assigned(10, 20251110).
out_assigned(11, 20251110).
out_assigned(12, 20251110).
out_assigned(14, 20251110).
out_assigned(16, 20251110).
out_assigned(17, 20251110).
out_assigned(18, 20251110).
out_assigned(19, 20251110).
out_assigned(20, 20251110).
out_assigned(21, 20251110).
out_assigned(22, 20251110).
out_assigned(23, 20251110).
out_assigned(24, 20251110).
out_assigned(25, 20251110).
out_assigned(26, 20251110).
out_assigned(27, 20251110).
out_assigned(28, 20251110).
out_assigned(29, 20251110).
out_assigned(30, 20251110).
out_assigned(32, 20251110).
out_assigned(33, 20251110).
out_assigned(34, 20251110).
out_assigned(35, 20251110).
out_assigned(36, 20251110).
out_assigned(37, 20251110).
out_assigned(38, 20251110).
out_assigned(39, 20251110).
out_assigned(1, 20251111).
out_assigned(3, 20251111).
out_assigned(4, 20251111).
out_assigned(5, 20251111).
out_assigned(7, 20251111).
out_assigned(8, 20251111).
out_assigned(9, 20251111).
out_assigned(10, 20251111).
out_assigned(11, 20251111).
out_assigned(12, 20251111).
out_assigned(13, 20251111).
out_assigned(14, 20251111).
out_assigned(15, 20251111).
out_assigned(16, 20251111).
out_assigned(17, 20251111).
out_assigned(18, 20251111).
out_assigned(19, 20251111).
out_assigned(20, 20251111).
out_assigned(21, 20251111).
out_assigned(22, 20251111).
out_assigned(23, 20251111).
out_assigned(24, 20251111).
out_assigned(25, 20251111).
out_assigned(26, 20251111).
out_assigned(27, 20251111).
out_assigned(29, 20251111).
out_assigned(30, 20251111).
out_assigned(31, 20251111).
out_assigned(32, 20251111).
out_assigned(33, 20251111).
out_assigned(34, 20251111).
out_assigned(35, 20251111).
out_assigned(36, 20251111).
out_assigned(37, 20251111).
out_assigned(38, 20251111).
out_assigned(39, 20251111).
out_assigned(1, 20251112).
out_assigned(2, 20251112).
out_assigned(3, 20251112).
out_assigned(4, 20251112).
out_assigned(5, 20251112).
out_assigned(6, 20251112).
out_assigned(7, 20251112).
out_assigned(8, 20251112).
out_assigned(9, 20251112).
out_assigned(10, 20251112).
out_assigned(11, 20251112).
out_assigned(12, 20251112).
out_assigned(13, 20251112).
out_assigned(14, 20251112).
out_assigned(15, 20251112).
out_assigned(16, 20251112).
out_assigned(17, 20251112).
out_assigned(19, 20251112).
out_assigned(20, 20251112).
out_assigned(21, 20251112).
out_assigned(22, 20251112).
out_assigned(23, 20251112).
out_assigned(24, 20251112).
out_assigned(25, 20251112).
out_assigned(27, 20251112).
out_assigned(28, 20251112).
out_assigned(29, 20251112).
out_assigned(30, 20251112).
out_assigned(31, 20251112).
out_assigned(32, 20251112).
out_assigned(33, 20251112).
out_assigned(34, 20251112).
out_assigned(35, 20251112).
out_assigned(36, 20251112).
out_assigned(37, 20251112).
out_assigned(38, 20251112).
out_assigned(39, 20251112).
out_assigned(1, 20251113).
out_assigned(2, 20251113).
out_assigned(3, 20251113).
out_assigned(4, 20251113).
out_assigned(5, 20251113).
out_assigned(7, 20251113).
out_assigned(8, 20251113).
out_assigned(9, 20251113).
out_assigned(10, 20251113).
out_assigned(11, 20251113).
out_assigned(12, 20251113).
out_assigned(13, 20251113).
out_assigned(14, 20251113).
out_assigned(15, 20251113).
out_assigned(16, 20251113).
out_assigned(17, 20251113).
out_assigned(18, 20251113).
out_assigned(19, 20251113).
out_assigned(20, 20251113).
out_assigned(21, 20251113).
out_assigned(22, 20251113).
out_assigned(23, 20251113).
out_assigned(24, 20251113).
out_assigned(25, 20251113).
out_assigned(27, 20251113).
out_assigned(28, 20251113).
out_assigned(29, 20251113).
out_assigned(30, 20251113).
out_assigned(31, 20251113).
out_assigned(32, 20251113).
out_assigned(33, 20251113).
out_assigned(34, 20251113).
out_assigned(35, 20251113).
out_assigned(36, 20251113).
out_assigned(37, 20251113).
out_assigned(38, 20251113).
out_assigned(39, 20251113).
out_assigned(2, 20251114).
out_assigned(3, 20251114).
out_assigned(4, 20251114).
out_assigned(5, 20251114).
out_assigned(6, 20251114).
out_assigned(7, 20251114).
out_assigned(8, 20251114).
out_assigned(9, 20251114).
out_assigned(10, 20251114).
out_assigned(11, 20251114).
out_assigned(12, 20251114).
out_assigned(15, 20251114).
out_assigned(16, 20251114).
out_assigned(17, 20251114).
out_assigned(18, 20251114).
out_assigned(19, 20251114).
out_assigned(20, 20251114).
out_assigned(22, 20251114).
out_assigned(24, 20251114).
out_assigned(25, 20251114).
out_assigned(27, 20251114).
out_assigned(28, 20251114).
out_assigned(29, 20251114).
out_assigned(30, 20251114).
out_assigned(31, 20251114).
out_assigned(32, 20251114).
out_assigned(33, 20251114).
out_assigned(34, 20251114).
out_assigned(35, 20251114).
out_assigned(36, 20251114).
out_assigned(37, 20251114).
out_assigned(38, 20251114).
out_assigned(39, 20251114).
out_assigned(1, 20251115).
out_assigned(2, 20251115).
out_assigned(3, 20251115).
out_assigned(4, 20251115).
out_assigned(5, 20251115).
out_assigned(6, 20251115).
out_assigned(7, 20251115).
out_assigned(8, 20251115).
out_assigned(9, 20251115).
out_assigned(10, 20251115).
out_assigned(11, 20251115).
out_assigned(12, 20251115).
out_assigned(14, 20251115).
out_assigned(15, 20251115).
out_assigned(16, 20251115).
out_assigned(17, 20251115).
out_assigned(18, 20251115).
out_assigned(19, 20251115).
out_assigned(20, 20251115).
out_assigned(22, 20251115).
out_assigned(23, 20251115).
out_assigned(25, 20251115).
out_assigned(27, 20251115).
out_assigned(28, 20251115).
out_assigned(29, 20251115).
out_assigned(30, 20251115).
out_assigned(31, 20251115).
out_assigned(32, 20251115).
out_assigned(33, 20251115).
out_assigned(34, 20251115).
out_assigned(35, 20251115).
out_assigned(36, 20251115).
out_assigned(37, 20251115).
out_assigned(38, 20251115).
out_assigned(39, 20251115).
out_assigned(34, 20251005).
out_assigned(35, 20251005).
out_assigned(36, 20251005).
out_assigned(37, 20251005).
out_assigned(38, 20251005).
out_assigned(39, 20251005).
out_assigned(35, 20251006).
out_assigned(36, 20251006).
out_assigned(37, 20251006).
out_assigned(38, 20251006).
out_assigned(39, 20251006).
out_assigned(35, 20251007).
out_assigned(36, 20251007).
out_assigned(37, 20251007).
out_assigned(38, 20251007).
out_assigned(39, 20251007).
out_assigned(35, 20251008).
out_assigned(36, 20251008).
out_assigned(37, 20251008).
out_assigned(38, 20251008).
out_assigned(39, 20251008).
out_assigned(34, 20251009).
out_assigned(35, 20251009).
out_assigned(36, 20251009).
out_assigned(37, 20251009).
out_assigned(38, 20251009).
out_assigned(39, 20251009).
out_assigned(35, 20251010).
out_assigned(36, 20251010).
out_assigned(37, 20251010).
out_assigned(38, 20251010).
out_assigned(39, 20251010).
out_assigned(34, 20251011).
out_assigned(35, 20251011).
out_assigned(36, 20251011).
out_assigned(37, 20251011).
out_assigned(38, 20251011).
out_assigned(39, 20251011).
out_date(20251013, "Ho").
out_date(20251103, "Ho").
out_date(20251012, "Su").
out_date(20251014, "Tu").
out_date(20251015, "We").
out_date(20251016, "Th").
out_date(20251017, "Fr").
out_date(20251018, "Sa").
out_date(20251019, "Su").
out_date(20251020, "Mo").
out_date(20251021, "Tu").
out_date(20251022, "We").
out_date(20251023, "Th").
out_date(20251024, "Fr").
out_date(20251025, "Sa").
out_date(20251026, "Su").
out_date(20251027, "Mo").
out_date(20251028, "Tu").
out_date(20251029, "We").
out_date(20251030, "Th").
out_date(20251031, "Fr").
out_date(20251101, "Sa").
out_date(20251102, "Su").
out_date(20251104, "Tu").
out_date(20251105, "We").
out_date(20251106, "Th").
out_date(20251107, "Fr").
out_date(20251108, "Sa").
out_date(20251109, "Su").
out_date(20251110, "Mo").
out_date(20251111, "Tu").
out_date(20251112, "We").
out_date(20251113, "Th").
out_date(20251114, "Fr").
out_date(20251115, "Sa").
out_date(20251005, "Su").
out_date(20251006, "Mo").
out_date(20251007, "Tu").
out_date(20251008, "We").
out_date(20251009, "Th").
out_date(20251010, "Fr").
out_date(20251011, "Sa").
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
out_assigned(1, 20251012, "WR").
out_assigned(34, 20251012, "WR").
out_assigned(35, 20251012, "WR").
out_assigned(36, 20251012, "WR").
out_assigned(37, 20251012, "WR").
out_assigned(38, 20251012, "WR").
out_assigned(39, 20251012, "WR").
out_assigned(1, 20251013, "PH").
out_assigned(34, 20251013, "PH").
out_assigned(35, 20251013, "PH").
out_assigned(36, 20251013, "PH").
out_assigned(37, 20251013, "PH").
out_assigned(38, 20251013, "PH").
out_assigned(39, 20251013, "PH").
out_assigned(1, 20251014, "D").
out_assigned(34, 20251014, "WR").
out_assigned(35, 20251014, "WR").
out_assigned(36, 20251014, "WR").
out_assigned(37, 20251014, "WR").
out_assigned(38, 20251014, "WR").
out_assigned(39, 20251014, "WR").
out_assigned(1, 20251015, "D").
out_assigned(34, 20251015, "WR").
out_assigned(35, 20251015, "WR").
out_assigned(36, 20251015, "WR").
out_assigned(37, 20251015, "WR").
out_assigned(38, 20251015, "WR").
out_assigned(39, 20251015, "WR").
out_assigned(1, 20251016, "D").
out_assigned(34, 20251016, "WR").
out_assigned(35, 20251016, "WR").
out_assigned(36, 20251016, "WR").
out_assigned(37, 20251016, "WR").
out_assigned(38, 20251016, "WR").
out_assigned(39, 20251016, "WR").
out_assigned(1, 20251017, "D").
out_assigned(34, 20251017, "WR").
out_assigned(35, 20251017, "WR").
out_assigned(36, 20251017, "WR").
out_assigned(37, 20251017, "WR").
out_assigned(38, 20251017, "WR").
out_assigned(39, 20251017, "WR").
out_assigned(1, 20251018, "WR").
out_assigned(34, 20251018, "WR").
out_assigned(35, 20251018, "WR").
out_assigned(36, 20251018, "WR").
out_assigned(37, 20251018, "WR").
out_assigned(38, 20251018, "WR").
out_assigned(39, 20251018, "WR").
out_assigned(1, 20251019, "WR").
out_assigned(34, 20251019, "WR").
out_assigned(35, 20251019, "WR").
out_assigned(36, 20251019, "WR").
out_assigned(37, 20251019, "WR").
out_assigned(38, 20251019, "WR").
out_assigned(39, 20251019, "WR").
out_assigned(34, 20251020, "WR").
out_assigned(35, 20251020, "WR").
out_assigned(36, 20251020, "WR").
out_assigned(37, 20251020, "WR").
out_assigned(38, 20251020, "WR").
out_assigned(39, 20251020, "WR").
out_assigned(34, 20251021, "WR").
out_assigned(35, 20251021, "WR").
out_assigned(36, 20251021, "WR").
out_assigned(37, 20251021, "WR").
out_assigned(38, 20251021, "WR").
out_assigned(39, 20251021, "WR").
out_assigned(34, 20251022, "WR").
out_assigned(35, 20251022, "WR").
out_assigned(36, 20251022, "WR").
out_assigned(37, 20251022, "WR").
out_assigned(38, 20251022, "WR").
out_assigned(39, 20251022, "WR").
out_assigned(1, 20251023, "WR").
out_assigned(34, 20251023, "WR").
out_assigned(35, 20251023, "WR").
out_assigned(36, 20251023, "WR").
out_assigned(37, 20251023, "WR").
out_assigned(38, 20251023, "WR").
out_assigned(39, 20251023, "WR").
out_assigned(1, 20251024, "LD").
out_assigned(34, 20251024, "WR").
out_assigned(35, 20251024, "WR").
out_assigned(36, 20251024, "WR").
out_assigned(37, 20251024, "WR").
out_assigned(38, 20251024, "WR").
out_assigned(39, 20251024, "WR").
out_assigned(34, 20251025, "WR").
out_assigned(35, 20251025, "WR").
out_assigned(36, 20251025, "WR").
out_assigned(37, 20251025, "WR").
out_assigned(38, 20251025, "WR").
out_assigned(39, 20251025, "WR").
out_assigned(1, 20251026, "SN").
out_assigned(34, 20251026, "WR").
out_assigned(35, 20251026, "WR").
out_assigned(36, 20251026, "WR").
out_assigned(37, 20251026, "WR").
out_assigned(38, 20251026, "WR").
out_assigned(39, 20251026, "WR").
out_assigned(1, 20251027, "WR").
out_assigned(34, 20251027, "WR").
out_assigned(35, 20251027, "WR").
out_assigned(36, 20251027, "WR").
out_assigned(37, 20251027, "WR").
out_assigned(38, 20251027, "WR").
out_assigned(39, 20251027, "WR").
out_assigned(34, 20251028, "WR").
out_assigned(35, 20251028, "WR").
out_assigned(36, 20251028, "WR").
out_assigned(37, 20251028, "WR").
out_assigned(38, 20251028, "WR").
out_assigned(39, 20251028, "WR").
out_assigned(34, 20251029, "WR").
out_assigned(35, 20251029, "WR").
out_assigned(36, 20251029, "WR").
out_assigned(37, 20251029, "WR").
out_assigned(38, 20251029, "WR").
out_assigned(39, 20251029, "WR").
out_assigned(1, 20251030, "HL").
out_assigned(34, 20251030, "WR").
out_assigned(35, 20251030, "WR").
out_assigned(36, 20251030, "WR").
out_assigned(37, 20251030, "WR").
out_assigned(38, 20251030, "WR").
out_assigned(39, 20251030, "WR").
out_assigned(1, 20251031, "HL").
out_assigned(34, 20251031, "WR").
out_assigned(35, 20251031, "WR").
out_assigned(36, 20251031, "WR").
out_assigned(37, 20251031, "WR").
out_assigned(38, 20251031, "WR").
out_assigned(39, 20251031, "WR").
out_assigned(1, 20251101, "WR").
out_assigned(34, 20251101, "WR").
out_assigned(35, 20251101, "WR").
out_assigned(36, 20251101, "WR").
out_assigned(37, 20251101, "WR").
out_assigned(38, 20251101, "WR").
out_assigned(39, 20251101, "WR").
out_assigned(1, 20251102, "WR").
out_assigned(34, 20251102, "WR").
out_assigned(35, 20251102, "WR").
out_assigned(36, 20251102, "WR").
out_assigned(37, 20251102, "WR").
out_assigned(38, 20251102, "WR").
out_assigned(39, 20251102, "WR").
out_assigned(1, 20251103, "PH").
out_assigned(34, 20251103, "PH").
out_assigned(35, 20251103, "PH").
out_assigned(36, 20251103, "PH").
out_assigned(37, 20251103, "PH").
out_assigned(38, 20251103, "PH").
out_assigned(39, 20251103, "PH").
out_assigned(34, 20251104, "WR").
out_assigned(35, 20251104, "WR").
out_assigned(36, 20251104, "WR").
out_assigned(37, 20251104, "WR").
out_assigned(38, 20251104, "WR").
out_assigned(39, 20251104, "WR").
out_assigned(34, 20251105, "WR").
out_assigned(35, 20251105, "WR").
out_assigned(36, 20251105, "WR").
out_assigned(37, 20251105, "WR").
out_assigned(38, 20251105, "WR").
out_assigned(39, 20251105, "WR").
out_assigned(1, 20251106, "D").
out_assigned(34, 20251106, "WR").
out_assigned(35, 20251106, "WR").
out_assigned(36, 20251106, "WR").
out_assigned(37, 20251106, "WR").
out_assigned(38, 20251106, "WR").
out_assigned(39, 20251106, "WR").
out_assigned(1, 20251107, "WR").
out_assigned(34, 20251107, "WR").
out_assigned(35, 20251107, "WR").
out_assigned(36, 20251107, "WR").
out_assigned(37, 20251107, "WR").
out_assigned(38, 20251107, "WR").
out_assigned(39, 20251107, "WR").
out_assigned(1, 20251108, "D").
out_assigned(34, 20251108, "WR").
out_assigned(35, 20251108, "WR").
out_assigned(36, 20251108, "WR").
out_assigned(37, 20251108, "WR").
out_assigned(38, 20251108, "WR").
out_assigned(39, 20251108, "WR").
out_assigned(8, 20251109, "WR").
out_assigned(15, 20251109, "WR").
out_assigned(31, 20251109, "WR").
out_assigned(15, 20251110, "WR").
out_assigned(31, 20251110, "AL").
out_assigned(13, 20251110, "D").
out_assigned(28, 20251111, "WR").
out_assigned(2, 20251111, "D").
out_assigned(6, 20251111, "D").
out_assigned(26, 20251112, "WR").
out_assigned(18, 20251112, "D").
out_assigned(26, 20251113, "HL").
out_assigned(6, 20251113, "D").
out_assigned(1, 20251114, "LD").
out_assigned(13, 20251114, "WR").
out_assigned(26, 20251114, "AL").
out_assigned(14, 20251114, "D").
out_assigned(21, 20251114, "D").
out_assigned(23, 20251114, "D").
out_assigned(13, 20251115, "WR").
out_assigned(21, 20251115, "WR").
out_assigned(24, 20251115, "WR").
out_assigned(26, 20251115, "WR").
out_assigned(1, 20251005, "WR").
out_assigned(2, 20251005, "SN").
out_assigned(3, 20251005, "WR").
out_assigned(4, 20251005, "LD").
out_assigned(5, 20251005, "WR").
out_assigned(6, 20251005, "WR").
out_assigned(7, 20251005, "SE").
out_assigned(8, 20251005, "WR").
out_assigned(9, 20251005, "WR").
out_assigned(10, 20251005, "D").
out_assigned(11, 20251005, "LD").
out_assigned(12, 20251005, "WR").
out_assigned(13, 20251005, "WR").
out_assigned(14, 20251005, "WR").
out_assigned(15, 20251005, "SE").
out_assigned(16, 20251005, "D").
out_assigned(17, 20251005, "SN").
out_assigned(18, 20251005, "D").
out_assigned(19, 20251005, "WR").
out_assigned(20, 20251005, "LD").
out_assigned(21, 20251005, "SN").
out_assigned(22, 20251005, "AL").
out_assigned(23, 20251005, "SE").
out_assigned(24, 20251005, "D").
out_assigned(25, 20251005, "WR").
out_assigned(26, 20251005, "WR").
out_assigned(27, 20251005, "LD").
out_assigned(28, 20251005, "WR").
out_assigned(29, 20251005, "SN").
out_assigned(30, 20251005, "WR").
out_assigned(31, 20251005, "WR").
out_assigned(32, 20251005, "WR").
out_assigned(33, 20251005, "SE").
out_assigned(1, 20251006, "D").
out_assigned(2, 20251006, "AL").
out_assigned(3, 20251006, "D").
out_assigned(4, 20251006, "SE").
out_assigned(5, 20251006, "D").
out_assigned(6, 20251006, "LD").
out_assigned(7, 20251006, "SN").
out_assigned(8, 20251006, "WR").
out_assigned(9, 20251006, "LD").
out_assigned(10, 20251006, "D").
out_assigned(11, 20251006, "SE").
out_assigned(12, 20251006, "WR").
out_assigned(13, 20251006, "D").
out_assigned(14, 20251006, "D").
out_assigned(15, 20251006, "SN").
out_assigned(16, 20251006, "D").
out_assigned(17, 20251006, "WR").
out_assigned(18, 20251006, "D").
out_assigned(19, 20251006, "D").
out_assigned(20, 20251006, "SE").
out_assigned(21, 20251006, "WR").
out_assigned(22, 20251006, "D").
out_assigned(23, 20251006, "SN").
out_assigned(24, 20251006, "D").
out_assigned(25, 20251006, "D").
out_assigned(26, 20251006, "LD").
out_assigned(27, 20251006, "SE").
out_assigned(28, 20251006, "WR").
out_assigned(29, 20251006, "WR").
out_assigned(30, 20251006, "LD").
out_assigned(31, 20251006, "D").
out_assigned(32, 20251006, "D").
out_assigned(33, 20251006, "SN").
out_assigned(34, 20251006, "SL").
out_assigned(1, 20251007, "D").
out_assigned(2, 20251007, "HL").
out_assigned(3, 20251007, "D").
out_assigned(4, 20251007, "SN").
out_assigned(5, 20251007, "D").
out_assigned(6, 20251007, "SE").
out_assigned(7, 20251007, "AL").
out_assigned(8, 20251007, "SE").
out_assigned(9, 20251007, "WR").
out_assigned(10, 20251007, "LD").
out_assigned(11, 20251007, "SN").
out_assigned(12, 20251007, "WR").
out_assigned(13, 20251007, "D").
out_assigned(14, 20251007, "LD").
out_assigned(15, 20251007, "WR").
out_assigned(16, 20251007, "D").
out_assigned(17, 20251007, "D").
out_assigned(18, 20251007, "D").
out_assigned(19, 20251007, "WR").
out_assigned(20, 20251007, "SN").
out_assigned(21, 20251007, "LD").
out_assigned(22, 20251007, "SE").
out_assigned(23, 20251007, "WR").
out_assigned(24, 20251007, "LD").
out_assigned(25, 20251007, "D").
out_assigned(26, 20251007, "WR").
out_assigned(27, 20251007, "SN").
out_assigned(28, 20251007, "D").
out_assigned(29, 20251007, "AL").
out_assigned(30, 20251007, "SE").
out_assigned(31, 20251007, "D").
out_assigned(32, 20251007, "D").
out_assigned(33, 20251007, "WR").
out_assigned(34, 20251007, "SL").
out_assigned(1, 20251008, "D").
out_assigned(2, 20251008, "WR").
out_assigned(3, 20251008, "D").
out_assigned(4, 20251008, "WR").
out_assigned(5, 20251008, "D").
out_assigned(6, 20251008, "SN").
out_assigned(7, 20251008, "WR").
out_assigned(8, 20251008, "SN").
out_assigned(9, 20251008, "WR").
out_assigned(10, 20251008, "SE").
out_assigned(11, 20251008, "WR").
out_assigned(12, 20251008, "D").
out_assigned(13, 20251008, "LD").
out_assigned(14, 20251008, "SE").
out_assigned(15, 20251008, "LD").
out_assigned(16, 20251008, "WR").
out_assigned(17, 20251008, "D").
out_assigned(18, 20251008, "D").
out_assigned(19, 20251008, "D").
out_assigned(20, 20251008, "WR").
out_assigned(21, 20251008, "WR").
out_assigned(22, 20251008, "SN").
out_assigned(23, 20251008, "LD").
out_assigned(24, 20251008, "SE").
out_assigned(25, 20251008, "SE").
out_assigned(26, 20251008, "D").
out_assigned(27, 20251008, "WR").
out_assigned(28, 20251008, "LD").
out_assigned(29, 20251008, "AL").
out_assigned(30, 20251008, "SN").
out_assigned(31, 20251008, "D").
out_assigned(32, 20251008, "D").
out_assigned(33, 20251008, "D").
out_assigned(34, 20251008, "SL").
out_assigned(1, 20251009, "D").
out_assigned(2, 20251009, "WR").
out_assigned(3, 20251009, "D").
out_assigned(4, 20251009, "D").
out_assigned(5, 20251009, "D").
out_assigned(6, 20251009, "WR").
out_assigned(7, 20251009, "WR").
out_assigned(8, 20251009, "WR").
out_assigned(9, 20251009, "LD").
out_assigned(10, 20251009, "SN").
out_assigned(11, 20251009, "D").
out_assigned(12, 20251009, "LD").
out_assigned(13, 20251009, "SE").
out_assigned(14, 20251009, "SN").
out_assigned(15, 20251009, "SE").
out_assigned(16, 20251009, "D").
out_assigned(17, 20251009, "D").
out_assigned(18, 20251009, "WR").
out_assigned(19, 20251009, "HL").
out_assigned(20, 20251009, "LD").
out_assigned(21, 20251009, "D").
out_assigned(22, 20251009, "WR").
out_assigned(23, 20251009, "WR").
out_assigned(24, 20251009, "SN").
out_assigned(25, 20251009, "SN").
out_assigned(26, 20251009, "SE").
out_assigned(27, 20251009, "LD").
out_assigned(28, 20251009, "SE").
out_assigned(29, 20251009, "HL").
out_assigned(30, 20251009, "WR").
out_assigned(31, 20251009, "D").
out_assigned(32, 20251009, "D").
out_assigned(33, 20251009, "D").
out_assigned(1, 20251010, "D").
out_assigned(2, 20251010, "HL").
out_assigned(3, 20251010, "WR").
out_assigned(4, 20251010, "LD").
out_assigned(5, 20251010, "WR").
out_assigned(6, 20251010, "D").
out_assigned(7, 20251010, "D").
out_assigned(8, 20251010, "D").
out_assigned(9, 20251010, "SE").
out_assigned(10, 20251010, "WR").
out_assigned(11, 20251010, "LD").
out_assigned(12, 20251010, "SE").
out_assigned(13, 20251010, "SN").
out_assigned(14, 20251010, "WR").
out_assigned(15, 20251010, "SN").
out_assigned(16, 20251010, "D").
out_assigned(17, 20251010, "WR").
out_assigned(18, 20251010, "D").
out_assigned(19, 20251010, "SE").
out_assigned(20, 20251010, "WR").
out_assigned(21, 20251010, "D").
out_assigned(22, 20251010, "LD").
out_assigned(23, 20251010, "D").
out_assigned(24, 20251010, "WR").
out_assigned(25, 20251010, "WR").
out_assigned(26, 20251010, "SN").
out_assigned(27, 20251010, "SE").
out_assigned(28, 20251010, "SN").
out_assigned(29, 20251010, "HL").
out_assigned(30, 20251010, "LD").
out_assigned(31, 20251010, "HL").
out_assigned(32, 20251010, "D").
out_assigned(33, 20251010, "D").
out_assigned(34, 20251010, "SL").
out_assigned(1, 20251011, "WR").
out_assigned(2, 20251011, "LD").
out_assigned(3, 20251011, "D").
out_assigned(4, 20251011, "WR").
out_assigned(5, 20251011, "AL").
out_assigned(6, 20251011, "SE").
out_assigned(7, 20251011, "LD").
out_assigned(8, 20251011, "WR").
out_assigned(9, 20251011, "SN").
out_assigned(10, 20251011, "WR").
out_assigned(11, 20251011, "SE").
out_assigned(12, 20251011, "SN").
out_assigned(13, 20251011, "WR").
out_assigned(14, 20251011, "D").
out_assigned(15, 20251011, "WR").
out_assigned(16, 20251011, "WR").
out_assigned(17, 20251011, "WR").
out_assigned(18, 20251011, "WR").
out_assigned(19, 20251011, "SN").
out_assigned(20, 20251011, "D").
out_assigned(21, 20251011, "D").
out_assigned(22, 20251011, "SE").
out_assigned(23, 20251011, "WR").
out_assigned(24, 20251011, "WR").
out_assigned(25, 20251011, "LD").
out_assigned(26, 20251011, "WR").
out_assigned(27, 20251011, "SN").
out_assigned(28, 20251011, "AL").
out_assigned(29, 20251011, "LD").
out_assigned(30, 20251011, "SE").
out_assigned(31, 20251011, "WR").
out_assigned(32, 20251011, "WR").
out_assigned(33, 20251011, "WR").
out_pos_request(1, 20251012, "WR").
out_pos_request(5, 20251012, "HL").
out_pos_request(7, 20251012, "WR").
out_pos_request(12, 20251012, "WR").
out_pos_request(13, 20251012, "WR").
out_pos_request(17, 20251012, "AL").
out_pos_request(23, 20251012, "HL").
out_pos_request(28, 20251012, "WR").
out_pos_request(31, 20251012, "WR").
out_pos_request(1, 20251013, "PH").
out_pos_request(5, 20251013, "PH").
out_pos_request(7, 20251013, "PH").
out_pos_request(12, 20251013, "PH").
out_pos_request(17, 20251013, "PH").
out_pos_request(23, 20251013, "PH").
out_pos_request(26, 20251013, "PH").
out_pos_request(28, 20251013, "PH").
out_pos_request(31, 20251013, "PH").
out_pos_request(1, 20251014, "D").
out_pos_request(2, 20251014, "D").
out_pos_request(9, 20251014, "D").
out_pos_request(12, 20251014, "WR").
out_pos_request(18, 20251014, "D").
out_pos_request(23, 20251014, "D").
out_pos_request(26, 20251014, "AL").
out_pos_request(27, 20251014, "AL").
out_pos_request(28, 20251014, "WR").
out_pos_request(1, 20251015, "D").
out_pos_request(4, 20251015, "D").
out_pos_request(18, 20251015, "D").
out_pos_request(20, 20251015, "D").
out_pos_request(21, 20251015, "HL").
out_pos_request(22, 20251015, "WR").
out_pos_request(26, 20251015, "WR").
out_pos_request(27, 20251015, "WR").
out_pos_request(1, 20251016, "D").
out_pos_request(19, 20251016, "AL").
out_pos_request(27, 20251016, "AL").
out_pos_request(1, 20251017, "D").
out_pos_request(5, 20251017, "WR").
out_pos_request(1, 20251018, "WR").
out_pos_request(11, 20251018, "WR").
out_pos_request(12, 20251018, "HL").
out_pos_request(14, 20251018, "WR").
out_pos_request(15, 20251018, "WR").
out_pos_request(17, 20251018, "AL").
out_pos_request(29, 20251018, "WR").
out_pos_request(32, 20251018, "WR").
out_pos_request(33, 20251018, "WR").
out_pos_request(1, 20251019, "WR").
out_pos_request(11, 20251019, "WR").
out_pos_request(14, 20251019, "WR").
out_pos_request(15, 20251019, "WR").
out_pos_request(26, 20251019, "AL").
out_pos_request(29, 20251019, "WR").
out_pos_request(32, 20251019, "AL").
out_pos_request(33, 20251019, "AL").
out_pos_request(1, 20251020, "D").
out_pos_request(21, 20251020, "D").
out_pos_request(26, 20251020, "AL").
out_pos_request(32, 20251020, "D").
out_pos_request(1, 20251021, "D").
out_pos_request(8, 20251021, "WR").
out_pos_request(16, 20251021, "D").
out_pos_request(21, 20251021, "WR").
out_pos_request(23, 20251021, "AL").
out_pos_request(1, 20251022, "D").
out_pos_request(8, 20251022, "WR").
out_pos_request(9, 20251022, "AL").
out_pos_request(18, 20251022, "D").
out_pos_request(30, 20251022, "HL").
out_pos_request(31, 20251022, "D").
out_pos_request(33, 20251022, "WR").
out_pos_request(8, 20251023, "WR").
out_pos_request(9, 20251023, "AL").
out_pos_request(14, 20251023, "D").
out_pos_request(15, 20251023, "D").
out_pos_request(17, 20251023, "D").
out_pos_request(19, 20251023, "D").
out_pos_request(20, 20251023, "WR").
out_pos_request(21, 20251023, "WR").
out_pos_request(27, 20251023, "D").
out_pos_request(29, 20251023, "D").
out_pos_request(30, 20251023, "AL").
out_pos_request(31, 20251023, "AL").
out_pos_request(1, 20251024, "LD").
out_pos_request(20, 20251024, "D").
out_pos_request(1, 20251025, "SE").
out_pos_request(2, 20251025, "WR").
out_pos_request(4, 20251025, "HL").
out_pos_request(5, 20251025, "WR").
out_pos_request(17, 20251025, "AL").
out_pos_request(32, 20251025, "WR").
out_pos_request(33, 20251025, "WR").
out_pos_request(1, 20251026, "SN").
out_pos_request(2, 20251026, "WR").
out_pos_request(4, 20251026, "HL").
out_pos_request(17, 20251026, "AL").
out_pos_request(20, 20251026, "AL").
out_pos_request(21, 20251026, "WR").
out_pos_request(32, 20251026, "AL").
out_pos_request(1, 20251027, "WR").
out_pos_request(2, 20251027, "WR").
out_pos_request(4, 20251027, "D").
out_pos_request(14, 20251027, "WR").
out_pos_request(18, 20251027, "HL").
out_pos_request(19, 20251027, "D").
out_pos_request(32, 20251027, "HL").
out_pos_request(1, 20251028, "D").
out_pos_request(14, 20251028, "AL").
out_pos_request(18, 20251028, "AL").
out_pos_request(20, 20251028, "D").
out_pos_request(22, 20251028, "WR").
out_pos_request(1, 20251029, "D").
out_pos_request(3, 20251029, "WR").
out_pos_request(14, 20251029, "HL").
out_pos_request(16, 20251029, "AL").
out_pos_request(22, 20251029, "WR").
out_pos_request(1, 20251030, "HL").
out_pos_request(3, 20251030, "AL").
out_pos_request(14, 20251030, "HL").
out_pos_request(16, 20251030, "AL").
out_pos_request(22, 20251030, "AL").
out_pos_request(33, 20251030, "AL").
out_pos_request(1, 20251031, "HL").
out_pos_request(3, 20251031, "WR").
out_pos_request(5, 20251031, "SE").
out_pos_request(12, 20251031, "D").
out_pos_request(16, 20251031, "AL").
out_pos_request(21, 20251031, "D").
out_pos_request(22, 20251031, "AL").
out_pos_request(26, 20251031, "D").
out_pos_request(29, 20251031, "D").
out_pos_request(1, 20251101, "WR").
out_pos_request(4, 20251101, "WR").
out_pos_request(5, 20251101, "SN").
out_pos_request(11, 20251101, "WR").
out_pos_request(16, 20251101, "WR").
out_pos_request(17, 20251101, "WR").
out_pos_request(28, 20251101, "HL").
out_pos_request(1, 20251102, "WR").
out_pos_request(4, 20251102, "AL").
out_pos_request(11, 20251102, "AL").
out_pos_request(16, 20251102, "WR").
out_pos_request(20, 20251102, "WR").
out_pos_request(22, 20251102, "HL").
out_pos_request(28, 20251102, "HL").
out_pos_request(1, 20251103, "PH").
out_pos_request(11, 20251103, "AL").
out_pos_request(20, 20251103, "PH").
out_pos_request(26, 20251103, "PH").
out_pos_request(1, 20251104, "D").
out_pos_request(3, 20251104, "D").
out_pos_request(1, 20251105, "D").
out_pos_request(19, 20251105, "D").
out_pos_request(1, 20251106, "D").
out_pos_request(6, 20251106, "WR").
out_pos_request(7, 20251106, "D").
out_pos_request(3, 20251107, "D").
out_pos_request(6, 20251107, "HL").
out_pos_request(7, 20251107, "D").
out_pos_request(13, 20251107, "WR").
out_pos_request(15, 20251107, "D").
out_pos_request(25, 20251107, "D").
out_pos_request(28, 20251107, "D").
out_pos_request(31, 20251107, "HL").
out_pos_request(1, 20251108, "D").
out_pos_request(3, 20251108, "D").
out_pos_request(4, 20251108, "WR").
out_pos_request(5, 20251108, "LD").
out_pos_request(6, 20251108, "HL").
out_pos_request(7, 20251108, "WR").
out_pos_request(27, 20251108, "WR").
out_pos_request(31, 20251108, "WR").
out_pos_request(8, 20251109, "WR").
out_pos_request(15, 20251109, "WR").
out_pos_request(31, 20251109, "WR").
out_pos_request(13, 20251110, "D").
out_pos_request(15, 20251110, "WR").
out_pos_request(31, 20251110, "AL").
out_pos_request(2, 20251111, "D").
out_pos_request(6, 20251111, "D").
out_pos_request(28, 20251111, "WR").
out_pos_request(18, 20251112, "D").
out_pos_request(26, 20251112, "WR").
out_pos_request(6, 20251113, "D").
out_pos_request(26, 20251113, "HL").
out_pos_request(1, 20251114, "LD").
out_pos_request(13, 20251114, "WR").
out_pos_request(14, 20251114, "D").
out_pos_request(21, 20251114, "D").
out_pos_request(23, 20251114, "D").
out_pos_request(26, 20251114, "AL").
out_pos_request(13, 20251115, "WR").
out_pos_request(21, 20251115, "WR").
out_pos_request(24, 20251115, "WR").
out_pos_request(26, 20251115, "WR").
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("EM", "LD", "Staffs").
vertical_constraint_type("EM", "SE", "Staffs").
vertical_constraint_type("EM", "SN", "Staffs").
vertical_constraint_type("リ－ダ－", "D", "Staffs").
vertical_constraint_type("B", "LD", "Staffs").
vertical_constraint_type("B", "SE", "Staffs").
vertical_constraint_type("B", "SN", "Staffs").
vertical_constraint_type("Newcomers", "SE", "Staffs").
vertical_constraint_type("Newcomers", "SN", "Staffs").
reward(weekend_offs(1, 0), staff(1), 4, 1).
reward(weekend_offs(34, 0), staff(34), 4, 1).
reward(weekend_offs(35, 0), staff(35), 4, 1).
reward(weekend_offs(36, 0), staff(36), 4, 1).
reward(weekend_offs(37, 0), staff(37), 4, 1).
reward(weekend_offs(38, 0), staff(38), 4, 1).
reward(weekend_offs(39, 0), staff(39), 4, 1).
reward(consecutive_holidays(1, 0), staff(1), 4, 1).
reward(national_holiday_offs(1, 0), staff(1), 2, 1).
reward(national_holiday_offs(34, 0), staff(34), 2, 1).
reward(national_holiday_offs(35, 0), staff(35), 2, 1).
reward(national_holiday_offs(36, 0), staff(36), 2, 1).
reward(national_holiday_offs(37, 0), staff(37), 2, 1).
reward(national_holiday_offs(38, 0), staff(38), 2, 1).
reward(national_holiday_offs(39, 0), staff(39), 2, 1).
staff(1, "茶谷　直子", "Heads", "0039121", 0).
staff(2, "深沢　泉", "Deputy Chief Nurse", "0192253", 0).
staff(3, "長沼　みづき", "Nurse", "0055030", 0).
staff(4, "中込　結", "Nurse", "0312931", 0).
staff(5, "丹澤　いず美", "Nurse", "0237017", 0).
staff(6, "田口　沙紀", "Nurse", "0301733", 0).
staff(7, "長田　七彩", "Nurse", "0370203", 0).
staff(8, "竹内　七菜", "Nurse", "0384528", 0).
staff(9, "小原　莉央", "Nurse", "0384368", 0).
staff(10, "杉山　ちはる", "Nurse", "0400254", 0).
staff(11, "今沢　優希", "Nurse", "0400221", 0).
staff(12, "中嶋　千智", "Nurse", "0418973", 0).
staff(13, "北村　莉々", "Nurse", "0418995", 0).
staff(14, "奥脇　香子", "Nurse", "0372342", 0).
staff(15, "駒井　渚", "Nurse", "0469508", 0).
staff(16, "遠藤　遥香", "Nurse", "0606151", 0).
staff(17, "岡本　莉里衣", "Nurse", "0606173", 0).
staff(18, "三枝　栄江", "Deputy Chief Nurse", "0019822", 0).
staff(19, "渡邉　裕美", "Deputy Chief Nurse", "0150172", 0).
staff(20, "山寺　智美", "Nurse", "0312760", 0).
staff(21, "岡本　こずえ", "Nurse", "0355317", 0).
staff(22, "小林　利里愛", "Nurse", "0384448", 0).
staff(23, "長谷川　千華", "Nurse", "0329441", 0).
staff(24, "初鹿　春花", "Nurse", "0384596", 0).
staff(25, "福田　優花", "Nurse", "0400152", 0).
staff(26, "長嶺　世奈", "Nurse", "0399696", 0).
staff(27, "名取　遥", "Nurse", "0418962", 0).
staff(28, "望月　夢音香", "Nurse", "0418940", 0).
staff(29, "田中　理子", "Nurse", "0469576", 0).
staff(30, "麻川　のん子", "Nurse", "0469372", 0).
staff(31, "太田　真叶", "Nurse", "0606162", 0).
staff(32, "安藤　優李", "Nurse", "0606116", 0).
staff(33, "平山　綾乃", "Nurse", "0606548", 0).
staff(34, "堀内　裕香", "Nurse", "0212176", 0).
staff(35, "前嶋　歩美", "Nurse", "0269415", 0).
staff(36, "補）入倉　留美", "補助者", "0367788", 0).
staff(37, "派）三木　由加里", "補助者", "0604251", 0).
staff(38, "補）千塚　久子", "補助者", "0033875", 0).
staff(39, "派）小林　祐子", "補助者", "0630628", 0).
out_assigned(4, 20251012, "SE").
out_assigned(6, 20251012, "SN").
out_assigned(8, 20251012, "D").
out_assigned(10, 20251012, "D").
out_assigned(11, 20251012, "SN").
out_assigned(14, 20251012, "LD").
out_assigned(15, 20251012, "LD").
out_assigned(16, 20251012, "SE").
out_assigned(20, 20251012, "LD").
out_assigned(21, 20251012, "LD").
out_assigned(22, 20251012, "SN").
out_assigned(24, 20251012, "D").
out_assigned(25, 20251012, "SE").
out_assigned(29, 20251012, "SE").
out_assigned(30, 20251012, "SN").
out_assigned(32, 20251012, "D").
out_assigned(5, 20251012, "HL").
out_assigned(7, 20251012, "WR").
out_assigned(12, 20251012, "WR").
out_assigned(13, 20251012, "WR").
out_assigned(17, 20251012, "AL").
out_assigned(23, 20251012, "HL").
out_assigned(28, 20251012, "WR").
out_assigned(31, 20251012, "WR").
out_assigned(2, 20251012, "WR").
out_assigned(3, 20251012, "WR").
out_assigned(9, 20251012, "WR").
out_assigned(18, 20251012, "WR").
out_assigned(19, 20251012, "WR").
out_assigned(26, 20251012, "WR").
out_assigned(27, 20251012, "WR").
out_assigned(33, 20251012, "WR").
out_assigned(2, 20251013, "D").
out_assigned(4, 20251013, "SN").
out_assigned(8, 20251013, "LD").
out_assigned(10, 20251013, "LD").
out_assigned(13, 20251013, "SE").
out_assigned(14, 20251013, "SE").
out_assigned(16, 20251013, "SN").
out_assigned(18, 20251013, "D").
out_assigned(19, 20251013, "LD").
out_assigned(21, 20251013, "SE").
out_assigned(24, 20251013, "LD").
out_assigned(25, 20251013, "SN").
out_assigned(27, 20251013, "D").
out_assigned(29, 20251013, "SN").
out_assigned(32, 20251013, "SE").
out_assigned(33, 20251013, "D").
out_assigned(5, 20251013, "PH").
out_assigned(7, 20251013, "PH").
out_assigned(12, 20251013, "PH").
out_assigned(17, 20251013, "PH").
out_assigned(23, 20251013, "PH").
out_assigned(26, 20251013, "PH").
out_assigned(28, 20251013, "PH").
out_assigned(31, 20251013, "PH").
out_assigned(3, 20251013, "PH").
out_assigned(6, 20251013, "PH").
out_assigned(9, 20251013, "PH").
out_assigned(11, 20251013, "PH").
out_assigned(15, 20251013, "PH").
out_assigned(20, 20251013, "PH").
out_assigned(22, 20251013, "PH").
out_assigned(30, 20251013, "PH").
out_assigned(2, 20251014, "D").
out_assigned(3, 20251014, "LM").
out_assigned(5, 20251014, "D").
out_assigned(7, 20251014, "LD").
out_assigned(9, 20251014, "D").
out_assigned(10, 20251014, "SE").
out_assigned(11, 20251014, "SE").
out_assigned(13, 20251014, "SN").
out_assigned(14, 20251014, "SN").
out_assigned(15, 20251014, "D").
out_assigned(17, 20251014, "LD").
out_assigned(18, 20251014, "D").
out_assigned(19, 20251014, "SE").
out_assigned(21, 20251014, "SN").
out_assigned(22, 20251014, "LD").
out_assigned(23, 20251014, "D").
out_assigned(24, 20251014, "SE").
out_assigned(30, 20251014, "D").
out_assigned(31, 20251014, "D").
out_assigned(32, 20251014, "SN").
out_assigned(33, 20251014, "LD").
out_assigned(12, 20251014, "WR").
out_assigned(26, 20251014, "AL").
out_assigned(27, 20251014, "AL").
out_assigned(28, 20251014, "WR").
out_assigned(4, 20251014, "WR").
out_assigned(6, 20251014, "WR").
out_assigned(8, 20251014, "WR").
out_assigned(16, 20251014, "WR").
out_assigned(20, 20251014, "WR").
out_assigned(25, 20251014, "WR").
out_assigned(29, 20251014, "WR").
out_assigned(3, 20251015, "SE").
out_assigned(4, 20251015, "D").
out_assigned(5, 20251015, "D").
out_assigned(8, 20251015, "LD").
out_assigned(9, 20251015, "SE").
out_assigned(10, 20251015, "SN").
out_assigned(11, 20251015, "SN").
out_assigned(12, 20251015, "LD").
out_assigned(15, 20251015, "D").
out_assigned(16, 20251015, "D").
out_assigned(18, 20251015, "D").
out_assigned(19, 20251015, "SN").
out_assigned(20, 20251015, "D").
out_assigned(23, 20251015, "LD").
out_assigned(24, 20251015, "SN").
out_assigned(25, 20251015, "LD").
out_assigned(28, 20251015, "SE").
out_assigned(29, 20251015, "SE").
out_assigned(30, 20251015, "D").
out_assigned(31, 20251015, "D").
out_assigned(33, 20251015, "LM").
out_assigned(21, 20251015, "HL").
out_assigned(22, 20251015, "WR").
out_assigned(26, 20251015, "WR").
out_assigned(27, 20251015, "WR").
out_assigned(2, 20251015, "WR").
out_assigned(6, 20251015, "WR").
out_assigned(7, 20251015, "WR").
out_assigned(13, 20251015, "WR").
out_assigned(14, 20251015, "WR").
out_assigned(17, 20251015, "WR").
out_assigned(32, 20251015, "WR").
out_assigned(2, 20251016, "LD").
out_assigned(3, 20251016, "SN").
out_assigned(4, 20251016, "D").
out_assigned(5, 20251016, "D").
out_assigned(6, 20251016, "LM").
out_assigned(7, 20251016, "LD").
out_assigned(8, 20251016, "SE").
out_assigned(9, 20251016, "SN").
out_assigned(12, 20251016, "SE").
out_assigned(13, 20251016, "D").
out_assigned(14, 20251016, "D").
out_assigned(15, 20251016, "D").
out_assigned(18, 20251016, "SE").
out_assigned(20, 20251016, "LD").
out_assigned(21, 20251016, "D").
out_assigned(23, 20251016, "SE").
out_assigned(28, 20251016, "SN").
out_assigned(29, 20251016, "SN").
out_assigned(30, 20251016, "LD").
out_assigned(31, 20251016, "D").
out_assigned(32, 20251016, "D").
out_assigned(19, 20251016, "AL").
out_assigned(27, 20251016, "AL").
out_assigned(10, 20251016, "WR").
out_assigned(11, 20251016, "WR").
out_assigned(16, 20251016, "WR").
out_assigned(17, 20251016, "WR").
out_assigned(22, 20251016, "WR").
out_assigned(24, 20251016, "WR").
out_assigned(25, 20251016, "WR").
out_assigned(26, 20251016, "WR").
out_assigned(33, 20251016, "WR").
out_assigned(2, 20251017, "LM").
out_assigned(7, 20251017, "SE").
out_assigned(8, 20251017, "SN").
out_assigned(10, 20251017, "SE").
out_assigned(11, 20251017, "LD").
out_assigned(12, 20251017, "SN").
out_assigned(13, 20251017, "LD").
out_assigned(15, 20251017, "D").
out_assigned(17, 20251017, "D").
out_assigned(18, 20251017, "SN").
out_assigned(19, 20251017, "D").
out_assigned(20, 20251017, "SE").
out_assigned(21, 20251017, "D").
out_assigned(22, 20251017, "LD").
out_assigned(23, 20251017, "SN").
out_assigned(24, 20251017, "D").
out_assigned(25, 20251017, "LD").
out_assigned(26, 20251017, "D").
out_assigned(30, 20251017, "SE").
out_assigned(31, 20251017, "D").
out_assigned(32, 20251017, "D").
out_assigned(5, 20251017, "WR").
out_assigned(3, 20251017, "WR").
out_assigned(4, 20251017, "WR").
out_assigned(6, 20251017, "WR").
out_assigned(9, 20251017, "WR").
out_assigned(14, 20251017, "WR").
out_assigned(16, 20251017, "WR").
out_assigned(27, 20251017, "WR").
out_assigned(28, 20251017, "WR").
out_assigned(29, 20251017, "WR").
out_assigned(33, 20251017, "WR").
out_assigned(2, 20251018, "SE").
out_assigned(3, 20251018, "D").
out_assigned(5, 20251018, "D").
out_assigned(6, 20251018, "D").
out_assigned(7, 20251018, "SN").
out_assigned(9, 20251018, "LD").
out_assigned(10, 20251018, "SN").
out_assigned(13, 20251018, "SE").
out_assigned(16, 20251018, "LD").
out_assigned(19, 20251018, "D").
out_assigned(20, 20251018, "SN").
out_assigned(22, 20251018, "SE").
out_assigned(24, 20251018, "LD").
out_assigned(25, 20251018, "SE").
out_assigned(26, 20251018, "LD").
out_assigned(30, 20251018, "SN").
out_assigned(11, 20251018, "WR").
out_assigned(12, 20251018, "HL").
out_assigned(14, 20251018, "WR").
out_assigned(15, 20251018, "WR").
out_assigned(17, 20251018, "AL").
out_assigned(29, 20251018, "WR").
out_assigned(32, 20251018, "WR").
out_assigned(33, 20251018, "WR").
out_assigned(4, 20251018, "WR").
out_assigned(8, 20251018, "WR").
out_assigned(18, 20251018, "WR").
out_assigned(21, 20251018, "WR").
out_assigned(23, 20251018, "WR").
out_assigned(27, 20251018, "WR").
out_assigned(28, 20251018, "WR").
out_assigned(31, 20251018, "WR").
out_assigned(2, 20251019, "SN").
out_assigned(4, 20251019, "LD").
out_assigned(5, 20251019, "D").
out_assigned(8, 20251019, "SE").
out_assigned(12, 20251019, "SE").
out_assigned(13, 20251019, "SN").
out_assigned(17, 20251019, "LD").
out_assigned(18, 20251019, "LD").
out_assigned(19, 20251019, "D").
out_assigned(22, 20251019, "SN").
out_assigned(23, 20251019, "D").
out_assigned(24, 20251019, "SE").
out_assigned(25, 20251019, "SN").
out_assigned(27, 20251019, "D").
out_assigned(28, 20251019, "LD").
out_assigned(31, 20251019, "SE").
out_assigned(11, 20251019, "WR").
out_assigned(14, 20251019, "WR").
out_assigned(15, 20251019, "WR").
out_assigned(26, 20251019, "AL").
out_assigned(29, 20251019, "WR").
out_assigned(32, 20251019, "AL").
out_assigned(33, 20251019, "AL").
out_assigned(3, 20251019, "WR").
out_assigned(6, 20251019, "WR").
out_assigned(7, 20251019, "WR").
out_assigned(9, 20251019, "WR").
out_assigned(10, 20251019, "WR").
out_assigned(16, 20251019, "WR").
out_assigned(20, 20251019, "WR").
out_assigned(21, 20251019, "WR").
out_assigned(30, 20251019, "WR").
out_assigned(1, 20251020, "D").
out_assigned(3, 20251020, "SE").
out_assigned(5, 20251020, "D").
out_assigned(6, 20251020, "D").
out_assigned(7, 20251020, "LD").
out_assigned(8, 20251020, "SN").
out_assigned(10, 20251020, "SE").
out_assigned(11, 20251020, "LD").
out_assigned(12, 20251020, "SN").
out_assigned(14, 20251020, "D").
out_assigned(15, 20251020, "D").
out_assigned(16, 20251020, "D").
out_assigned(19, 20251020, "SE").
out_assigned(20, 20251020, "LM").
out_assigned(21, 20251020, "D").
out_assigned(23, 20251020, "LD").
out_assigned(24, 20251020, "SN").
out_assigned(27, 20251020, "D").
out_assigned(29, 20251020, "LD").
out_assigned(31, 20251020, "SN").
out_assigned(32, 20251020, "D").
out_assigned(33, 20251020, "SE").
out_assigned(26, 20251020, "AL").
out_assigned(2, 20251020, "WR").
out_assigned(4, 20251020, "WR").
out_assigned(9, 20251020, "WR").
out_assigned(13, 20251020, "WR").
out_assigned(17, 20251020, "WR").
out_assigned(18, 20251020, "WR").
out_assigned(22, 20251020, "WR").
out_assigned(25, 20251020, "WR").
out_assigned(28, 20251020, "WR").
out_assigned(30, 20251020, "WR").
out_assigned(1, 20251021, "D").
out_assigned(2, 20251021, "SE").
out_assigned(3, 20251021, "SN").
out_assigned(5, 20251021, "D").
out_assigned(6, 20251021, "D").
out_assigned(7, 20251021, "SE").
out_assigned(9, 20251021, "LD").
out_assigned(10, 20251021, "SN").
out_assigned(11, 20251021, "LM").
out_assigned(14, 20251021, "LD").
out_assigned(15, 20251021, "D").
out_assigned(16, 20251021, "D").
out_assigned(18, 20251021, "D").
out_assigned(19, 20251021, "SN").
out_assigned(20, 20251021, "SE").
out_assigned(22, 20251021, "D").
out_assigned(25, 20251021, "SE").
out_assigned(26, 20251021, "LD").
out_assigned(27, 20251021, "D").
out_assigned(28, 20251021, "LD").
out_assigned(32, 20251021, "D").
out_assigned(33, 20251021, "SN").
out_assigned(8, 20251021, "WR").
out_assigned(21, 20251021, "WR").
out_assigned(23, 20251021, "AL").
out_assigned(4, 20251021, "WR").
out_assigned(12, 20251021, "WR").
out_assigned(13, 20251021, "WR").
out_assigned(17, 20251021, "WR").
out_assigned(24, 20251021, "WR").
out_assigned(29, 20251021, "WR").
out_assigned(30, 20251021, "WR").
out_assigned(31, 20251021, "WR").
out_assigned(1, 20251022, "D").
out_assigned(2, 20251022, "SN").
out_assigned(4, 20251022, "LD").
out_assigned(5, 20251022, "D").
out_assigned(6, 20251022, "LD").
out_assigned(7, 20251022, "SN").
out_assigned(11, 20251022, "SE").
out_assigned(12, 20251022, "SE").
out_assigned(15, 20251022, "D").
out_assigned(16, 20251022, "LM").
out_assigned(17, 20251022, "D").
out_assigned(18, 20251022, "D").
out_assigned(20, 20251022, "SN").
out_assigned(21, 20251022, "LD").
out_assigned(22, 20251022, "D").
out_assigned(23, 20251022, "D").
out_assigned(24, 20251022, "LD").
out_assigned(25, 20251022, "SN").
out_assigned(26, 20251022, "SE").
out_assigned(29, 20251022, "D").
out_assigned(31, 20251022, "D").
out_assigned(32, 20251022, "SE").
out_assigned(8, 20251022, "WR").
out_assigned(9, 20251022, "AL").
out_assigned(30, 20251022, "HL").
out_assigned(33, 20251022, "WR").
out_assigned(3, 20251022, "WR").
out_assigned(10, 20251022, "WR").
out_assigned(13, 20251022, "WR").
out_assigned(14, 20251022, "WR").
out_assigned(19, 20251022, "WR").
out_assigned(27, 20251022, "WR").
out_assigned(28, 20251022, "WR").
out_assigned(3, 20251023, "LD").
out_assigned(4, 20251023, "SE").
out_assigned(5, 20251023, "LM").
out_assigned(6, 20251023, "SE").
out_assigned(10, 20251023, "LD").
out_assigned(11, 20251023, "SN").
out_assigned(12, 20251023, "SN").
out_assigned(13, 20251023, "D").
out_assigned(14, 20251023, "D").
out_assigned(15, 20251023, "D").
out_assigned(17, 20251023, "D").
out_assigned(18, 20251023, "SE").
out_assigned(19, 20251023, "D").
out_assigned(22, 20251023, "LD").
out_assigned(23, 20251023, "SE").
out_assigned(26, 20251023, "SN").
out_assigned(27, 20251023, "D").
out_assigned(28, 20251023, "LD").
out_assigned(29, 20251023, "D").
out_assigned(32, 20251023, "SN").
out_assigned(33, 20251023, "D").
out_assigned(8, 20251023, "WR").
out_assigned(9, 20251023, "AL").
out_assigned(20, 20251023, "WR").
out_assigned(21, 20251023, "WR").
out_assigned(30, 20251023, "AL").
out_assigned(31, 20251023, "AL").
out_assigned(2, 20251023, "WR").
out_assigned(7, 20251023, "WR").
out_assigned(16, 20251023, "WR").
out_assigned(24, 20251023, "WR").
out_assigned(25, 20251023, "WR").
out_assigned(2, 20251024, "LM").
out_assigned(4, 20251024, "SN").
out_assigned(6, 20251024, "SN").
out_assigned(7, 20251024, "SE").
out_assigned(8, 20251024, "SE").
out_assigned(9, 20251024, "LD").
out_assigned(13, 20251024, "D").
out_assigned(14, 20251024, "D").
out_assigned(15, 20251024, "LD").
out_assigned(17, 20251024, "D").
out_assigned(18, 20251024, "SN").
out_assigned(19, 20251024, "D").
out_assigned(20, 20251024, "D").
out_assigned(21, 20251024, "SE").
out_assigned(23, 20251024, "SN").
out_assigned(24, 20251024, "D").
out_assigned(25, 20251024, "LD").
out_assigned(27, 20251024, "D").
out_assigned(29, 20251024, "LD").
out_assigned(30, 20251024, "SE").
out_assigned(33, 20251024, "D").
out_assigned(3, 20251024, "WR").
out_assigned(5, 20251024, "WR").
out_assigned(10, 20251024, "WR").
out_assigned(11, 20251024, "WR").
out_assigned(12, 20251024, "WR").
out_assigned(16, 20251024, "WR").
out_assigned(22, 20251024, "WR").
out_assigned(26, 20251024, "WR").
out_assigned(28, 20251024, "WR").
out_assigned(31, 20251024, "WR").
out_assigned(32, 20251024, "WR").
out_assigned(1, 20251025, "SE").
out_assigned(3, 20251025, "LD").
out_assigned(7, 20251025, "SN").
out_assigned(8, 20251025, "SN").
out_assigned(9, 20251025, "LM").
out_assigned(11, 20251025, "LD").
out_assigned(13, 20251025, "SE").
out_assigned(14, 20251025, "SE").
out_assigned(16, 20251025, "D").
out_assigned(19, 20251025, "D").
out_assigned(20, 20251025, "LD").
out_assigned(21, 20251025, "SN").
out_assigned(24, 20251025, "D").
out_assigned(26, 20251025, "LD").
out_assigned(27, 20251025, "D").
out_assigned(28, 20251025, "SE").
out_assigned(29, 20251025, "SE").
out_assigned(30, 20251025, "SN").
out_assigned(2, 20251025, "WR").
out_assigned(4, 20251025, "HL").
out_assigned(5, 20251025, "WR").
out_assigned(17, 20251025, "AL").
out_assigned(32, 20251025, "WR").
out_assigned(33, 20251025, "WR").
out_assigned(6, 20251025, "WR").
out_assigned(10, 20251025, "WR").
out_assigned(12, 20251025, "WR").
out_assigned(15, 20251025, "WR").
out_assigned(18, 20251025, "WR").
out_assigned(22, 20251025, "WR").
out_assigned(23, 20251025, "WR").
out_assigned(25, 20251025, "WR").
out_assigned(31, 20251025, "WR").
out_assigned(3, 20251026, "SE").
out_assigned(6, 20251026, "D").
out_assigned(9, 20251026, "SE").
out_assigned(10, 20251026, "LD").
out_assigned(12, 20251026, "LD").
out_assigned(13, 20251026, "SN").
out_assigned(14, 20251026, "SN").
out_assigned(18, 20251026, "LD").
out_assigned(22, 20251026, "SE").
out_assigned(23, 20251026, "D").
out_assigned(24, 20251026, "LD").
out_assigned(26, 20251026, "SE").
out_assigned(28, 20251026, "SN").
out_assigned(29, 20251026, "SN").
out_assigned(31, 20251026, "D").
out_assigned(33, 20251026, "D").
out_assigned(2, 20251026, "WR").
out_assigned(4, 20251026, "HL").
out_assigned(17, 20251026, "AL").
out_assigned(20, 20251026, "AL").
out_assigned(21, 20251026, "WR").
out_assigned(32, 20251026, "AL").
out_assigned(5, 20251026, "WR").
out_assigned(7, 20251026, "WR").
out_assigned(8, 20251026, "WR").
out_assigned(11, 20251026, "WR").
out_assigned(15, 20251026, "WR").
out_assigned(16, 20251026, "WR").
out_assigned(19, 20251026, "WR").
out_assigned(25, 20251026, "WR").
out_assigned(27, 20251026, "WR").
out_assigned(30, 20251026, "WR").
out_assigned(3, 20251027, "SN").
out_assigned(4, 20251027, "D").
out_assigned(5, 20251027, "D").
out_assigned(6, 20251027, "D").
out_assigned(7, 20251027, "LD").
out_assigned(8, 20251027, "LM").
out_assigned(9, 20251027, "SN").
out_assigned(10, 20251027, "SE").
out_assigned(11, 20251027, "LD").
out_assigned(16, 20251027, "SE").
out_assigned(17, 20251027, "D").
out_assigned(19, 20251027, "D").
out_assigned(20, 20251027, "D").
out_assigned(22, 20251027, "SN").
out_assigned(23, 20251027, "LD").
out_assigned(24, 20251027, "SE").
out_assigned(25, 20251027, "SE").
out_assigned(26, 20251027, "SN").
out_assigned(27, 20251027, "D").
out_assigned(31, 20251027, "D").
out_assigned(33, 20251027, "LD").
out_assigned(2, 20251027, "WR").
out_assigned(14, 20251027, "WR").
out_assigned(18, 20251027, "HL").
out_assigned(32, 20251027, "HL").
out_assigned(12, 20251027, "WR").
out_assigned(13, 20251027, "WR").
out_assigned(15, 20251027, "WR").
out_assigned(21, 20251027, "WR").
out_assigned(28, 20251027, "WR").
out_assigned(29, 20251027, "WR").
out_assigned(30, 20251027, "WR").
out_assigned(1, 20251028, "D").
out_assigned(2, 20251028, "LM").
out_assigned(4, 20251028, "LD").
out_assigned(5, 20251028, "D").
out_assigned(7, 20251028, "SE").
out_assigned(10, 20251028, "SN").
out_assigned(11, 20251028, "SE").
out_assigned(12, 20251028, "D").
out_assigned(13, 20251028, "LD").
out_assigned(15, 20251028, "D").
out_assigned(16, 20251028, "SN").
out_assigned(17, 20251028, "D").
out_assigned(19, 20251028, "LD").
out_assigned(20, 20251028, "D").
out_assigned(23, 20251028, "SE").
out_assigned(24, 20251028, "SN").
out_assigned(25, 20251028, "SN").
out_assigned(27, 20251028, "LD").
out_assigned(29, 20251028, "EM").
out_assigned(30, 20251028, "D").
out_assigned(31, 20251028, "D").
out_assigned(32, 20251028, "D").
out_assigned(33, 20251028, "SE").
out_assigned(14, 20251028, "AL").
out_assigned(18, 20251028, "AL").
out_assigned(22, 20251028, "WR").
out_assigned(3, 20251028, "WR").
out_assigned(6, 20251028, "WR").
out_assigned(8, 20251028, "WR").
out_assigned(9, 20251028, "WR").
out_assigned(21, 20251028, "WR").
out_assigned(26, 20251028, "WR").
out_assigned(28, 20251028, "WR").
out_assigned(1, 20251029, "D").
out_assigned(4, 20251029, "LM").
out_assigned(5, 20251029, "D").
out_assigned(6, 20251029, "SE").
out_assigned(7, 20251029, "SN").
out_assigned(8, 20251029, "LD").
out_assigned(9, 20251029, "SE").
out_assigned(11, 20251029, "SN").
out_assigned(12, 20251029, "LD").
out_assigned(15, 20251029, "D").
out_assigned(17, 20251029, "D").
out_assigned(18, 20251029, "SE").
out_assigned(19, 20251029, "SE").
out_assigned(20, 20251029, "D").
out_assigned(21, 20251029, "D").
out_assigned(23, 20251029, "SN").
out_assigned(26, 20251029, "D").
out_assigned(28, 20251029, "LD").
out_assigned(30, 20251029, "D").
out_assigned(31, 20251029, "LD").
out_assigned(32, 20251029, "D").
out_assigned(33, 20251029, "SN").
out_assigned(3, 20251029, "WR").
out_assigned(14, 20251029, "HL").
out_assigned(16, 20251029, "AL").
out_assigned(22, 20251029, "WR").
out_assigned(2, 20251029, "WR").
out_assigned(10, 20251029, "WR").
out_assigned(13, 20251029, "WR").
out_assigned(24, 20251029, "WR").
out_assigned(25, 20251029, "WR").
out_assigned(27, 20251029, "WR").
out_assigned(29, 20251029, "WR").
out_assigned(2, 20251030, "LD").
out_assigned(4, 20251030, "SE").
out_assigned(5, 20251030, "D").
out_assigned(6, 20251030, "SN").
out_assigned(9, 20251030, "SN").
out_assigned(10, 20251030, "D").
out_assigned(13, 20251030, "LD").
out_assigned(15, 20251030, "SE").
out_assigned(17, 20251030, "D").
out_assigned(18, 20251030, "SN").
out_assigned(19, 20251030, "SN").
out_assigned(20, 20251030, "LD").
out_assigned(21, 20251030, "D").
out_assigned(24, 20251030, "SE").
out_assigned(25, 20251030, "LM").
out_assigned(26, 20251030, "D").
out_assigned(27, 20251030, "LD").
out_assigned(28, 20251030, "SE").
out_assigned(29, 20251030, "D").
out_assigned(30, 20251030, "D").
out_assigned(32, 20251030, "D").
out_assigned(3, 20251030, "AL").
out_assigned(14, 20251030, "HL").
out_assigned(16, 20251030, "AL").
out_assigned(22, 20251030, "AL").
out_assigned(33, 20251030, "AL").
out_assigned(7, 20251030, "WR").
out_assigned(8, 20251030, "WR").
out_assigned(11, 20251030, "WR").
out_assigned(12, 20251030, "WR").
out_assigned(23, 20251030, "WR").
out_assigned(31, 20251030, "WR").
out_assigned(2, 20251031, "SE").
out_assigned(4, 20251031, "SN").
out_assigned(5, 20251031, "SE").
out_assigned(8, 20251031, "LD").
out_assigned(10, 20251031, "LD").
out_assigned(12, 20251031, "D").
out_assigned(13, 20251031, "LM").
out_assigned(14, 20251031, "D").
out_assigned(15, 20251031, "SN").
out_assigned(17, 20251031, "D").
out_assigned(21, 20251031, "D").
out_assigned(23, 20251031, "LD").
out_assigned(24, 20251031, "SN").
out_assigned(25, 20251031, "SE").
out_assigned(26, 20251031, "D").
out_assigned(28, 20251031, "SN").
out_assigned(29, 20251031, "D").
out_assigned(30, 20251031, "SE").
out_assigned(31, 20251031, "LD").
out_assigned(32, 20251031, "D").
out_assigned(33, 20251031, "D").
out_assigned(3, 20251031, "WR").
out_assigned(16, 20251031, "AL").
out_assigned(22, 20251031, "AL").
out_assigned(6, 20251031, "WR").
out_assigned(7, 20251031, "WR").
out_assigned(9, 20251031, "WR").
out_assigned(11, 20251031, "WR").
out_assigned(18, 20251031, "WR").
out_assigned(19, 20251031, "WR").
out_assigned(20, 20251031, "WR").
out_assigned(27, 20251031, "WR").
out_assigned(2, 20251101, "SN").
out_assigned(5, 20251101, "SN").
out_assigned(7, 20251101, "LD").
out_assigned(8, 20251101, "SE").
out_assigned(9, 20251101, "LD").
out_assigned(12, 20251101, "D").
out_assigned(13, 20251101, "SE").
out_assigned(14, 20251101, "D").
out_assigned(18, 20251101, "LD").
out_assigned(21, 20251101, "D").
out_assigned(22, 20251101, "LD").
out_assigned(25, 20251101, "SN").
out_assigned(26, 20251101, "SE").
out_assigned(30, 20251101, "SN").
out_assigned(31, 20251101, "SE").
out_assigned(33, 20251101, "D").
out_assigned(4, 20251101, "WR").
out_assigned(11, 20251101, "WR").
out_assigned(16, 20251101, "WR").
out_assigned(17, 20251101, "WR").
out_assigned(28, 20251101, "HL").
out_assigned(3, 20251101, "WR").
out_assigned(6, 20251101, "WR").
out_assigned(10, 20251101, "WR").
out_assigned(15, 20251101, "WR").
out_assigned(19, 20251101, "WR").
out_assigned(20, 20251101, "WR").
out_assigned(23, 20251101, "WR").
out_assigned(24, 20251101, "WR").
out_assigned(27, 20251101, "WR").
out_assigned(29, 20251101, "WR").
out_assigned(32, 20251101, "WR").
out_assigned(3, 20251102, "LD").
out_assigned(6, 20251102, "D").
out_assigned(7, 20251102, "SE").
out_assigned(8, 20251102, "SN").
out_assigned(10, 20251102, "D").
out_assigned(12, 20251102, "LD").
out_assigned(13, 20251102, "SN").
out_assigned(14, 20251102, "D").
out_assigned(15, 20251102, "SE").
out_assigned(21, 20251102, "SE").
out_assigned(23, 20251102, "D").
out_assigned(26, 20251102, "SN").
out_assigned(27, 20251102, "SE").
out_assigned(29, 20251102, "LD").
out_assigned(31, 20251102, "SN").
out_assigned(32, 20251102, "LD").
out_assigned(4, 20251102, "AL").
out_assigned(11, 20251102, "AL").
out_assigned(16, 20251102, "WR").
out_assigned(20, 20251102, "WR").
out_assigned(22, 20251102, "HL").
out_assigned(28, 20251102, "HL").
out_assigned(2, 20251102, "WR").
out_assigned(5, 20251102, "WR").
out_assigned(9, 20251102, "WR").
out_assigned(17, 20251102, "WR").
out_assigned(18, 20251102, "WR").
out_assigned(19, 20251102, "WR").
out_assigned(24, 20251102, "WR").
out_assigned(25, 20251102, "WR").
out_assigned(30, 20251102, "WR").
out_assigned(33, 20251102, "WR").
out_assigned(2, 20251103, "D").
out_assigned(4, 20251103, "LD").
out_assigned(6, 20251103, "D").
out_assigned(7, 20251103, "SN").
out_assigned(12, 20251103, "SE").
out_assigned(15, 20251103, "SN").
out_assigned(16, 20251103, "LD").
out_assigned(17, 20251103, "SE").
out_assigned(18, 20251103, "D").
out_assigned(21, 20251103, "SN").
out_assigned(22, 20251103, "SE").
out_assigned(27, 20251103, "SN").
out_assigned(28, 20251103, "LD").
out_assigned(29, 20251103, "SE").
out_assigned(30, 20251103, "LD").
out_assigned(33, 20251103, "D").
out_assigned(11, 20251103, "AL").
out_assigned(20, 20251103, "PH").
out_assigned(26, 20251103, "PH").
out_assigned(3, 20251103, "PH").
out_assigned(5, 20251103, "PH").
out_assigned(8, 20251103, "PH").
out_assigned(9, 20251103, "PH").
out_assigned(10, 20251103, "PH").
out_assigned(13, 20251103, "PH").
out_assigned(14, 20251103, "PH").
out_assigned(19, 20251103, "PH").
out_assigned(23, 20251103, "PH").
out_assigned(24, 20251103, "PH").
out_assigned(25, 20251103, "PH").
out_assigned(31, 20251103, "PH").
out_assigned(32, 20251103, "PH").
out_assigned(1, 20251104, "D").
out_assigned(2, 20251104, "D").
out_assigned(3, 20251104, "D").
out_assigned(5, 20251104, "D").
out_assigned(6, 20251104, "LD").
out_assigned(9, 20251104, "SE").
out_assigned(11, 20251104, "SE").
out_assigned(12, 20251104, "SN").
out_assigned(13, 20251104, "LD").
out_assigned(14, 20251104, "D").
out_assigned(17, 20251104, "SN").
out_assigned(18, 20251104, "D").
out_assigned(19, 20251104, "D").
out_assigned(20, 20251104, "SE").
out_assigned(22, 20251104, "SN").
out_assigned(24, 20251104, "LD").
out_assigned(26, 20251104, "LM").
out_assigned(28, 20251104, "SE").
out_assigned(29, 20251104, "SN").
out_assigned(31, 20251104, "D").
out_assigned(32, 20251104, "LD").
out_assigned(33, 20251104, "D").
out_assigned(4, 20251104, "WR").
out_assigned(7, 20251104, "WR").
out_assigned(8, 20251104, "WR").
out_assigned(10, 20251104, "WR").
out_assigned(15, 20251104, "WR").
out_assigned(16, 20251104, "WR").
out_assigned(21, 20251104, "WR").
out_assigned(23, 20251104, "WR").
out_assigned(25, 20251104, "WR").
out_assigned(27, 20251104, "WR").
out_assigned(30, 20251104, "WR").
out_assigned(1, 20251105, "D").
out_assigned(2, 20251105, "LD").
out_assigned(4, 20251105, "SE").
out_assigned(5, 20251105, "D").
out_assigned(6, 20251105, "LM").
out_assigned(8, 20251105, "LD").
out_assigned(9, 20251105, "SN").
out_assigned(11, 20251105, "SN").
out_assigned(13, 20251105, "SE").
out_assigned(14, 20251105, "D").
out_assigned(15, 20251105, "D").
out_assigned(16, 20251105, "D").
out_assigned(18, 20251105, "D").
out_assigned(19, 20251105, "D").
out_assigned(20, 20251105, "SN").
out_assigned(24, 20251105, "SE").
out_assigned(25, 20251105, "LD").
out_assigned(26, 20251105, "SE").
out_assigned(28, 20251105, "SN").
out_assigned(30, 20251105, "LD").
out_assigned(31, 20251105, "D").
out_assigned(33, 20251105, "D").
out_assigned(3, 20251105, "WR").
out_assigned(7, 20251105, "WR").
out_assigned(10, 20251105, "WR").
out_assigned(12, 20251105, "WR").
out_assigned(17, 20251105, "WR").
out_assigned(21, 20251105, "WR").
out_assigned(22, 20251105, "WR").
out_assigned(23, 20251105, "WR").
out_assigned(27, 20251105, "WR").
out_assigned(29, 20251105, "WR").
out_assigned(32, 20251105, "WR").
out_assigned(2, 20251106, "LM").
out_assigned(3, 20251106, "D").
out_assigned(4, 20251106, "SN").
out_assigned(5, 20251106, "D").
out_assigned(7, 20251106, "D").
out_assigned(8, 20251106, "SE").
out_assigned(10, 20251106, "LD").
out_assigned(12, 20251106, "LD").
out_assigned(13, 20251106, "SN").
out_assigned(14, 20251106, "D").
out_assigned(15, 20251106, "D").
out_assigned(16, 20251106, "D").
out_assigned(17, 20251106, "SE").
out_assigned(19, 20251106, "LD").
out_assigned(21, 20251106, "LD").
out_assigned(22, 20251106, "SE").
out_assigned(24, 20251106, "SN").
out_assigned(26, 20251106, "SN").
out_assigned(27, 20251106, "SE").
out_assigned(31, 20251106, "D").
out_assigned(33, 20251106, "D").
out_assigned(6, 20251106, "WR").
out_assigned(9, 20251106, "WR").
out_assigned(11, 20251106, "WR").
out_assigned(18, 20251106, "WR").
out_assigned(20, 20251106, "WR").
out_assigned(23, 20251106, "WR").
out_assigned(25, 20251106, "WR").
out_assigned(28, 20251106, "WR").
out_assigned(29, 20251106, "WR").
out_assigned(30, 20251106, "WR").
out_assigned(32, 20251106, "WR").
out_assigned(3, 20251107, "D").
out_assigned(5, 20251107, "D").
out_assigned(7, 20251107, "D").
out_assigned(8, 20251107, "SN").
out_assigned(9, 20251107, "LD").
out_assigned(10, 20251107, "SE").
out_assigned(11, 20251107, "LD").
out_assigned(12, 20251107, "SE").
out_assigned(15, 20251107, "D").
out_assigned(16, 20251107, "D").
out_assigned(17, 20251107, "SN").
out_assigned(19, 20251107, "LM").
out_assigned(20, 20251107, "SE").
out_assigned(22, 20251107, "SN").
out_assigned(23, 20251107, "SE").
out_assigned(25, 20251107, "D").
out_assigned(27, 20251107, "SN").
out_assigned(28, 20251107, "D").
out_assigned(29, 20251107, "LD").
out_assigned(30, 20251107, "LD").
out_assigned(33, 20251107, "D").
out_assigned(6, 20251107, "HL").
out_assigned(13, 20251107, "WR").
out_assigned(31, 20251107, "HL").
out_assigned(2, 20251107, "WR").
out_assigned(4, 20251107, "WR").
out_assigned(14, 20251107, "WR").
out_assigned(18, 20251107, "WR").
out_assigned(21, 20251107, "WR").
out_assigned(24, 20251107, "WR").
out_assigned(26, 20251107, "WR").
out_assigned(32, 20251107, "WR").
out_assigned(3, 20251108, "D").
out_assigned(5, 20251108, "LD").
out_assigned(9, 20251108, "SE").
out_assigned(10, 20251108, "SN").
out_assigned(11, 20251108, "SE").
out_assigned(12, 20251108, "SN").
out_assigned(13, 20251108, "LD").
out_assigned(16, 20251108, "D").
out_assigned(20, 20251108, "SN").
out_assigned(21, 20251108, "D").
out_assigned(23, 20251108, "SN").
out_assigned(25, 20251108, "LD").
out_assigned(26, 20251108, "LD").
out_assigned(28, 20251108, "SE").
out_assigned(29, 20251108, "LM").
out_assigned(30, 20251108, "SE").
out_assigned(32, 20251108, "D").
out_assigned(33, 20251108, "LM").
out_assigned(4, 20251108, "WR").
out_assigned(6, 20251108, "HL").
out_assigned(7, 20251108, "WR").
out_assigned(27, 20251108, "WR").
out_assigned(31, 20251108, "WR").
out_assigned(2, 20251108, "WR").
out_assigned(8, 20251108, "WR").
out_assigned(14, 20251108, "WR").
out_assigned(15, 20251108, "WR").
out_assigned(17, 20251108, "WR").
out_assigned(18, 20251108, "WR").
out_assigned(19, 20251108, "WR").
out_assigned(22, 20251108, "WR").
out_assigned(24, 20251108, "WR").
penalty(staff_hard_ub(20251014, "D", "リ－ダ－", 3, 4), num_staffs(2, "D", "リ－ダ－"), 1, 10).
penalty(staff_hard_ub(20251107, "D", "リ－ダ－", 3, 4), num_staffs(26, "D", "リ－ダ－"), 1, 10).
out_stats(num_consecutive_holidays(2, 0, 2, 2)).
out_stats(num_consecutive_holidays(3, 0, 2, 2)).
out_stats(num_consecutive_holidays(4, 0, 3, 3)).
out_stats(num_consecutive_holidays(5, 0, 2, 2)).
out_stats(num_consecutive_holidays(6, 0, 2, 2)).
out_stats(num_consecutive_holidays(7, 0, 3, 3)).
out_stats(num_consecutive_holidays(8, 0, 2, 2)).
out_stats(num_consecutive_holidays(9, 0, 3, 3)).
out_stats(num_consecutive_holidays(10, 0, 2, 2)).
out_stats(num_consecutive_holidays(11, 0, 2, 2)).
out_stats(num_consecutive_holidays(12, 0, 2, 2)).
out_stats(num_consecutive_holidays(13, 0, 2, 2)).
out_stats(num_consecutive_holidays(14, 0, 2, 2)).
out_stats(num_consecutive_holidays(15, 0, 2, 2)).
out_stats(num_consecutive_holidays(16, 0, 2, 2)).
out_stats(num_consecutive_holidays(17, 0, 3, 3)).
out_stats(num_consecutive_holidays(18, 0, 2, 2)).
out_stats(num_consecutive_holidays(19, 0, 1, 1)).
out_stats(num_consecutive_holidays(20, 0, 2, 2)).
out_stats(num_consecutive_holidays(21, 0, 3, 3)).
out_stats(num_consecutive_holidays(22, 0, 3, 3)).
out_stats(num_consecutive_holidays(23, 0, 1, 1)).
out_stats(num_consecutive_holidays(24, 0, 2, 2)).
out_stats(num_consecutive_holidays(25, 0, 2, 2)).
out_stats(num_consecutive_holidays(26, 0, 1, 1)).
out_stats(num_consecutive_holidays(27, 0, 2, 2)).
out_stats(num_consecutive_holidays(28, 0, 2, 2)).
out_stats(num_consecutive_holidays(29, 0, 2, 2)).
out_stats(num_consecutive_holidays(30, 0, 2, 2)).
out_stats(num_consecutive_holidays(31, 0, 0, 0)).
out_stats(num_consecutive_holidays(32, 0, 2, 2)).
out_stats(num_consecutive_holidays(33, 0, 2, 2)).
out_stats(num_national_holiday_offs(2, 0, 0, 0)).
out_stats(num_national_holiday_offs(3, 0, 2, 2)).
out_stats(num_national_holiday_offs(4, 0, 0, 0)).
out_stats(num_national_holiday_offs(5, 0, 2, 2)).
out_stats(num_national_holiday_offs(6, 0, 1, 1)).
out_stats(num_national_holiday_offs(7, 0, 1, 1)).
out_stats(num_national_holiday_offs(8, 0, 1, 1)).
out_stats(num_national_holiday_offs(9, 0, 2, 2)).
out_stats(num_national_holiday_offs(10, 0, 1, 1)).
out_stats(num_national_holiday_offs(11, 0, 1, 1)).
out_stats(num_national_holiday_offs(12, 0, 1, 1)).
out_stats(num_national_holiday_offs(13, 0, 1, 1)).
out_stats(num_national_holiday_offs(14, 0, 1, 1)).
out_stats(num_national_holiday_offs(15, 0, 1, 1)).
out_stats(num_national_holiday_offs(16, 0, 0, 0)).
out_stats(num_national_holiday_offs(17, 0, 1, 1)).
out_stats(num_national_holiday_offs(18, 0, 0, 0)).
out_stats(num_national_holiday_offs(19, 0, 1, 1)).
out_stats(num_national_holiday_offs(20, 0, 2, 2)).
out_stats(num_national_holiday_offs(21, 0, 0, 0)).
out_stats(num_national_holiday_offs(22, 0, 1, 1)).
out_stats(num_national_holiday_offs(23, 0, 2, 2)).
out_stats(num_national_holiday_offs(24, 0, 1, 1)).
out_stats(num_national_holiday_offs(25, 0, 1, 1)).
out_stats(num_national_holiday_offs(26, 0, 2, 2)).
out_stats(num_national_holiday_offs(27, 0, 0, 0)).
out_stats(num_national_holiday_offs(28, 0, 1, 1)).
out_stats(num_national_holiday_offs(29, 0, 0, 0)).
out_stats(num_national_holiday_offs(30, 0, 1, 1)).
out_stats(num_national_holiday_offs(31, 0, 2, 2)).
out_stats(num_national_holiday_offs(32, 0, 1, 1)).
out_stats(num_national_holiday_offs(33, 0, 0, 0)).
out_stats(num_weekend_offs(2, 0, 1, 1)).
out_stats(num_weekend_offs(3, 0, 0, 0)).
out_stats(num_weekend_offs(4, 0, 0, 0)).
out_stats(num_weekend_offs(5, 0, 1, 1)).
out_stats(num_weekend_offs(6, 0, 0, 0)).
out_stats(num_weekend_offs(7, 0, 0, 0)).
out_stats(num_weekend_offs(8, 0, 0, 0)).
out_stats(num_weekend_offs(9, 0, 0, 0)).
out_stats(num_weekend_offs(10, 0, 0, 0)).
out_stats(num_weekend_offs(11, 0, 1, 1)).
out_stats(num_weekend_offs(12, 0, 0, 0)).
out_stats(num_weekend_offs(13, 0, 1, 1)).
out_stats(num_weekend_offs(14, 0, 1, 1)).
out_stats(num_weekend_offs(15, 0, 2, 2)).
out_stats(num_weekend_offs(16, 0, 1, 1)).
out_stats(num_weekend_offs(17, 0, 1, 1)).
out_stats(num_weekend_offs(18, 0, 1, 1)).
out_stats(num_weekend_offs(19, 0, 1, 1)).
out_stats(num_weekend_offs(20, 0, 1, 1)).
out_stats(num_weekend_offs(21, 0, 1, 1)).
out_stats(num_weekend_offs(22, 0, 0, 0)).
out_stats(num_weekend_offs(23, 0, 0, 0)).
out_stats(num_weekend_offs(24, 0, 1, 1)).
out_stats(num_weekend_offs(25, 0, 1, 1)).
out_stats(num_weekend_offs(26, 0, 1, 1)).
out_stats(num_weekend_offs(27, 0, 0, 0)).
out_stats(num_weekend_offs(28, 0, 0, 0)).
out_stats(num_weekend_offs(29, 0, 1, 1)).
out_stats(num_weekend_offs(30, 0, 0, 0)).
out_stats(num_weekend_offs(31, 0, 1, 1)).
out_stats(num_weekend_offs(32, 0, 0, 0)).
out_stats(num_weekend_offs(33, 0, 1, 1)).
reward(weekend_offs(2, 0), staff(2), 4, 1).
reward(weekend_offs(5, 0), staff(5), 4, 1).
reward(weekend_offs(11, 0), staff(11), 4, 1).
reward(weekend_offs(13, 0), staff(13), 4, 1).
reward(weekend_offs(14, 0), staff(14), 4, 1).
reward(weekend_offs(15, 0), staff(15), 4, 1).
reward(weekend_offs(16, 0), staff(16), 4, 1).
reward(weekend_offs(17, 0), staff(17), 4, 1).
reward(weekend_offs(18, 0), staff(18), 4, 1).
reward(weekend_offs(19, 0), staff(19), 4, 1).
reward(weekend_offs(20, 0), staff(20), 4, 1).
reward(weekend_offs(21, 0), staff(21), 4, 1).
reward(weekend_offs(24, 0), staff(24), 4, 1).
reward(weekend_offs(25, 0), staff(25), 4, 1).
reward(weekend_offs(26, 0), staff(26), 4, 1).
reward(weekend_offs(29, 0), staff(29), 4, 1).
reward(weekend_offs(31, 0), staff(31), 4, 1).
reward(weekend_offs(33, 0), staff(33), 4, 1).
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
reward(consecutive_holidays(27, 0), staff(27), 4, 1).
reward(consecutive_holidays(28, 0), staff(28), 4, 1).
reward(consecutive_holidays(29, 0), staff(29), 4, 1).
reward(consecutive_holidays(30, 0), staff(30), 4, 1).
reward(consecutive_holidays(32, 0), staff(32), 4, 1).
reward(consecutive_holidays(33, 0), staff(33), 4, 1).
reward(national_holiday_offs(3, 0), staff(3), 2, 1).
reward(national_holiday_offs(5, 0), staff(5), 2, 1).
reward(national_holiday_offs(6, 0), staff(6), 2, 1).
reward(national_holiday_offs(7, 0), staff(7), 2, 1).
reward(national_holiday_offs(8, 0), staff(8), 2, 1).
reward(national_holiday_offs(9, 0), staff(9), 2, 1).
reward(national_holiday_offs(10, 0), staff(10), 2, 1).
reward(national_holiday_offs(11, 0), staff(11), 2, 1).
reward(national_holiday_offs(12, 0), staff(12), 2, 1).
reward(national_holiday_offs(13, 0), staff(13), 2, 1).
reward(national_holiday_offs(14, 0), staff(14), 2, 1).
reward(national_holiday_offs(15, 0), staff(15), 2, 1).
reward(national_holiday_offs(17, 0), staff(17), 2, 1).
reward(national_holiday_offs(19, 0), staff(19), 2, 1).
reward(national_holiday_offs(20, 0), staff(20), 2, 1).
reward(national_holiday_offs(22, 0), staff(22), 2, 1).
reward(national_holiday_offs(23, 0), staff(23), 2, 1).
reward(national_holiday_offs(24, 0), staff(24), 2, 1).
reward(national_holiday_offs(25, 0), staff(25), 2, 1).
reward(national_holiday_offs(26, 0), staff(26), 2, 1).
reward(national_holiday_offs(28, 0), staff(28), 2, 1).
reward(national_holiday_offs(30, 0), staff(30), 2, 1).
reward(national_holiday_offs(31, 0), staff(31), 2, 1).
reward(national_holiday_offs(32, 0), staff(32), 2, 1).
penalty(staff_soft_lb(20251020, "D", "All", 9, 8), num_staffs(8, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251027, "D", "All", 9, 8), num_staffs(15, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251014, "D", "All", 9, 8), num_staffs(2, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251021, "D", "All", 9, 8), num_staffs(9, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251028, "D", "All", 9, 8), num_staffs(16, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251104, "D", "All", 9, 8), num_staffs(23, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251015, "D", "All", 9, 8), num_staffs(3, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251022, "D", "All", 9, 8), num_staffs(10, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251029, "D", "All", 9, 8), num_staffs(17, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251105, "D", "All", 9, 8), num_staffs(24, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251016, "D", "All", 9, 8), num_staffs(4, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251023, "D", "All", 9, 8), num_staffs(11, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251030, "D", "All", 9, 8), num_staffs(18, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251106, "D", "All", 9, 8), num_staffs(25, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251017, "D", "All", 9, 8), num_staffs(5, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251024, "D", "All", 9, 8), num_staffs(12, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251031, "D", "All", 9, 8), num_staffs(19, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251107, "D", "All", 9, 8), num_staffs(26, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251018, "D", "All", 5, 4), num_staffs(6, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251025, "D", "All", 5, 4), num_staffs(13, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251101, "D", "All", 5, 4), num_staffs(20, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251108, "D", "All", 5, 4), num_staffs(27, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251012, "D", "All", 5, 4), num_staffs(0, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251019, "D", "All", 5, 4), num_staffs(7, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251026, "D", "All", 5, 4), num_staffs(14, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251102, "D", "All", 5, 4), num_staffs(21, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251013, "D", "All", 5, 4), num_staffs(1, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251103, "D", "All", 5, 4), num_staffs(22, "D", "全員"), 1, 2).
penalty(staff_soft_lb(20251105, "D", "リ－ダ－", 2, 1), num_staffs(24, "D", "リ－ダ－"), 1, 2).
penalty(staff_soft_lb(20251023, "D", "リ－ダ－", 2, 1), num_staffs(11, "D", "リ－ダ－"), 1, 2).
penalty(staff_soft_lb(20251025, "D", "リ－ダ－", 2, 1), num_staffs(13, "D", "リ－ダ－"), 1, 2).
penalty(staff_soft_lb(20251019, "D", "リ－ダ－", 2, 1), num_staffs(7, "D", "リ－ダ－"), 1, 2).
penalty(staff_soft_lb(20251014, "SE", "Newcomers", 1, 0), num_staffs(2, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251021, "SE", "Newcomers", 1, 0), num_staffs(9, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251104, "SE", "Newcomers", 1, 0), num_staffs(23, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251015, "SE", "Newcomers", 1, 0), num_staffs(3, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251029, "SE", "Newcomers", 1, 0), num_staffs(17, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251105, "SE", "Newcomers", 1, 0), num_staffs(24, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251016, "SE", "Newcomers", 1, 0), num_staffs(4, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251023, "SE", "Newcomers", 1, 0), num_staffs(11, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251030, "SE", "Newcomers", 1, 0), num_staffs(18, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251017, "SE", "Newcomers", 1, 0), num_staffs(5, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251024, "SE", "Newcomers", 1, 0), num_staffs(12, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251031, "SE", "Newcomers", 1, 0), num_staffs(19, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251107, "SE", "Newcomers", 1, 0), num_staffs(26, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251018, "SE", "Newcomers", 1, 0), num_staffs(6, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251025, "SE", "Newcomers", 1, 0), num_staffs(13, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251108, "SE", "Newcomers", 1, 0), num_staffs(27, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251026, "SE", "Newcomers", 1, 0), num_staffs(14, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251102, "SE", "Newcomers", 1, 0), num_staffs(21, "SE", "新人"), 1, 2).
penalty(staff_soft_lb(20251027, "SN", "Newcomers", 1, 0), num_staffs(15, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251015, "SN", "Newcomers", 1, 0), num_staffs(3, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251022, "SN", "Newcomers", 1, 0), num_staffs(10, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251105, "SN", "Newcomers", 1, 0), num_staffs(24, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251016, "SN", "Newcomers", 1, 0), num_staffs(4, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251030, "SN", "Newcomers", 1, 0), num_staffs(18, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251106, "SN", "Newcomers", 1, 0), num_staffs(25, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251017, "SN", "Newcomers", 1, 0), num_staffs(5, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251024, "SN", "Newcomers", 1, 0), num_staffs(12, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251031, "SN", "Newcomers", 1, 0), num_staffs(19, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251018, "SN", "Newcomers", 1, 0), num_staffs(6, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251025, "SN", "Newcomers", 1, 0), num_staffs(13, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251101, "SN", "Newcomers", 1, 0), num_staffs(20, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251108, "SN", "Newcomers", 1, 0), num_staffs(27, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251012, "SN", "Newcomers", 1, 0), num_staffs(0, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251019, "SN", "Newcomers", 1, 0), num_staffs(7, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251026, "SN", "Newcomers", 1, 0), num_staffs(14, "SN", "新人"), 1, 2).
penalty(staff_soft_lb(20251103, "SN", "Newcomers", 1, 0), num_staffs(22, "SN", "新人"), 1, 2).
penalty(staff_soft_ub(20251014, "D", "リ－ダ－", 3, 4), num_staffs(2, "D", "リ－ダ－"), 1, 2).
penalty(staff_soft_ub(20251107, "D", "リ－ダ－", 3, 4), num_staffs(26, "D", "リ－ダ－"), 1, 2).
penalty(shifts_soft_ub(2, "LM", 1, 4), num_shifts(2, "P"), 9, 2).
penalty(shifts_soft_ub(5, "LM", 0, 1), num_shifts(5, "P"), 1, 2).
penalty(shifts_soft_ub(6, "LM", 1, 2), num_shifts(6, "P"), 1, 2).
penalty(shifts_soft_ub(16, "LM", 0, 1), num_shifts(16, "P"), 1, 2).
penalty(shifts_soft_ub(18, "LD", 2, 3), num_shifts(18, "N"), 1, 2).
penalty(shifts_soft_ub(18, "SE", 2, 3), num_shifts(18, "J"), 1, 2).
penalty(shifts_soft_ub(18, "SN", 2, 3), num_shifts(18, "S"), 1, 2).
penalty(shifts_soft_ub(19, "LD", 2, 3), num_shifts(19, "N"), 1, 2).
penalty(shifts_soft_ub(19, "SE", 2, 3), num_shifts(19, "J"), 1, 2).
penalty(shifts_soft_ub(19, "SN", 2, 3), num_shifts(19, "S"), 1, 2).
penalty(shifts_soft_ub(29, "LM", 0, 1), num_shifts(29, "P"), 1, 2).
penalty(shifts_soft_ub(33, "LM", 0, 2), num_shifts(33, "P"), 4, 2).
penalty(isolated_working_day(15, 20251012), table(15, 0), 1, 1).
penalty(isolated_working_day(27, 20251013), table(27, 1), 1, 1).
penalty(isolated_working_day(7, 20251014), table(7, 2), 1, 1).
penalty(isolated_working_day(17, 20251014), table(17, 2), 1, 1).
penalty(isolated_working_day(22, 20251014), table(22, 2), 1, 1).
penalty(isolated_working_day(16, 20251015), table(16, 3), 1, 1).
penalty(isolated_working_day(25, 20251015), table(25, 3), 1, 1).
penalty(isolated_working_day(6, 20251016), table(6, 4), 1, 1).
penalty(isolated_working_day(14, 20251016), table(14, 4), 1, 1).
penalty(isolated_working_day(11, 20251017), table(11, 5), 1, 1).
penalty(isolated_working_day(17, 20251017), table(17, 5), 1, 1).
penalty(isolated_working_day(3, 20251018), table(3, 6), 1, 1).
penalty(isolated_working_day(6, 20251018), table(6, 6), 1, 1).
penalty(isolated_working_day(9, 20251018), table(9, 6), 1, 1).
penalty(isolated_working_day(16, 20251018), table(16, 6), 1, 1).
penalty(isolated_working_day(4, 20251019), table(4, 7), 1, 1).
penalty(isolated_working_day(17, 20251019), table(17, 7), 1, 1).
penalty(isolated_working_day(18, 20251019), table(18, 7), 1, 1).
penalty(isolated_working_day(28, 20251019), table(28, 7), 1, 1).
penalty(isolated_working_day(21, 20251020), table(21, 8), 1, 1).
penalty(isolated_working_day(29, 20251020), table(29, 8), 1, 1).
penalty(isolated_working_day(9, 20251021), table(9, 9), 1, 1).
penalty(isolated_working_day(28, 20251021), table(28, 9), 1, 1).
penalty(isolated_working_day(21, 20251022), table(21, 10), 1, 1).
penalty(isolated_working_day(24, 20251022), table(24, 10), 1, 1).
penalty(isolated_working_day(31, 20251022), table(31, 10), 1, 1).
penalty(isolated_working_day(3, 20251023), table(3, 11), 1, 1).
penalty(isolated_working_day(10, 20251023), table(10, 11), 1, 1).
penalty(isolated_working_day(28, 20251023), table(28, 11), 1, 1).
penalty(isolated_working_day(2, 20251024), table(2, 12), 1, 1).
penalty(isolated_working_day(25, 20251024), table(25, 12), 1, 1).
penalty(isolated_working_day(11, 20251025), table(11, 13), 1, 1).
penalty(isolated_working_day(16, 20251025), table(16, 13), 1, 1).
penalty(isolated_working_day(12, 20251026), table(12, 14), 1, 1).
penalty(isolated_working_day(18, 20251026), table(18, 14), 1, 1).
penalty(isolated_working_day(8, 20251027), table(8, 15), 1, 1).
penalty(isolated_working_day(2, 20251028), table(2, 16), 1, 1).
penalty(isolated_working_day(13, 20251028), table(13, 16), 1, 1).
penalty(isolated_working_day(29, 20251028), table(29, 16), 1, 1).
penalty(isolated_working_day(8, 20251029), table(8, 17), 1, 1).
penalty(isolated_working_day(27, 20251030), table(27, 18), 1, 1).
penalty(isolated_working_day(23, 20251031), table(23, 19), 1, 1).
penalty(isolated_working_day(9, 20251101), table(9, 20), 1, 1).
penalty(isolated_working_day(18, 20251101), table(18, 20), 1, 1).
penalty(isolated_working_day(22, 20251101), table(22, 20), 1, 1).
penalty(isolated_working_day(3, 20251102), table(3, 21), 1, 1).
penalty(isolated_working_day(10, 20251102), table(10, 21), 1, 1).
penalty(isolated_working_day(23, 20251102), table(23, 21), 1, 1).
penalty(isolated_working_day(32, 20251102), table(32, 21), 1, 1).
penalty(isolated_working_day(4, 20251103), table(4, 22), 1, 1).
penalty(isolated_working_day(16, 20251103), table(16, 22), 1, 1).
penalty(isolated_working_day(30, 20251103), table(30, 22), 1, 1).
penalty(isolated_working_day(3, 20251104), table(3, 23), 1, 1).
penalty(isolated_working_day(32, 20251104), table(32, 23), 1, 1).
penalty(isolated_working_day(25, 20251105), table(25, 24), 1, 1).
penalty(isolated_working_day(30, 20251105), table(30, 24), 1, 1).
penalty(isolated_working_day(21, 20251106), table(21, 25), 1, 1).
penalty(isolated_working_day(3, 20251011), table(3, -1), 1, 1).
penalty(isolated_working_day(2, 20251011), table(2, -1), 1, 1).
penalty(pred_shift(4, 20251012, "J"), table(4, 0), 1, 4).
penalty(pred_shift(16, 20251012, "J"), table(16, 0), 1, 4).
penalty(pred_shift(8, 20251012, "日"), table(8, 0), 1, 4).
penalty(pred_shift(10, 20251012, "日"), table(10, 0), 1, 4).
penalty(pred_shift(24, 20251012, "日"), table(24, 0), 1, 4).
penalty(pred_shift(32, 20251012, "日"), table(32, 0), 1, 4).
penalty(pred_shift(15, 20251012, "N"), table(15, 0), 1, 4).
penalty(adjacent_holiday(21, 20251015, 20251015), table(21, 3), 1, 1).
penalty(adjacent_holiday(19, 20251016, 20251016), table(19, 4), 1, 1).
penalty(adjacent_holiday(12, 20251018, 20251018), table(12, 6), 1, 1).
penalty(adjacent_holiday(17, 20251018, 20251018), table(17, 6), 1, 1).
penalty(adjacent_holiday(26, 20251019, 20251020), table(26, 7), 1, 1).
penalty(adjacent_holiday(23, 20251021, 20251021), table(23, 9), 1, 1).
penalty(adjacent_holiday(9, 20251022, 20251023), table(9, 10), 1, 1).
penalty(adjacent_holiday(4, 20251025, 20251026), table(4, 13), 1, 1).
penalty(adjacent_holiday(17, 20251025, 20251026), table(17, 13), 1, 1).
penalty(adjacent_holiday(20, 20251026, 20251026), table(20, 14), 1, 1).
penalty(adjacent_holiday(18, 20251027, 20251028), table(18, 15), 1, 1).
penalty(adjacent_holiday(33, 20251030, 20251030), table(33, 18), 1, 1).
penalty(adjacent_holiday(28, 20251101, 20251102), table(28, 20), 1, 1).
penalty(adjacent_holiday(22, 20251102, 20251102), table(22, 21), 1, 1).
