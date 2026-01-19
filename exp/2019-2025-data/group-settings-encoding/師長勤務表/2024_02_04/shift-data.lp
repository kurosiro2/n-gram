% Past shift data ----------------------------------------------------
shift_data("44444442", "内田　純子", 20240203, "N").
shift_data("44444441", "伊藤　由香", 20240201, "N").
shift_data("44444441", "伊藤　由香", 20240202, "J").
shift_data("44444441", "伊藤　由香", 20240203, "S").
shift_data("44444443", "磯野　絵美", 20240202, "日").
shift_data("4444443", "穴水　美和", 20240201, "S").
shift_data("55555551", "櫻本　かおり", 20240203, "○").
shift_data("55555592", "竹田  礼子", 20240201, "日").
shift_data("55555591", "鈴木 聖美", 20240203, "日").
shift_data("66666900", "戸栗　宏子", 20240201, "日").
shift_data("66666671", "大森　ゆかり", 20240202, "N").
shift_data("66666671", "大森　ゆかり", 20240203, "J").
shift_data("55555593", "山本  智子", 20240201, "J").
shift_data("55555593", "山本  智子", 20240202, "S").
shift_data("55555593", "山本  智子", 20240203, "○").
shift_data("5555566", "蓮沼　知津子", 20240202, "日").
% Current shift data -------------------------------------------------
staff_request(1, D, "○") :- base_date(D, 20240210, _).
staff_request(1, D, "○") :- base_date(D, 20240211, _).
staff_request(1, D, "日") :- base_date(D, 20240214, _).
staff_request(1, D, "年") :- base_date(D, 20240221, _).
staff_request(1, D, "○") :- base_date(D, 20240224, _).
staff_request(1, D, "○") :- base_date(D, 20240225, _).
staff_request(1, D, "○") :- base_date(D, 20240302, _).
staff_request(1, D, "○") :- base_date(D, 20240303, _).
staff_request(2, D, "◎") :- base_date(D, 20240223, _).
staff_request(2, D, "○") :- base_date(D, 20240224, _).
staff_request(2, D, "○") :- base_date(D, 20240225, _).
staff_request(2, D, "○") :- base_date(D, 20240309, _).
staff_request(4, D, "日") :- base_date(D, 20240205, _).
staff_request(4, D, "日") :- base_date(D, 20240207, _).
staff_request(4, D, "日") :- base_date(D, 20240219, _).
staff_request(4, D, "日") :- base_date(D, 20240221, _).
staff_request(4, D, "日") :- base_date(D, 20240306, _).
staff_request(5, D, "○") :- base_date(D, 20240211, _).
staff_request(5, D, "◎") :- base_date(D, 20240212, _).
staff_request(5, D, "日") :- base_date(D, 20240216, _).
staff_request(5, D, "日") :- base_date(D, 20240221, _).
staff_request(6, D, "日") :- base_date(D, 20240301, _).
staff_request(8, D, "日") :- base_date(D, 20240208, _).
staff_request(8, D, "日") :- base_date(D, 20240215, _).
staff_request(12, D, "◎") :- base_date(D, 20240212, _).
staff_request(12, D, "○") :- base_date(D, 20240213, _).
staff_request(12, D, "○") :- base_date(D, 20240309, _).
staff_request(13, D, "日") :- base_date(D, 20240208, _).
staff_request(13, D, "日") :- base_date(D, 20240213, _).
staff_request(13, D, "日") :- base_date(D, 20240215, _).
staff_request(13, D, "日") :- base_date(D, 20240226, _).
staff_request(13, D, "日") :- base_date(D, 20240228, _).
staff_request(13, D, "日") :- base_date(D, 20240301, _).
staff_request(13, D, "研") :- base_date(D, 20240308, _).
staff_request(16, D, "日") :- base_date(D, 20240208, _).
staff_request(16, D, "日") :- base_date(D, 20240216, _).
staff_request(17, D, "日") :- base_date(D, 20240208, _).
staff_request(17, D, "日") :- base_date(D, 20240210, _).
staff_request(17, D, "日") :- base_date(D, 20240214, _).
staff_request(17, D, "日") :- base_date(D, 20240217, _).
staff_request(17, D, "日") :- base_date(D, 20240219, _).
staff_request(17, D, "日") :- base_date(D, 20240227, _).
staff_request(17, D, "日") :- base_date(D, 20240307, _).
staff_request(17, D, "研") :- base_date(D, 20240308, _).
staff_request(18, D, "日") :- base_date(D, 20240206, _).
staff_request(18, D, "日") :- base_date(D, 20240208, _).
staff_request(18, D, "日") :- base_date(D, 20240213, _).
staff_request(18, D, "日") :- base_date(D, 20240214, _).
staff_request(18, D, "日") :- base_date(D, 20240215, _).
staff_request(18, D, "研") :- base_date(D, 20240216, _).
staff_request(18, D, "日") :- base_date(D, 20240217, _).
staff_request(18, D, "日") :- base_date(D, 20240220, _).
staff_request(18, D, "日") :- base_date(D, 20240227, _).
staff_request(18, D, "日") :- base_date(D, 20240229, _).
staff_request(18, D, "日") :- base_date(D, 20240305, _).
staff_request(18, D, "日") :- base_date(D, 20240306, _).
staff_request(18, D, "日") :- base_date(D, 20240307, _).
staff_request(19, D, "日") :- base_date(D, 20240210, _).
staff_request(19, D, "日") :- base_date(D, 20240213, _).
staff_request(19, D, "日") :- base_date(D, 20240221, _).
staff_request(19, D, "日") :- base_date(D, 20240222, _).
staff_request(19, D, "日") :- base_date(D, 20240226, _).
staff_request(19, D, "○") :- base_date(D, 20240305, _).
staff_request(20, D, "S") :- base_date(D, 20240204, _).
staff_request(20, D, "日") :- base_date(D, 20240210, _).
staff_request(20, D, "N") :- base_date(D, 20240222, _).
staff_request(20, D, "J") :- base_date(D, 20240223, _).
staff_request(20, D, "S") :- base_date(D, 20240224, _).
staff_request(20, D, "日") :- base_date(D, 20240226, _).
staff_request(21, D, "N") :- base_date(D, 20240209, _).
staff_request(21, D, "J") :- base_date(D, 20240210, _).
staff_request(21, D, "S") :- base_date(D, 20240211, _).
staff_request(21, D, "日") :- base_date(D, 20240215, _).
staff_request(21, D, "日") :- base_date(D, 20240219, _).
staff_request(21, D, "日") :- base_date(D, 20240223, _).
staff_request(21, D, "日") :- base_date(D, 20240226, _).
staff_request(22, D, "N") :- base_date(D, 20240208, _).
staff_request(22, D, "J") :- base_date(D, 20240209, _).
staff_request(22, D, "S") :- base_date(D, 20240210, _).
staff_request(22, D, "○") :- base_date(D, 20240213, _).
staff_request(22, D, "日") :- base_date(D, 20240221, _).
staff_request(22, D, "日") :- base_date(D, 20240226, _).
staff_request(22, D, "N") :- base_date(D, 20240308, _).
staff_request(22, D, "J") :- base_date(D, 20240309, _).
staff_request(23, D, "日") :- base_date(D, 20240222, _).
staff_request(23, D, "◎") :- base_date(D, 20240223, _).
staff_request(23, D, "日") :- base_date(D, 20240224, _).
staff_request(23, D, "日") :- base_date(D, 20240225, _).
staff_request(23, D, "N") :- base_date(D, 20240226, _).
staff_request(23, D, "J") :- base_date(D, 20240227, _).
staff_request(23, D, "S") :- base_date(D, 20240228, _).
staff_request(23, D, "研") :- base_date(D, 20240308, _).
staff_request(24, D, "日") :- base_date(D, 20240207, _).
staff_request(24, D, "日") :- base_date(D, 20240221, _).
staff_request(24, D, "日") :- base_date(D, 20240227, _).
staff_request(24, D, "日") :- base_date(D, 20240306, _).
staff_request(25, D, "日") :- base_date(D, 20240206, _).
staff_request(25, D, "日") :- base_date(D, 20240207, _).
staff_request(25, D, "日") :- base_date(D, 20240213, _).
staff_request(25, D, "日") :- base_date(D, 20240215, _).
staff_request(25, D, "日") :- base_date(D, 20240226, _).
staff_request(25, D, "日") :- base_date(D, 20240305, _).
staff_request(25, D, "日") :- base_date(D, 20240306, _).
staff_request(26, D, "日") :- base_date(D, 20240206, _).
staff_request(26, D, "日") :- base_date(D, 20240214, _).
staff_request(26, D, "日") :- base_date(D, 20240220, _).
staff_request(26, D, "日") :- base_date(D, 20240222, _).
staff_request(26, D, "日") :- base_date(D, 20240305, _).
