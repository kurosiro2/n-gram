% Past shift data ----------------------------------------------------
shift_data("0021574", "上原　良江", 20240201, "J").
shift_data("0021574", "上原　良江", 20240202, "S").
shift_data("0021574", "上原　良江", 20240203, "○").
shift_data("0163361", "齊藤　渚", 20240201, "J").
shift_data("0163361", "齊藤　渚", 20240202, "S").
shift_data("0163361", "齊藤　渚", 20240203, "○").
shift_data("0163521", "伊藤　祥子", 20240201, "日").
shift_data("0163521", "伊藤　祥子", 20240202, "日").
shift_data("0163521", "伊藤　祥子", 20240203, "N").
shift_data("0236676", "野澤　美季", 20240201, "日").
shift_data("0236676", "野澤　美季", 20240202, "日").
shift_data("0236676", "野澤　美季", 20240203, "N").
shift_data("0056417", "松山　和佳", 20240201, "P").
shift_data("0056417", "松山　和佳", 20240202, "○").
shift_data("0056417", "松山　和佳", 20240203, "○").
shift_data("0123486", "藤巻　夕香里", 20240201, "日").
shift_data("0123486", "藤巻　夕香里", 20240202, "J").
shift_data("0123486", "藤巻　夕香里", 20240203, "S").
shift_data("0354841", "佐藤　恵理香", 20240201, "J").
shift_data("0354841", "佐藤　恵理香", 20240202, "S").
shift_data("0354841", "佐藤　恵理香", 20240203, "○").
shift_data("0250721", "甘蔗　華子", 20240201, "S").
shift_data("0250721", "甘蔗　華子", 20240202, "○").
shift_data("0250721", "甘蔗　華子", 20240203, "○").
shift_data("0269483", "菅原　梨沙", 20240201, "○").
shift_data("0269483", "菅原　梨沙", 20240202, "N").
shift_data("0269483", "菅原　梨沙", 20240203, "日").
shift_data("0340514", "田上　貴士", 20240201, "N").
shift_data("0340514", "田上　貴士", 20240202, "○").
shift_data("0340514", "田上　貴士", 20240203, "J").
shift_data("0341036", "土橋　美佳", 20240201, "日").
shift_data("0341036", "土橋　美佳", 20240202, "日").
shift_data("0341036", "土橋　美佳", 20240203, "○").
shift_data("0370522", "大畠　綾香", 20240201, "S").
shift_data("0370522", "大畠　綾香", 20240202, "○").
shift_data("0370522", "大畠　綾香", 20240203, "N").
shift_data("0370668", "伊東　美来", 20240201, "日").
shift_data("0370668", "伊東　美来", 20240202, "N").
shift_data("0370668", "伊東　美来", 20240203, "J").
shift_data("0315796", "萩原　万友佳", 20240201, "N").
shift_data("0315796", "萩原　万友佳", 20240202, "J").
shift_data("0315796", "萩原　万友佳", 20240203, "S").
shift_data("0384233", "雨宮　周実", 20240203, "○").
shift_data("0384723", "渡邉　樹里", 20240201, "○").
shift_data("0384723", "渡邉　樹里", 20240202, "J").
shift_data("0384723", "渡邉　樹里", 20240203, "S").
shift_data("0349720", "伊東　未緒", 20240201, "○").
shift_data("0349720", "伊東　未緒", 20240202, "○").
shift_data("0399960", "中野　佑麻", 20240201, "日").
shift_data("0399960", "中野　佑麻", 20240202, "日").
shift_data("0399960", "中野　佑麻", 20240203, "J").
shift_data("0400072", "岩波　みさと", 20240201, "S").
shift_data("0400072", "岩波　みさと", 20240202, "○").
shift_data("0400072", "岩波　みさと", 20240203, "○").
shift_data("0400265", "芳賀　友香", 20240201, "○").
shift_data("0400265", "芳賀　友香", 20240202, "日").
shift_data("0400265", "芳賀　友香", 20240203, "○").
shift_data("0357890", "高井　彩", 20240201, "○").
shift_data("0357890", "高井　彩", 20240202, "N").
shift_data("0357890", "高井　彩", 20240203, "○").
shift_data("0418860", "渡邊　陽子", 20240201, "N").
shift_data("0418860", "渡邊　陽子", 20240202, "○").
shift_data("0418860", "渡邊　陽子", 20240203, "日").
% Current shift data -------------------------------------------------
staff_request(1, D, "◎") :- base_date(D, 20240212, _).
staff_request(1, D, "○") :- base_date(D, 20240213, _).
staff_request(2, D, "○") :- base_date(D, 20240204, _).
staff_request(2, D, "○") :- base_date(D, 20240205, _).
staff_request(2, D, "日") :- base_date(D, 20240213, _).
staff_request(2, D, "年") :- base_date(D, 20240214, _).
staff_request(2, D, "○") :- base_date(D, 20240215, _).
staff_request(3, D, "日") :- base_date(D, 20240214, _).
staff_request(3, D, "○") :- base_date(D, 20240224, _).
staff_request(3, D, "○") :- base_date(D, 20240225, _).
staff_request(3, D, "日") :- base_date(D, 20240228, _).
staff_request(4, D, "○") :- base_date(D, 20240204, _).
staff_request(4, D, "N") :- base_date(D, 20240208, _).
staff_request(4, D, "J") :- base_date(D, 20240209, _).
staff_request(4, D, "S") :- base_date(D, 20240210, _).
staff_request(4, D, "◎") :- base_date(D, 20240212, _).
staff_request(4, D, "日") :- base_date(D, 20240213, _).
staff_request(4, D, "N") :- base_date(D, 20240214, _).
staff_request(4, D, "J") :- base_date(D, 20240215, _).
staff_request(4, D, "S") :- base_date(D, 20240216, _).
staff_request(4, D, "○") :- base_date(D, 20240218, _).
staff_request(4, D, "○") :- base_date(D, 20240219, _).
staff_request(4, D, "日") :- base_date(D, 20240222, _).
staff_request(4, D, "N") :- base_date(D, 20240226, _).
staff_request(4, D, "J") :- base_date(D, 20240227, _).
staff_request(4, D, "S") :- base_date(D, 20240228, _).
staff_request(4, D, "○") :- base_date(D, 20240229, _).
staff_request(4, D, "日") :- base_date(D, 20240301, _).
staff_request(4, D, "○") :- base_date(D, 20240302, _).
staff_request(6, D, "○") :- base_date(D, 20240208, _).
staff_request(6, D, "日") :- base_date(D, 20240227, _).
staff_request(6, D, "○") :- base_date(D, 20240301, _).
staff_request(6, D, "○") :- base_date(D, 20240305, _).
staff_request(7, D, "年") :- base_date(D, 20240208, _).
staff_request(7, D, "○") :- base_date(D, 20240209, _).
staff_request(7, D, "N") :- base_date(D, 20240210, _).
staff_request(7, D, "J") :- base_date(D, 20240216, _).
staff_request(7, D, "S") :- base_date(D, 20240217, _).
staff_request(7, D, "日") :- base_date(D, 20240223, _).
staff_request(7, D, "N") :- base_date(D, 20240224, _).
staff_request(7, D, "○") :- base_date(D, 20240227, _).
staff_request(7, D, "J") :- base_date(D, 20240301, _).
staff_request(7, D, "S") :- base_date(D, 20240302, _).
staff_request(8, D, "P") :- base_date(D, 20240205, _).
staff_request(8, D, "年") :- base_date(D, 20240207, _).
staff_request(8, D, "P") :- base_date(D, 20240213, _).
staff_request(8, D, "P") :- base_date(D, 20240219, _).
staff_request(8, D, "P") :- base_date(D, 20240226, _).
staff_request(8, D, "P") :- base_date(D, 20240304, _).
staff_request(8, D, "○") :- base_date(D, 20240309, _).
staff_request(10, D, "○") :- base_date(D, 20240206, _).
staff_request(10, D, "○") :- base_date(D, 20240207, _).
staff_request(10, D, "P") :- base_date(D, 20240208, _).
staff_request(10, D, "N") :- base_date(D, 20240209, _).
staff_request(10, D, "○") :- base_date(D, 20240211, _).
staff_request(10, D, "N") :- base_date(D, 20240217, _).
staff_request(10, D, "N") :- base_date(D, 20240218, _).
staff_request(10, D, "J") :- base_date(D, 20240220, _).
staff_request(10, D, "S") :- base_date(D, 20240221, _).
staff_request(10, D, "J") :- base_date(D, 20240223, _).
staff_request(10, D, "S") :- base_date(D, 20240224, _).
staff_request(10, D, "J") :- base_date(D, 20240227, _).
staff_request(10, D, "S") :- base_date(D, 20240228, _).
staff_request(11, D, "○") :- base_date(D, 20240211, _).
staff_request(11, D, "◎") :- base_date(D, 20240212, _).
staff_request(11, D, "年") :- base_date(D, 20240213, _).
staff_request(13, D, "日") :- base_date(D, 20240216, _).
staff_request(15, D, "日") :- base_date(D, 20240226, _).
staff_request(15, D, "日") :- base_date(D, 20240307, _).
staff_request(17, D, "○") :- base_date(D, 20240210, _).
staff_request(17, D, "○") :- base_date(D, 20240211, _).
staff_request(18, D, "○") :- base_date(D, 20240303, _).
staff_request(22, D, "○") :- base_date(D, 20240219, _).
staff_request(22, D, "日") :- base_date(D, 20240306, _).
staff_request(23, D, "日") :- base_date(D, 20240306, _).
