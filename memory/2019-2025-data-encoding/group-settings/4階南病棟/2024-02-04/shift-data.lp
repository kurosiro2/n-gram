% Past shift data ----------------------------------------------------
shift_data("74666","Timothy Ochoa",20240201,"SE").
shift_data("74666","Timothy Ochoa",20240202,"SN").
shift_data("74666","Timothy Ochoa",20240203,"WR").
shift_data("91727","Marcus Baker",20240201,"SE").
shift_data("91727","Marcus Baker",20240202,"SN").
shift_data("91727","Marcus Baker",20240203,"WR").
shift_data("98601","Christopher Chavez",20240201,"D").
shift_data("98601","Christopher Chavez",20240202,"D").
shift_data("98601","Christopher Chavez",20240203,"LD").
shift_data("98943","Haley Thomas",20240201,"D").
shift_data("98943","Haley Thomas",20240202,"D").
shift_data("98943","Haley Thomas",20240203,"LD").
shift_data("89988","Maria Phillips",20240201,"LM").
shift_data("89988","Maria Phillips",20240202,"WR").
shift_data("89988","Maria Phillips",20240203,"WR").
shift_data("92686","Natasha Stanley",20240201,"D").
shift_data("92686","Natasha Stanley",20240202,"SE").
shift_data("92686","Natasha Stanley",20240203,"SN").
shift_data("87705","James Pena",20240201,"SE").
shift_data("87705","James Pena",20240202,"SN").
shift_data("87705","James Pena",20240203,"WR").
shift_data("87534","Rebecca Crosby",20240201,"SN").
shift_data("87534","Rebecca Crosby",20240202,"WR").
shift_data("87534","Rebecca Crosby",20240203,"WR").
shift_data("45714","Michelle Nelson",20240201,"WR").
shift_data("45714","Michelle Nelson",20240202,"LD").
shift_data("45714","Michelle Nelson",20240203,"D").
shift_data("65213","Scott Kennedy",20240201,"LD").
shift_data("65213","Scott Kennedy",20240202,"WR").
shift_data("65213","Scott Kennedy",20240203,"SE").
shift_data("52823","Brandon Nelson",20240201,"D").
shift_data("52823","Brandon Nelson",20240202,"D").
shift_data("52823","Brandon Nelson",20240203,"WR").
shift_data("77977","Andrew Walker",20240201,"SN").
shift_data("77977","Andrew Walker",20240202,"WR").
shift_data("77977","Andrew Walker",20240203,"LD").
shift_data("68154","Amanda Jacobs",20240201,"D").
shift_data("68154","Amanda Jacobs",20240202,"LD").
shift_data("68154","Amanda Jacobs",20240203,"SE").
shift_data("81952","Derek Logan",20240201,"LD").
shift_data("81952","Derek Logan",20240202,"SE").
shift_data("81952","Derek Logan",20240203,"SN").
shift_data("37798","Allison White",20240203,"WR").
shift_data("01760","Brittney Shepard",20240201,"WR").
shift_data("01760","Brittney Shepard",20240202,"SE").
shift_data("01760","Brittney Shepard",20240203,"SN").
shift_data("95122","Erica Cooper",20240201,"WR").
shift_data("95122","Erica Cooper",20240202,"WR").
shift_data("25984","Ashley Stewart",20240201,"D").
shift_data("25984","Ashley Stewart",20240202,"D").
shift_data("25984","Ashley Stewart",20240203,"SE").
shift_data("74419","Tracey Davis",20240201,"SN").
shift_data("74419","Tracey Davis",20240202,"WR").
shift_data("74419","Tracey Davis",20240203,"WR").
shift_data("71150","Sara Thompson",20240201,"WR").
shift_data("71150","Sara Thompson",20240202,"D").
shift_data("71150","Sara Thompson",20240203,"WR").
shift_data("74978","William Stevens",20240201,"WR").
shift_data("74978","William Stevens",20240202,"LD").
shift_data("74978","William Stevens",20240203,"WR").
shift_data("10869","Tammie Miller",20240201,"LD").
shift_data("10869","Tammie Miller",20240202,"WR").
shift_data("10869","Tammie Miller",20240203,"D").
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
