% Staffs ---------------------------------------------------
staff(1, "Stacey Rhodes", "Deputy Chief Nurse", "07622", 0).
staff(2, "Christy Collins", "Deputy Chief Nurse", "61026", 0).
staff(3, "Edward Alvarez", "Nurse", "04741", 0).
staff(4, "Lori Perez", "Nurse", "71316", 0).
staff(5, "Jared Howe", "Nurse", "62161", 0).
staff(6, "Jaclyn Sanchez", "Nurse", "57921", 0).
staff(7, "Micheal Sexton", "Nurse", "61638", 0).
staff(8, "Amanda Brennan", "Nurse", "21414", 0).
staff(9, "Jennifer Le", "Nurse", "03874", 0).
staff(10, "Brittney Chan", "Nurse", "13320", 0).
staff(11, "Dr. Michelle Phillips", "Nurse", "45749", 0).
staff(12, "Brandon Gill", "Nurse", "86161", 0).
staff(13, "Carla Adams", "Nurse", "60822", 0).
staff(14, "Lisa Ford", "Nurse", "80153", 0).
staff(15, "Mindy Gordon", "Nurse", "66780", 0).
staff(16, "Melanie Stephens", "Nurse", "41836", 0).
staff(17, "Jennifer Maldonado", "Deputy Chief Nurse", "25435", 0).
staff(18, "Michael Garner", "Deputy Chief Nurse", "75479", 0).
staff(19, "Garrett Bean", "Nurse", "67709", 0).
staff(20, "Natalie Kennedy", "Nurse", "34770", 0).
staff(21, "Brittney Mcdowell", "Nurse", "55629", 0).
staff(22, "James Williams", "Nurse", "21745", 0).
staff(23, "Stacy Tucker", "Nurse", "94027", 0).
staff(24, "Amy Lester", "Nurse", "94975", 0).
staff(25, "James Jones", "Nurse", "65419", 0).
staff(26, "Sonya Cooper", "Nurse", "24150", 0).
staff(27, "Arthur Morrow", "Nurse", "46820", 0).
staff(28, "Jerry Wood", "Nurse", "67943", 0).
staff(29, "Douglas Williams", "Nurse", "95040", 0).
staff(30, "Jill Merritt", "Nurse", "03189", 0).
% Groups ---------------------------------------------------
group("All").
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
group("Heads").
group("師長代行").
group("師長代行", 1).
group("師長代行", 2).
group("師長代行", 4).
group("師長代行", 10).
group("師長代行", 14).
group("師長代行", 16).
group("師長代行", 17).
group("師長代行", 18).
group("師長代行", 19).
group("師長代行", 20).
group("師長代行", 21).
group("A").
group("A", 1).
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
group("A", 13).
group("A", 14).
group("A", 15).
group("A", 16).
group("B").
group("B", 17).
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
group("Newcomers").
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
group("Night", 30).
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
forbidden_pattern("LD-D").
pattern("LD-D", 2).
pattern("LD-D", 0, "LD").
pattern("LD-D", 1, "D").
forbidden_pattern("LD-LD").
pattern("LD-LD", 2).
pattern("LD-LD", 0, "LD").
pattern("LD-LD", 1, "LD").
forbidden_pattern("LD-TR").
pattern("LD-TR", 2).
pattern("LD-TR", 0, "LD").
pattern("LD-TR", 1, "TR").
forbidden_pattern("LM-D").
pattern("LM-D", 2).
pattern("LM-D", 0, "LM").
pattern("LM-D", 1, "D").
forbidden_pattern("LM-LD").
pattern("LM-LD", 2).
pattern("LM-LD", 0, "LM").
pattern("LM-LD", 1, "LD").
forbidden_pattern("LM-TR").
pattern("LM-TR", 2).
pattern("LM-TR", 0, "LM").
pattern("LM-TR", 1, "TR").
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
forbidden_pattern("D-D-D-D-TR-LD").
pattern("D-D-D-D-TR-LD", 6).
pattern("D-D-D-D-TR-LD", 0, "D").
pattern("D-D-D-D-TR-LD", 1, "D").
pattern("D-D-D-D-TR-LD", 2, "D").
pattern("D-D-D-D-TR-LD", 3, "D").
pattern("D-D-D-D-TR-LD", 4, "TR").
pattern("D-D-D-D-TR-LD", 5, "LD").
forbidden_pattern("D-D-D-LD-LM-LM").
pattern("D-D-D-LD-LM-LM", 6).
pattern("D-D-D-LD-LM-LM", 0, "D").
pattern("D-D-D-LD-LM-LM", 1, "D").
pattern("D-D-D-LD-LM-LM", 2, "D").
pattern("D-D-D-LD-LM-LM", 3, "LD").
pattern("D-D-D-LD-LM-LM", 4, "LM").
pattern("D-D-D-LD-LM-LM", 5, "LM").
forbidden_pattern("D-D-D-TR-D-LD").
pattern("D-D-D-TR-D-LD", 6).
pattern("D-D-D-TR-D-LD", 0, "D").
pattern("D-D-D-TR-D-LD", 1, "D").
pattern("D-D-D-TR-D-LD", 2, "D").
pattern("D-D-D-TR-D-LD", 3, "TR").
pattern("D-D-D-TR-D-LD", 4, "D").
pattern("D-D-D-TR-D-LD", 5, "LD").
forbidden_pattern("D-D-D-TR-LD-LM").
pattern("D-D-D-TR-LD-LM", 6).
pattern("D-D-D-TR-LD-LM", 0, "D").
pattern("D-D-D-TR-LD-LM", 1, "D").
pattern("D-D-D-TR-LD-LM", 2, "D").
pattern("D-D-D-TR-LD-LM", 3, "TR").
pattern("D-D-D-TR-LD-LM", 4, "LD").
pattern("D-D-D-TR-LD-LM", 5, "LM").
forbidden_pattern("D-D-D-TR-TR-LD").
pattern("D-D-D-TR-TR-LD", 6).
pattern("D-D-D-TR-TR-LD", 0, "D").
pattern("D-D-D-TR-TR-LD", 1, "D").
pattern("D-D-D-TR-TR-LD", 2, "D").
pattern("D-D-D-TR-TR-LD", 3, "TR").
pattern("D-D-D-TR-TR-LD", 4, "TR").
pattern("D-D-D-TR-TR-LD", 5, "LD").
forbidden_pattern("D-D-LD-LM-LM-LM").
pattern("D-D-LD-LM-LM-LM", 6).
pattern("D-D-LD-LM-LM-LM", 0, "D").
pattern("D-D-LD-LM-LM-LM", 1, "D").
pattern("D-D-LD-LM-LM-LM", 2, "LD").
pattern("D-D-LD-LM-LM-LM", 3, "LM").
pattern("D-D-LD-LM-LM-LM", 4, "LM").
pattern("D-D-LD-LM-LM-LM", 5, "LM").
forbidden_pattern("D-D-TR-D-D-LD").
pattern("D-D-TR-D-D-LD", 6).
pattern("D-D-TR-D-D-LD", 0, "D").
pattern("D-D-TR-D-D-LD", 1, "D").
pattern("D-D-TR-D-D-LD", 2, "TR").
pattern("D-D-TR-D-D-LD", 3, "D").
pattern("D-D-TR-D-D-LD", 4, "D").
pattern("D-D-TR-D-D-LD", 5, "LD").
forbidden_pattern("D-D-TR-D-LD-LM").
pattern("D-D-TR-D-LD-LM", 6).
pattern("D-D-TR-D-LD-LM", 0, "D").
pattern("D-D-TR-D-LD-LM", 1, "D").
pattern("D-D-TR-D-LD-LM", 2, "TR").
pattern("D-D-TR-D-LD-LM", 3, "D").
pattern("D-D-TR-D-LD-LM", 4, "LD").
pattern("D-D-TR-D-LD-LM", 5, "LM").
forbidden_pattern("D-D-TR-D-TR-LD").
pattern("D-D-TR-D-TR-LD", 6).
pattern("D-D-TR-D-TR-LD", 0, "D").
pattern("D-D-TR-D-TR-LD", 1, "D").
pattern("D-D-TR-D-TR-LD", 2, "TR").
pattern("D-D-TR-D-TR-LD", 3, "D").
pattern("D-D-TR-D-TR-LD", 4, "TR").
pattern("D-D-TR-D-TR-LD", 5, "LD").
forbidden_pattern("D-D-TR-LD-LM-LM").
pattern("D-D-TR-LD-LM-LM", 6).
pattern("D-D-TR-LD-LM-LM", 0, "D").
pattern("D-D-TR-LD-LM-LM", 1, "D").
pattern("D-D-TR-LD-LM-LM", 2, "TR").
pattern("D-D-TR-LD-LM-LM", 3, "LD").
pattern("D-D-TR-LD-LM-LM", 4, "LM").
pattern("D-D-TR-LD-LM-LM", 5, "LM").
forbidden_pattern("D-D-TR-TR-D-LD").
pattern("D-D-TR-TR-D-LD", 6).
pattern("D-D-TR-TR-D-LD", 0, "D").
pattern("D-D-TR-TR-D-LD", 1, "D").
pattern("D-D-TR-TR-D-LD", 2, "TR").
pattern("D-D-TR-TR-D-LD", 3, "TR").
pattern("D-D-TR-TR-D-LD", 4, "D").
pattern("D-D-TR-TR-D-LD", 5, "LD").
forbidden_pattern("D-D-TR-TR-LD-LM").
pattern("D-D-TR-TR-LD-LM", 6).
pattern("D-D-TR-TR-LD-LM", 0, "D").
pattern("D-D-TR-TR-LD-LM", 1, "D").
pattern("D-D-TR-TR-LD-LM", 2, "TR").
pattern("D-D-TR-TR-LD-LM", 3, "TR").
pattern("D-D-TR-TR-LD-LM", 4, "LD").
pattern("D-D-TR-TR-LD-LM", 5, "LM").
forbidden_pattern("D-D-TR-TR-TR-LD").
pattern("D-D-TR-TR-TR-LD", 6).
pattern("D-D-TR-TR-TR-LD", 0, "D").
pattern("D-D-TR-TR-TR-LD", 1, "D").
pattern("D-D-TR-TR-TR-LD", 2, "TR").
pattern("D-D-TR-TR-TR-LD", 3, "TR").
pattern("D-D-TR-TR-TR-LD", 4, "TR").
pattern("D-D-TR-TR-TR-LD", 5, "LD").
forbidden_pattern("D-LD-LM-LM-LM-LM").
pattern("D-LD-LM-LM-LM-LM", 6).
pattern("D-LD-LM-LM-LM-LM", 0, "D").
pattern("D-LD-LM-LM-LM-LM", 1, "LD").
pattern("D-LD-LM-LM-LM-LM", 2, "LM").
pattern("D-LD-LM-LM-LM-LM", 3, "LM").
pattern("D-LD-LM-LM-LM-LM", 4, "LM").
pattern("D-LD-LM-LM-LM-LM", 5, "LM").
forbidden_pattern("D-TR-D-D-D-LD").
pattern("D-TR-D-D-D-LD", 6).
pattern("D-TR-D-D-D-LD", 0, "D").
pattern("D-TR-D-D-D-LD", 1, "TR").
pattern("D-TR-D-D-D-LD", 2, "D").
pattern("D-TR-D-D-D-LD", 3, "D").
pattern("D-TR-D-D-D-LD", 4, "D").
pattern("D-TR-D-D-D-LD", 5, "LD").
forbidden_pattern("D-TR-D-D-LD-LM").
pattern("D-TR-D-D-LD-LM", 6).
pattern("D-TR-D-D-LD-LM", 0, "D").
pattern("D-TR-D-D-LD-LM", 1, "TR").
pattern("D-TR-D-D-LD-LM", 2, "D").
pattern("D-TR-D-D-LD-LM", 3, "D").
pattern("D-TR-D-D-LD-LM", 4, "LD").
pattern("D-TR-D-D-LD-LM", 5, "LM").
forbidden_pattern("D-TR-D-D-TR-LD").
pattern("D-TR-D-D-TR-LD", 6).
pattern("D-TR-D-D-TR-LD", 0, "D").
pattern("D-TR-D-D-TR-LD", 1, "TR").
pattern("D-TR-D-D-TR-LD", 2, "D").
pattern("D-TR-D-D-TR-LD", 3, "D").
pattern("D-TR-D-D-TR-LD", 4, "TR").
pattern("D-TR-D-D-TR-LD", 5, "LD").
forbidden_pattern("D-TR-D-LD-LM-LM").
pattern("D-TR-D-LD-LM-LM", 6).
pattern("D-TR-D-LD-LM-LM", 0, "D").
pattern("D-TR-D-LD-LM-LM", 1, "TR").
pattern("D-TR-D-LD-LM-LM", 2, "D").
pattern("D-TR-D-LD-LM-LM", 3, "LD").
pattern("D-TR-D-LD-LM-LM", 4, "LM").
pattern("D-TR-D-LD-LM-LM", 5, "LM").
forbidden_pattern("D-TR-D-TR-D-LD").
pattern("D-TR-D-TR-D-LD", 6).
pattern("D-TR-D-TR-D-LD", 0, "D").
pattern("D-TR-D-TR-D-LD", 1, "TR").
pattern("D-TR-D-TR-D-LD", 2, "D").
pattern("D-TR-D-TR-D-LD", 3, "TR").
pattern("D-TR-D-TR-D-LD", 4, "D").
pattern("D-TR-D-TR-D-LD", 5, "LD").
forbidden_pattern("D-TR-D-TR-LD-LM").
pattern("D-TR-D-TR-LD-LM", 6).
pattern("D-TR-D-TR-LD-LM", 0, "D").
pattern("D-TR-D-TR-LD-LM", 1, "TR").
pattern("D-TR-D-TR-LD-LM", 2, "D").
pattern("D-TR-D-TR-LD-LM", 3, "TR").
pattern("D-TR-D-TR-LD-LM", 4, "LD").
pattern("D-TR-D-TR-LD-LM", 5, "LM").
forbidden_pattern("D-TR-D-TR-TR-LD").
pattern("D-TR-D-TR-TR-LD", 6).
pattern("D-TR-D-TR-TR-LD", 0, "D").
pattern("D-TR-D-TR-TR-LD", 1, "TR").
pattern("D-TR-D-TR-TR-LD", 2, "D").
pattern("D-TR-D-TR-TR-LD", 3, "TR").
pattern("D-TR-D-TR-TR-LD", 4, "TR").
pattern("D-TR-D-TR-TR-LD", 5, "LD").
forbidden_pattern("D-TR-LD-LM-LM-LM").
pattern("D-TR-LD-LM-LM-LM", 6).
pattern("D-TR-LD-LM-LM-LM", 0, "D").
pattern("D-TR-LD-LM-LM-LM", 1, "TR").
pattern("D-TR-LD-LM-LM-LM", 2, "LD").
pattern("D-TR-LD-LM-LM-LM", 3, "LM").
pattern("D-TR-LD-LM-LM-LM", 4, "LM").
pattern("D-TR-LD-LM-LM-LM", 5, "LM").
forbidden_pattern("D-TR-TR-D-D-LD").
pattern("D-TR-TR-D-D-LD", 6).
pattern("D-TR-TR-D-D-LD", 0, "D").
pattern("D-TR-TR-D-D-LD", 1, "TR").
pattern("D-TR-TR-D-D-LD", 2, "TR").
pattern("D-TR-TR-D-D-LD", 3, "D").
pattern("D-TR-TR-D-D-LD", 4, "D").
pattern("D-TR-TR-D-D-LD", 5, "LD").
forbidden_pattern("D-TR-TR-D-LD-LM").
pattern("D-TR-TR-D-LD-LM", 6).
pattern("D-TR-TR-D-LD-LM", 0, "D").
pattern("D-TR-TR-D-LD-LM", 1, "TR").
pattern("D-TR-TR-D-LD-LM", 2, "TR").
pattern("D-TR-TR-D-LD-LM", 3, "D").
pattern("D-TR-TR-D-LD-LM", 4, "LD").
pattern("D-TR-TR-D-LD-LM", 5, "LM").
forbidden_pattern("D-TR-TR-D-TR-LD").
pattern("D-TR-TR-D-TR-LD", 6).
pattern("D-TR-TR-D-TR-LD", 0, "D").
pattern("D-TR-TR-D-TR-LD", 1, "TR").
pattern("D-TR-TR-D-TR-LD", 2, "TR").
pattern("D-TR-TR-D-TR-LD", 3, "D").
pattern("D-TR-TR-D-TR-LD", 4, "TR").
pattern("D-TR-TR-D-TR-LD", 5, "LD").
forbidden_pattern("D-TR-TR-LD-LM-LM").
pattern("D-TR-TR-LD-LM-LM", 6).
pattern("D-TR-TR-LD-LM-LM", 0, "D").
pattern("D-TR-TR-LD-LM-LM", 1, "TR").
pattern("D-TR-TR-LD-LM-LM", 2, "TR").
pattern("D-TR-TR-LD-LM-LM", 3, "LD").
pattern("D-TR-TR-LD-LM-LM", 4, "LM").
pattern("D-TR-TR-LD-LM-LM", 5, "LM").
forbidden_pattern("D-TR-TR-TR-D-LD").
pattern("D-TR-TR-TR-D-LD", 6).
pattern("D-TR-TR-TR-D-LD", 0, "D").
pattern("D-TR-TR-TR-D-LD", 1, "TR").
pattern("D-TR-TR-TR-D-LD", 2, "TR").
pattern("D-TR-TR-TR-D-LD", 3, "TR").
pattern("D-TR-TR-TR-D-LD", 4, "D").
pattern("D-TR-TR-TR-D-LD", 5, "LD").
forbidden_pattern("D-TR-TR-TR-LD-LM").
pattern("D-TR-TR-TR-LD-LM", 6).
pattern("D-TR-TR-TR-LD-LM", 0, "D").
pattern("D-TR-TR-TR-LD-LM", 1, "TR").
pattern("D-TR-TR-TR-LD-LM", 2, "TR").
pattern("D-TR-TR-TR-LD-LM", 3, "TR").
pattern("D-TR-TR-TR-LD-LM", 4, "LD").
pattern("D-TR-TR-TR-LD-LM", 5, "LM").
forbidden_pattern("D-TR-TR-TR-TR-LD").
pattern("D-TR-TR-TR-TR-LD", 6).
pattern("D-TR-TR-TR-TR-LD", 0, "D").
pattern("D-TR-TR-TR-TR-LD", 1, "TR").
pattern("D-TR-TR-TR-TR-LD", 2, "TR").
pattern("D-TR-TR-TR-TR-LD", 3, "TR").
pattern("D-TR-TR-TR-TR-LD", 4, "TR").
pattern("D-TR-TR-TR-TR-LD", 5, "LD").
forbidden_pattern("LD-LM-LM-LM-LM-LM").
pattern("LD-LM-LM-LM-LM-LM", 6).
pattern("LD-LM-LM-LM-LM-LM", 0, "LD").
pattern("LD-LM-LM-LM-LM-LM", 1, "LM").
pattern("LD-LM-LM-LM-LM-LM", 2, "LM").
pattern("LD-LM-LM-LM-LM-LM", 3, "LM").
pattern("LD-LM-LM-LM-LM-LM", 4, "LM").
pattern("LD-LM-LM-LM-LM-LM", 5, "LM").
forbidden_pattern("TR-D-D-D-D-LD").
pattern("TR-D-D-D-D-LD", 6).
pattern("TR-D-D-D-D-LD", 0, "TR").
pattern("TR-D-D-D-D-LD", 1, "D").
pattern("TR-D-D-D-D-LD", 2, "D").
pattern("TR-D-D-D-D-LD", 3, "D").
pattern("TR-D-D-D-D-LD", 4, "D").
pattern("TR-D-D-D-D-LD", 5, "LD").
forbidden_pattern("TR-D-D-D-LD-LM").
pattern("TR-D-D-D-LD-LM", 6).
pattern("TR-D-D-D-LD-LM", 0, "TR").
pattern("TR-D-D-D-LD-LM", 1, "D").
pattern("TR-D-D-D-LD-LM", 2, "D").
pattern("TR-D-D-D-LD-LM", 3, "D").
pattern("TR-D-D-D-LD-LM", 4, "LD").
pattern("TR-D-D-D-LD-LM", 5, "LM").
forbidden_pattern("TR-D-D-D-TR-LD").
pattern("TR-D-D-D-TR-LD", 6).
pattern("TR-D-D-D-TR-LD", 0, "TR").
pattern("TR-D-D-D-TR-LD", 1, "D").
pattern("TR-D-D-D-TR-LD", 2, "D").
pattern("TR-D-D-D-TR-LD", 3, "D").
pattern("TR-D-D-D-TR-LD", 4, "TR").
pattern("TR-D-D-D-TR-LD", 5, "LD").
forbidden_pattern("TR-D-D-LD-LM-LM").
pattern("TR-D-D-LD-LM-LM", 6).
pattern("TR-D-D-LD-LM-LM", 0, "TR").
pattern("TR-D-D-LD-LM-LM", 1, "D").
pattern("TR-D-D-LD-LM-LM", 2, "D").
pattern("TR-D-D-LD-LM-LM", 3, "LD").
pattern("TR-D-D-LD-LM-LM", 4, "LM").
pattern("TR-D-D-LD-LM-LM", 5, "LM").
forbidden_pattern("TR-D-D-TR-D-LD").
pattern("TR-D-D-TR-D-LD", 6).
pattern("TR-D-D-TR-D-LD", 0, "TR").
pattern("TR-D-D-TR-D-LD", 1, "D").
pattern("TR-D-D-TR-D-LD", 2, "D").
pattern("TR-D-D-TR-D-LD", 3, "TR").
pattern("TR-D-D-TR-D-LD", 4, "D").
pattern("TR-D-D-TR-D-LD", 5, "LD").
forbidden_pattern("TR-D-D-TR-LD-LM").
pattern("TR-D-D-TR-LD-LM", 6).
pattern("TR-D-D-TR-LD-LM", 0, "TR").
pattern("TR-D-D-TR-LD-LM", 1, "D").
pattern("TR-D-D-TR-LD-LM", 2, "D").
pattern("TR-D-D-TR-LD-LM", 3, "TR").
pattern("TR-D-D-TR-LD-LM", 4, "LD").
pattern("TR-D-D-TR-LD-LM", 5, "LM").
forbidden_pattern("TR-D-D-TR-TR-LD").
pattern("TR-D-D-TR-TR-LD", 6).
pattern("TR-D-D-TR-TR-LD", 0, "TR").
pattern("TR-D-D-TR-TR-LD", 1, "D").
pattern("TR-D-D-TR-TR-LD", 2, "D").
pattern("TR-D-D-TR-TR-LD", 3, "TR").
pattern("TR-D-D-TR-TR-LD", 4, "TR").
pattern("TR-D-D-TR-TR-LD", 5, "LD").
forbidden_pattern("TR-D-LD-LM-LM-LM").
pattern("TR-D-LD-LM-LM-LM", 6).
pattern("TR-D-LD-LM-LM-LM", 0, "TR").
pattern("TR-D-LD-LM-LM-LM", 1, "D").
pattern("TR-D-LD-LM-LM-LM", 2, "LD").
pattern("TR-D-LD-LM-LM-LM", 3, "LM").
pattern("TR-D-LD-LM-LM-LM", 4, "LM").
pattern("TR-D-LD-LM-LM-LM", 5, "LM").
forbidden_pattern("TR-D-TR-D-D-LD").
pattern("TR-D-TR-D-D-LD", 6).
pattern("TR-D-TR-D-D-LD", 0, "TR").
pattern("TR-D-TR-D-D-LD", 1, "D").
pattern("TR-D-TR-D-D-LD", 2, "TR").
pattern("TR-D-TR-D-D-LD", 3, "D").
pattern("TR-D-TR-D-D-LD", 4, "D").
pattern("TR-D-TR-D-D-LD", 5, "LD").
forbidden_pattern("TR-D-TR-D-LD-LM").
pattern("TR-D-TR-D-LD-LM", 6).
pattern("TR-D-TR-D-LD-LM", 0, "TR").
pattern("TR-D-TR-D-LD-LM", 1, "D").
pattern("TR-D-TR-D-LD-LM", 2, "TR").
pattern("TR-D-TR-D-LD-LM", 3, "D").
pattern("TR-D-TR-D-LD-LM", 4, "LD").
pattern("TR-D-TR-D-LD-LM", 5, "LM").
forbidden_pattern("TR-D-TR-D-TR-LD").
pattern("TR-D-TR-D-TR-LD", 6).
pattern("TR-D-TR-D-TR-LD", 0, "TR").
pattern("TR-D-TR-D-TR-LD", 1, "D").
pattern("TR-D-TR-D-TR-LD", 2, "TR").
pattern("TR-D-TR-D-TR-LD", 3, "D").
pattern("TR-D-TR-D-TR-LD", 4, "TR").
pattern("TR-D-TR-D-TR-LD", 5, "LD").
forbidden_pattern("TR-D-TR-LD-LM-LM").
pattern("TR-D-TR-LD-LM-LM", 6).
pattern("TR-D-TR-LD-LM-LM", 0, "TR").
pattern("TR-D-TR-LD-LM-LM", 1, "D").
pattern("TR-D-TR-LD-LM-LM", 2, "TR").
pattern("TR-D-TR-LD-LM-LM", 3, "LD").
pattern("TR-D-TR-LD-LM-LM", 4, "LM").
pattern("TR-D-TR-LD-LM-LM", 5, "LM").
forbidden_pattern("TR-D-TR-TR-D-LD").
pattern("TR-D-TR-TR-D-LD", 6).
pattern("TR-D-TR-TR-D-LD", 0, "TR").
pattern("TR-D-TR-TR-D-LD", 1, "D").
pattern("TR-D-TR-TR-D-LD", 2, "TR").
pattern("TR-D-TR-TR-D-LD", 3, "TR").
pattern("TR-D-TR-TR-D-LD", 4, "D").
pattern("TR-D-TR-TR-D-LD", 5, "LD").
forbidden_pattern("TR-D-TR-TR-LD-LM").
pattern("TR-D-TR-TR-LD-LM", 6).
pattern("TR-D-TR-TR-LD-LM", 0, "TR").
pattern("TR-D-TR-TR-LD-LM", 1, "D").
pattern("TR-D-TR-TR-LD-LM", 2, "TR").
pattern("TR-D-TR-TR-LD-LM", 3, "TR").
pattern("TR-D-TR-TR-LD-LM", 4, "LD").
pattern("TR-D-TR-TR-LD-LM", 5, "LM").
forbidden_pattern("TR-D-TR-TR-TR-LD").
pattern("TR-D-TR-TR-TR-LD", 6).
pattern("TR-D-TR-TR-TR-LD", 0, "TR").
pattern("TR-D-TR-TR-TR-LD", 1, "D").
pattern("TR-D-TR-TR-TR-LD", 2, "TR").
pattern("TR-D-TR-TR-TR-LD", 3, "TR").
pattern("TR-D-TR-TR-TR-LD", 4, "TR").
pattern("TR-D-TR-TR-TR-LD", 5, "LD").
forbidden_pattern("TR-LD-LM-LM-LM-LM").
pattern("TR-LD-LM-LM-LM-LM", 6).
pattern("TR-LD-LM-LM-LM-LM", 0, "TR").
pattern("TR-LD-LM-LM-LM-LM", 1, "LD").
pattern("TR-LD-LM-LM-LM-LM", 2, "LM").
pattern("TR-LD-LM-LM-LM-LM", 3, "LM").
pattern("TR-LD-LM-LM-LM-LM", 4, "LM").
pattern("TR-LD-LM-LM-LM-LM", 5, "LM").
forbidden_pattern("TR-TR-D-D-D-LD").
pattern("TR-TR-D-D-D-LD", 6).
pattern("TR-TR-D-D-D-LD", 0, "TR").
pattern("TR-TR-D-D-D-LD", 1, "TR").
pattern("TR-TR-D-D-D-LD", 2, "D").
pattern("TR-TR-D-D-D-LD", 3, "D").
pattern("TR-TR-D-D-D-LD", 4, "D").
pattern("TR-TR-D-D-D-LD", 5, "LD").
forbidden_pattern("TR-TR-D-D-LD-LM").
pattern("TR-TR-D-D-LD-LM", 6).
pattern("TR-TR-D-D-LD-LM", 0, "TR").
pattern("TR-TR-D-D-LD-LM", 1, "TR").
pattern("TR-TR-D-D-LD-LM", 2, "D").
pattern("TR-TR-D-D-LD-LM", 3, "D").
pattern("TR-TR-D-D-LD-LM", 4, "LD").
pattern("TR-TR-D-D-LD-LM", 5, "LM").
forbidden_pattern("TR-TR-D-D-TR-LD").
pattern("TR-TR-D-D-TR-LD", 6).
pattern("TR-TR-D-D-TR-LD", 0, "TR").
pattern("TR-TR-D-D-TR-LD", 1, "TR").
pattern("TR-TR-D-D-TR-LD", 2, "D").
pattern("TR-TR-D-D-TR-LD", 3, "D").
pattern("TR-TR-D-D-TR-LD", 4, "TR").
pattern("TR-TR-D-D-TR-LD", 5, "LD").
forbidden_pattern("TR-TR-D-LD-LM-LM").
pattern("TR-TR-D-LD-LM-LM", 6).
pattern("TR-TR-D-LD-LM-LM", 0, "TR").
pattern("TR-TR-D-LD-LM-LM", 1, "TR").
pattern("TR-TR-D-LD-LM-LM", 2, "D").
pattern("TR-TR-D-LD-LM-LM", 3, "LD").
pattern("TR-TR-D-LD-LM-LM", 4, "LM").
pattern("TR-TR-D-LD-LM-LM", 5, "LM").
forbidden_pattern("TR-TR-D-TR-D-LD").
pattern("TR-TR-D-TR-D-LD", 6).
pattern("TR-TR-D-TR-D-LD", 0, "TR").
pattern("TR-TR-D-TR-D-LD", 1, "TR").
pattern("TR-TR-D-TR-D-LD", 2, "D").
pattern("TR-TR-D-TR-D-LD", 3, "TR").
pattern("TR-TR-D-TR-D-LD", 4, "D").
pattern("TR-TR-D-TR-D-LD", 5, "LD").
forbidden_pattern("TR-TR-D-TR-LD-LM").
pattern("TR-TR-D-TR-LD-LM", 6).
pattern("TR-TR-D-TR-LD-LM", 0, "TR").
pattern("TR-TR-D-TR-LD-LM", 1, "TR").
pattern("TR-TR-D-TR-LD-LM", 2, "D").
pattern("TR-TR-D-TR-LD-LM", 3, "TR").
pattern("TR-TR-D-TR-LD-LM", 4, "LD").
pattern("TR-TR-D-TR-LD-LM", 5, "LM").
forbidden_pattern("TR-TR-D-TR-TR-LD").
pattern("TR-TR-D-TR-TR-LD", 6).
pattern("TR-TR-D-TR-TR-LD", 0, "TR").
pattern("TR-TR-D-TR-TR-LD", 1, "TR").
pattern("TR-TR-D-TR-TR-LD", 2, "D").
pattern("TR-TR-D-TR-TR-LD", 3, "TR").
pattern("TR-TR-D-TR-TR-LD", 4, "TR").
pattern("TR-TR-D-TR-TR-LD", 5, "LD").
forbidden_pattern("TR-TR-LD-LM-LM-LM").
pattern("TR-TR-LD-LM-LM-LM", 6).
pattern("TR-TR-LD-LM-LM-LM", 0, "TR").
pattern("TR-TR-LD-LM-LM-LM", 1, "TR").
pattern("TR-TR-LD-LM-LM-LM", 2, "LD").
pattern("TR-TR-LD-LM-LM-LM", 3, "LM").
pattern("TR-TR-LD-LM-LM-LM", 4, "LM").
pattern("TR-TR-LD-LM-LM-LM", 5, "LM").
forbidden_pattern("TR-TR-TR-D-D-LD").
pattern("TR-TR-TR-D-D-LD", 6).
pattern("TR-TR-TR-D-D-LD", 0, "TR").
pattern("TR-TR-TR-D-D-LD", 1, "TR").
pattern("TR-TR-TR-D-D-LD", 2, "TR").
pattern("TR-TR-TR-D-D-LD", 3, "D").
pattern("TR-TR-TR-D-D-LD", 4, "D").
pattern("TR-TR-TR-D-D-LD", 5, "LD").
forbidden_pattern("TR-TR-TR-D-LD-LM").
pattern("TR-TR-TR-D-LD-LM", 6).
pattern("TR-TR-TR-D-LD-LM", 0, "TR").
pattern("TR-TR-TR-D-LD-LM", 1, "TR").
pattern("TR-TR-TR-D-LD-LM", 2, "TR").
pattern("TR-TR-TR-D-LD-LM", 3, "D").
pattern("TR-TR-TR-D-LD-LM", 4, "LD").
pattern("TR-TR-TR-D-LD-LM", 5, "LM").
forbidden_pattern("TR-TR-TR-D-TR-LD").
pattern("TR-TR-TR-D-TR-LD", 6).
pattern("TR-TR-TR-D-TR-LD", 0, "TR").
pattern("TR-TR-TR-D-TR-LD", 1, "TR").
pattern("TR-TR-TR-D-TR-LD", 2, "TR").
pattern("TR-TR-TR-D-TR-LD", 3, "D").
pattern("TR-TR-TR-D-TR-LD", 4, "TR").
pattern("TR-TR-TR-D-TR-LD", 5, "LD").
forbidden_pattern("TR-TR-TR-LD-LM-LM").
pattern("TR-TR-TR-LD-LM-LM", 6).
pattern("TR-TR-TR-LD-LM-LM", 0, "TR").
pattern("TR-TR-TR-LD-LM-LM", 1, "TR").
pattern("TR-TR-TR-LD-LM-LM", 2, "TR").
pattern("TR-TR-TR-LD-LM-LM", 3, "LD").
pattern("TR-TR-TR-LD-LM-LM", 4, "LM").
pattern("TR-TR-TR-LD-LM-LM", 5, "LM").
forbidden_pattern("TR-TR-TR-TR-D-LD").
pattern("TR-TR-TR-TR-D-LD", 6).
pattern("TR-TR-TR-TR-D-LD", 0, "TR").
pattern("TR-TR-TR-TR-D-LD", 1, "TR").
pattern("TR-TR-TR-TR-D-LD", 2, "TR").
pattern("TR-TR-TR-TR-D-LD", 3, "TR").
pattern("TR-TR-TR-TR-D-LD", 4, "D").
pattern("TR-TR-TR-TR-D-LD", 5, "LD").
forbidden_pattern("TR-TR-TR-TR-LD-LM").
pattern("TR-TR-TR-TR-LD-LM", 6).
pattern("TR-TR-TR-TR-LD-LM", 0, "TR").
pattern("TR-TR-TR-TR-LD-LM", 1, "TR").
pattern("TR-TR-TR-TR-LD-LM", 2, "TR").
pattern("TR-TR-TR-TR-LD-LM", 3, "TR").
pattern("TR-TR-TR-TR-LD-LM", 4, "LD").
pattern("TR-TR-TR-TR-LD-LM", 5, "LM").
forbidden_pattern("TR-TR-TR-TR-TR-LD").
pattern("TR-TR-TR-TR-TR-LD", 6).
pattern("TR-TR-TR-TR-TR-LD", 0, "TR").
pattern("TR-TR-TR-TR-TR-LD", 1, "TR").
pattern("TR-TR-TR-TR-TR-LD", 2, "TR").
pattern("TR-TR-TR-TR-TR-LD", 3, "TR").
pattern("TR-TR-TR-TR-TR-LD", 4, "TR").
pattern("TR-TR-TR-TR-TR-LD", 5, "LD").
% 前日・翌日可能シフト
pred_shift("SE", "SN").
succ_shift("SN", "SE").
succ_shift("SN", "WR").
succ_shift("SN", "PH").
pred_shift("LD", "SE").
pred_shift("LM", "SE").
succ_shift("SE", "SN").
% パターン割当数
patterns("WR-WR", "WR-WR").
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("○○", 0).
patterns_soft_lb("○○", 1).
patterns_soft_ub("○○", 3).
patterns_hard_ub("○○", 4).
% Vertical constraints -------------------------------------
shifts("D").
shifts("D", "D").
staff_dweek_hard_lb("All", "D", "Weekday", 5).
staff_dweek_hard_lb("All", "D", "Weekend", 3).
staff_dweek_hard_lb("All", "D", "Holiday", 3).
staff_dweek_soft_lb("All", "D", "Weekday", 6).
staff_dweek_soft_lb("All", "D", "Weekend", 4).
staff_dweek_soft_lb("All", "D", "Holiday", 4).
staff_dweek_soft_ub("All", "D", "Weekday", 10).
staff_dweek_soft_ub("All", "D", "Weekend", 6).
staff_dweek_soft_ub("All", "D", "Holiday", 6).
staff_dweek_hard_ub("All", "D", "Weekday", 11).
staff_dweek_hard_ub("All", "D", "Weekend", 7).
staff_dweek_hard_ub("All", "D", "Holiday", 7).
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
staff_dweek_soft_lb("All", "LD", "Weekday", 3).
staff_dweek_soft_lb("All", "SE", "Weekday", 3).
staff_dweek_soft_lb("All", "SN", "Weekday", 3).
staff_dweek_soft_lb("All", "LD", "Weekend", 3).
staff_dweek_soft_lb("All", "SE", "Weekend", 3).
staff_dweek_soft_lb("All", "SN", "Weekend", 3).
staff_dweek_soft_lb("All", "LD", "Holiday", 3).
staff_dweek_soft_lb("All", "SE", "Holiday", 3).
staff_dweek_soft_lb("All", "SN", "Holiday", 3).
staff_dweek_soft_ub("All", "LD", "Weekday", 4).
staff_dweek_soft_ub("All", "SE", "Weekday", 4).
staff_dweek_soft_ub("All", "SN", "Weekday", 4).
staff_dweek_soft_ub("All", "LD", "Weekend", 4).
staff_dweek_soft_ub("All", "SE", "Weekend", 4).
staff_dweek_soft_ub("All", "SN", "Weekend", 4).
staff_dweek_soft_ub("All", "LD", "Holiday", 4).
staff_dweek_soft_ub("All", "SE", "Holiday", 4).
staff_dweek_soft_ub("All", "SN", "Holiday", 4).
staff_dweek_hard_ub("All", "LD", "Weekday", 5).
staff_dweek_hard_ub("All", "SE", "Weekday", 5).
staff_dweek_hard_ub("All", "SN", "Weekday", 5).
staff_dweek_hard_ub("All", "LD", "Weekend", 5).
staff_dweek_hard_ub("All", "SE", "Weekend", 5).
staff_dweek_hard_ub("All", "SN", "Weekend", 5).
staff_dweek_hard_ub("All", "LD", "Holiday", 5).
staff_dweek_hard_ub("All", "SE", "Holiday", 5).
staff_dweek_hard_ub("All", "SN", "Holiday", 5).
shifts("E").
shifts("E", "E").
staff_dweek_hard_lb("All", "E", "Weekday", 0).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("All", "N", "Weekday", 0).
staff_dweek_hard_lb("All", "E", "Weekend", 0).
staff_dweek_hard_lb("All", "N", "Weekend", 0).
staff_dweek_hard_lb("All", "E", "Holiday", 0).
staff_dweek_hard_lb("All", "N", "Holiday", 0).
staff_dweek_soft_lb("All", "E", "Weekday", 0).
staff_dweek_soft_lb("All", "N", "Weekday", 0).
staff_dweek_soft_lb("All", "E", "Weekend", 0).
staff_dweek_soft_lb("All", "N", "Weekend", 0).
staff_dweek_soft_lb("All", "E", "Holiday", 0).
staff_dweek_soft_lb("All", "N", "Holiday", 0).
staff_dweek_soft_ub("All", "E", "Weekday", 0).
staff_dweek_soft_ub("All", "N", "Weekday", 0).
staff_dweek_soft_ub("All", "E", "Weekend", 0).
staff_dweek_soft_ub("All", "N", "Weekend", 0).
staff_dweek_soft_ub("All", "E", "Holiday", 0).
staff_dweek_soft_ub("All", "N", "Holiday", 0).
staff_dweek_hard_ub("All", "E", "Weekday", 0).
staff_dweek_hard_ub("All", "N", "Weekday", 0).
staff_dweek_hard_ub("All", "E", "Weekend", 0).
staff_dweek_hard_ub("All", "N", "Weekend", 0).
staff_dweek_hard_ub("All", "E", "Holiday", 0).
staff_dweek_hard_ub("All", "N", "Holiday", 0).
shifts("LM").
shifts("LM", "LM").
staff_dweek_hard_lb("All", "LM", "Weekday", 0).
staff_dweek_hard_lb("All", "LM", "Weekend", 0).
staff_dweek_hard_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_lb("All", "LM", "Weekday", 0).
staff_dweek_soft_lb("All", "LM", "Weekend", 0).
staff_dweek_soft_lb("All", "LM", "Holiday", 0).
staff_dweek_soft_ub("All", "LM", "Weekday", 0).
staff_dweek_soft_ub("All", "LM", "Weekend", 0).
staff_dweek_soft_ub("All", "LM", "Holiday", 0).
staff_dweek_hard_ub("All", "LM", "Weekday", 0).
staff_dweek_hard_ub("All", "LM", "Weekend", 0).
staff_dweek_hard_ub("All", "LM", "Holiday", 0).
shifts("LD+LM+E").
shifts("LD+LM+E", "LD").
shifts("LD+LM+E", "LM").
shifts("LD+LM+E", "E").
staff_dweek_hard_lb("All", "LD+LM+E", "Weekday", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_hard_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekday", 3).
staff_dweek_soft_lb("All", "LD+LM+E", "Weekend", 3).
staff_dweek_soft_lb("All", "LD+LM+E", "Holiday", 3).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekday", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Weekend", 4).
staff_dweek_soft_ub("All", "LD+LM+E", "Holiday", 4).
staff_dweek_hard_ub("All", "LD+LM+E", "Weekday", 5).
staff_dweek_hard_ub("All", "LD+LM+E", "Weekend", 5).
staff_dweek_hard_ub("All", "LD+LM+E", "Holiday", 5).
staff_dweek_hard_lb("師長代行", "D", "Weekday", 1).
staff_dweek_hard_lb("師長代行", "D", "Weekend", 0).
staff_dweek_hard_lb("師長代行", "D", "Holiday", 0).
staff_dweek_soft_lb("師長代行", "D", "Weekday", 2).
staff_dweek_soft_lb("師長代行", "D", "Weekend", 1).
staff_dweek_soft_lb("師長代行", "D", "Holiday", 1).
staff_dweek_soft_ub("師長代行", "D", "Weekday", 6).
staff_dweek_soft_ub("師長代行", "D", "Weekend", 4).
staff_dweek_soft_ub("師長代行", "D", "Holiday", 4).
staff_dweek_hard_ub("師長代行", "D", "Weekday", 7).
staff_dweek_hard_ub("師長代行", "D", "Weekend", 5).
staff_dweek_hard_ub("師長代行", "D", "Holiday", 5).
staff_dweek_hard_lb("師長代行", "SE", "Weekday", 0).
staff_dweek_hard_lb("師長代行", "SN", "Weekday", 0).
staff_dweek_hard_lb("師長代行", "SE", "Weekend", 0).
staff_dweek_hard_lb("師長代行", "SN", "Weekend", 0).
staff_dweek_hard_lb("師長代行", "SE", "Holiday", 0).
staff_dweek_hard_lb("師長代行", "SN", "Holiday", 0).
staff_dweek_soft_lb("師長代行", "SE", "Weekday", 0).
staff_dweek_soft_lb("師長代行", "SN", "Weekday", 0).
staff_dweek_soft_lb("師長代行", "SE", "Weekend", 0).
staff_dweek_soft_lb("師長代行", "SN", "Weekend", 0).
staff_dweek_soft_lb("師長代行", "SE", "Holiday", 0).
staff_dweek_soft_lb("師長代行", "SN", "Holiday", 0).
staff_dweek_soft_ub("師長代行", "SE", "Weekday", 2).
staff_dweek_soft_ub("師長代行", "SN", "Weekday", 2).
staff_dweek_soft_ub("師長代行", "SE", "Weekend", 2).
staff_dweek_soft_ub("師長代行", "SN", "Weekend", 2).
staff_dweek_soft_ub("師長代行", "SE", "Holiday", 2).
staff_dweek_soft_ub("師長代行", "SN", "Holiday", 2).
staff_dweek_hard_ub("師長代行", "SE", "Weekday", 4).
staff_dweek_hard_ub("師長代行", "SN", "Weekday", 4).
staff_dweek_hard_ub("師長代行", "SE", "Weekend", 3).
staff_dweek_hard_ub("師長代行", "SN", "Weekend", 3).
staff_dweek_hard_ub("師長代行", "SE", "Holiday", 3).
staff_dweek_hard_ub("師長代行", "SN", "Holiday", 3).
staff_dweek_hard_lb("A", "SE", "Weekday", 1).
staff_dweek_hard_lb("A", "SN", "Weekday", 1).
staff_dweek_hard_lb("A", "SE", "Weekend", 1).
staff_dweek_hard_lb("A", "SN", "Weekend", 1).
staff_dweek_hard_lb("A", "SE", "Holiday", 1).
staff_dweek_hard_lb("A", "SN", "Holiday", 1).
staff_dweek_soft_lb("A", "SE", "Weekday", 1).
staff_dweek_soft_lb("A", "SN", "Weekday", 1).
staff_dweek_soft_lb("A", "SE", "Weekend", 1).
staff_dweek_soft_lb("A", "SN", "Weekend", 1).
staff_dweek_soft_lb("A", "SE", "Holiday", 1).
staff_dweek_soft_lb("A", "SN", "Holiday", 1).
staff_dweek_soft_ub("A", "SE", "Weekday", 3).
staff_dweek_soft_ub("A", "SN", "Weekday", 3).
staff_dweek_soft_ub("A", "SE", "Weekend", 2).
staff_dweek_soft_ub("A", "SN", "Weekend", 2).
staff_dweek_soft_ub("A", "SE", "Holiday", 2).
staff_dweek_soft_ub("A", "SN", "Holiday", 2).
staff_dweek_hard_ub("A", "SE", "Weekday", 3).
staff_dweek_hard_ub("A", "SN", "Weekday", 3).
staff_dweek_hard_ub("A", "SE", "Weekend", 3).
staff_dweek_hard_ub("A", "SN", "Weekend", 3).
staff_dweek_hard_ub("A", "SE", "Holiday", 3).
staff_dweek_hard_ub("A", "SN", "Holiday", 3).
staff_dweek_hard_lb("B", "SE", "Weekday", 1).
staff_dweek_hard_lb("B", "SN", "Weekday", 1).
staff_dweek_hard_lb("B", "SE", "Weekend", 1).
staff_dweek_hard_lb("B", "SN", "Weekend", 1).
staff_dweek_hard_lb("B", "SE", "Holiday", 1).
staff_dweek_hard_lb("B", "SN", "Holiday", 1).
staff_dweek_soft_lb("B", "SE", "Weekday", 1).
staff_dweek_soft_lb("B", "SN", "Weekday", 1).
staff_dweek_soft_lb("B", "SE", "Weekend", 1).
staff_dweek_soft_lb("B", "SN", "Weekend", 1).
staff_dweek_soft_lb("B", "SE", "Holiday", 1).
staff_dweek_soft_lb("B", "SN", "Holiday", 1).
staff_dweek_soft_ub("B", "SE", "Weekday", 3).
staff_dweek_soft_ub("B", "SN", "Weekday", 3).
staff_dweek_soft_ub("B", "SE", "Weekend", 3).
staff_dweek_soft_ub("B", "SN", "Weekend", 3).
staff_dweek_soft_ub("B", "SE", "Holiday", 3).
staff_dweek_soft_ub("B", "SN", "Holiday", 3).
staff_dweek_hard_ub("B", "SE", "Weekday", 3).
staff_dweek_hard_ub("B", "SN", "Weekday", 3).
staff_dweek_hard_ub("B", "SE", "Weekend", 3).
staff_dweek_hard_ub("B", "SN", "Weekend", 3).
staff_dweek_hard_ub("B", "SE", "Holiday", 3).
staff_dweek_hard_ub("B", "SN", "Holiday", 3).
shifts("SE+E").
shifts("SE+E", "SE").
shifts("SE+E", "E").
staff_dweek_hard_lb("Night", "SE+E", "Weekday", 3).
shifts("SN+N").
shifts("SN+N", "SN").
shifts("SN+N", "N").
staff_dweek_hard_lb("Night", "SN+N", "Weekday", 3).
staff_dweek_hard_lb("Night", "SE+E", "Weekend", 3).
staff_dweek_hard_lb("Night", "SN+N", "Weekend", 3).
staff_dweek_hard_lb("Night", "SE+E", "Holiday", 3).
staff_dweek_hard_lb("Night", "SN+N", "Holiday", 3).
staff_dweek_soft_lb("Night", "SE+E", "Weekday", 3).
staff_dweek_soft_lb("Night", "SN+N", "Weekday", 3).
staff_dweek_soft_lb("Night", "SE+E", "Weekend", 3).
staff_dweek_soft_lb("Night", "SN+N", "Weekend", 3).
staff_dweek_soft_lb("Night", "SE+E", "Holiday", 3).
staff_dweek_soft_lb("Night", "SN+N", "Holiday", 3).
staff_dweek_soft_ub("Night", "SE+E", "Weekday", 4).
staff_dweek_soft_ub("Night", "SN+N", "Weekday", 4).
staff_dweek_soft_ub("Night", "SE+E", "Weekend", 4).
staff_dweek_soft_ub("Night", "SN+N", "Weekend", 4).
staff_dweek_soft_ub("Night", "SE+E", "Holiday", 4).
staff_dweek_soft_ub("Night", "SN+N", "Holiday", 4).
staff_dweek_hard_ub("Night", "SE+E", "Weekday", 5).
staff_dweek_hard_ub("Night", "SN+N", "Weekday", 5).
staff_dweek_hard_ub("Night", "SE+E", "Weekend", 5).
staff_dweek_hard_ub("Night", "SN+N", "Weekend", 5).
staff_dweek_hard_ub("Night", "SE+E", "Holiday", 5).
staff_dweek_hard_ub("Night", "SN+N", "Holiday", 5).
vertical_constraint_type("All", "D", "Staffs").
vertical_constraint_type("All", "LD", "Staffs").
vertical_constraint_type("All", "SE", "Staffs").
vertical_constraint_type("All", "SN", "Staffs").
vertical_constraint_type("All", "E", "Staffs").
vertical_constraint_type("All", "N", "Staffs").
vertical_constraint_type("All", "LM", "Staffs").
vertical_constraint_type("All", "LD+LM+E", "Staffs").
vertical_constraint_type("師長代行", "D", "Staffs").
vertical_constraint_type("師長代行", "SE", "Staffs").
vertical_constraint_type("師長代行", "SN", "Staffs").
vertical_constraint_type("A", "SE", "Staffs").
vertical_constraint_type("A", "SN", "Staffs").
vertical_constraint_type("B", "SE", "Staffs").
vertical_constraint_type("B", "SN", "Staffs").
vertical_constraint_type("Night", "SE+E", "Staffs").
vertical_constraint_type("Night", "SN+N", "Staffs").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_hard_lb(1, "D", 14).
shifts_soft_lb(1, "D", 14).
shifts_soft_ub(1, "D", 16).
shifts_hard_ub(1, "D", 17).
shifts_hard_lb(1, "LD", 1).
shifts_hard_lb(1, "SE", 1).
shifts_hard_lb(1, "SN", 1).
shifts_soft_lb(1, "LD", 1).
shifts_soft_lb(1, "SE", 1).
shifts_soft_lb(1, "SN", 1).
shifts_soft_ub(1, "LD", 1).
shifts_soft_ub(1, "SE", 1).
shifts_soft_ub(1, "SN", 1).
shifts_hard_ub(1, "LD", 1).
shifts_hard_ub(1, "SE", 1).
shifts_hard_ub(1, "SN", 1).
shifts_hard_lb(1, "E", 0).
shifts_hard_lb(1, "N", 0).
shifts_hard_lb(1, "LM", 0).
shifts_soft_lb(1, "E", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_lb(1, "LM", 0).
shifts_soft_ub(1, "E", 0).
shifts_soft_ub(1, "N", 0).
shifts_soft_ub(1, "LM", 0).
shifts_hard_ub(1, "E", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "LM", 0).
shifts("EM").
shifts("EM", "EM").
shifts_hard_lb(1, "EM", 0).
shifts_soft_lb(1, "EM", 0).
shifts_soft_ub(1, "EM", 0).
shifts_hard_ub(1, "EM", 0).
% Staff 2:
shifts_hard_lb(2, "D", 6).
shifts_soft_lb(2, "D", 7).
shifts_soft_ub(2, "D", 16).
shifts_hard_ub(2, "D", 19).
shifts_hard_lb(2, "LD", 0).
shifts_hard_lb(2, "SE", 0).
shifts_hard_lb(2, "SN", 0).
shifts_soft_lb(2, "LD", 0).
shifts_soft_lb(2, "SE", 0).
shifts_soft_lb(2, "SN", 0).
shifts_soft_ub(2, "LD", 4).
shifts_soft_ub(2, "SE", 4).
shifts_soft_ub(2, "SN", 4).
shifts_hard_ub(2, "LD", 4).
shifts_hard_ub(2, "SE", 4).
shifts_hard_ub(2, "SN", 4).
shifts_hard_lb(2, "E", 0).
shifts_hard_lb(2, "N", 0).
shifts_hard_lb(2, "LM", 0).
shifts_soft_lb(2, "E", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "LM", 0).
shifts_soft_ub(2, "E", 0).
shifts_soft_ub(2, "N", 0).
shifts_soft_ub(2, "LM", 0).
shifts_hard_ub(2, "E", 0).
shifts_hard_ub(2, "N", 0).
shifts_hard_ub(2, "LM", 0).
shifts_hard_lb(2, "EM", 0).
shifts_soft_lb(2, "EM", 0).
shifts_soft_ub(2, "EM", 0).
shifts_hard_ub(2, "EM", 0).
% Staff 3:
shifts_hard_lb(3, "D", 3).
shifts_soft_lb(3, "D", 3).
shifts_soft_ub(3, "D", 8).
shifts_hard_ub(3, "D", 9).
shifts_hard_lb(3, "LD", 3).
shifts_hard_lb(3, "SE", 3).
shifts_hard_lb(3, "SN", 3).
shifts_soft_lb(3, "LD", 3).
shifts_soft_lb(3, "SE", 3).
shifts_soft_lb(3, "SN", 3).
shifts_soft_ub(3, "LD", 5).
shifts_soft_ub(3, "SE", 5).
shifts_soft_ub(3, "SN", 5).
shifts_hard_ub(3, "LD", 5).
shifts_hard_ub(3, "SE", 5).
shifts_hard_ub(3, "SN", 5).
shifts_hard_lb(3, "E", 0).
shifts_hard_lb(3, "N", 0).
shifts_hard_lb(3, "LM", 0).
shifts_soft_lb(3, "E", 0).
shifts_soft_lb(3, "N", 0).
shifts_soft_lb(3, "LM", 0).
shifts_soft_ub(3, "E", 0).
shifts_soft_ub(3, "N", 0).
shifts_soft_ub(3, "LM", 0).
shifts_hard_ub(3, "E", 1).
shifts_hard_ub(3, "N", 1).
shifts_hard_ub(3, "LM", 1).
shifts_hard_lb(3, "EM", 0).
shifts_soft_lb(3, "EM", 0).
shifts_soft_ub(3, "EM", 0).
shifts_hard_ub(3, "EM", 0).
% Staff 4:
shifts_hard_lb(4, "D", 3).
shifts_soft_lb(4, "D", 3).
shifts_soft_ub(4, "D", 9).
shifts_hard_ub(4, "D", 9).
shifts_hard_lb(4, "LD", 2).
shifts_hard_lb(4, "SE", 2).
shifts_hard_lb(4, "SN", 2).
shifts_soft_lb(4, "LD", 3).
shifts_soft_lb(4, "SE", 3).
shifts_soft_lb(4, "SN", 3).
shifts_soft_ub(4, "LD", 5).
shifts_soft_ub(4, "SE", 5).
shifts_soft_ub(4, "SN", 5).
shifts_hard_ub(4, "LD", 5).
shifts_hard_ub(4, "SE", 5).
shifts_hard_ub(4, "SN", 5).
shifts_hard_lb(4, "E", 0).
shifts_hard_lb(4, "N", 0).
shifts_hard_lb(4, "LM", 0).
shifts_soft_lb(4, "E", 0).
shifts_soft_lb(4, "N", 0).
shifts_soft_lb(4, "LM", 0).
shifts_soft_ub(4, "E", 0).
shifts_soft_ub(4, "N", 0).
shifts_soft_ub(4, "LM", 0).
shifts_hard_ub(4, "E", 1).
shifts_hard_ub(4, "N", 1).
shifts_hard_ub(4, "LM", 1).
shifts_hard_lb(4, "EM", 0).
shifts_soft_lb(4, "EM", 0).
shifts_soft_ub(4, "EM", 0).
shifts_hard_ub(4, "EM", 0).
% Staff 5:
shifts_hard_lb(5, "D", 3).
shifts_soft_lb(5, "D", 4).
shifts_soft_ub(5, "D", 6).
shifts_hard_ub(5, "D", 9).
shifts_hard_lb(5, "LD", 3).
shifts_hard_lb(5, "SE", 3).
shifts_hard_lb(5, "SN", 3).
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
shifts_hard_lb(5, "LM", 0).
shifts_soft_lb(5, "E", 0).
shifts_soft_lb(5, "N", 0).
shifts_soft_lb(5, "LM", 0).
shifts_soft_ub(5, "E", 0).
shifts_soft_ub(5, "N", 0).
shifts_soft_ub(5, "LM", 0).
shifts_hard_ub(5, "E", 0).
shifts_hard_ub(5, "N", 0).
shifts_hard_ub(5, "LM", 0).
shifts_hard_lb(5, "EM", 0).
shifts_soft_lb(5, "EM", 0).
shifts_soft_ub(5, "EM", 0).
shifts_hard_ub(5, "EM", 0).
% Staff 6:
shifts_hard_lb(6, "D", 4).
shifts_soft_lb(6, "D", 4).
shifts_soft_ub(6, "D", 11).
shifts_hard_ub(6, "D", 12).
shifts_hard_lb(6, "LD", 2).
shifts_hard_lb(6, "SE", 2).
shifts_hard_lb(6, "SN", 2).
shifts_soft_lb(6, "LD", 3).
shifts_soft_lb(6, "SE", 3).
shifts_soft_lb(6, "SN", 3).
shifts_soft_ub(6, "LD", 5).
shifts_soft_ub(6, "SE", 5).
shifts_soft_ub(6, "SN", 5).
shifts_hard_ub(6, "LD", 5).
shifts_hard_ub(6, "SE", 5).
shifts_hard_ub(6, "SN", 5).
shifts_hard_lb(6, "E", 0).
shifts_hard_lb(6, "N", 0).
shifts_hard_lb(6, "LM", 0).
shifts_soft_lb(6, "E", 0).
shifts_soft_lb(6, "N", 0).
shifts_soft_lb(6, "LM", 0).
shifts_soft_ub(6, "E", 0).
shifts_soft_ub(6, "N", 0).
shifts_soft_ub(6, "LM", 0).
shifts_hard_ub(6, "E", 0).
shifts_hard_ub(6, "N", 0).
shifts_hard_ub(6, "LM", 0).
shifts_hard_lb(6, "EM", 0).
shifts_soft_lb(6, "EM", 0).
shifts_soft_ub(6, "EM", 0).
shifts_hard_ub(6, "EM", 0).
% Staff 7:
shifts_hard_lb(7, "D", 1).
shifts_soft_lb(7, "D", 1).
shifts_soft_ub(7, "D", 7).
shifts_hard_ub(7, "D", 8).
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
shifts_hard_lb(7, "E", 0).
shifts_hard_lb(7, "N", 0).
shifts_hard_lb(7, "LM", 0).
shifts_soft_lb(7, "E", 0).
shifts_soft_lb(7, "N", 0).
shifts_soft_lb(7, "LM", 0).
shifts_soft_ub(7, "E", 0).
shifts_soft_ub(7, "N", 0).
shifts_soft_ub(7, "LM", 0).
shifts_hard_ub(7, "E", 0).
shifts_hard_ub(7, "N", 0).
shifts_hard_ub(7, "LM", 0).
shifts_hard_lb(7, "EM", 0).
shifts_soft_lb(7, "EM", 0).
shifts_soft_ub(7, "EM", 0).
shifts_hard_ub(7, "EM", 0).
% Staff 8:
shifts_hard_lb(8, "E", 0).
shifts_hard_lb(8, "N", 0).
shifts_hard_lb(8, "LM", 0).
shifts_soft_lb(8, "E", 0).
shifts_soft_lb(8, "N", 0).
shifts_soft_lb(8, "LM", 0).
shifts_soft_ub(8, "E", 0).
shifts_soft_ub(8, "N", 0).
shifts_soft_ub(8, "LM", 0).
shifts_hard_ub(8, "E", 0).
shifts_hard_ub(8, "N", 0).
shifts_hard_ub(8, "LM", 0).
shifts_hard_lb(8, "EM", 0).
shifts_soft_lb(8, "EM", 0).
shifts_soft_ub(8, "EM", 0).
shifts_hard_ub(8, "EM", 0).
% Staff 9:
shifts_hard_lb(9, "D", 2).
shifts_soft_lb(9, "D", 3).
shifts_soft_ub(9, "D", 8).
shifts_hard_ub(9, "D", 8).
shifts_hard_lb(9, "LD", 3).
shifts_hard_lb(9, "SE", 3).
shifts_hard_lb(9, "SN", 3).
shifts_soft_lb(9, "LD", 3).
shifts_soft_lb(9, "SE", 3).
shifts_soft_lb(9, "SN", 3).
shifts_soft_ub(9, "LD", 5).
shifts_soft_ub(9, "SE", 5).
shifts_soft_ub(9, "SN", 5).
shifts_hard_ub(9, "LD", 5).
shifts_hard_ub(9, "SE", 5).
shifts_hard_ub(9, "SN", 5).
shifts_hard_lb(9, "E", 0).
shifts_hard_lb(9, "N", 0).
shifts_hard_lb(9, "LM", 0).
shifts_soft_lb(9, "E", 0).
shifts_soft_lb(9, "N", 0).
shifts_soft_lb(9, "LM", 0).
shifts_soft_ub(9, "E", 0).
shifts_soft_ub(9, "N", 0).
shifts_soft_ub(9, "LM", 0).
shifts_hard_ub(9, "E", 0).
shifts_hard_ub(9, "N", 0).
shifts_hard_ub(9, "LM", 0).
shifts_hard_lb(9, "EM", 0).
shifts_soft_lb(9, "EM", 0).
shifts_soft_ub(9, "EM", 0).
shifts_hard_ub(9, "EM", 0).
% Staff 10:
shifts_hard_lb(10, "D", 2).
shifts_soft_lb(10, "D", 4).
shifts_soft_ub(10, "D", 8).
shifts_hard_ub(10, "D", 10).
shifts_hard_lb(10, "LD", 0).
shifts_hard_lb(10, "SE", 0).
shifts_hard_lb(10, "SN", 0).
shifts_soft_lb(10, "LD", 3).
shifts_soft_lb(10, "SE", 3).
shifts_soft_lb(10, "SN", 3).
shifts_soft_ub(10, "LD", 5).
shifts_soft_ub(10, "SE", 5).
shifts_soft_ub(10, "SN", 5).
shifts_hard_ub(10, "LD", 6).
shifts_hard_ub(10, "SE", 6).
shifts_hard_ub(10, "SN", 6).
shifts_hard_lb(10, "E", 0).
shifts_hard_lb(10, "N", 0).
shifts_hard_lb(10, "LM", 0).
shifts_soft_lb(10, "E", 0).
shifts_soft_lb(10, "N", 0).
shifts_soft_lb(10, "LM", 0).
shifts_soft_ub(10, "E", 0).
shifts_soft_ub(10, "N", 0).
shifts_soft_ub(10, "LM", 0).
shifts_hard_ub(10, "E", 0).
shifts_hard_ub(10, "N", 0).
shifts_hard_ub(10, "LM", 0).
shifts_hard_lb(10, "EM", 0).
shifts_soft_lb(10, "EM", 0).
shifts_soft_ub(10, "EM", 0).
shifts_hard_ub(10, "EM", 0).
% Staff 11:
shifts_hard_lb(11, "D", 3).
shifts_soft_lb(11, "D", 3).
shifts_soft_ub(11, "D", 8).
shifts_hard_ub(11, "D", 9).
shifts_hard_lb(11, "LD", 3).
shifts_hard_lb(11, "SE", 3).
shifts_hard_lb(11, "SN", 3).
shifts_soft_lb(11, "LD", 3).
shifts_soft_lb(11, "SE", 3).
shifts_soft_lb(11, "SN", 3).
shifts_soft_ub(11, "LD", 5).
shifts_soft_ub(11, "SE", 5).
shifts_soft_ub(11, "SN", 5).
shifts_hard_ub(11, "LD", 5).
shifts_hard_ub(11, "SE", 5).
shifts_hard_ub(11, "SN", 5).
shifts_hard_lb(11, "E", 0).
shifts_hard_lb(11, "N", 0).
shifts_hard_lb(11, "LM", 0).
shifts_soft_lb(11, "E", 0).
shifts_soft_lb(11, "N", 0).
shifts_soft_lb(11, "LM", 0).
shifts_soft_ub(11, "E", 0).
shifts_soft_ub(11, "N", 0).
shifts_soft_ub(11, "LM", 0).
shifts_hard_ub(11, "E", 0).
shifts_hard_ub(11, "N", 0).
shifts_hard_ub(11, "LM", 0).
shifts_hard_lb(11, "EM", 0).
shifts_soft_lb(11, "EM", 0).
shifts_soft_ub(11, "EM", 0).
shifts_hard_ub(11, "EM", 0).
% Staff 12:
shifts_hard_lb(12, "E", 0).
shifts_hard_lb(12, "N", 0).
shifts_hard_lb(12, "LM", 0).
shifts_soft_lb(12, "E", 0).
shifts_soft_lb(12, "N", 0).
shifts_soft_lb(12, "LM", 0).
shifts_soft_ub(12, "E", 0).
shifts_soft_ub(12, "N", 0).
shifts_soft_ub(12, "LM", 0).
shifts_hard_ub(12, "E", 0).
shifts_hard_ub(12, "N", 0).
shifts_hard_ub(12, "LM", 0).
shifts_hard_lb(12, "EM", 0).
shifts_soft_lb(12, "EM", 0).
shifts_soft_ub(12, "EM", 0).
shifts_hard_ub(12, "EM", 0).
% Staff 13:
shifts_hard_lb(13, "D", 2).
shifts_soft_lb(13, "D", 3).
shifts_soft_ub(13, "D", 7).
shifts_hard_ub(13, "D", 8).
shifts_hard_lb(13, "LD", 4).
shifts_hard_lb(13, "SE", 4).
shifts_hard_lb(13, "SN", 4).
shifts_soft_lb(13, "LD", 4).
shifts_soft_lb(13, "SE", 4).
shifts_soft_lb(13, "SN", 4).
shifts_soft_ub(13, "LD", 6).
shifts_soft_ub(13, "SE", 6).
shifts_soft_ub(13, "SN", 6).
shifts_hard_ub(13, "LD", 6).
shifts_hard_ub(13, "SE", 6).
shifts_hard_ub(13, "SN", 6).
shifts_hard_lb(13, "E", 0).
shifts_hard_lb(13, "N", 0).
shifts_hard_lb(13, "LM", 0).
shifts_soft_lb(13, "E", 0).
shifts_soft_lb(13, "N", 0).
shifts_soft_lb(13, "LM", 0).
shifts_soft_ub(13, "E", 0).
shifts_soft_ub(13, "N", 0).
shifts_soft_ub(13, "LM", 0).
shifts_hard_ub(13, "E", 0).
shifts_hard_ub(13, "N", 0).
shifts_hard_ub(13, "LM", 0).
shifts_hard_lb(13, "EM", 0).
shifts_soft_lb(13, "EM", 0).
shifts_soft_ub(13, "EM", 0).
shifts_hard_ub(13, "EM", 0).
% Staff 14:
shifts_hard_lb(14, "D", 0).
shifts_soft_lb(14, "D", 2).
shifts_soft_ub(14, "D", 9).
shifts_hard_ub(14, "D", 12).
shifts_hard_lb(14, "LD", 2).
shifts_hard_lb(14, "SE", 2).
shifts_hard_lb(14, "SN", 2).
shifts_soft_lb(14, "LD", 2).
shifts_soft_lb(14, "SE", 2).
shifts_soft_lb(14, "SN", 2).
shifts_soft_ub(14, "LD", 6).
shifts_soft_ub(14, "SE", 6).
shifts_soft_ub(14, "SN", 6).
shifts_hard_ub(14, "LD", 6).
shifts_hard_ub(14, "SE", 6).
shifts_hard_ub(14, "SN", 6).
shifts_hard_lb(14, "E", 0).
shifts_hard_lb(14, "N", 0).
shifts_hard_lb(14, "LM", 0).
shifts_soft_lb(14, "E", 0).
shifts_soft_lb(14, "N", 0).
shifts_soft_lb(14, "LM", 0).
shifts_soft_ub(14, "E", 0).
shifts_soft_ub(14, "N", 0).
shifts_soft_ub(14, "LM", 0).
shifts_hard_ub(14, "E", 0).
shifts_hard_ub(14, "N", 0).
shifts_hard_ub(14, "LM", 0).
shifts_hard_lb(14, "EM", 0).
shifts_soft_lb(14, "EM", 0).
shifts_soft_ub(14, "EM", 0).
shifts_hard_ub(14, "EM", 0).
% Staff 15:
shifts_hard_lb(15, "D", 5).
shifts_soft_lb(15, "D", 6).
shifts_soft_ub(15, "D", 17).
shifts_hard_ub(15, "D", 19).
shifts_hard_lb(15, "LD", 0).
shifts_hard_lb(15, "SE", 0).
shifts_hard_lb(15, "SN", 0).
shifts_soft_lb(15, "LD", 0).
shifts_soft_lb(15, "SE", 0).
shifts_soft_lb(15, "SN", 0).
shifts_soft_ub(15, "LD", 3).
shifts_soft_ub(15, "SE", 3).
shifts_soft_ub(15, "SN", 3).
shifts_hard_ub(15, "LD", 4).
shifts_hard_ub(15, "SE", 4).
shifts_hard_ub(15, "SN", 4).
shifts_hard_lb(15, "E", 0).
shifts_hard_lb(15, "N", 0).
shifts_hard_lb(15, "LM", 0).
shifts_soft_lb(15, "E", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_lb(15, "LM", 0).
shifts_soft_ub(15, "E", 0).
shifts_soft_ub(15, "N", 0).
shifts_soft_ub(15, "LM", 0).
shifts_hard_ub(15, "E", 0).
shifts_hard_ub(15, "N", 0).
shifts_hard_ub(15, "LM", 0).
shifts_hard_lb(15, "EM", 0).
shifts_soft_lb(15, "EM", 0).
shifts_soft_ub(15, "EM", 0).
shifts_hard_ub(15, "EM", 0).
% Staff 16:
shifts_hard_lb(16, "D", 10).
shifts_soft_lb(16, "D", 11).
shifts_soft_ub(16, "D", 17).
shifts_hard_ub(16, "D", 19).
shifts_hard_lb(16, "LD", 0).
shifts_hard_lb(16, "SE", 0).
shifts_hard_lb(16, "SN", 0).
shifts_soft_lb(16, "LD", 0).
shifts_soft_lb(16, "SE", 0).
shifts_soft_lb(16, "SN", 0).
shifts_soft_ub(16, "LD", 1).
shifts_soft_ub(16, "SE", 1).
shifts_soft_ub(16, "SN", 1).
shifts_hard_ub(16, "LD", 1).
shifts_hard_ub(16, "SE", 1).
shifts_hard_ub(16, "SN", 1).
shifts_hard_lb(16, "E", 0).
shifts_hard_lb(16, "N", 0).
shifts_hard_lb(16, "LM", 0).
shifts_soft_lb(16, "E", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_lb(16, "LM", 0).
shifts_soft_ub(16, "E", 0).
shifts_soft_ub(16, "N", 0).
shifts_soft_ub(16, "LM", 0).
shifts_hard_ub(16, "E", 0).
shifts_hard_ub(16, "N", 0).
shifts_hard_ub(16, "LM", 0).
shifts_hard_lb(16, "EM", 0).
shifts_soft_lb(16, "EM", 0).
shifts_soft_ub(16, "EM", 0).
shifts_hard_ub(16, "EM", 0).
% Staff 17:
shifts_hard_lb(17, "D", 7).
shifts_soft_lb(17, "D", 7).
shifts_soft_ub(17, "D", 14).
shifts_hard_ub(17, "D", 14).
shifts_hard_lb(17, "LD", 1).
shifts_hard_lb(17, "SE", 1).
shifts_hard_lb(17, "SN", 1).
shifts_soft_lb(17, "LD", 2).
shifts_soft_lb(17, "SE", 2).
shifts_soft_lb(17, "SN", 2).
shifts_soft_ub(17, "LD", 3).
shifts_soft_ub(17, "SE", 3).
shifts_soft_ub(17, "SN", 3).
shifts_hard_ub(17, "LD", 4).
shifts_hard_ub(17, "SE", 4).
shifts_hard_ub(17, "SN", 4).
shifts_hard_lb(17, "E", 0).
shifts_hard_lb(17, "N", 0).
shifts_hard_lb(17, "LM", 0).
shifts_soft_lb(17, "E", 0).
shifts_soft_lb(17, "N", 0).
shifts_soft_lb(17, "LM", 0).
shifts_soft_ub(17, "E", 0).
shifts_soft_ub(17, "N", 0).
shifts_soft_ub(17, "LM", 0).
shifts_hard_ub(17, "E", 0).
shifts_hard_ub(17, "N", 0).
shifts_hard_ub(17, "LM", 0).
shifts_hard_lb(17, "EM", 0).
shifts_soft_lb(17, "EM", 0).
shifts_soft_ub(17, "EM", 0).
shifts_hard_ub(17, "EM", 0).
% Staff 18:
shifts_hard_lb(18, "D", 6).
shifts_soft_lb(18, "D", 7).
shifts_soft_ub(18, "D", 13).
shifts_hard_ub(18, "D", 18).
shifts_hard_lb(18, "LD", 0).
shifts_hard_lb(18, "SE", 0).
shifts_hard_lb(18, "SN", 0).
shifts_soft_lb(18, "LD", 2).
shifts_soft_lb(18, "SE", 2).
shifts_soft_lb(18, "SN", 2).
shifts_soft_ub(18, "LD", 4).
shifts_soft_ub(18, "SE", 4).
shifts_soft_ub(18, "SN", 4).
shifts_hard_ub(18, "LD", 4).
shifts_hard_ub(18, "SE", 4).
shifts_hard_ub(18, "SN", 4).
shifts_hard_lb(18, "E", 0).
shifts_hard_lb(18, "N", 0).
shifts_hard_lb(18, "LM", 0).
shifts_soft_lb(18, "E", 0).
shifts_soft_lb(18, "N", 0).
shifts_soft_lb(18, "LM", 0).
shifts_soft_ub(18, "E", 0).
shifts_soft_ub(18, "N", 0).
shifts_soft_ub(18, "LM", 0).
shifts_hard_ub(18, "E", 0).
shifts_hard_ub(18, "N", 0).
shifts_hard_ub(18, "LM", 0).
shifts_hard_lb(18, "EM", 0).
shifts_soft_lb(18, "EM", 0).
shifts_soft_ub(18, "EM", 0).
shifts_hard_ub(18, "EM", 0).
% Staff 19:
shifts_hard_lb(19, "D", 3).
shifts_soft_lb(19, "D", 4).
shifts_soft_ub(19, "D", 8).
shifts_hard_ub(19, "D", 10).
shifts_hard_lb(19, "LD", 3).
shifts_hard_lb(19, "SE", 3).
shifts_hard_lb(19, "SN", 3).
shifts_soft_lb(19, "LD", 3).
shifts_soft_lb(19, "SE", 3).
shifts_soft_lb(19, "SN", 3).
shifts_soft_ub(19, "LD", 5).
shifts_soft_ub(19, "SE", 5).
shifts_soft_ub(19, "SN", 5).
shifts_hard_ub(19, "LD", 5).
shifts_hard_ub(19, "SE", 5).
shifts_hard_ub(19, "SN", 5).
shifts_hard_lb(19, "E", 0).
shifts_hard_lb(19, "N", 0).
shifts_hard_lb(19, "LM", 0).
shifts_soft_lb(19, "E", 0).
shifts_soft_lb(19, "N", 0).
shifts_soft_lb(19, "LM", 0).
shifts_soft_ub(19, "E", 0).
shifts_soft_ub(19, "N", 0).
shifts_soft_ub(19, "LM", 0).
shifts_hard_ub(19, "E", 1).
shifts_hard_ub(19, "N", 1).
shifts_hard_ub(19, "LM", 1).
shifts_hard_lb(19, "EM", 0).
shifts_soft_lb(19, "EM", 0).
shifts_soft_ub(19, "EM", 0).
shifts_hard_ub(19, "EM", 0).
% Staff 20:
shifts_hard_lb(20, "D", 3).
shifts_soft_lb(20, "D", 3).
shifts_soft_ub(20, "D", 9).
shifts_hard_ub(20, "D", 13).
shifts_hard_lb(20, "LD", 2).
shifts_hard_lb(20, "SE", 2).
shifts_hard_lb(20, "SN", 2).
shifts_soft_lb(20, "LD", 2).
shifts_soft_lb(20, "SE", 2).
shifts_soft_lb(20, "SN", 2).
shifts_soft_ub(20, "LD", 5).
shifts_soft_ub(20, "SE", 5).
shifts_soft_ub(20, "SN", 5).
shifts_hard_ub(20, "LD", 6).
shifts_hard_ub(20, "SE", 6).
shifts_hard_ub(20, "SN", 6).
shifts_hard_lb(20, "E", 0).
shifts_hard_lb(20, "N", 0).
shifts_hard_lb(20, "LM", 0).
shifts_soft_lb(20, "E", 0).
shifts_soft_lb(20, "N", 0).
shifts_soft_lb(20, "LM", 0).
shifts_soft_ub(20, "E", 0).
shifts_soft_ub(20, "N", 0).
shifts_soft_ub(20, "LM", 0).
shifts_hard_ub(20, "E", 1).
shifts_hard_ub(20, "N", 1).
shifts_hard_ub(20, "LM", 1).
shifts_hard_lb(20, "EM", 0).
shifts_soft_lb(20, "EM", 0).
shifts_soft_ub(20, "EM", 0).
shifts_hard_ub(20, "EM", 0).
% Staff 21:
shifts_hard_lb(21, "D", 2).
shifts_soft_lb(21, "D", 3).
shifts_soft_ub(21, "D", 11).
shifts_hard_ub(21, "D", 13).
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
shifts_hard_lb(21, "LM", 0).
shifts_soft_lb(21, "E", 0).
shifts_soft_lb(21, "N", 0).
shifts_soft_lb(21, "LM", 0).
shifts_soft_ub(21, "E", 0).
shifts_soft_ub(21, "N", 0).
shifts_soft_ub(21, "LM", 0).
shifts_hard_ub(21, "E", 0).
shifts_hard_ub(21, "N", 0).
shifts_hard_ub(21, "LM", 0).
shifts_hard_lb(21, "EM", 0).
shifts_soft_lb(21, "EM", 0).
shifts_soft_ub(21, "EM", 0).
shifts_hard_ub(21, "EM", 0).
% Staff 22:
shifts_hard_lb(22, "D", 4).
shifts_soft_lb(22, "D", 4).
shifts_soft_ub(22, "D", 7).
shifts_hard_ub(22, "D", 7).
shifts_hard_lb(22, "LD", 4).
shifts_hard_lb(22, "SE", 4).
shifts_hard_lb(22, "SN", 4).
shifts_soft_lb(22, "LD", 4).
shifts_soft_lb(22, "SE", 4).
shifts_soft_lb(22, "SN", 4).
shifts_soft_ub(22, "LD", 5).
shifts_soft_ub(22, "SE", 5).
shifts_soft_ub(22, "SN", 5).
shifts_hard_ub(22, "LD", 5).
shifts_hard_ub(22, "SE", 5).
shifts_hard_ub(22, "SN", 5).
shifts_hard_lb(22, "E", 0).
shifts_hard_lb(22, "N", 0).
shifts_hard_lb(22, "LM", 0).
shifts_soft_lb(22, "E", 0).
shifts_soft_lb(22, "N", 0).
shifts_soft_lb(22, "LM", 0).
shifts_soft_ub(22, "E", 0).
shifts_soft_ub(22, "N", 0).
shifts_soft_ub(22, "LM", 0).
shifts_hard_ub(22, "E", 0).
shifts_hard_ub(22, "N", 0).
shifts_hard_ub(22, "LM", 0).
shifts_hard_lb(22, "EM", 0).
shifts_soft_lb(22, "EM", 0).
shifts_soft_ub(22, "EM", 0).
shifts_hard_ub(22, "EM", 0).
% Staff 23:
shifts_hard_lb(23, "D", 2).
shifts_soft_lb(23, "D", 3).
shifts_soft_ub(23, "D", 6).
shifts_hard_ub(23, "D", 7).
shifts_hard_lb(23, "LD", 4).
shifts_hard_lb(23, "SE", 4).
shifts_hard_lb(23, "SN", 4).
shifts_soft_lb(23, "LD", 4).
shifts_soft_lb(23, "SE", 4).
shifts_soft_lb(23, "SN", 4).
shifts_soft_ub(23, "LD", 6).
shifts_soft_ub(23, "SE", 6).
shifts_soft_ub(23, "SN", 6).
shifts_hard_ub(23, "LD", 6).
shifts_hard_ub(23, "SE", 6).
shifts_hard_ub(23, "SN", 6).
shifts_hard_lb(23, "E", 0).
shifts_hard_lb(23, "N", 0).
shifts_hard_lb(23, "LM", 0).
shifts_soft_lb(23, "E", 0).
shifts_soft_lb(23, "N", 0).
shifts_soft_lb(23, "LM", 0).
shifts_soft_ub(23, "E", 0).
shifts_soft_ub(23, "N", 0).
shifts_soft_ub(23, "LM", 0).
shifts_hard_ub(23, "E", 0).
shifts_hard_ub(23, "N", 0).
shifts_hard_ub(23, "LM", 0).
shifts_hard_lb(23, "EM", 0).
shifts_soft_lb(23, "EM", 0).
shifts_soft_ub(23, "EM", 0).
shifts_hard_ub(23, "EM", 0).
% Staff 24:
shifts_hard_lb(24, "D", 4).
shifts_soft_lb(24, "D", 4).
shifts_soft_ub(24, "D", 8).
shifts_hard_ub(24, "D", 8).
shifts_hard_lb(24, "LD", 3).
shifts_hard_lb(24, "SE", 3).
shifts_hard_lb(24, "SN", 3).
shifts_soft_lb(24, "LD", 3).
shifts_soft_lb(24, "SE", 3).
shifts_soft_lb(24, "SN", 3).
shifts_soft_ub(24, "LD", 5).
shifts_soft_ub(24, "SE", 5).
shifts_soft_ub(24, "SN", 5).
shifts_hard_ub(24, "LD", 5).
shifts_hard_ub(24, "SE", 5).
shifts_hard_ub(24, "SN", 5).
shifts_hard_lb(24, "E", 0).
shifts_hard_lb(24, "N", 0).
shifts_hard_lb(24, "LM", 0).
shifts_soft_lb(24, "E", 0).
shifts_soft_lb(24, "N", 0).
shifts_soft_lb(24, "LM", 0).
shifts_soft_ub(24, "E", 0).
shifts_soft_ub(24, "N", 0).
shifts_soft_ub(24, "LM", 0).
shifts_hard_ub(24, "E", 0).
shifts_hard_ub(24, "N", 0).
shifts_hard_ub(24, "LM", 0).
shifts_hard_lb(24, "EM", 0).
shifts_soft_lb(24, "EM", 0).
shifts_soft_ub(24, "EM", 0).
shifts_hard_ub(24, "EM", 0).
% Staff 25:
shifts_hard_lb(25, "D", 2).
shifts_soft_lb(25, "D", 2).
shifts_soft_ub(25, "D", 8).
shifts_hard_ub(25, "D", 12).
shifts_hard_lb(25, "LD", 2).
shifts_hard_lb(25, "SE", 2).
shifts_hard_lb(25, "SN", 2).
shifts_soft_lb(25, "LD", 3).
shifts_soft_lb(25, "SE", 3).
shifts_soft_lb(25, "SN", 3).
shifts_soft_ub(25, "LD", 6).
shifts_soft_ub(25, "SE", 6).
shifts_soft_ub(25, "SN", 6).
shifts_hard_ub(25, "LD", 6).
shifts_hard_ub(25, "SE", 6).
shifts_hard_ub(25, "SN", 6).
shifts_hard_lb(25, "E", 0).
shifts_hard_lb(25, "N", 0).
shifts_hard_lb(25, "LM", 0).
shifts_soft_lb(25, "E", 0).
shifts_soft_lb(25, "N", 0).
shifts_soft_lb(25, "LM", 0).
shifts_soft_ub(25, "E", 0).
shifts_soft_ub(25, "N", 0).
shifts_soft_ub(25, "LM", 0).
shifts_hard_ub(25, "E", 0).
shifts_hard_ub(25, "N", 0).
shifts_hard_ub(25, "LM", 0).
shifts_hard_lb(25, "EM", 0).
shifts_soft_lb(25, "EM", 0).
shifts_soft_ub(25, "EM", 0).
shifts_hard_ub(25, "EM", 0).
% Staff 26:
shifts_hard_lb(26, "D", 2).
shifts_soft_lb(26, "D", 3).
shifts_soft_ub(26, "D", 8).
shifts_hard_ub(26, "D", 8).
shifts_hard_lb(26, "LD", 3).
shifts_hard_lb(26, "SE", 3).
shifts_hard_lb(26, "SN", 3).
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
shifts_hard_lb(26, "LM", 0).
shifts_soft_lb(26, "E", 0).
shifts_soft_lb(26, "N", 0).
shifts_soft_lb(26, "LM", 0).
shifts_soft_ub(26, "E", 0).
shifts_soft_ub(26, "N", 0).
shifts_soft_ub(26, "LM", 0).
shifts_hard_ub(26, "E", 0).
shifts_hard_ub(26, "N", 0).
shifts_hard_ub(26, "LM", 0).
shifts_hard_lb(26, "EM", 0).
shifts_soft_lb(26, "EM", 0).
shifts_soft_ub(26, "EM", 0).
shifts_hard_ub(26, "EM", 0).
% Staff 27:
shifts_hard_lb(27, "D", 2).
shifts_soft_lb(27, "D", 3).
shifts_soft_ub(27, "D", 6).
shifts_hard_ub(27, "D", 8).
shifts_hard_lb(27, "LD", 4).
shifts_hard_lb(27, "SE", 4).
shifts_hard_lb(27, "SN", 4).
shifts_soft_lb(27, "LD", 4).
shifts_soft_lb(27, "SE", 4).
shifts_soft_lb(27, "SN", 4).
shifts_soft_ub(27, "LD", 5).
shifts_soft_ub(27, "SE", 5).
shifts_soft_ub(27, "SN", 5).
shifts_hard_ub(27, "LD", 6).
shifts_hard_ub(27, "SE", 6).
shifts_hard_ub(27, "SN", 6).
shifts_hard_lb(27, "E", 0).
shifts_hard_lb(27, "N", 0).
shifts_hard_lb(27, "LM", 0).
shifts_soft_lb(27, "E", 0).
shifts_soft_lb(27, "N", 0).
shifts_soft_lb(27, "LM", 0).
shifts_soft_ub(27, "E", 0).
shifts_soft_ub(27, "N", 0).
shifts_soft_ub(27, "LM", 0).
shifts_hard_ub(27, "E", 0).
shifts_hard_ub(27, "N", 0).
shifts_hard_ub(27, "LM", 0).
shifts_hard_lb(27, "EM", 0).
shifts_soft_lb(27, "EM", 0).
shifts_soft_ub(27, "EM", 0).
shifts_hard_ub(27, "EM", 0).
% Staff 28:
shifts_hard_lb(28, "D", 3).
shifts_soft_lb(28, "D", 3).
shifts_soft_ub(28, "D", 6).
shifts_hard_ub(28, "D", 8).
shifts_hard_lb(28, "LD", 4).
shifts_hard_lb(28, "SE", 4).
shifts_hard_lb(28, "SN", 4).
shifts_soft_lb(28, "LD", 4).
shifts_soft_lb(28, "SE", 4).
shifts_soft_lb(28, "SN", 4).
shifts_soft_ub(28, "LD", 5).
shifts_soft_ub(28, "SE", 5).
shifts_soft_ub(28, "SN", 5).
shifts_hard_ub(28, "LD", 6).
shifts_hard_ub(28, "SE", 6).
shifts_hard_ub(28, "SN", 6).
shifts_hard_lb(28, "E", 0).
shifts_hard_lb(28, "N", 0).
shifts_hard_lb(28, "LM", 0).
shifts_soft_lb(28, "E", 0).
shifts_soft_lb(28, "N", 0).
shifts_soft_lb(28, "LM", 0).
shifts_soft_ub(28, "E", 0).
shifts_soft_ub(28, "N", 0).
shifts_soft_ub(28, "LM", 0).
shifts_hard_ub(28, "E", 0).
shifts_hard_ub(28, "N", 0).
shifts_hard_ub(28, "LM", 0).
shifts_hard_lb(28, "EM", 0).
shifts_soft_lb(28, "EM", 0).
shifts_soft_ub(28, "EM", 0).
shifts_hard_ub(28, "EM", 0).
% Staff 29:
shifts_hard_lb(29, "D", 6).
shifts_soft_lb(29, "D", 7).
shifts_soft_ub(29, "D", 19).
shifts_hard_ub(29, "D", 20).
shifts_hard_lb(29, "LD", 0).
shifts_hard_lb(29, "SE", 0).
shifts_hard_lb(29, "SN", 0).
shifts_soft_lb(29, "LD", 0).
shifts_soft_lb(29, "SE", 0).
shifts_soft_lb(29, "SN", 0).
shifts_soft_ub(29, "LD", 4).
shifts_soft_ub(29, "SE", 4).
shifts_soft_ub(29, "SN", 4).
shifts_hard_ub(29, "LD", 4).
shifts_hard_ub(29, "SE", 4).
shifts_hard_ub(29, "SN", 4).
shifts_hard_lb(29, "E", 0).
shifts_hard_lb(29, "N", 0).
shifts_hard_lb(29, "LM", 0).
shifts_soft_lb(29, "E", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_lb(29, "LM", 0).
shifts_soft_ub(29, "E", 0).
shifts_soft_ub(29, "N", 0).
shifts_soft_ub(29, "LM", 0).
shifts_hard_ub(29, "E", 0).
shifts_hard_ub(29, "N", 0).
shifts_hard_ub(29, "LM", 0).
shifts_hard_lb(29, "EM", 0).
shifts_soft_lb(29, "EM", 0).
shifts_soft_ub(29, "EM", 0).
shifts_hard_ub(29, "EM", 0).
% Staff 30:
shifts_hard_lb(30, "D", 4).
shifts_soft_lb(30, "D", 4).
shifts_soft_ub(30, "D", 15).
shifts_hard_ub(30, "D", 16).
shifts_hard_lb(30, "LD", 0).
shifts_hard_lb(30, "SE", 0).
shifts_hard_lb(30, "SN", 0).
shifts_soft_lb(30, "LD", 0).
shifts_soft_lb(30, "SE", 0).
shifts_soft_lb(30, "SN", 0).
shifts_soft_ub(30, "LD", 5).
shifts_soft_ub(30, "SE", 5).
shifts_soft_ub(30, "SN", 5).
shifts_hard_ub(30, "LD", 5).
shifts_hard_ub(30, "SE", 5).
shifts_hard_ub(30, "SN", 5).
shifts_hard_lb(30, "E", 0).
shifts_hard_lb(30, "N", 0).
shifts_hard_lb(30, "LM", 0).
shifts_soft_lb(30, "E", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_lb(30, "LM", 0).
shifts_soft_ub(30, "E", 0).
shifts_soft_ub(30, "N", 0).
shifts_soft_ub(30, "LM", 0).
shifts_hard_ub(30, "E", 0).
shifts_hard_ub(30, "N", 0).
shifts_hard_ub(30, "LM", 0).
shifts_hard_lb(30, "EM", 0).
shifts_soft_lb(30, "EM", 0).
shifts_soft_ub(30, "EM", 0).
shifts_hard_ub(30, "EM", 0).
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
