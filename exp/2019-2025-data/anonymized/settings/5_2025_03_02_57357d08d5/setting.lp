% Staffs ---------------------------------------------------
staff(1, "Scott Johnson", "Deputy Chief Nurse", "50584", 0).
staff(2, "Lindsey Baker", "Deputy Chief Nurse", "14591", 0).
staff(3, "Michelle Mitchell", "Nurse", "36466", 0).
staff(4, "Christine Taylor", "Nurse", "89033", 0).
staff(5, "Richard Stephens", "Nurse", "45891", 0).
staff(6, "Jean Pennington", "Nurse", "28871", 0).
staff(7, "Victor Vargas", "Nurse", "39748", 0).
staff(8, "Deanna Turner", "Nurse", "42007", 0).
staff(9, "Heather Lowe", "Nurse", "16954", 0).
staff(10, "Edward Hines", "Nurse", "42223", 0).
staff(11, "Aaron Washington", "Nurse", "62223", 0).
staff(12, "Danielle Mcbride", "Nurse", "09002", 0).
staff(13, "Patricia Levy", "Nurse", "41516", 0).
staff(14, "Lee Blake", "Nurse", "02013", 0).
staff(15, "Kimberly Moody", "Nurse", "55630", 0).
staff(16, "Ashlee Watson", "Nurse", "37568", 0).
staff(17, "Laura Watson", "Deputy Chief Nurse", "82132", 0).
staff(18, "Kristin Roberson", "Deputy Chief Nurse", "64798", 0).
staff(19, "Jimmy Davis", "Nurse", "74243", 0).
staff(20, "Nicholas Price", "Nurse", "70636", 0).
staff(21, "Donna Casey", "Nurse", "13905", 0).
staff(22, "Alexander Jackson", "Nurse", "29654", 0).
staff(23, "Ann Allen DDS", "Nurse", "41192", 0).
staff(24, "Larry Porter", "Nurse", "05417", 0).
staff(25, "Tyler Tran", "Nurse", "54907", 0).
staff(26, "Alex Torres", "Nurse", "76023", 0).
staff(27, "Allison Hull", "Nurse", "69635", 0).
staff(28, "James Evans", "Nurse", "29319", 0).
staff(29, "Elizabeth Estes", "Nurse", "94132", 0).
staff(30, "Sandra Joseph", "Nurse", "21510", 0).
% Groups ---------------------------------------------------
staff_group("All").
staff_group("All", 2).
staff_group("All", 3).
staff_group("All", 4).
staff_group("All", 5).
staff_group("All", 6).
staff_group("All", 7).
staff_group("All", 8).
staff_group("All", 9).
staff_group("All", 10).
staff_group("All", 11).
staff_group("All", 12).
staff_group("All", 13).
staff_group("All", 14).
staff_group("All", 15).
staff_group("All", 16).
staff_group("All", 17).
staff_group("All", 18).
staff_group("All", 19).
staff_group("All", 20).
staff_group("All", 21).
staff_group("All", 22).
staff_group("All", 23).
staff_group("All", 24).
staff_group("All", 25).
staff_group("All", 26).
staff_group("All", 27).
staff_group("All", 28).
staff_group("All", 29).
staff_group("All", 30).
staff_group("Heads").
% Skipped nurse group column "師長代行" (no mapping).
% Skipped nurse group column "A" (no mapping).
% Skipped nurse group column "B" (no mapping).
staff_group("Newcomers").
staff_group("Night").
staff_group("Night", 1).
staff_group("Night", 2).
staff_group("Night", 3).
staff_group("Night", 4).
staff_group("Night", 5).
staff_group("Night", 6).
staff_group("Night", 7).
staff_group("Night", 8).
staff_group("Night", 9).
staff_group("Night", 10).
staff_group("Night", 11).
staff_group("Night", 12).
staff_group("Night", 13).
staff_group("Night", 14).
staff_group("Night", 15).
staff_group("Night", 16).
staff_group("Night", 17).
staff_group("Night", 18).
staff_group("Night", 19).
staff_group("Night", 20).
staff_group("Night", 21).
staff_group("Night", 22).
staff_group("Night", 23).
staff_group("Night", 24).
staff_group("Night", 25).
staff_group("Night", 26).
staff_group("Night", 27).
staff_group("Night", 28).
staff_group("Night", 29).
staff_group("Night", 30).
% Shift constraints ----------------------------------------
% Maximum consecutive working days
consecutive_work_ub(hard, 6).
% Forbidden patterns
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
% Assignable previous/next shifts
prev_shift(hard, "SN", "SE").
next_shift(hard, "SN", "SE").
next_shift(hard, "SN", "WR").
next_shift(hard, "SN", "PH").
prev_shift(hard, "SE", "LD").
prev_shift(hard, "SE", "LM").
next_shift(hard, "SE", "SN").
% Pattern occurrences
pattern("WR-WR", 2).
pattern("WR-WR", 0, "WR").
pattern("WR-WR", 1, "WR").
patterns_hard_lb("WR-WR", 0).
patterns_soft_lb("WR-WR", 1).
patterns_soft_ub("WR-WR", 3).
patterns_hard_ub("WR-WR", 4).
% Vertical constraints -------------------------------------
shift_group("D").
shift_group("D", "D").
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
shift_group("LD").
shift_group("LD", "LD").
staff_dweek_hard_lb("All", "LD", "Weekday", 3).
shift_group("SE").
shift_group("SE", "SE").
staff_dweek_hard_lb("All", "SE", "Weekday", 3).
shift_group("SN").
shift_group("SN", "SN").
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
shift_group("E").
shift_group("E", "E").
staff_dweek_hard_lb("All", "E", "Weekday", 0).
shift_group("N").
shift_group("N", "N").
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
shift_group("LM").
shift_group("LM", "LM").
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
shift_group("LD+LM+E").
shift_group("LD+LM+E", "LD").
shift_group("LD+LM+E", "LM").
shift_group("LD+LM+E", "E").
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
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "師長代行" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "A" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
% Skipped vertical constraint for nurse group "B" (no mapping).
shift_group("SE+E").
shift_group("SE+E", "SE").
shift_group("SE+E", "E").
staff_dweek_hard_lb("Night", "SE+E", "Weekday", 3).
shift_group("SN+N").
shift_group("SN+N", "SN").
shift_group("SN+N", "N").
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
vertical_constraint_type("All","D","Staffs").
vertical_constraint_type("All","LD","Staffs").
vertical_constraint_type("All","SE","Staffs").
vertical_constraint_type("All","SN","Staffs").
vertical_constraint_type("All","E","Staffs").
vertical_constraint_type("All","N","Staffs").
vertical_constraint_type("All","LM","Staffs").
vertical_constraint_type("All","LD+LM+E","Staffs").
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "師長代行" (no mapping).
% Skipped vertical_constraint_type for nurse group "A" (no mapping).
% Skipped vertical_constraint_type for nurse group "B" (no mapping).
vertical_constraint_type("Night","SE+E","Staffs").
vertical_constraint_type("Night","SN+N","Staffs").
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
shift_group("EM").
shift_group("EM", "EM").
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
