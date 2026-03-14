% Staffs ---------------------------------------------------
staff(1, "清水　美紀", "副師長", "0123544", 0).
staff(2, "日向　恵", "副師長", "0051593", 0).
staff(3, "小林　伊吹", "看護師", "0355727", 0).
staff(4, "小佐野　まりな", "看護師", "0312576", 0).
staff(5, "武藤　裕紀", "看護師", "0370954", 0).
staff(6, "細井　一成", "看護師", "0418791", 0).
staff(7, "堀内　美希", "看護師", "0370340", 0).
staff(8, "髙山　りえ", "看護師", "0329337", 0).
staff(9, "阿部　知依奈", "看護師", "0350460", 0).
staff(10, "志村　実優", "看護師", "0399801", 0).
staff(11, "金野　舞子", "看護師", "0418825", 0).
staff(12, "中島　夏希", "看護師", "0341708", 0).
staff(13, "鈴木　百花", "看護師", "0418803", 0).
staff(14, "河西　笑歩", "看護師", "0399823", 0).
staff(15, "福岡　円華", "看護師", "0469623", 0).
staff(16, "笠井　優里香", "看護師", "0312703", 0).
staff(17, "神宮司　文", "副師長", "0146427", 0).
staff(18, "渡邉　祐将", "副師長", "0173250", 0).
staff(19, "中村　香織", "看護師", "0283504", 0).
staff(20, "青山　桃子", "看護師", "0282161", 0).
staff(21, "本田　愛梨香", "看護師", "0313054", 0).
staff(22, "成田　彩月", "看護師", "0355556", 0).
staff(23, "三井　歩夢", "看護師", "0384643", 0).
staff(24, "石橋　朋華", "看護師", "0370761", 0).
staff(25, "南　かれん", "看護師", "0370588", 0).
staff(26, "渡邉　栞里", "看護師", "0384712", 0).
staff(27, "矢野　杏藍", "看護師", "0399641", 0).
staff(28, "信田　舞奈", "看護師", "0418404", 0).
staff(29, "鶴田　裕花", "看護師", "0372615", 0).
staff(30, "清水　優花", "看護師", "0418814", 0).
% Groups ---------------------------------------------------
group("全員").
group("全員", 2).
group("全員", 3).
group("全員", 4).
group("全員", 5).
group("全員", 6).
group("全員", 7).
group("全員", 8).
group("全員", 9).
group("全員", 10).
group("全員", 11).
group("全員", 12).
group("全員", 13).
group("全員", 14).
group("全員", 15).
group("全員", 16).
group("全員", 17).
group("全員", 18).
group("全員", 19).
group("全員", 20).
group("全員", 21).
group("全員", 22).
group("全員", 23).
group("全員", 24).
group("全員", 25).
group("全員", 26).
group("全員", 27).
group("全員", 28).
group("全員", 29).
group("全員", 30).
group("師長").
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
group("新人").
group("夜勤").
group("夜勤", 1).
group("夜勤", 2).
group("夜勤", 3).
group("夜勤", 4).
group("夜勤", 5).
group("夜勤", 6).
group("夜勤", 7).
group("夜勤", 8).
group("夜勤", 9).
group("夜勤", 10).
group("夜勤", 11).
group("夜勤", 12).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
group("夜勤", 18).
group("夜勤", 19).
group("夜勤", 20).
group("夜勤", 21).
group("夜勤", 22).
group("夜勤", 23).
group("夜勤", 24).
group("夜勤", 25).
group("夜勤", 26).
group("夜勤", 27).
group("夜勤", 28).
group("夜勤", 29).
group("夜勤", 30).
% Shifts ---------------------------------------------------
shift("日").
shift("N").
shift("J").
shift("S").
shift("★").
shift("☆").
shift("A").
shift("P").
shift("○").
shift("◎").
shift("研").
shift("年").
shift("健").
shift("特").
shift("/").
% Shift constraints ----------------------------------------
% 連続勤務可能日数
consecutive_work_ub(6).
% 禁止パターン
forbidden_pattern("N日").
pattern("N日", 2).
pattern("N日", 0, "N").
pattern("N日", 1, "日").
forbidden_pattern("NN").
pattern("NN", 2).
pattern("NN", 0, "N").
pattern("NN", 1, "N").
forbidden_pattern("N研").
pattern("N研", 2).
pattern("N研", 0, "N").
pattern("N研", 1, "研").
forbidden_pattern("P日").
pattern("P日", 2).
pattern("P日", 0, "P").
pattern("P日", 1, "日").
forbidden_pattern("PN").
pattern("PN", 2).
pattern("PN", 0, "P").
pattern("PN", 1, "N").
forbidden_pattern("P研").
pattern("P研", 2).
pattern("P研", 0, "P").
pattern("P研", 1, "研").
forbidden_pattern("日日日日日N").
pattern("日日日日日N", 6).
pattern("日日日日日N", 0, "日").
pattern("日日日日日N", 1, "日").
pattern("日日日日日N", 2, "日").
pattern("日日日日日N", 3, "日").
pattern("日日日日日N", 4, "日").
pattern("日日日日日N", 5, "N").
forbidden_pattern("日日日日NP").
pattern("日日日日NP", 6).
pattern("日日日日NP", 0, "日").
pattern("日日日日NP", 1, "日").
pattern("日日日日NP", 2, "日").
pattern("日日日日NP", 3, "日").
pattern("日日日日NP", 4, "N").
pattern("日日日日NP", 5, "P").
forbidden_pattern("日日日日研N").
pattern("日日日日研N", 6).
pattern("日日日日研N", 0, "日").
pattern("日日日日研N", 1, "日").
pattern("日日日日研N", 2, "日").
pattern("日日日日研N", 3, "日").
pattern("日日日日研N", 4, "研").
pattern("日日日日研N", 5, "N").
forbidden_pattern("日日日NPP").
pattern("日日日NPP", 6).
pattern("日日日NPP", 0, "日").
pattern("日日日NPP", 1, "日").
pattern("日日日NPP", 2, "日").
pattern("日日日NPP", 3, "N").
pattern("日日日NPP", 4, "P").
pattern("日日日NPP", 5, "P").
forbidden_pattern("日日日研日N").
pattern("日日日研日N", 6).
pattern("日日日研日N", 0, "日").
pattern("日日日研日N", 1, "日").
pattern("日日日研日N", 2, "日").
pattern("日日日研日N", 3, "研").
pattern("日日日研日N", 4, "日").
pattern("日日日研日N", 5, "N").
forbidden_pattern("日日日研NP").
pattern("日日日研NP", 6).
pattern("日日日研NP", 0, "日").
pattern("日日日研NP", 1, "日").
pattern("日日日研NP", 2, "日").
pattern("日日日研NP", 3, "研").
pattern("日日日研NP", 4, "N").
pattern("日日日研NP", 5, "P").
forbidden_pattern("日日日研研N").
pattern("日日日研研N", 6).
pattern("日日日研研N", 0, "日").
pattern("日日日研研N", 1, "日").
pattern("日日日研研N", 2, "日").
pattern("日日日研研N", 3, "研").
pattern("日日日研研N", 4, "研").
pattern("日日日研研N", 5, "N").
forbidden_pattern("日日NPPP").
pattern("日日NPPP", 6).
pattern("日日NPPP", 0, "日").
pattern("日日NPPP", 1, "日").
pattern("日日NPPP", 2, "N").
pattern("日日NPPP", 3, "P").
pattern("日日NPPP", 4, "P").
pattern("日日NPPP", 5, "P").
forbidden_pattern("日日研日日N").
pattern("日日研日日N", 6).
pattern("日日研日日N", 0, "日").
pattern("日日研日日N", 1, "日").
pattern("日日研日日N", 2, "研").
pattern("日日研日日N", 3, "日").
pattern("日日研日日N", 4, "日").
pattern("日日研日日N", 5, "N").
forbidden_pattern("日日研日NP").
pattern("日日研日NP", 6).
pattern("日日研日NP", 0, "日").
pattern("日日研日NP", 1, "日").
pattern("日日研日NP", 2, "研").
pattern("日日研日NP", 3, "日").
pattern("日日研日NP", 4, "N").
pattern("日日研日NP", 5, "P").
forbidden_pattern("日日研日研N").
pattern("日日研日研N", 6).
pattern("日日研日研N", 0, "日").
pattern("日日研日研N", 1, "日").
pattern("日日研日研N", 2, "研").
pattern("日日研日研N", 3, "日").
pattern("日日研日研N", 4, "研").
pattern("日日研日研N", 5, "N").
forbidden_pattern("日日研NPP").
pattern("日日研NPP", 6).
pattern("日日研NPP", 0, "日").
pattern("日日研NPP", 1, "日").
pattern("日日研NPP", 2, "研").
pattern("日日研NPP", 3, "N").
pattern("日日研NPP", 4, "P").
pattern("日日研NPP", 5, "P").
forbidden_pattern("日日研研日N").
pattern("日日研研日N", 6).
pattern("日日研研日N", 0, "日").
pattern("日日研研日N", 1, "日").
pattern("日日研研日N", 2, "研").
pattern("日日研研日N", 3, "研").
pattern("日日研研日N", 4, "日").
pattern("日日研研日N", 5, "N").
forbidden_pattern("日日研研NP").
pattern("日日研研NP", 6).
pattern("日日研研NP", 0, "日").
pattern("日日研研NP", 1, "日").
pattern("日日研研NP", 2, "研").
pattern("日日研研NP", 3, "研").
pattern("日日研研NP", 4, "N").
pattern("日日研研NP", 5, "P").
forbidden_pattern("日日研研研N").
pattern("日日研研研N", 6).
pattern("日日研研研N", 0, "日").
pattern("日日研研研N", 1, "日").
pattern("日日研研研N", 2, "研").
pattern("日日研研研N", 3, "研").
pattern("日日研研研N", 4, "研").
pattern("日日研研研N", 5, "N").
forbidden_pattern("日NPPPP").
pattern("日NPPPP", 6).
pattern("日NPPPP", 0, "日").
pattern("日NPPPP", 1, "N").
pattern("日NPPPP", 2, "P").
pattern("日NPPPP", 3, "P").
pattern("日NPPPP", 4, "P").
pattern("日NPPPP", 5, "P").
forbidden_pattern("日研日日日N").
pattern("日研日日日N", 6).
pattern("日研日日日N", 0, "日").
pattern("日研日日日N", 1, "研").
pattern("日研日日日N", 2, "日").
pattern("日研日日日N", 3, "日").
pattern("日研日日日N", 4, "日").
pattern("日研日日日N", 5, "N").
forbidden_pattern("日研日日NP").
pattern("日研日日NP", 6).
pattern("日研日日NP", 0, "日").
pattern("日研日日NP", 1, "研").
pattern("日研日日NP", 2, "日").
pattern("日研日日NP", 3, "日").
pattern("日研日日NP", 4, "N").
pattern("日研日日NP", 5, "P").
forbidden_pattern("日研日日研N").
pattern("日研日日研N", 6).
pattern("日研日日研N", 0, "日").
pattern("日研日日研N", 1, "研").
pattern("日研日日研N", 2, "日").
pattern("日研日日研N", 3, "日").
pattern("日研日日研N", 4, "研").
pattern("日研日日研N", 5, "N").
forbidden_pattern("日研日NPP").
pattern("日研日NPP", 6).
pattern("日研日NPP", 0, "日").
pattern("日研日NPP", 1, "研").
pattern("日研日NPP", 2, "日").
pattern("日研日NPP", 3, "N").
pattern("日研日NPP", 4, "P").
pattern("日研日NPP", 5, "P").
forbidden_pattern("日研日研日N").
pattern("日研日研日N", 6).
pattern("日研日研日N", 0, "日").
pattern("日研日研日N", 1, "研").
pattern("日研日研日N", 2, "日").
pattern("日研日研日N", 3, "研").
pattern("日研日研日N", 4, "日").
pattern("日研日研日N", 5, "N").
forbidden_pattern("日研日研NP").
pattern("日研日研NP", 6).
pattern("日研日研NP", 0, "日").
pattern("日研日研NP", 1, "研").
pattern("日研日研NP", 2, "日").
pattern("日研日研NP", 3, "研").
pattern("日研日研NP", 4, "N").
pattern("日研日研NP", 5, "P").
forbidden_pattern("日研日研研N").
pattern("日研日研研N", 6).
pattern("日研日研研N", 0, "日").
pattern("日研日研研N", 1, "研").
pattern("日研日研研N", 2, "日").
pattern("日研日研研N", 3, "研").
pattern("日研日研研N", 4, "研").
pattern("日研日研研N", 5, "N").
forbidden_pattern("日研NPPP").
pattern("日研NPPP", 6).
pattern("日研NPPP", 0, "日").
pattern("日研NPPP", 1, "研").
pattern("日研NPPP", 2, "N").
pattern("日研NPPP", 3, "P").
pattern("日研NPPP", 4, "P").
pattern("日研NPPP", 5, "P").
forbidden_pattern("日研研日日N").
pattern("日研研日日N", 6).
pattern("日研研日日N", 0, "日").
pattern("日研研日日N", 1, "研").
pattern("日研研日日N", 2, "研").
pattern("日研研日日N", 3, "日").
pattern("日研研日日N", 4, "日").
pattern("日研研日日N", 5, "N").
forbidden_pattern("日研研日NP").
pattern("日研研日NP", 6).
pattern("日研研日NP", 0, "日").
pattern("日研研日NP", 1, "研").
pattern("日研研日NP", 2, "研").
pattern("日研研日NP", 3, "日").
pattern("日研研日NP", 4, "N").
pattern("日研研日NP", 5, "P").
forbidden_pattern("日研研日研N").
pattern("日研研日研N", 6).
pattern("日研研日研N", 0, "日").
pattern("日研研日研N", 1, "研").
pattern("日研研日研N", 2, "研").
pattern("日研研日研N", 3, "日").
pattern("日研研日研N", 4, "研").
pattern("日研研日研N", 5, "N").
forbidden_pattern("日研研NPP").
pattern("日研研NPP", 6).
pattern("日研研NPP", 0, "日").
pattern("日研研NPP", 1, "研").
pattern("日研研NPP", 2, "研").
pattern("日研研NPP", 3, "N").
pattern("日研研NPP", 4, "P").
pattern("日研研NPP", 5, "P").
forbidden_pattern("日研研研日N").
pattern("日研研研日N", 6).
pattern("日研研研日N", 0, "日").
pattern("日研研研日N", 1, "研").
pattern("日研研研日N", 2, "研").
pattern("日研研研日N", 3, "研").
pattern("日研研研日N", 4, "日").
pattern("日研研研日N", 5, "N").
forbidden_pattern("日研研研NP").
pattern("日研研研NP", 6).
pattern("日研研研NP", 0, "日").
pattern("日研研研NP", 1, "研").
pattern("日研研研NP", 2, "研").
pattern("日研研研NP", 3, "研").
pattern("日研研研NP", 4, "N").
pattern("日研研研NP", 5, "P").
forbidden_pattern("日研研研研N").
pattern("日研研研研N", 6).
pattern("日研研研研N", 0, "日").
pattern("日研研研研N", 1, "研").
pattern("日研研研研N", 2, "研").
pattern("日研研研研N", 3, "研").
pattern("日研研研研N", 4, "研").
pattern("日研研研研N", 5, "N").
forbidden_pattern("NPPPPP").
pattern("NPPPPP", 6).
pattern("NPPPPP", 0, "N").
pattern("NPPPPP", 1, "P").
pattern("NPPPPP", 2, "P").
pattern("NPPPPP", 3, "P").
pattern("NPPPPP", 4, "P").
pattern("NPPPPP", 5, "P").
forbidden_pattern("研日日日日N").
pattern("研日日日日N", 6).
pattern("研日日日日N", 0, "研").
pattern("研日日日日N", 1, "日").
pattern("研日日日日N", 2, "日").
pattern("研日日日日N", 3, "日").
pattern("研日日日日N", 4, "日").
pattern("研日日日日N", 5, "N").
forbidden_pattern("研日日日NP").
pattern("研日日日NP", 6).
pattern("研日日日NP", 0, "研").
pattern("研日日日NP", 1, "日").
pattern("研日日日NP", 2, "日").
pattern("研日日日NP", 3, "日").
pattern("研日日日NP", 4, "N").
pattern("研日日日NP", 5, "P").
forbidden_pattern("研日日日研N").
pattern("研日日日研N", 6).
pattern("研日日日研N", 0, "研").
pattern("研日日日研N", 1, "日").
pattern("研日日日研N", 2, "日").
pattern("研日日日研N", 3, "日").
pattern("研日日日研N", 4, "研").
pattern("研日日日研N", 5, "N").
forbidden_pattern("研日日NPP").
pattern("研日日NPP", 6).
pattern("研日日NPP", 0, "研").
pattern("研日日NPP", 1, "日").
pattern("研日日NPP", 2, "日").
pattern("研日日NPP", 3, "N").
pattern("研日日NPP", 4, "P").
pattern("研日日NPP", 5, "P").
forbidden_pattern("研日日研日N").
pattern("研日日研日N", 6).
pattern("研日日研日N", 0, "研").
pattern("研日日研日N", 1, "日").
pattern("研日日研日N", 2, "日").
pattern("研日日研日N", 3, "研").
pattern("研日日研日N", 4, "日").
pattern("研日日研日N", 5, "N").
forbidden_pattern("研日日研NP").
pattern("研日日研NP", 6).
pattern("研日日研NP", 0, "研").
pattern("研日日研NP", 1, "日").
pattern("研日日研NP", 2, "日").
pattern("研日日研NP", 3, "研").
pattern("研日日研NP", 4, "N").
pattern("研日日研NP", 5, "P").
forbidden_pattern("研日日研研N").
pattern("研日日研研N", 6).
pattern("研日日研研N", 0, "研").
pattern("研日日研研N", 1, "日").
pattern("研日日研研N", 2, "日").
pattern("研日日研研N", 3, "研").
pattern("研日日研研N", 4, "研").
pattern("研日日研研N", 5, "N").
forbidden_pattern("研日NPPP").
pattern("研日NPPP", 6).
pattern("研日NPPP", 0, "研").
pattern("研日NPPP", 1, "日").
pattern("研日NPPP", 2, "N").
pattern("研日NPPP", 3, "P").
pattern("研日NPPP", 4, "P").
pattern("研日NPPP", 5, "P").
forbidden_pattern("研日研日日N").
pattern("研日研日日N", 6).
pattern("研日研日日N", 0, "研").
pattern("研日研日日N", 1, "日").
pattern("研日研日日N", 2, "研").
pattern("研日研日日N", 3, "日").
pattern("研日研日日N", 4, "日").
pattern("研日研日日N", 5, "N").
forbidden_pattern("研日研日NP").
pattern("研日研日NP", 6).
pattern("研日研日NP", 0, "研").
pattern("研日研日NP", 1, "日").
pattern("研日研日NP", 2, "研").
pattern("研日研日NP", 3, "日").
pattern("研日研日NP", 4, "N").
pattern("研日研日NP", 5, "P").
forbidden_pattern("研日研日研N").
pattern("研日研日研N", 6).
pattern("研日研日研N", 0, "研").
pattern("研日研日研N", 1, "日").
pattern("研日研日研N", 2, "研").
pattern("研日研日研N", 3, "日").
pattern("研日研日研N", 4, "研").
pattern("研日研日研N", 5, "N").
forbidden_pattern("研日研NPP").
pattern("研日研NPP", 6).
pattern("研日研NPP", 0, "研").
pattern("研日研NPP", 1, "日").
pattern("研日研NPP", 2, "研").
pattern("研日研NPP", 3, "N").
pattern("研日研NPP", 4, "P").
pattern("研日研NPP", 5, "P").
forbidden_pattern("研日研研日N").
pattern("研日研研日N", 6).
pattern("研日研研日N", 0, "研").
pattern("研日研研日N", 1, "日").
pattern("研日研研日N", 2, "研").
pattern("研日研研日N", 3, "研").
pattern("研日研研日N", 4, "日").
pattern("研日研研日N", 5, "N").
forbidden_pattern("研日研研NP").
pattern("研日研研NP", 6).
pattern("研日研研NP", 0, "研").
pattern("研日研研NP", 1, "日").
pattern("研日研研NP", 2, "研").
pattern("研日研研NP", 3, "研").
pattern("研日研研NP", 4, "N").
pattern("研日研研NP", 5, "P").
forbidden_pattern("研日研研研N").
pattern("研日研研研N", 6).
pattern("研日研研研N", 0, "研").
pattern("研日研研研N", 1, "日").
pattern("研日研研研N", 2, "研").
pattern("研日研研研N", 3, "研").
pattern("研日研研研N", 4, "研").
pattern("研日研研研N", 5, "N").
forbidden_pattern("研NPPPP").
pattern("研NPPPP", 6).
pattern("研NPPPP", 0, "研").
pattern("研NPPPP", 1, "N").
pattern("研NPPPP", 2, "P").
pattern("研NPPPP", 3, "P").
pattern("研NPPPP", 4, "P").
pattern("研NPPPP", 5, "P").
forbidden_pattern("研研日日日N").
pattern("研研日日日N", 6).
pattern("研研日日日N", 0, "研").
pattern("研研日日日N", 1, "研").
pattern("研研日日日N", 2, "日").
pattern("研研日日日N", 3, "日").
pattern("研研日日日N", 4, "日").
pattern("研研日日日N", 5, "N").
forbidden_pattern("研研日日NP").
pattern("研研日日NP", 6).
pattern("研研日日NP", 0, "研").
pattern("研研日日NP", 1, "研").
pattern("研研日日NP", 2, "日").
pattern("研研日日NP", 3, "日").
pattern("研研日日NP", 4, "N").
pattern("研研日日NP", 5, "P").
forbidden_pattern("研研日日研N").
pattern("研研日日研N", 6).
pattern("研研日日研N", 0, "研").
pattern("研研日日研N", 1, "研").
pattern("研研日日研N", 2, "日").
pattern("研研日日研N", 3, "日").
pattern("研研日日研N", 4, "研").
pattern("研研日日研N", 5, "N").
forbidden_pattern("研研日NPP").
pattern("研研日NPP", 6).
pattern("研研日NPP", 0, "研").
pattern("研研日NPP", 1, "研").
pattern("研研日NPP", 2, "日").
pattern("研研日NPP", 3, "N").
pattern("研研日NPP", 4, "P").
pattern("研研日NPP", 5, "P").
forbidden_pattern("研研日研日N").
pattern("研研日研日N", 6).
pattern("研研日研日N", 0, "研").
pattern("研研日研日N", 1, "研").
pattern("研研日研日N", 2, "日").
pattern("研研日研日N", 3, "研").
pattern("研研日研日N", 4, "日").
pattern("研研日研日N", 5, "N").
forbidden_pattern("研研日研NP").
pattern("研研日研NP", 6).
pattern("研研日研NP", 0, "研").
pattern("研研日研NP", 1, "研").
pattern("研研日研NP", 2, "日").
pattern("研研日研NP", 3, "研").
pattern("研研日研NP", 4, "N").
pattern("研研日研NP", 5, "P").
forbidden_pattern("研研日研研N").
pattern("研研日研研N", 6).
pattern("研研日研研N", 0, "研").
pattern("研研日研研N", 1, "研").
pattern("研研日研研N", 2, "日").
pattern("研研日研研N", 3, "研").
pattern("研研日研研N", 4, "研").
pattern("研研日研研N", 5, "N").
forbidden_pattern("研研NPPP").
pattern("研研NPPP", 6).
pattern("研研NPPP", 0, "研").
pattern("研研NPPP", 1, "研").
pattern("研研NPPP", 2, "N").
pattern("研研NPPP", 3, "P").
pattern("研研NPPP", 4, "P").
pattern("研研NPPP", 5, "P").
forbidden_pattern("研研研日日N").
pattern("研研研日日N", 6).
pattern("研研研日日N", 0, "研").
pattern("研研研日日N", 1, "研").
pattern("研研研日日N", 2, "研").
pattern("研研研日日N", 3, "日").
pattern("研研研日日N", 4, "日").
pattern("研研研日日N", 5, "N").
forbidden_pattern("研研研日NP").
pattern("研研研日NP", 6).
pattern("研研研日NP", 0, "研").
pattern("研研研日NP", 1, "研").
pattern("研研研日NP", 2, "研").
pattern("研研研日NP", 3, "日").
pattern("研研研日NP", 4, "N").
pattern("研研研日NP", 5, "P").
forbidden_pattern("研研研日研N").
pattern("研研研日研N", 6).
pattern("研研研日研N", 0, "研").
pattern("研研研日研N", 1, "研").
pattern("研研研日研N", 2, "研").
pattern("研研研日研N", 3, "日").
pattern("研研研日研N", 4, "研").
pattern("研研研日研N", 5, "N").
forbidden_pattern("研研研NPP").
pattern("研研研NPP", 6).
pattern("研研研NPP", 0, "研").
pattern("研研研NPP", 1, "研").
pattern("研研研NPP", 2, "研").
pattern("研研研NPP", 3, "N").
pattern("研研研NPP", 4, "P").
pattern("研研研NPP", 5, "P").
forbidden_pattern("研研研研日N").
pattern("研研研研日N", 6).
pattern("研研研研日N", 0, "研").
pattern("研研研研日N", 1, "研").
pattern("研研研研日N", 2, "研").
pattern("研研研研日N", 3, "研").
pattern("研研研研日N", 4, "日").
pattern("研研研研日N", 5, "N").
forbidden_pattern("研研研研NP").
pattern("研研研研NP", 6).
pattern("研研研研NP", 0, "研").
pattern("研研研研NP", 1, "研").
pattern("研研研研NP", 2, "研").
pattern("研研研研NP", 3, "研").
pattern("研研研研NP", 4, "N").
pattern("研研研研NP", 5, "P").
forbidden_pattern("研研研研研N").
pattern("研研研研研N", 6).
pattern("研研研研研N", 0, "研").
pattern("研研研研研N", 1, "研").
pattern("研研研研研N", 2, "研").
pattern("研研研研研N", 3, "研").
pattern("研研研研研N", 4, "研").
pattern("研研研研研N", 5, "N").
% 前日・翌日可能シフト
pred_shift("J", "S").
succ_shift("S", "J").
succ_shift("S", "○").
succ_shift("S", "◎").
pred_shift("N", "J").
pred_shift("P", "J").
succ_shift("J", "S").
% パターン割当数
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
patterns_hard_lb("○○", 0).
patterns_soft_lb("○○", 1).
patterns_soft_ub("○○", 3).
patterns_hard_ub("○○", 4).
% Vertical constraints -------------------------------------
shifts("日").
shifts("日", "日").
staff_dweek_hard_lb("全員", "日", "平日", 5).
staff_dweek_hard_lb("全員", "日", "土日", 3).
staff_dweek_hard_lb("全員", "日", "祝日", 3).
staff_dweek_soft_lb("全員", "日", "平日", 6).
staff_dweek_soft_lb("全員", "日", "土日", 4).
staff_dweek_soft_lb("全員", "日", "祝日", 4).
staff_dweek_soft_ub("全員", "日", "平日", 10).
staff_dweek_soft_ub("全員", "日", "土日", 6).
staff_dweek_soft_ub("全員", "日", "祝日", 6).
staff_dweek_hard_ub("全員", "日", "平日", 11).
staff_dweek_hard_ub("全員", "日", "土日", 7).
staff_dweek_hard_ub("全員", "日", "祝日", 7).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 3).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 3).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 3).
staff_dweek_hard_lb("全員", "N", "土日", 3).
staff_dweek_hard_lb("全員", "J", "土日", 3).
staff_dweek_hard_lb("全員", "S", "土日", 3).
staff_dweek_hard_lb("全員", "N", "祝日", 3).
staff_dweek_hard_lb("全員", "J", "祝日", 3).
staff_dweek_hard_lb("全員", "S", "祝日", 3).
staff_dweek_soft_lb("全員", "N", "平日", 3).
staff_dweek_soft_lb("全員", "J", "平日", 3).
staff_dweek_soft_lb("全員", "S", "平日", 3).
staff_dweek_soft_lb("全員", "N", "土日", 3).
staff_dweek_soft_lb("全員", "J", "土日", 3).
staff_dweek_soft_lb("全員", "S", "土日", 3).
staff_dweek_soft_lb("全員", "N", "祝日", 3).
staff_dweek_soft_lb("全員", "J", "祝日", 3).
staff_dweek_soft_lb("全員", "S", "祝日", 3).
staff_dweek_soft_ub("全員", "N", "平日", 4).
staff_dweek_soft_ub("全員", "J", "平日", 4).
staff_dweek_soft_ub("全員", "S", "平日", 4).
staff_dweek_soft_ub("全員", "N", "土日", 4).
staff_dweek_soft_ub("全員", "J", "土日", 4).
staff_dweek_soft_ub("全員", "S", "土日", 4).
staff_dweek_soft_ub("全員", "N", "祝日", 4).
staff_dweek_soft_ub("全員", "J", "祝日", 4).
staff_dweek_soft_ub("全員", "S", "祝日", 4).
staff_dweek_hard_ub("全員", "N", "平日", 5).
staff_dweek_hard_ub("全員", "J", "平日", 5).
staff_dweek_hard_ub("全員", "S", "平日", 5).
staff_dweek_hard_ub("全員", "N", "土日", 5).
staff_dweek_hard_ub("全員", "J", "土日", 5).
staff_dweek_hard_ub("全員", "S", "土日", 5).
staff_dweek_hard_ub("全員", "N", "祝日", 5).
staff_dweek_hard_ub("全員", "J", "祝日", 5).
staff_dweek_hard_ub("全員", "S", "祝日", 5).
shifts("★").
shifts("★", "★").
staff_dweek_hard_lb("全員", "★", "平日", 0).
shifts("☆").
shifts("☆", "☆").
staff_dweek_hard_lb("全員", "☆", "平日", 0).
staff_dweek_hard_lb("全員", "★", "土日", 0).
staff_dweek_hard_lb("全員", "☆", "土日", 0).
staff_dweek_hard_lb("全員", "★", "祝日", 0).
staff_dweek_hard_lb("全員", "☆", "祝日", 0).
staff_dweek_soft_lb("全員", "★", "平日", 0).
staff_dweek_soft_lb("全員", "☆", "平日", 0).
staff_dweek_soft_lb("全員", "★", "土日", 0).
staff_dweek_soft_lb("全員", "☆", "土日", 0).
staff_dweek_soft_lb("全員", "★", "祝日", 0).
staff_dweek_soft_lb("全員", "☆", "祝日", 0).
staff_dweek_soft_ub("全員", "★", "平日", 0).
staff_dweek_soft_ub("全員", "☆", "平日", 0).
staff_dweek_soft_ub("全員", "★", "土日", 0).
staff_dweek_soft_ub("全員", "☆", "土日", 0).
staff_dweek_soft_ub("全員", "★", "祝日", 0).
staff_dweek_soft_ub("全員", "☆", "祝日", 0).
staff_dweek_hard_ub("全員", "★", "平日", 0).
staff_dweek_hard_ub("全員", "☆", "平日", 0).
staff_dweek_hard_ub("全員", "★", "土日", 0).
staff_dweek_hard_ub("全員", "☆", "土日", 0).
staff_dweek_hard_ub("全員", "★", "祝日", 0).
staff_dweek_hard_ub("全員", "☆", "祝日", 0).
shifts("P").
shifts("P", "P").
staff_dweek_hard_lb("全員", "P", "平日", 0).
staff_dweek_hard_lb("全員", "P", "土日", 0).
staff_dweek_hard_lb("全員", "P", "祝日", 0).
staff_dweek_soft_lb("全員", "P", "平日", 0).
staff_dweek_soft_lb("全員", "P", "土日", 0).
staff_dweek_soft_lb("全員", "P", "祝日", 0).
staff_dweek_soft_ub("全員", "P", "平日", 0).
staff_dweek_soft_ub("全員", "P", "土日", 0).
staff_dweek_soft_ub("全員", "P", "祝日", 0).
staff_dweek_hard_ub("全員", "P", "平日", 0).
staff_dweek_hard_ub("全員", "P", "土日", 0).
staff_dweek_hard_ub("全員", "P", "祝日", 0).
shifts("N+P+★").
shifts("N+P+★", "N").
shifts("N+P+★", "P").
shifts("N+P+★", "★").
staff_dweek_hard_lb("全員", "N+P+★", "平日", 3).
staff_dweek_hard_lb("全員", "N+P+★", "土日", 3).
staff_dweek_hard_lb("全員", "N+P+★", "祝日", 3).
staff_dweek_soft_lb("全員", "N+P+★", "平日", 3).
staff_dweek_soft_lb("全員", "N+P+★", "土日", 3).
staff_dweek_soft_lb("全員", "N+P+★", "祝日", 3).
staff_dweek_soft_ub("全員", "N+P+★", "平日", 4).
staff_dweek_soft_ub("全員", "N+P+★", "土日", 4).
staff_dweek_soft_ub("全員", "N+P+★", "祝日", 4).
staff_dweek_hard_ub("全員", "N+P+★", "平日", 5).
staff_dweek_hard_ub("全員", "N+P+★", "土日", 5).
staff_dweek_hard_ub("全員", "N+P+★", "祝日", 5).
staff_dweek_hard_lb("師長代行", "日", "平日", 1).
staff_dweek_hard_lb("師長代行", "日", "土日", 0).
staff_dweek_hard_lb("師長代行", "日", "祝日", 0).
staff_dweek_soft_lb("師長代行", "日", "平日", 2).
staff_dweek_soft_lb("師長代行", "日", "土日", 1).
staff_dweek_soft_lb("師長代行", "日", "祝日", 1).
staff_dweek_soft_ub("師長代行", "日", "平日", 6).
staff_dweek_soft_ub("師長代行", "日", "土日", 4).
staff_dweek_soft_ub("師長代行", "日", "祝日", 4).
staff_dweek_hard_ub("師長代行", "日", "平日", 7).
staff_dweek_hard_ub("師長代行", "日", "土日", 5).
staff_dweek_hard_ub("師長代行", "日", "祝日", 5).
staff_dweek_hard_lb("師長代行", "J", "平日", 0).
staff_dweek_hard_lb("師長代行", "S", "平日", 0).
staff_dweek_hard_lb("師長代行", "J", "土日", 0).
staff_dweek_hard_lb("師長代行", "S", "土日", 0).
staff_dweek_hard_lb("師長代行", "J", "祝日", 0).
staff_dweek_hard_lb("師長代行", "S", "祝日", 0).
staff_dweek_soft_lb("師長代行", "J", "平日", 0).
staff_dweek_soft_lb("師長代行", "S", "平日", 0).
staff_dweek_soft_lb("師長代行", "J", "土日", 0).
staff_dweek_soft_lb("師長代行", "S", "土日", 0).
staff_dweek_soft_lb("師長代行", "J", "祝日", 0).
staff_dweek_soft_lb("師長代行", "S", "祝日", 0).
staff_dweek_soft_ub("師長代行", "J", "平日", 2).
staff_dweek_soft_ub("師長代行", "S", "平日", 2).
staff_dweek_soft_ub("師長代行", "J", "土日", 2).
staff_dweek_soft_ub("師長代行", "S", "土日", 2).
staff_dweek_soft_ub("師長代行", "J", "祝日", 2).
staff_dweek_soft_ub("師長代行", "S", "祝日", 2).
staff_dweek_hard_ub("師長代行", "J", "平日", 4).
staff_dweek_hard_ub("師長代行", "S", "平日", 4).
staff_dweek_hard_ub("師長代行", "J", "土日", 3).
staff_dweek_hard_ub("師長代行", "S", "土日", 3).
staff_dweek_hard_ub("師長代行", "J", "祝日", 3).
staff_dweek_hard_ub("師長代行", "S", "祝日", 3).
staff_dweek_hard_lb("A", "J", "平日", 1).
staff_dweek_hard_lb("A", "S", "平日", 1).
staff_dweek_hard_lb("A", "J", "土日", 1).
staff_dweek_hard_lb("A", "S", "土日", 1).
staff_dweek_hard_lb("A", "J", "祝日", 1).
staff_dweek_hard_lb("A", "S", "祝日", 1).
staff_dweek_soft_lb("A", "J", "平日", 1).
staff_dweek_soft_lb("A", "S", "平日", 1).
staff_dweek_soft_lb("A", "J", "土日", 1).
staff_dweek_soft_lb("A", "S", "土日", 1).
staff_dweek_soft_lb("A", "J", "祝日", 1).
staff_dweek_soft_lb("A", "S", "祝日", 1).
staff_dweek_soft_ub("A", "J", "平日", 3).
staff_dweek_soft_ub("A", "S", "平日", 3).
staff_dweek_soft_ub("A", "J", "土日", 2).
staff_dweek_soft_ub("A", "S", "土日", 2).
staff_dweek_soft_ub("A", "J", "祝日", 2).
staff_dweek_soft_ub("A", "S", "祝日", 2).
staff_dweek_hard_ub("A", "J", "平日", 3).
staff_dweek_hard_ub("A", "S", "平日", 3).
staff_dweek_hard_ub("A", "J", "土日", 3).
staff_dweek_hard_ub("A", "S", "土日", 3).
staff_dweek_hard_ub("A", "J", "祝日", 3).
staff_dweek_hard_ub("A", "S", "祝日", 3).
staff_dweek_hard_lb("B", "J", "平日", 1).
staff_dweek_hard_lb("B", "S", "平日", 1).
staff_dweek_hard_lb("B", "J", "土日", 1).
staff_dweek_hard_lb("B", "S", "土日", 1).
staff_dweek_hard_lb("B", "J", "祝日", 1).
staff_dweek_hard_lb("B", "S", "祝日", 1).
staff_dweek_soft_lb("B", "J", "平日", 1).
staff_dweek_soft_lb("B", "S", "平日", 1).
staff_dweek_soft_lb("B", "J", "土日", 1).
staff_dweek_soft_lb("B", "S", "土日", 1).
staff_dweek_soft_lb("B", "J", "祝日", 1).
staff_dweek_soft_lb("B", "S", "祝日", 1).
staff_dweek_soft_ub("B", "J", "平日", 3).
staff_dweek_soft_ub("B", "S", "平日", 3).
staff_dweek_soft_ub("B", "J", "土日", 3).
staff_dweek_soft_ub("B", "S", "土日", 3).
staff_dweek_soft_ub("B", "J", "祝日", 3).
staff_dweek_soft_ub("B", "S", "祝日", 3).
staff_dweek_hard_ub("B", "J", "平日", 3).
staff_dweek_hard_ub("B", "S", "平日", 3).
staff_dweek_hard_ub("B", "J", "土日", 3).
staff_dweek_hard_ub("B", "S", "土日", 3).
staff_dweek_hard_ub("B", "J", "祝日", 3).
staff_dweek_hard_ub("B", "S", "祝日", 3).
shifts("J+★").
shifts("J+★", "J").
shifts("J+★", "★").
staff_dweek_hard_lb("夜勤", "J+★", "平日", 3).
shifts("S+☆").
shifts("S+☆", "S").
shifts("S+☆", "☆").
staff_dweek_hard_lb("夜勤", "S+☆", "平日", 3).
staff_dweek_hard_lb("夜勤", "J+★", "土日", 3).
staff_dweek_hard_lb("夜勤", "S+☆", "土日", 3).
staff_dweek_hard_lb("夜勤", "J+★", "祝日", 3).
staff_dweek_hard_lb("夜勤", "S+☆", "祝日", 3).
staff_dweek_soft_lb("夜勤", "J+★", "平日", 3).
staff_dweek_soft_lb("夜勤", "S+☆", "平日", 3).
staff_dweek_soft_lb("夜勤", "J+★", "土日", 3).
staff_dweek_soft_lb("夜勤", "S+☆", "土日", 3).
staff_dweek_soft_lb("夜勤", "J+★", "祝日", 3).
staff_dweek_soft_lb("夜勤", "S+☆", "祝日", 3).
staff_dweek_soft_ub("夜勤", "J+★", "平日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "平日", 4).
staff_dweek_soft_ub("夜勤", "J+★", "土日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "土日", 4).
staff_dweek_soft_ub("夜勤", "J+★", "祝日", 4).
staff_dweek_soft_ub("夜勤", "S+☆", "祝日", 4).
staff_dweek_hard_ub("夜勤", "J+★", "平日", 5).
staff_dweek_hard_ub("夜勤", "S+☆", "平日", 5).
staff_dweek_hard_ub("夜勤", "J+★", "土日", 5).
staff_dweek_hard_ub("夜勤", "S+☆", "土日", 5).
staff_dweek_hard_ub("夜勤", "J+★", "祝日", 5).
staff_dweek_hard_ub("夜勤", "S+☆", "祝日", 5).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("全員","★","人数").
vertical_constraint_type("全員","☆","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("師長代行","日","人数").
vertical_constraint_type("師長代行","J","人数").
vertical_constraint_type("師長代行","S","人数").
vertical_constraint_type("A","J","人数").
vertical_constraint_type("A","S","人数").
vertical_constraint_type("B","J","人数").
vertical_constraint_type("B","S","人数").
vertical_constraint_type("夜勤","J+★","人数").
vertical_constraint_type("夜勤","S+☆","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_hard_lb(1, "日", 14).
shifts_soft_lb(1, "日", 14).
shifts_soft_ub(1, "日", 16).
shifts_hard_ub(1, "日", 17).
shifts_hard_lb(1, "N", 1).
shifts_hard_lb(1, "J", 1).
shifts_hard_lb(1, "S", 1).
shifts_soft_lb(1, "N", 1).
shifts_soft_lb(1, "J", 1).
shifts_soft_lb(1, "S", 1).
shifts_soft_ub(1, "N", 1).
shifts_soft_ub(1, "J", 1).
shifts_soft_ub(1, "S", 1).
shifts_hard_ub(1, "N", 1).
shifts_hard_ub(1, "J", 1).
shifts_hard_ub(1, "S", 1).
shifts_hard_lb(1, "★", 0).
shifts_hard_lb(1, "☆", 0).
shifts_hard_lb(1, "P", 0).
shifts_soft_lb(1, "★", 0).
shifts_soft_lb(1, "☆", 0).
shifts_soft_lb(1, "P", 0).
shifts_soft_ub(1, "★", 0).
shifts_soft_ub(1, "☆", 0).
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_hard_ub(1, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_lb(1, "A", 0).
shifts_soft_lb(1, "A", 0).
shifts_soft_ub(1, "A", 0).
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_hard_lb(2, "日", 6).
shifts_soft_lb(2, "日", 7).
shifts_soft_ub(2, "日", 16).
shifts_hard_ub(2, "日", 19).
shifts_hard_lb(2, "N", 0).
shifts_hard_lb(2, "J", 0).
shifts_hard_lb(2, "S", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "J", 0).
shifts_soft_lb(2, "S", 0).
shifts_soft_ub(2, "N", 4).
shifts_soft_ub(2, "J", 4).
shifts_soft_ub(2, "S", 4).
shifts_hard_ub(2, "N", 4).
shifts_hard_ub(2, "J", 4).
shifts_hard_ub(2, "S", 4).
shifts_hard_lb(2, "★", 0).
shifts_hard_lb(2, "☆", 0).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "★", 0).
shifts_soft_ub(2, "☆", 0).
shifts_soft_ub(2, "P", 0).
shifts_hard_ub(2, "★", 0).
shifts_hard_ub(2, "☆", 0).
shifts_hard_ub(2, "P", 0).
shifts_hard_lb(2, "A", 0).
shifts_soft_lb(2, "A", 0).
shifts_soft_ub(2, "A", 0).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_hard_lb(3, "日", 3).
shifts_soft_lb(3, "日", 3).
shifts_soft_ub(3, "日", 8).
shifts_hard_ub(3, "日", 9).
shifts_hard_lb(3, "N", 3).
shifts_hard_lb(3, "J", 3).
shifts_hard_lb(3, "S", 3).
shifts_soft_lb(3, "N", 3).
shifts_soft_lb(3, "J", 3).
shifts_soft_lb(3, "S", 3).
shifts_soft_ub(3, "N", 5).
shifts_soft_ub(3, "J", 5).
shifts_soft_ub(3, "S", 5).
shifts_hard_ub(3, "N", 5).
shifts_hard_ub(3, "J", 5).
shifts_hard_ub(3, "S", 5).
shifts_hard_lb(3, "★", 0).
shifts_hard_lb(3, "☆", 0).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "★", 0).
shifts_soft_lb(3, "☆", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "★", 0).
shifts_soft_ub(3, "☆", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "★", 1).
shifts_hard_ub(3, "☆", 1).
shifts_hard_ub(3, "P", 1).
shifts_hard_lb(3, "A", 0).
shifts_soft_lb(3, "A", 0).
shifts_soft_ub(3, "A", 0).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_hard_lb(4, "日", 3).
shifts_soft_lb(4, "日", 3).
shifts_soft_ub(4, "日", 9).
shifts_hard_ub(4, "日", 9).
shifts_hard_lb(4, "N", 2).
shifts_hard_lb(4, "J", 2).
shifts_hard_lb(4, "S", 2).
shifts_soft_lb(4, "N", 3).
shifts_soft_lb(4, "J", 3).
shifts_soft_lb(4, "S", 3).
shifts_soft_ub(4, "N", 5).
shifts_soft_ub(4, "J", 5).
shifts_soft_ub(4, "S", 5).
shifts_hard_ub(4, "N", 5).
shifts_hard_ub(4, "J", 5).
shifts_hard_ub(4, "S", 5).
shifts_hard_lb(4, "★", 0).
shifts_hard_lb(4, "☆", 0).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "★", 0).
shifts_soft_lb(4, "☆", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "★", 0).
shifts_soft_ub(4, "☆", 0).
shifts_soft_ub(4, "P", 0).
shifts_hard_ub(4, "★", 1).
shifts_hard_ub(4, "☆", 1).
shifts_hard_ub(4, "P", 1).
shifts_hard_lb(4, "A", 0).
shifts_soft_lb(4, "A", 0).
shifts_soft_ub(4, "A", 0).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_hard_lb(5, "日", 3).
shifts_soft_lb(5, "日", 4).
shifts_soft_ub(5, "日", 6).
shifts_hard_ub(5, "日", 9).
shifts_hard_lb(5, "N", 3).
shifts_hard_lb(5, "J", 3).
shifts_hard_lb(5, "S", 3).
shifts_soft_lb(5, "N", 3).
shifts_soft_lb(5, "J", 3).
shifts_soft_lb(5, "S", 3).
shifts_soft_ub(5, "N", 5).
shifts_soft_ub(5, "J", 5).
shifts_soft_ub(5, "S", 5).
shifts_hard_ub(5, "N", 6).
shifts_hard_ub(5, "J", 6).
shifts_hard_ub(5, "S", 6).
shifts_hard_lb(5, "★", 0).
shifts_hard_lb(5, "☆", 0).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "★", 0).
shifts_soft_ub(5, "☆", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_hard_ub(5, "P", 0).
shifts_hard_lb(5, "A", 0).
shifts_soft_lb(5, "A", 0).
shifts_soft_ub(5, "A", 0).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_hard_lb(6, "日", 4).
shifts_soft_lb(6, "日", 4).
shifts_soft_ub(6, "日", 11).
shifts_hard_ub(6, "日", 12).
shifts_hard_lb(6, "N", 2).
shifts_hard_lb(6, "J", 2).
shifts_hard_lb(6, "S", 2).
shifts_soft_lb(6, "N", 3).
shifts_soft_lb(6, "J", 3).
shifts_soft_lb(6, "S", 3).
shifts_soft_ub(6, "N", 5).
shifts_soft_ub(6, "J", 5).
shifts_soft_ub(6, "S", 5).
shifts_hard_ub(6, "N", 5).
shifts_hard_ub(6, "J", 5).
shifts_hard_ub(6, "S", 5).
shifts_hard_lb(6, "★", 0).
shifts_hard_lb(6, "☆", 0).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "★", 0).
shifts_soft_ub(6, "☆", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "★", 0).
shifts_hard_ub(6, "☆", 0).
shifts_hard_ub(6, "P", 0).
shifts_hard_lb(6, "A", 0).
shifts_soft_lb(6, "A", 0).
shifts_soft_ub(6, "A", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_hard_lb(7, "日", 1).
shifts_soft_lb(7, "日", 1).
shifts_soft_ub(7, "日", 7).
shifts_hard_ub(7, "日", 8).
shifts_hard_lb(7, "N", 3).
shifts_hard_lb(7, "J", 3).
shifts_hard_lb(7, "S", 3).
shifts_soft_lb(7, "N", 4).
shifts_soft_lb(7, "J", 4).
shifts_soft_lb(7, "S", 4).
shifts_soft_ub(7, "N", 5).
shifts_soft_ub(7, "J", 5).
shifts_soft_ub(7, "S", 5).
shifts_hard_ub(7, "N", 6).
shifts_hard_ub(7, "J", 6).
shifts_hard_ub(7, "S", 6).
shifts_hard_lb(7, "★", 0).
shifts_hard_lb(7, "☆", 0).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "★", 0).
shifts_soft_lb(7, "☆", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "★", 0).
shifts_soft_ub(7, "☆", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_hard_ub(7, "P", 0).
shifts_hard_lb(7, "A", 0).
shifts_soft_lb(7, "A", 0).
shifts_soft_ub(7, "A", 0).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_hard_lb(8, "★", 0).
shifts_hard_lb(8, "☆", 0).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "★", 0).
shifts_soft_lb(8, "☆", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "★", 0).
shifts_soft_ub(8, "☆", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "★", 0).
shifts_hard_ub(8, "☆", 0).
shifts_hard_ub(8, "P", 0).
shifts_hard_lb(8, "A", 0).
shifts_soft_lb(8, "A", 0).
shifts_soft_ub(8, "A", 0).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_hard_lb(9, "日", 2).
shifts_soft_lb(9, "日", 3).
shifts_soft_ub(9, "日", 8).
shifts_hard_ub(9, "日", 8).
shifts_hard_lb(9, "N", 3).
shifts_hard_lb(9, "J", 3).
shifts_hard_lb(9, "S", 3).
shifts_soft_lb(9, "N", 3).
shifts_soft_lb(9, "J", 3).
shifts_soft_lb(9, "S", 3).
shifts_soft_ub(9, "N", 5).
shifts_soft_ub(9, "J", 5).
shifts_soft_ub(9, "S", 5).
shifts_hard_ub(9, "N", 5).
shifts_hard_ub(9, "J", 5).
shifts_hard_ub(9, "S", 5).
shifts_hard_lb(9, "★", 0).
shifts_hard_lb(9, "☆", 0).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "★", 0).
shifts_soft_lb(9, "☆", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "★", 0).
shifts_soft_ub(9, "☆", 0).
shifts_soft_ub(9, "P", 0).
shifts_hard_ub(9, "★", 0).
shifts_hard_ub(9, "☆", 0).
shifts_hard_ub(9, "P", 0).
shifts_hard_lb(9, "A", 0).
shifts_soft_lb(9, "A", 0).
shifts_soft_ub(9, "A", 0).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_hard_lb(10, "日", 2).
shifts_soft_lb(10, "日", 4).
shifts_soft_ub(10, "日", 8).
shifts_hard_ub(10, "日", 10).
shifts_hard_lb(10, "N", 0).
shifts_hard_lb(10, "J", 0).
shifts_hard_lb(10, "S", 0).
shifts_soft_lb(10, "N", 3).
shifts_soft_lb(10, "J", 3).
shifts_soft_lb(10, "S", 3).
shifts_soft_ub(10, "N", 5).
shifts_soft_ub(10, "J", 5).
shifts_soft_ub(10, "S", 5).
shifts_hard_ub(10, "N", 6).
shifts_hard_ub(10, "J", 6).
shifts_hard_ub(10, "S", 6).
shifts_hard_lb(10, "★", 0).
shifts_hard_lb(10, "☆", 0).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "★", 0).
shifts_soft_ub(10, "☆", 0).
shifts_soft_ub(10, "P", 0).
shifts_hard_ub(10, "★", 0).
shifts_hard_ub(10, "☆", 0).
shifts_hard_ub(10, "P", 0).
shifts_hard_lb(10, "A", 0).
shifts_soft_lb(10, "A", 0).
shifts_soft_ub(10, "A", 0).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_hard_lb(11, "日", 3).
shifts_soft_lb(11, "日", 3).
shifts_soft_ub(11, "日", 8).
shifts_hard_ub(11, "日", 9).
shifts_hard_lb(11, "N", 3).
shifts_hard_lb(11, "J", 3).
shifts_hard_lb(11, "S", 3).
shifts_soft_lb(11, "N", 3).
shifts_soft_lb(11, "J", 3).
shifts_soft_lb(11, "S", 3).
shifts_soft_ub(11, "N", 5).
shifts_soft_ub(11, "J", 5).
shifts_soft_ub(11, "S", 5).
shifts_hard_ub(11, "N", 5).
shifts_hard_ub(11, "J", 5).
shifts_hard_ub(11, "S", 5).
shifts_hard_lb(11, "★", 0).
shifts_hard_lb(11, "☆", 0).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "★", 0).
shifts_soft_lb(11, "☆", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "★", 0).
shifts_soft_ub(11, "☆", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_hard_ub(11, "P", 0).
shifts_hard_lb(11, "A", 0).
shifts_soft_lb(11, "A", 0).
shifts_soft_ub(11, "A", 0).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_hard_lb(12, "★", 0).
shifts_hard_lb(12, "☆", 0).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "★", 0).
shifts_soft_ub(12, "☆", 0).
shifts_soft_ub(12, "P", 0).
shifts_hard_ub(12, "★", 0).
shifts_hard_ub(12, "☆", 0).
shifts_hard_ub(12, "P", 0).
shifts_hard_lb(12, "A", 0).
shifts_soft_lb(12, "A", 0).
shifts_soft_ub(12, "A", 0).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_hard_lb(13, "日", 2).
shifts_soft_lb(13, "日", 3).
shifts_soft_ub(13, "日", 7).
shifts_hard_ub(13, "日", 8).
shifts_hard_lb(13, "N", 4).
shifts_hard_lb(13, "J", 4).
shifts_hard_lb(13, "S", 4).
shifts_soft_lb(13, "N", 4).
shifts_soft_lb(13, "J", 4).
shifts_soft_lb(13, "S", 4).
shifts_soft_ub(13, "N", 6).
shifts_soft_ub(13, "J", 6).
shifts_soft_ub(13, "S", 6).
shifts_hard_ub(13, "N", 6).
shifts_hard_ub(13, "J", 6).
shifts_hard_ub(13, "S", 6).
shifts_hard_lb(13, "★", 0).
shifts_hard_lb(13, "☆", 0).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "★", 0).
shifts_soft_ub(13, "☆", 0).
shifts_soft_ub(13, "P", 0).
shifts_hard_ub(13, "★", 0).
shifts_hard_ub(13, "☆", 0).
shifts_hard_ub(13, "P", 0).
shifts_hard_lb(13, "A", 0).
shifts_soft_lb(13, "A", 0).
shifts_soft_ub(13, "A", 0).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_hard_lb(14, "日", 0).
shifts_soft_lb(14, "日", 2).
shifts_soft_ub(14, "日", 9).
shifts_hard_ub(14, "日", 12).
shifts_hard_lb(14, "N", 2).
shifts_hard_lb(14, "J", 2).
shifts_hard_lb(14, "S", 2).
shifts_soft_lb(14, "N", 2).
shifts_soft_lb(14, "J", 2).
shifts_soft_lb(14, "S", 2).
shifts_soft_ub(14, "N", 6).
shifts_soft_ub(14, "J", 6).
shifts_soft_ub(14, "S", 6).
shifts_hard_ub(14, "N", 6).
shifts_hard_ub(14, "J", 6).
shifts_hard_ub(14, "S", 6).
shifts_hard_lb(14, "★", 0).
shifts_hard_lb(14, "☆", 0).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "★", 0).
shifts_soft_ub(14, "☆", 0).
shifts_soft_ub(14, "P", 0).
shifts_hard_ub(14, "★", 0).
shifts_hard_ub(14, "☆", 0).
shifts_hard_ub(14, "P", 0).
shifts_hard_lb(14, "A", 0).
shifts_soft_lb(14, "A", 0).
shifts_soft_ub(14, "A", 0).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_hard_lb(15, "日", 5).
shifts_soft_lb(15, "日", 6).
shifts_soft_ub(15, "日", 17).
shifts_hard_ub(15, "日", 19).
shifts_hard_lb(15, "N", 0).
shifts_hard_lb(15, "J", 0).
shifts_hard_lb(15, "S", 0).
shifts_soft_lb(15, "N", 0).
shifts_soft_lb(15, "J", 0).
shifts_soft_lb(15, "S", 0).
shifts_soft_ub(15, "N", 3).
shifts_soft_ub(15, "J", 3).
shifts_soft_ub(15, "S", 3).
shifts_hard_ub(15, "N", 4).
shifts_hard_ub(15, "J", 4).
shifts_hard_ub(15, "S", 4).
shifts_hard_lb(15, "★", 0).
shifts_hard_lb(15, "☆", 0).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "★", 0).
shifts_soft_ub(15, "☆", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_hard_ub(15, "P", 0).
shifts_hard_lb(15, "A", 0).
shifts_soft_lb(15, "A", 0).
shifts_soft_ub(15, "A", 0).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_hard_lb(16, "日", 10).
shifts_soft_lb(16, "日", 11).
shifts_soft_ub(16, "日", 17).
shifts_hard_ub(16, "日", 19).
shifts_hard_lb(16, "N", 0).
shifts_hard_lb(16, "J", 0).
shifts_hard_lb(16, "S", 0).
shifts_soft_lb(16, "N", 0).
shifts_soft_lb(16, "J", 0).
shifts_soft_lb(16, "S", 0).
shifts_soft_ub(16, "N", 1).
shifts_soft_ub(16, "J", 1).
shifts_soft_ub(16, "S", 1).
shifts_hard_ub(16, "N", 1).
shifts_hard_ub(16, "J", 1).
shifts_hard_ub(16, "S", 1).
shifts_hard_lb(16, "★", 0).
shifts_hard_lb(16, "☆", 0).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "★", 0).
shifts_soft_lb(16, "☆", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "★", 0).
shifts_soft_ub(16, "☆", 0).
shifts_soft_ub(16, "P", 0).
shifts_hard_ub(16, "★", 0).
shifts_hard_ub(16, "☆", 0).
shifts_hard_ub(16, "P", 0).
shifts_hard_lb(16, "A", 0).
shifts_soft_lb(16, "A", 0).
shifts_soft_ub(16, "A", 0).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_hard_lb(17, "日", 7).
shifts_soft_lb(17, "日", 7).
shifts_soft_ub(17, "日", 14).
shifts_hard_ub(17, "日", 14).
shifts_hard_lb(17, "N", 1).
shifts_hard_lb(17, "J", 1).
shifts_hard_lb(17, "S", 1).
shifts_soft_lb(17, "N", 2).
shifts_soft_lb(17, "J", 2).
shifts_soft_lb(17, "S", 2).
shifts_soft_ub(17, "N", 3).
shifts_soft_ub(17, "J", 3).
shifts_soft_ub(17, "S", 3).
shifts_hard_ub(17, "N", 4).
shifts_hard_ub(17, "J", 4).
shifts_hard_ub(17, "S", 4).
shifts_hard_lb(17, "★", 0).
shifts_hard_lb(17, "☆", 0).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "★", 0).
shifts_soft_lb(17, "☆", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "★", 0).
shifts_soft_ub(17, "☆", 0).
shifts_soft_ub(17, "P", 0).
shifts_hard_ub(17, "★", 0).
shifts_hard_ub(17, "☆", 0).
shifts_hard_ub(17, "P", 0).
shifts_hard_lb(17, "A", 0).
shifts_soft_lb(17, "A", 0).
shifts_soft_ub(17, "A", 0).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_hard_lb(18, "日", 6).
shifts_soft_lb(18, "日", 7).
shifts_soft_ub(18, "日", 13).
shifts_hard_ub(18, "日", 18).
shifts_hard_lb(18, "N", 0).
shifts_hard_lb(18, "J", 0).
shifts_hard_lb(18, "S", 0).
shifts_soft_lb(18, "N", 2).
shifts_soft_lb(18, "J", 2).
shifts_soft_lb(18, "S", 2).
shifts_soft_ub(18, "N", 4).
shifts_soft_ub(18, "J", 4).
shifts_soft_ub(18, "S", 4).
shifts_hard_ub(18, "N", 4).
shifts_hard_ub(18, "J", 4).
shifts_hard_ub(18, "S", 4).
shifts_hard_lb(18, "★", 0).
shifts_hard_lb(18, "☆", 0).
shifts_hard_lb(18, "P", 0).
shifts_soft_lb(18, "★", 0).
shifts_soft_lb(18, "☆", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "★", 0).
shifts_soft_ub(18, "☆", 0).
shifts_soft_ub(18, "P", 0).
shifts_hard_ub(18, "★", 0).
shifts_hard_ub(18, "☆", 0).
shifts_hard_ub(18, "P", 0).
shifts_hard_lb(18, "A", 0).
shifts_soft_lb(18, "A", 0).
shifts_soft_ub(18, "A", 0).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_hard_lb(19, "日", 3).
shifts_soft_lb(19, "日", 4).
shifts_soft_ub(19, "日", 8).
shifts_hard_ub(19, "日", 10).
shifts_hard_lb(19, "N", 3).
shifts_hard_lb(19, "J", 3).
shifts_hard_lb(19, "S", 3).
shifts_soft_lb(19, "N", 3).
shifts_soft_lb(19, "J", 3).
shifts_soft_lb(19, "S", 3).
shifts_soft_ub(19, "N", 5).
shifts_soft_ub(19, "J", 5).
shifts_soft_ub(19, "S", 5).
shifts_hard_ub(19, "N", 5).
shifts_hard_ub(19, "J", 5).
shifts_hard_ub(19, "S", 5).
shifts_hard_lb(19, "★", 0).
shifts_hard_lb(19, "☆", 0).
shifts_hard_lb(19, "P", 0).
shifts_soft_lb(19, "★", 0).
shifts_soft_lb(19, "☆", 0).
shifts_soft_lb(19, "P", 0).
shifts_soft_ub(19, "★", 0).
shifts_soft_ub(19, "☆", 0).
shifts_soft_ub(19, "P", 0).
shifts_hard_ub(19, "★", 1).
shifts_hard_ub(19, "☆", 1).
shifts_hard_ub(19, "P", 1).
shifts_hard_lb(19, "A", 0).
shifts_soft_lb(19, "A", 0).
shifts_soft_ub(19, "A", 0).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_hard_lb(20, "日", 3).
shifts_soft_lb(20, "日", 3).
shifts_soft_ub(20, "日", 9).
shifts_hard_ub(20, "日", 13).
shifts_hard_lb(20, "N", 2).
shifts_hard_lb(20, "J", 2).
shifts_hard_lb(20, "S", 2).
shifts_soft_lb(20, "N", 2).
shifts_soft_lb(20, "J", 2).
shifts_soft_lb(20, "S", 2).
shifts_soft_ub(20, "N", 5).
shifts_soft_ub(20, "J", 5).
shifts_soft_ub(20, "S", 5).
shifts_hard_ub(20, "N", 6).
shifts_hard_ub(20, "J", 6).
shifts_hard_ub(20, "S", 6).
shifts_hard_lb(20, "★", 0).
shifts_hard_lb(20, "☆", 0).
shifts_hard_lb(20, "P", 0).
shifts_soft_lb(20, "★", 0).
shifts_soft_lb(20, "☆", 0).
shifts_soft_lb(20, "P", 0).
shifts_soft_ub(20, "★", 0).
shifts_soft_ub(20, "☆", 0).
shifts_soft_ub(20, "P", 0).
shifts_hard_ub(20, "★", 1).
shifts_hard_ub(20, "☆", 1).
shifts_hard_ub(20, "P", 1).
shifts_hard_lb(20, "A", 0).
shifts_soft_lb(20, "A", 0).
shifts_soft_ub(20, "A", 0).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_hard_lb(21, "日", 2).
shifts_soft_lb(21, "日", 3).
shifts_soft_ub(21, "日", 11).
shifts_hard_ub(21, "日", 13).
shifts_hard_lb(21, "N", 2).
shifts_hard_lb(21, "J", 2).
shifts_hard_lb(21, "S", 2).
shifts_soft_lb(21, "N", 3).
shifts_soft_lb(21, "J", 3).
shifts_soft_lb(21, "S", 3).
shifts_soft_ub(21, "N", 5).
shifts_soft_ub(21, "J", 5).
shifts_soft_ub(21, "S", 5).
shifts_hard_ub(21, "N", 6).
shifts_hard_ub(21, "J", 6).
shifts_hard_ub(21, "S", 6).
shifts_hard_lb(21, "★", 0).
shifts_hard_lb(21, "☆", 0).
shifts_hard_lb(21, "P", 0).
shifts_soft_lb(21, "★", 0).
shifts_soft_lb(21, "☆", 0).
shifts_soft_lb(21, "P", 0).
shifts_soft_ub(21, "★", 0).
shifts_soft_ub(21, "☆", 0).
shifts_soft_ub(21, "P", 0).
shifts_hard_ub(21, "★", 0).
shifts_hard_ub(21, "☆", 0).
shifts_hard_ub(21, "P", 0).
shifts_hard_lb(21, "A", 0).
shifts_soft_lb(21, "A", 0).
shifts_soft_ub(21, "A", 0).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_hard_lb(22, "日", 4).
shifts_soft_lb(22, "日", 4).
shifts_soft_ub(22, "日", 7).
shifts_hard_ub(22, "日", 7).
shifts_hard_lb(22, "N", 4).
shifts_hard_lb(22, "J", 4).
shifts_hard_lb(22, "S", 4).
shifts_soft_lb(22, "N", 4).
shifts_soft_lb(22, "J", 4).
shifts_soft_lb(22, "S", 4).
shifts_soft_ub(22, "N", 5).
shifts_soft_ub(22, "J", 5).
shifts_soft_ub(22, "S", 5).
shifts_hard_ub(22, "N", 5).
shifts_hard_ub(22, "J", 5).
shifts_hard_ub(22, "S", 5).
shifts_hard_lb(22, "★", 0).
shifts_hard_lb(22, "☆", 0).
shifts_hard_lb(22, "P", 0).
shifts_soft_lb(22, "★", 0).
shifts_soft_lb(22, "☆", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "★", 0).
shifts_soft_ub(22, "☆", 0).
shifts_soft_ub(22, "P", 0).
shifts_hard_ub(22, "★", 0).
shifts_hard_ub(22, "☆", 0).
shifts_hard_ub(22, "P", 0).
shifts_hard_lb(22, "A", 0).
shifts_soft_lb(22, "A", 0).
shifts_soft_ub(22, "A", 0).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_hard_lb(23, "日", 2).
shifts_soft_lb(23, "日", 3).
shifts_soft_ub(23, "日", 6).
shifts_hard_ub(23, "日", 7).
shifts_hard_lb(23, "N", 4).
shifts_hard_lb(23, "J", 4).
shifts_hard_lb(23, "S", 4).
shifts_soft_lb(23, "N", 4).
shifts_soft_lb(23, "J", 4).
shifts_soft_lb(23, "S", 4).
shifts_soft_ub(23, "N", 6).
shifts_soft_ub(23, "J", 6).
shifts_soft_ub(23, "S", 6).
shifts_hard_ub(23, "N", 6).
shifts_hard_ub(23, "J", 6).
shifts_hard_ub(23, "S", 6).
shifts_hard_lb(23, "★", 0).
shifts_hard_lb(23, "☆", 0).
shifts_hard_lb(23, "P", 0).
shifts_soft_lb(23, "★", 0).
shifts_soft_lb(23, "☆", 0).
shifts_soft_lb(23, "P", 0).
shifts_soft_ub(23, "★", 0).
shifts_soft_ub(23, "☆", 0).
shifts_soft_ub(23, "P", 0).
shifts_hard_ub(23, "★", 0).
shifts_hard_ub(23, "☆", 0).
shifts_hard_ub(23, "P", 0).
shifts_hard_lb(23, "A", 0).
shifts_soft_lb(23, "A", 0).
shifts_soft_ub(23, "A", 0).
shifts_hard_ub(23, "A", 0).
% Staff 24:
shifts_hard_lb(24, "日", 4).
shifts_soft_lb(24, "日", 4).
shifts_soft_ub(24, "日", 8).
shifts_hard_ub(24, "日", 8).
shifts_hard_lb(24, "N", 3).
shifts_hard_lb(24, "J", 3).
shifts_hard_lb(24, "S", 3).
shifts_soft_lb(24, "N", 3).
shifts_soft_lb(24, "J", 3).
shifts_soft_lb(24, "S", 3).
shifts_soft_ub(24, "N", 5).
shifts_soft_ub(24, "J", 5).
shifts_soft_ub(24, "S", 5).
shifts_hard_ub(24, "N", 5).
shifts_hard_ub(24, "J", 5).
shifts_hard_ub(24, "S", 5).
shifts_hard_lb(24, "★", 0).
shifts_hard_lb(24, "☆", 0).
shifts_hard_lb(24, "P", 0).
shifts_soft_lb(24, "★", 0).
shifts_soft_lb(24, "☆", 0).
shifts_soft_lb(24, "P", 0).
shifts_soft_ub(24, "★", 0).
shifts_soft_ub(24, "☆", 0).
shifts_soft_ub(24, "P", 0).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_hard_ub(24, "P", 0).
shifts_hard_lb(24, "A", 0).
shifts_soft_lb(24, "A", 0).
shifts_soft_ub(24, "A", 0).
shifts_hard_ub(24, "A", 0).
% Staff 25:
shifts_hard_lb(25, "日", 2).
shifts_soft_lb(25, "日", 2).
shifts_soft_ub(25, "日", 8).
shifts_hard_ub(25, "日", 12).
shifts_hard_lb(25, "N", 2).
shifts_hard_lb(25, "J", 2).
shifts_hard_lb(25, "S", 2).
shifts_soft_lb(25, "N", 3).
shifts_soft_lb(25, "J", 3).
shifts_soft_lb(25, "S", 3).
shifts_soft_ub(25, "N", 6).
shifts_soft_ub(25, "J", 6).
shifts_soft_ub(25, "S", 6).
shifts_hard_ub(25, "N", 6).
shifts_hard_ub(25, "J", 6).
shifts_hard_ub(25, "S", 6).
shifts_hard_lb(25, "★", 0).
shifts_hard_lb(25, "☆", 0).
shifts_hard_lb(25, "P", 0).
shifts_soft_lb(25, "★", 0).
shifts_soft_lb(25, "☆", 0).
shifts_soft_lb(25, "P", 0).
shifts_soft_ub(25, "★", 0).
shifts_soft_ub(25, "☆", 0).
shifts_soft_ub(25, "P", 0).
shifts_hard_ub(25, "★", 0).
shifts_hard_ub(25, "☆", 0).
shifts_hard_ub(25, "P", 0).
shifts_hard_lb(25, "A", 0).
shifts_soft_lb(25, "A", 0).
shifts_soft_ub(25, "A", 0).
shifts_hard_ub(25, "A", 0).
% Staff 26:
shifts_hard_lb(26, "日", 2).
shifts_soft_lb(26, "日", 3).
shifts_soft_ub(26, "日", 8).
shifts_hard_ub(26, "日", 8).
shifts_hard_lb(26, "N", 3).
shifts_hard_lb(26, "J", 3).
shifts_hard_lb(26, "S", 3).
shifts_soft_lb(26, "N", 3).
shifts_soft_lb(26, "J", 3).
shifts_soft_lb(26, "S", 3).
shifts_soft_ub(26, "N", 5).
shifts_soft_ub(26, "J", 5).
shifts_soft_ub(26, "S", 5).
shifts_hard_ub(26, "N", 6).
shifts_hard_ub(26, "J", 6).
shifts_hard_ub(26, "S", 6).
shifts_hard_lb(26, "★", 0).
shifts_hard_lb(26, "☆", 0).
shifts_hard_lb(26, "P", 0).
shifts_soft_lb(26, "★", 0).
shifts_soft_lb(26, "☆", 0).
shifts_soft_lb(26, "P", 0).
shifts_soft_ub(26, "★", 0).
shifts_soft_ub(26, "☆", 0).
shifts_soft_ub(26, "P", 0).
shifts_hard_ub(26, "★", 0).
shifts_hard_ub(26, "☆", 0).
shifts_hard_ub(26, "P", 0).
shifts_hard_lb(26, "A", 0).
shifts_soft_lb(26, "A", 0).
shifts_soft_ub(26, "A", 0).
shifts_hard_ub(26, "A", 0).
% Staff 27:
shifts_hard_lb(27, "日", 2).
shifts_soft_lb(27, "日", 3).
shifts_soft_ub(27, "日", 6).
shifts_hard_ub(27, "日", 8).
shifts_hard_lb(27, "N", 4).
shifts_hard_lb(27, "J", 4).
shifts_hard_lb(27, "S", 4).
shifts_soft_lb(27, "N", 4).
shifts_soft_lb(27, "J", 4).
shifts_soft_lb(27, "S", 4).
shifts_soft_ub(27, "N", 5).
shifts_soft_ub(27, "J", 5).
shifts_soft_ub(27, "S", 5).
shifts_hard_ub(27, "N", 6).
shifts_hard_ub(27, "J", 6).
shifts_hard_ub(27, "S", 6).
shifts_hard_lb(27, "★", 0).
shifts_hard_lb(27, "☆", 0).
shifts_hard_lb(27, "P", 0).
shifts_soft_lb(27, "★", 0).
shifts_soft_lb(27, "☆", 0).
shifts_soft_lb(27, "P", 0).
shifts_soft_ub(27, "★", 0).
shifts_soft_ub(27, "☆", 0).
shifts_soft_ub(27, "P", 0).
shifts_hard_ub(27, "★", 0).
shifts_hard_ub(27, "☆", 0).
shifts_hard_ub(27, "P", 0).
shifts_hard_lb(27, "A", 0).
shifts_soft_lb(27, "A", 0).
shifts_soft_ub(27, "A", 0).
shifts_hard_ub(27, "A", 0).
% Staff 28:
shifts_hard_lb(28, "日", 3).
shifts_soft_lb(28, "日", 3).
shifts_soft_ub(28, "日", 6).
shifts_hard_ub(28, "日", 8).
shifts_hard_lb(28, "N", 4).
shifts_hard_lb(28, "J", 4).
shifts_hard_lb(28, "S", 4).
shifts_soft_lb(28, "N", 4).
shifts_soft_lb(28, "J", 4).
shifts_soft_lb(28, "S", 4).
shifts_soft_ub(28, "N", 5).
shifts_soft_ub(28, "J", 5).
shifts_soft_ub(28, "S", 5).
shifts_hard_ub(28, "N", 6).
shifts_hard_ub(28, "J", 6).
shifts_hard_ub(28, "S", 6).
shifts_hard_lb(28, "★", 0).
shifts_hard_lb(28, "☆", 0).
shifts_hard_lb(28, "P", 0).
shifts_soft_lb(28, "★", 0).
shifts_soft_lb(28, "☆", 0).
shifts_soft_lb(28, "P", 0).
shifts_soft_ub(28, "★", 0).
shifts_soft_ub(28, "☆", 0).
shifts_soft_ub(28, "P", 0).
shifts_hard_ub(28, "★", 0).
shifts_hard_ub(28, "☆", 0).
shifts_hard_ub(28, "P", 0).
shifts_hard_lb(28, "A", 0).
shifts_soft_lb(28, "A", 0).
shifts_soft_ub(28, "A", 0).
shifts_hard_ub(28, "A", 0).
% Staff 29:
shifts_hard_lb(29, "日", 6).
shifts_soft_lb(29, "日", 7).
shifts_soft_ub(29, "日", 19).
shifts_hard_ub(29, "日", 20).
shifts_hard_lb(29, "N", 0).
shifts_hard_lb(29, "J", 0).
shifts_hard_lb(29, "S", 0).
shifts_soft_lb(29, "N", 0).
shifts_soft_lb(29, "J", 0).
shifts_soft_lb(29, "S", 0).
shifts_soft_ub(29, "N", 4).
shifts_soft_ub(29, "J", 4).
shifts_soft_ub(29, "S", 4).
shifts_hard_ub(29, "N", 4).
shifts_hard_ub(29, "J", 4).
shifts_hard_ub(29, "S", 4).
shifts_hard_lb(29, "★", 0).
shifts_hard_lb(29, "☆", 0).
shifts_hard_lb(29, "P", 0).
shifts_soft_lb(29, "★", 0).
shifts_soft_lb(29, "☆", 0).
shifts_soft_lb(29, "P", 0).
shifts_soft_ub(29, "★", 0).
shifts_soft_ub(29, "☆", 0).
shifts_soft_ub(29, "P", 0).
shifts_hard_ub(29, "★", 0).
shifts_hard_ub(29, "☆", 0).
shifts_hard_ub(29, "P", 0).
shifts_hard_lb(29, "A", 0).
shifts_soft_lb(29, "A", 0).
shifts_soft_ub(29, "A", 0).
shifts_hard_ub(29, "A", 0).
% Staff 30:
shifts_hard_lb(30, "日", 4).
shifts_soft_lb(30, "日", 4).
shifts_soft_ub(30, "日", 15).
shifts_hard_ub(30, "日", 16).
shifts_hard_lb(30, "N", 0).
shifts_hard_lb(30, "J", 0).
shifts_hard_lb(30, "S", 0).
shifts_soft_lb(30, "N", 0).
shifts_soft_lb(30, "J", 0).
shifts_soft_lb(30, "S", 0).
shifts_soft_ub(30, "N", 5).
shifts_soft_ub(30, "J", 5).
shifts_soft_ub(30, "S", 5).
shifts_hard_ub(30, "N", 5).
shifts_hard_ub(30, "J", 5).
shifts_hard_ub(30, "S", 5).
shifts_hard_lb(30, "★", 0).
shifts_hard_lb(30, "☆", 0).
shifts_hard_lb(30, "P", 0).
shifts_soft_lb(30, "★", 0).
shifts_soft_lb(30, "☆", 0).
shifts_soft_lb(30, "P", 0).
shifts_soft_ub(30, "★", 0).
shifts_soft_ub(30, "☆", 0).
shifts_soft_ub(30, "P", 0).
shifts_hard_ub(30, "★", 0).
shifts_hard_ub(30, "☆", 0).
shifts_hard_ub(30, "P", 0).
shifts_hard_lb(30, "A", 0).
shifts_soft_lb(30, "A", 0).
shifts_soft_ub(30, "A", 0).
shifts_hard_ub(30, "A", 0).
horizontal_constraint_type("日").
horizontal_constraint_type("N").
horizontal_constraint_type("J").
horizontal_constraint_type("S").
horizontal_constraint_type("★").
horizontal_constraint_type("☆").
horizontal_constraint_type("P").
horizontal_constraint_type("A").
% Positive shifts ------------------------------------------
% Negative shifts ------------------------------------------
