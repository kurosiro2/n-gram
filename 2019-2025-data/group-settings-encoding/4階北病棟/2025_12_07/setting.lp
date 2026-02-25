% Staffs ---------------------------------------------------
staff(1, "長澤　良美", "師長", "0191206", 0).
staff(2, "望月　文香", "副師長", "0236665", 0).
staff(3, "河村　早紀", "助産師", "0269756", 0).
staff(4, "中込　美幸", "副師長", "0058294", 0).
staff(5, "橋本　瑞季", "副師長", "0272853", 0).
staff(6, "鶴田　真希", "助産師", "0250845", 0).
staff(7, "平岡　千夏", "助産師", "0261564", 0).
staff(8, "赤松　真実", "助産師", "0299354", 0).
staff(9, "笠井　知世", "助産師", "0212963", 0).
staff(10, "望月　舞", "助産師", "0341402", 0).
staff(11, "奥山　有紗", "助産師", "0370431", 0).
staff(12, "伊藤　晴美", "助産師", "0315433", 0).
staff(13, "小俣　りか", "助産師", "0494998", 0).
staff(14, "鷲見　舞", "助産師", "0329348", 0).
staff(15, "山口　奈津", "助産師", "0469680", 0).
staff(16, "田中　里実", "助産師", "0347784", 0).
staff(17, "原　奈々子", "助産師", "0347820", 0).
staff(18, "米山　菜月", "看護師", "0384621", 0).
staff(19, "岡本　結良", "助産師", "0357720", 0).
staff(20, "若杉　凜", "助産師", "0418858", 0).
staff(21, "松山　紗季", "看護師", "0418951", 0).
staff(22, "髙山　梓", "看護師", "0418984", 0).
staff(23, "小林　凜", "助産師", "0378202", 0).
staff(24, "道本　紗佑里", "助産師", "0487330", 0).
staff(25, "佐々木　佳子", "助産師", "0372524", 0).
staff(26, "内川　あずさ", "助産師", "0385677", 0).
staff(27, "小野　紅羽", "助産師", "0395748", 0).
staff(28, "古瀬村　綾乃", "助産師", "0395794", 0).
staff(29, "河野　鈴", "助産師", "0606242", 0).
staff(30, "三科　和香", "助産師", "0606594", 0).
staff(31, "大村　向日葵", "助産師", "0400037", 0).
staff(32, "大勝　めい", "助産師", "0399732", 0).
staff(33, "窪田　陽香", "助産師", "0384698", 0).
staff(34, "大木　麻衣", "パート", "0236643", 0).
staff(35, "石川　美好", "パート", "0150536", 0).
staff(36, "雨宮　美紀", "パート", "0222395", 0).
% Groups ---------------------------------------------------
group("全員").
group("全員", 1).
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
group("全員", 34).
group("全員", 35).
group("全員", 36).
group("代行").
group("代行", 3).
group("代行", 4).
group("代行", 5).
group("代行", 6).
group("代行", 7).
group("代行", 8).
group("代行", 9).
group("代行", 10).
group("リーダー").
group("リーダー", 3).
group("リーダー", 4).
group("リーダー", 5).
group("リーダー", 6).
group("リーダー", 7).
group("リーダー", 8).
group("リーダー", 9).
group("リーダー", 10).
group("リーダー", 11).
group("リーダー", 12).
group("リーダー", 14).
group("平日代行").
group("平日代行", 3).
group("平日代行", 4).
group("平日代行", 5).
group("平日代行", 6).
group("新人").
group("新人", 18).
group("新人", 21).
group("新人", 22).
group("新人", 25).
group("新人", 26).
group("新人", 27).
group("新人", 28).
group("新人", 29).
group("新人", 30).
group("夜勤").
group("夜勤", 3).
group("夜勤", 4).
group("夜勤", 5).
group("夜勤", 6).
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
forbidden_pattern("NN").
pattern("NN", 2).
pattern("NN", 0, "N").
pattern("NN", 1, "N").
forbidden_pattern("JSJS").
pattern("JSJS", 4).
pattern("JSJS", 0, "J").
pattern("JSJS", 1, "S").
pattern("JSJS", 2, "J").
pattern("JSJS", 3, "S").
forbidden_pattern("N日").
pattern("N日", 2).
pattern("N日", 0, "N").
pattern("N日", 1, "日").
forbidden_pattern("PN").
pattern("PN", 2).
pattern("PN", 0, "P").
pattern("PN", 1, "N").
forbidden_pattern("P日").
pattern("P日", 2).
pattern("P日", 0, "P").
pattern("P日", 1, "日").
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
forbidden_pattern("日日日日NJ").
pattern("日日日日NJ", 6).
pattern("日日日日NJ", 0, "日").
pattern("日日日日NJ", 1, "日").
pattern("日日日日NJ", 2, "日").
pattern("日日日日NJ", 3, "日").
pattern("日日日日NJ", 4, "N").
pattern("日日日日NJ", 5, "J").
forbidden_pattern("日日NJSJS").
pattern("日日NJSJS", 7).
pattern("日日NJSJS", 0, "日").
pattern("日日NJSJS", 1, "日").
pattern("日日NJSJS", 2, "N").
pattern("日日NJSJS", 3, "J").
pattern("日日NJSJS", 4, "S").
pattern("日日NJSJS", 5, "J").
pattern("日日NJSJS", 6, "S").
forbidden_pattern("日日日日日日").
pattern("日日日日日日", 6).
pattern("日日日日日日", 0, "日").
pattern("日日日日日日", 1, "日").
pattern("日日日日日日", 2, "日").
pattern("日日日日日日", 3, "日").
pattern("日日日日日日", 4, "日").
pattern("日日日日日日", 5, "日").
% 前日・翌日可能シフト
pred_shift("J", "S").
succ_shift("S", "J").
succ_shift("S", "○").
succ_shift("S", "◎").
pred_shift("N", "J").
pred_shift("P", "J").
pred_shift("日", "J").
pred_shift("○", "J").
pred_shift("◎", "J").
succ_shift("J", "S").
pred_shift("N", "★").
pred_shift("P", "★").
pred_shift("日", "★").
pred_shift("○", "★").
pred_shift("◎", "★").
succ_shift("★", "☆").
pred_shift("★", "☆").
succ_shift("☆", "○").
succ_shift("☆", "◎").
pred_shift("日", "日").
pred_shift("○", "日").
pred_shift("◎", "日").
succ_shift("日", "N").
succ_shift("日", "P").
succ_shift("日", "日").
succ_shift("日", "○").
succ_shift("日", "◎").
succ_shift("日", "J").
succ_shift("日", "★").
pred_shift("日", "N").
pred_shift("○", "N").
pred_shift("◎", "N").
succ_shift("N", "P").
succ_shift("N", "○").
succ_shift("N", "◎").
succ_shift("N", "J").
succ_shift("N", "★").
pred_shift("N", "P").
pred_shift("日", "P").
pred_shift("○", "P").
pred_shift("◎", "P").
succ_shift("P", "○").
succ_shift("P", "◎").
succ_shift("P", "J").
succ_shift("P", "★").
% パターン割当数
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
patterns_hard_lb("○○", 2).
patterns("N日", "N日").
patterns_soft_ub("N日", 0).
patterns("P日", "P日").
patterns_soft_ub("P日", 0).
patterns("PN", "PN").
patterns_soft_ub("PN", 0).
patterns("PP", "PP").
pattern("PP", 2).
pattern("PP", 0, "P").
pattern("PP", 1, "P").
patterns_soft_ub("PP", 0).
% Vertical constraints -------------------------------------
shifts("日").
shifts("日", "日").
staff_dweek_hard_lb("全員", "日", "月", 4).
staff_dweek_hard_lb("全員", "日", "火", 4).
staff_dweek_hard_lb("全員", "日", "水", 4).
staff_dweek_hard_lb("全員", "日", "木", 4).
staff_dweek_hard_lb("全員", "日", "金", 4).
staff_dweek_hard_lb("全員", "日", "土", 3).
staff_dweek_hard_lb("全員", "日", "日", 3).
staff_dweek_hard_lb("全員", "日", "祝", 3).
staff_dweek_soft_lb("全員", "日", "月", 4).
staff_dweek_soft_lb("全員", "日", "火", 4).
staff_dweek_soft_lb("全員", "日", "水", 4).
staff_dweek_soft_lb("全員", "日", "木", 4).
staff_dweek_soft_lb("全員", "日", "金", 4).
staff_dweek_soft_lb("全員", "日", "土", 3).
staff_dweek_soft_lb("全員", "日", "日", 3).
staff_dweek_soft_lb("全員", "日", "祝", 3).
staff_dweek_soft_ub("全員", "日", "月", 9).
staff_dweek_soft_ub("全員", "日", "火", 9).
staff_dweek_soft_ub("全員", "日", "水", 9).
staff_dweek_soft_ub("全員", "日", "木", 9).
staff_dweek_soft_ub("全員", "日", "金", 9).
staff_dweek_soft_ub("全員", "日", "土", 5).
staff_dweek_soft_ub("全員", "日", "日", 5).
staff_dweek_soft_ub("全員", "日", "祝", 5).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "月", 4).
staff_dweek_hard_lb("全員", "N", "火", 4).
staff_dweek_hard_lb("全員", "N", "水", 4).
staff_dweek_hard_lb("全員", "N", "木", 4).
staff_dweek_hard_lb("全員", "N", "金", 4).
staff_dweek_hard_lb("全員", "N", "土", 4).
staff_dweek_hard_lb("全員", "N", "日", 4).
staff_dweek_hard_lb("全員", "N", "祝", 4).
staff_dweek_soft_lb("全員", "N", "月", 4).
staff_dweek_soft_lb("全員", "N", "火", 4).
staff_dweek_soft_lb("全員", "N", "水", 4).
staff_dweek_soft_lb("全員", "N", "木", 4).
staff_dweek_soft_lb("全員", "N", "金", 4).
staff_dweek_soft_lb("全員", "N", "土", 4).
staff_dweek_soft_lb("全員", "N", "日", 4).
staff_dweek_soft_lb("全員", "N", "祝", 4).
staff_dweek_soft_ub("全員", "N", "月", 4).
staff_dweek_soft_ub("全員", "N", "火", 4).
staff_dweek_soft_ub("全員", "N", "水", 4).
staff_dweek_soft_ub("全員", "N", "木", 4).
staff_dweek_soft_ub("全員", "N", "金", 4).
staff_dweek_soft_ub("全員", "N", "土", 4).
staff_dweek_soft_ub("全員", "N", "日", 4).
staff_dweek_soft_ub("全員", "N", "祝", 4).
staff_dweek_hard_ub("全員", "N", "月", 4).
staff_dweek_hard_ub("全員", "N", "火", 4).
staff_dweek_hard_ub("全員", "N", "水", 4).
staff_dweek_hard_ub("全員", "N", "木", 4).
staff_dweek_hard_ub("全員", "N", "金", 4).
staff_dweek_hard_ub("全員", "N", "土", 4).
staff_dweek_hard_ub("全員", "N", "日", 4).
staff_dweek_hard_ub("全員", "N", "祝", 4).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "月", 4).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "月", 4).
staff_dweek_hard_lb("全員", "J", "火", 4).
staff_dweek_hard_lb("全員", "S", "火", 4).
staff_dweek_hard_lb("全員", "J", "水", 4).
staff_dweek_hard_lb("全員", "S", "水", 4).
staff_dweek_hard_lb("全員", "J", "木", 4).
staff_dweek_hard_lb("全員", "S", "木", 4).
staff_dweek_hard_lb("全員", "J", "金", 4).
staff_dweek_hard_lb("全員", "S", "金", 4).
staff_dweek_hard_lb("全員", "J", "土", 4).
staff_dweek_hard_lb("全員", "S", "土", 4).
staff_dweek_hard_lb("全員", "J", "日", 4).
staff_dweek_hard_lb("全員", "S", "日", 4).
staff_dweek_hard_lb("全員", "J", "祝", 4).
staff_dweek_hard_lb("全員", "S", "祝", 4).
staff_dweek_soft_lb("全員", "J", "月", 4).
staff_dweek_soft_lb("全員", "S", "月", 4).
staff_dweek_soft_lb("全員", "J", "火", 4).
staff_dweek_soft_lb("全員", "S", "火", 4).
staff_dweek_soft_lb("全員", "J", "水", 4).
staff_dweek_soft_lb("全員", "S", "水", 4).
staff_dweek_soft_lb("全員", "J", "木", 4).
staff_dweek_soft_lb("全員", "S", "木", 4).
staff_dweek_soft_lb("全員", "J", "金", 4).
staff_dweek_soft_lb("全員", "S", "金", 4).
staff_dweek_soft_lb("全員", "J", "土", 4).
staff_dweek_soft_lb("全員", "S", "土", 4).
staff_dweek_soft_lb("全員", "J", "日", 4).
staff_dweek_soft_lb("全員", "S", "日", 4).
staff_dweek_soft_lb("全員", "J", "祝", 4).
staff_dweek_soft_lb("全員", "S", "祝", 4).
staff_dweek_soft_ub("全員", "J", "月", 4).
staff_dweek_soft_ub("全員", "S", "月", 4).
staff_dweek_soft_ub("全員", "J", "火", 4).
staff_dweek_soft_ub("全員", "S", "火", 4).
staff_dweek_soft_ub("全員", "J", "水", 4).
staff_dweek_soft_ub("全員", "S", "水", 4).
staff_dweek_soft_ub("全員", "J", "木", 4).
staff_dweek_soft_ub("全員", "S", "木", 4).
staff_dweek_soft_ub("全員", "J", "金", 4).
staff_dweek_soft_ub("全員", "S", "金", 4).
staff_dweek_soft_ub("全員", "J", "土", 4).
staff_dweek_soft_ub("全員", "S", "土", 4).
staff_dweek_soft_ub("全員", "J", "日", 4).
staff_dweek_soft_ub("全員", "S", "日", 4).
staff_dweek_soft_ub("全員", "J", "祝", 4).
staff_dweek_soft_ub("全員", "S", "祝", 4).
staff_dweek_hard_ub("全員", "J", "月", 4).
staff_dweek_hard_ub("全員", "S", "月", 4).
staff_dweek_hard_ub("全員", "J", "火", 4).
staff_dweek_hard_ub("全員", "S", "火", 4).
staff_dweek_hard_ub("全員", "J", "水", 4).
staff_dweek_hard_ub("全員", "S", "水", 4).
staff_dweek_hard_ub("全員", "J", "木", 4).
staff_dweek_hard_ub("全員", "S", "木", 4).
staff_dweek_hard_ub("全員", "J", "金", 4).
staff_dweek_hard_ub("全員", "S", "金", 4).
staff_dweek_hard_ub("全員", "J", "土", 4).
staff_dweek_hard_ub("全員", "S", "土", 4).
staff_dweek_hard_ub("全員", "J", "日", 4).
staff_dweek_hard_ub("全員", "S", "日", 4).
staff_dweek_hard_ub("全員", "J", "祝", 4).
staff_dweek_hard_ub("全員", "S", "祝", 4).
shifts("P").
shifts("P", "P").
staff_dweek_hard_ub("全員", "P", "月", 0).
staff_dweek_hard_ub("全員", "P", "火", 0).
staff_dweek_hard_ub("全員", "P", "水", 0).
staff_dweek_hard_ub("全員", "P", "木", 0).
staff_dweek_hard_ub("全員", "P", "金", 0).
staff_dweek_hard_ub("全員", "P", "土", 0).
staff_dweek_hard_ub("全員", "P", "日", 0).
staff_dweek_hard_ub("全員", "P", "祝", 0).
staff_dweek_hard_lb("代行", "日", "月", 1).
staff_dweek_hard_lb("代行", "N", "月", 1).
staff_dweek_hard_lb("代行", "日", "火", 1).
staff_dweek_hard_lb("代行", "N", "火", 1).
staff_dweek_hard_lb("代行", "日", "水", 1).
staff_dweek_hard_lb("代行", "N", "水", 1).
staff_dweek_hard_lb("代行", "日", "木", 1).
staff_dweek_hard_lb("代行", "N", "木", 1).
staff_dweek_hard_lb("代行", "日", "金", 1).
staff_dweek_hard_lb("代行", "N", "金", 1).
staff_dweek_hard_lb("代行", "日", "土", 1).
staff_dweek_hard_lb("代行", "N", "土", 1).
staff_dweek_hard_lb("代行", "日", "日", 1).
staff_dweek_hard_lb("代行", "N", "日", 1).
staff_dweek_hard_lb("代行", "日", "祝", 1).
staff_dweek_hard_lb("代行", "N", "祝", 1).
staff_dweek_soft_lb("代行", "日", "月", 1).
staff_dweek_soft_lb("代行", "N", "月", 1).
staff_dweek_soft_lb("代行", "日", "火", 1).
staff_dweek_soft_lb("代行", "N", "火", 1).
staff_dweek_soft_lb("代行", "日", "水", 1).
staff_dweek_soft_lb("代行", "N", "水", 1).
staff_dweek_soft_lb("代行", "日", "木", 1).
staff_dweek_soft_lb("代行", "N", "木", 1).
staff_dweek_soft_lb("代行", "日", "金", 1).
staff_dweek_soft_lb("代行", "N", "金", 1).
staff_dweek_soft_lb("代行", "日", "土", 1).
staff_dweek_soft_lb("代行", "N", "土", 1).
staff_dweek_soft_lb("代行", "日", "日", 1).
staff_dweek_soft_lb("代行", "N", "日", 1).
staff_dweek_soft_lb("代行", "日", "祝", 1).
staff_dweek_soft_lb("代行", "N", "祝", 1).
staff_dweek_soft_ub("代行", "日", "月", 1).
staff_dweek_soft_ub("代行", "N", "月", 1).
staff_dweek_soft_ub("代行", "日", "火", 1).
staff_dweek_soft_ub("代行", "N", "火", 1).
staff_dweek_soft_ub("代行", "日", "水", 1).
staff_dweek_soft_ub("代行", "N", "水", 1).
staff_dweek_soft_ub("代行", "日", "木", 1).
staff_dweek_soft_ub("代行", "N", "木", 1).
staff_dweek_soft_ub("代行", "日", "金", 1).
staff_dweek_soft_ub("代行", "N", "金", 1).
staff_dweek_soft_ub("代行", "日", "土", 1).
staff_dweek_soft_ub("代行", "N", "土", 1).
staff_dweek_soft_ub("代行", "日", "日", 1).
staff_dweek_soft_ub("代行", "N", "日", 1).
staff_dweek_soft_ub("代行", "日", "祝", 1).
staff_dweek_soft_ub("代行", "N", "祝", 1).
staff_dweek_hard_lb("リーダー", "N", "月", 1).
staff_dweek_hard_lb("リーダー", "J", "月", 1).
staff_dweek_hard_lb("リーダー", "S", "月", 1).
staff_dweek_hard_lb("リーダー", "N", "火", 1).
staff_dweek_hard_lb("リーダー", "J", "火", 1).
staff_dweek_hard_lb("リーダー", "S", "火", 1).
staff_dweek_hard_lb("リーダー", "N", "水", 1).
staff_dweek_hard_lb("リーダー", "J", "水", 1).
staff_dweek_hard_lb("リーダー", "S", "水", 1).
staff_dweek_hard_lb("リーダー", "N", "木", 1).
staff_dweek_hard_lb("リーダー", "J", "木", 1).
staff_dweek_hard_lb("リーダー", "S", "木", 1).
staff_dweek_hard_lb("リーダー", "N", "金", 1).
staff_dweek_hard_lb("リーダー", "J", "金", 1).
staff_dweek_hard_lb("リーダー", "S", "金", 1).
staff_dweek_hard_lb("リーダー", "N", "土", 1).
staff_dweek_hard_lb("リーダー", "J", "土", 1).
staff_dweek_hard_lb("リーダー", "S", "土", 1).
staff_dweek_hard_lb("リーダー", "N", "日", 1).
staff_dweek_hard_lb("リーダー", "J", "日", 1).
staff_dweek_hard_lb("リーダー", "S", "日", 1).
staff_dweek_hard_lb("リーダー", "N", "祝", 1).
staff_dweek_hard_lb("リーダー", "J", "祝", 1).
staff_dweek_hard_lb("リーダー", "S", "祝", 1).
staff_dweek_soft_lb("リーダー", "N", "月", 1).
staff_dweek_soft_lb("リーダー", "J", "月", 1).
staff_dweek_soft_lb("リーダー", "S", "月", 1).
staff_dweek_soft_lb("リーダー", "N", "火", 1).
staff_dweek_soft_lb("リーダー", "J", "火", 1).
staff_dweek_soft_lb("リーダー", "S", "火", 1).
staff_dweek_soft_lb("リーダー", "N", "水", 1).
staff_dweek_soft_lb("リーダー", "J", "水", 1).
staff_dweek_soft_lb("リーダー", "S", "水", 1).
staff_dweek_soft_lb("リーダー", "N", "木", 1).
staff_dweek_soft_lb("リーダー", "J", "木", 1).
staff_dweek_soft_lb("リーダー", "S", "木", 1).
staff_dweek_soft_lb("リーダー", "N", "金", 1).
staff_dweek_soft_lb("リーダー", "J", "金", 1).
staff_dweek_soft_lb("リーダー", "S", "金", 1).
staff_dweek_soft_lb("リーダー", "N", "土", 1).
staff_dweek_soft_lb("リーダー", "J", "土", 1).
staff_dweek_soft_lb("リーダー", "S", "土", 1).
staff_dweek_soft_lb("リーダー", "N", "日", 1).
staff_dweek_soft_lb("リーダー", "J", "日", 1).
staff_dweek_soft_lb("リーダー", "S", "日", 1).
staff_dweek_soft_lb("リーダー", "N", "祝", 1).
staff_dweek_soft_lb("リーダー", "J", "祝", 1).
staff_dweek_soft_lb("リーダー", "S", "祝", 1).
staff_dweek_soft_ub("リーダー", "N", "月", 2).
staff_dweek_soft_ub("リーダー", "J", "月", 2).
staff_dweek_soft_ub("リーダー", "S", "月", 2).
staff_dweek_soft_ub("リーダー", "N", "火", 2).
staff_dweek_soft_ub("リーダー", "J", "火", 2).
staff_dweek_soft_ub("リーダー", "S", "火", 2).
staff_dweek_soft_ub("リーダー", "N", "水", 2).
staff_dweek_soft_ub("リーダー", "J", "水", 2).
staff_dweek_soft_ub("リーダー", "S", "水", 2).
staff_dweek_soft_ub("リーダー", "N", "木", 2).
staff_dweek_soft_ub("リーダー", "J", "木", 2).
staff_dweek_soft_ub("リーダー", "S", "木", 2).
staff_dweek_soft_ub("リーダー", "N", "金", 2).
staff_dweek_soft_ub("リーダー", "J", "金", 2).
staff_dweek_soft_ub("リーダー", "S", "金", 2).
staff_dweek_soft_ub("リーダー", "N", "土", 2).
staff_dweek_soft_ub("リーダー", "J", "土", 2).
staff_dweek_soft_ub("リーダー", "S", "土", 2).
staff_dweek_soft_ub("リーダー", "N", "日", 2).
staff_dweek_soft_ub("リーダー", "J", "日", 2).
staff_dweek_soft_ub("リーダー", "S", "日", 2).
staff_dweek_soft_ub("リーダー", "N", "祝", 2).
staff_dweek_soft_ub("リーダー", "J", "祝", 2).
staff_dweek_soft_ub("リーダー", "S", "祝", 2).
staff_dweek_hard_ub("リーダー", "N", "月", 3).
staff_dweek_hard_ub("リーダー", "J", "月", 3).
staff_dweek_hard_ub("リーダー", "S", "月", 3).
staff_dweek_hard_ub("リーダー", "N", "火", 3).
staff_dweek_hard_ub("リーダー", "J", "火", 3).
staff_dweek_hard_ub("リーダー", "S", "火", 3).
staff_dweek_hard_ub("リーダー", "N", "水", 3).
staff_dweek_hard_ub("リーダー", "J", "水", 3).
staff_dweek_hard_ub("リーダー", "S", "水", 3).
staff_dweek_hard_ub("リーダー", "N", "木", 3).
staff_dweek_hard_ub("リーダー", "J", "木", 3).
staff_dweek_hard_ub("リーダー", "S", "木", 3).
staff_dweek_hard_ub("リーダー", "N", "金", 3).
staff_dweek_hard_ub("リーダー", "J", "金", 3).
staff_dweek_hard_ub("リーダー", "S", "金", 3).
staff_dweek_hard_ub("リーダー", "N", "土", 3).
staff_dweek_hard_ub("リーダー", "J", "土", 3).
staff_dweek_hard_ub("リーダー", "S", "土", 3).
staff_dweek_hard_ub("リーダー", "N", "日", 3).
staff_dweek_hard_ub("リーダー", "J", "日", 3).
staff_dweek_hard_ub("リーダー", "S", "日", 3).
staff_dweek_hard_ub("リーダー", "N", "祝", 3).
staff_dweek_hard_ub("リーダー", "J", "祝", 3).
staff_dweek_hard_ub("リーダー", "S", "祝", 3).
staff_dweek_hard_lb("平日代行", "日", "月", 1).
staff_dweek_hard_lb("平日代行", "日", "火", 1).
staff_dweek_hard_lb("平日代行", "日", "水", 1).
staff_dweek_hard_lb("平日代行", "日", "木", 1).
staff_dweek_hard_lb("平日代行", "日", "金", 1).
staff_dweek_hard_lb("平日代行", "日", "土", 0).
staff_dweek_hard_lb("平日代行", "日", "日", 0).
staff_dweek_hard_lb("平日代行", "日", "祝", 0).
staff_dweek_soft_lb("平日代行", "日", "月", 1).
staff_dweek_soft_lb("平日代行", "日", "火", 1).
staff_dweek_soft_lb("平日代行", "日", "水", 1).
staff_dweek_soft_lb("平日代行", "日", "木", 1).
staff_dweek_soft_lb("平日代行", "日", "金", 1).
staff_dweek_soft_lb("平日代行", "日", "土", 0).
staff_dweek_soft_lb("平日代行", "日", "日", 0).
staff_dweek_soft_lb("平日代行", "日", "祝", 0).
staff_dweek_soft_ub("平日代行", "日", "月", 1).
staff_dweek_soft_ub("平日代行", "日", "火", 1).
staff_dweek_soft_ub("平日代行", "日", "水", 1).
staff_dweek_soft_ub("平日代行", "日", "木", 1).
staff_dweek_soft_ub("平日代行", "日", "金", 1).
staff_dweek_soft_ub("平日代行", "日", "土", 0).
staff_dweek_soft_ub("平日代行", "日", "日", 0).
staff_dweek_soft_ub("平日代行", "日", "祝", 0).
staff_dweek_hard_lb("新人", "J", "月", 0).
staff_dweek_hard_lb("新人", "S", "月", 0).
staff_dweek_hard_lb("新人", "J", "火", 0).
staff_dweek_hard_lb("新人", "S", "火", 0).
staff_dweek_hard_lb("新人", "J", "水", 0).
staff_dweek_hard_lb("新人", "S", "水", 0).
staff_dweek_hard_lb("新人", "J", "木", 0).
staff_dweek_hard_lb("新人", "S", "木", 0).
staff_dweek_hard_lb("新人", "J", "金", 0).
staff_dweek_hard_lb("新人", "S", "金", 0).
staff_dweek_hard_lb("新人", "J", "土", 0).
staff_dweek_hard_lb("新人", "S", "土", 0).
staff_dweek_hard_lb("新人", "J", "日", 0).
staff_dweek_hard_lb("新人", "S", "日", 0).
staff_dweek_hard_lb("新人", "J", "祝", 0).
staff_dweek_hard_lb("新人", "S", "祝", 0).
staff_dweek_soft_lb("新人", "J", "月", 1).
staff_dweek_soft_lb("新人", "S", "月", 1).
staff_dweek_soft_lb("新人", "J", "火", 1).
staff_dweek_soft_lb("新人", "S", "火", 1).
staff_dweek_soft_lb("新人", "J", "水", 1).
staff_dweek_soft_lb("新人", "S", "水", 1).
staff_dweek_soft_lb("新人", "J", "木", 1).
staff_dweek_soft_lb("新人", "S", "木", 1).
staff_dweek_soft_lb("新人", "J", "金", 1).
staff_dweek_soft_lb("新人", "S", "金", 1).
staff_dweek_soft_lb("新人", "J", "土", 1).
staff_dweek_soft_lb("新人", "S", "土", 1).
staff_dweek_soft_lb("新人", "J", "日", 1).
staff_dweek_soft_lb("新人", "S", "日", 1).
staff_dweek_soft_lb("新人", "J", "祝", 1).
staff_dweek_soft_lb("新人", "S", "祝", 1).
staff_dweek_soft_ub("新人", "J", "月", 1).
staff_dweek_soft_ub("新人", "S", "月", 1).
staff_dweek_soft_ub("新人", "J", "火", 1).
staff_dweek_soft_ub("新人", "S", "火", 1).
staff_dweek_soft_ub("新人", "J", "水", 1).
staff_dweek_soft_ub("新人", "S", "水", 1).
staff_dweek_soft_ub("新人", "J", "木", 1).
staff_dweek_soft_ub("新人", "S", "木", 1).
staff_dweek_soft_ub("新人", "J", "金", 1).
staff_dweek_soft_ub("新人", "S", "金", 1).
staff_dweek_soft_ub("新人", "J", "土", 1).
staff_dweek_soft_ub("新人", "S", "土", 1).
staff_dweek_soft_ub("新人", "J", "日", 1).
staff_dweek_soft_ub("新人", "S", "日", 1).
staff_dweek_soft_ub("新人", "J", "祝", 1).
staff_dweek_soft_ub("新人", "S", "祝", 1).
staff_dweek_hard_ub("新人", "J", "月", 1).
staff_dweek_hard_ub("新人", "S", "月", 1).
staff_dweek_hard_ub("新人", "J", "火", 1).
staff_dweek_hard_ub("新人", "S", "火", 1).
staff_dweek_hard_ub("新人", "J", "水", 1).
staff_dweek_hard_ub("新人", "S", "水", 1).
staff_dweek_hard_ub("新人", "J", "木", 1).
staff_dweek_hard_ub("新人", "S", "木", 1).
staff_dweek_hard_ub("新人", "J", "金", 1).
staff_dweek_hard_ub("新人", "S", "金", 1).
staff_dweek_hard_ub("新人", "J", "土", 1).
staff_dweek_hard_ub("新人", "S", "土", 1).
staff_dweek_hard_ub("新人", "J", "日", 1).
staff_dweek_hard_ub("新人", "S", "日", 1).
staff_dweek_hard_ub("新人", "J", "祝", 1).
staff_dweek_hard_ub("新人", "S", "祝", 1).
staff_dweek_hard_lb("夜勤", "J", "月", 4).
staff_dweek_hard_lb("夜勤", "S", "月", 4).
staff_dweek_hard_lb("夜勤", "J", "火", 4).
staff_dweek_hard_lb("夜勤", "S", "火", 4).
staff_dweek_hard_lb("夜勤", "J", "水", 4).
staff_dweek_hard_lb("夜勤", "S", "水", 4).
staff_dweek_hard_lb("夜勤", "J", "木", 4).
staff_dweek_hard_lb("夜勤", "S", "木", 4).
staff_dweek_hard_lb("夜勤", "J", "金", 4).
staff_dweek_hard_lb("夜勤", "S", "金", 4).
staff_dweek_hard_lb("夜勤", "J", "土", 4).
staff_dweek_hard_lb("夜勤", "S", "土", 4).
staff_dweek_hard_lb("夜勤", "J", "日", 4).
staff_dweek_hard_lb("夜勤", "S", "日", 4).
staff_dweek_hard_lb("夜勤", "J", "祝", 4).
staff_dweek_hard_lb("夜勤", "S", "祝", 4).
staff_dweek_soft_lb("夜勤", "J", "月", 4).
staff_dweek_soft_lb("夜勤", "S", "月", 4).
staff_dweek_soft_lb("夜勤", "J", "火", 4).
staff_dweek_soft_lb("夜勤", "S", "火", 4).
staff_dweek_soft_lb("夜勤", "J", "水", 4).
staff_dweek_soft_lb("夜勤", "S", "水", 4).
staff_dweek_soft_lb("夜勤", "J", "木", 4).
staff_dweek_soft_lb("夜勤", "S", "木", 4).
staff_dweek_soft_lb("夜勤", "J", "金", 4).
staff_dweek_soft_lb("夜勤", "S", "金", 4).
staff_dweek_soft_lb("夜勤", "J", "土", 4).
staff_dweek_soft_lb("夜勤", "S", "土", 4).
staff_dweek_soft_lb("夜勤", "J", "日", 4).
staff_dweek_soft_lb("夜勤", "S", "日", 4).
staff_dweek_soft_lb("夜勤", "J", "祝", 4).
staff_dweek_soft_lb("夜勤", "S", "祝", 4).
staff_dweek_soft_ub("夜勤", "J", "月", 4).
staff_dweek_soft_ub("夜勤", "S", "月", 4).
staff_dweek_soft_ub("夜勤", "J", "火", 4).
staff_dweek_soft_ub("夜勤", "S", "火", 4).
staff_dweek_soft_ub("夜勤", "J", "水", 4).
staff_dweek_soft_ub("夜勤", "S", "水", 4).
staff_dweek_soft_ub("夜勤", "J", "木", 4).
staff_dweek_soft_ub("夜勤", "S", "木", 4).
staff_dweek_soft_ub("夜勤", "J", "金", 4).
staff_dweek_soft_ub("夜勤", "S", "金", 4).
staff_dweek_soft_ub("夜勤", "J", "土", 4).
staff_dweek_soft_ub("夜勤", "S", "土", 4).
staff_dweek_soft_ub("夜勤", "J", "日", 4).
staff_dweek_soft_ub("夜勤", "S", "日", 4).
staff_dweek_soft_ub("夜勤", "J", "祝", 4).
staff_dweek_soft_ub("夜勤", "S", "祝", 4).
staff_dweek_hard_ub("夜勤", "J", "月", 4).
staff_dweek_hard_ub("夜勤", "S", "月", 4).
staff_dweek_hard_ub("夜勤", "J", "火", 4).
staff_dweek_hard_ub("夜勤", "S", "火", 4).
staff_dweek_hard_ub("夜勤", "J", "水", 4).
staff_dweek_hard_ub("夜勤", "S", "水", 4).
staff_dweek_hard_ub("夜勤", "J", "木", 4).
staff_dweek_hard_ub("夜勤", "S", "木", 4).
staff_dweek_hard_ub("夜勤", "J", "金", 4).
staff_dweek_hard_ub("夜勤", "S", "金", 4).
staff_dweek_hard_ub("夜勤", "J", "土", 4).
staff_dweek_hard_ub("夜勤", "S", "土", 4).
staff_dweek_hard_ub("夜勤", "J", "日", 4).
staff_dweek_hard_ub("夜勤", "S", "日", 4).
staff_dweek_hard_ub("夜勤", "J", "祝", 4).
staff_dweek_hard_ub("夜勤", "S", "祝", 4).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("代行","日","人数").
vertical_constraint_type("代行","N","人数").
vertical_constraint_type("リーダー","N","人数").
vertical_constraint_type("リーダー","J","人数").
vertical_constraint_type("リーダー","S","人数").
vertical_constraint_type("平日代行","日","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
vertical_constraint_type("夜勤","J","人数").
vertical_constraint_type("夜勤","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_hard_lb(1, "N", 0).
shifts_hard_lb(1, "J", 0).
shifts_hard_lb(1, "S", 0).
shifts_soft_lb(1, "N", 0).
shifts_soft_lb(1, "J", 0).
shifts_soft_lb(1, "S", 0).
shifts_soft_ub(1, "N", 0).
shifts_soft_ub(1, "J", 0).
shifts_soft_ub(1, "S", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "J", 0).
shifts_hard_ub(1, "S", 0).
shifts("★").
shifts("★", "★").
shifts_hard_lb(1, "★", 0).
shifts("☆").
shifts("☆", "☆").
shifts_hard_lb(1, "☆", 0).
shifts_soft_lb(1, "★", 0).
shifts_soft_lb(1, "☆", 0).
shifts_soft_ub(1, "★", 0).
shifts_soft_ub(1, "☆", 0).
shifts_hard_ub(1, "★", 0).
shifts_hard_ub(1, "☆", 0).
shifts_hard_lb(1, "P", 0).
shifts_soft_lb(1, "P", 0).
shifts_soft_ub(1, "P", 0).
shifts_hard_ub(1, "P", 0).
shifts("A").
shifts("A", "A").
shifts_hard_lb(1, "A", 0).
shifts_soft_lb(1, "A", 0).
shifts_soft_ub(1, "A", 0).
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_hard_lb(2, "N", 0).
shifts_hard_lb(2, "J", 0).
shifts_hard_lb(2, "S", 0).
shifts_soft_lb(2, "N", 0).
shifts_soft_lb(2, "J", 0).
shifts_soft_lb(2, "S", 0).
shifts_soft_ub(2, "N", 0).
shifts_soft_ub(2, "J", 0).
shifts_soft_ub(2, "S", 0).
shifts_hard_ub(2, "N", 0).
shifts_hard_ub(2, "J", 0).
shifts_hard_ub(2, "S", 0).
shifts_hard_lb(2, "★", 0).
shifts_hard_lb(2, "☆", 0).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_ub(2, "★", 0).
shifts_soft_ub(2, "☆", 0).
shifts_hard_ub(2, "★", 0).
shifts_hard_ub(2, "☆", 0).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "P", 0).
shifts_hard_ub(2, "P", 0).
shifts_hard_lb(2, "A", 0).
shifts_soft_lb(2, "A", 0).
shifts_soft_ub(2, "A", 0).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_hard_lb(3, "N", 2).
shifts_hard_lb(3, "J", 2).
shifts_hard_lb(3, "S", 2).
shifts_soft_lb(3, "N", 2).
shifts_soft_lb(3, "J", 2).
shifts_soft_lb(3, "S", 2).
shifts_soft_ub(3, "N", 2).
shifts_soft_ub(3, "J", 2).
shifts_soft_ub(3, "S", 2).
shifts_hard_ub(3, "N", 3).
shifts_hard_ub(3, "J", 3).
shifts_hard_ub(3, "S", 3).
shifts_hard_lb(3, "★", 0).
shifts_hard_lb(3, "☆", 0).
shifts_soft_lb(3, "★", 0).
shifts_soft_lb(3, "☆", 0).
shifts_soft_ub(3, "★", 0).
shifts_soft_ub(3, "☆", 0).
shifts_hard_ub(3, "★", 0).
shifts_hard_ub(3, "☆", 0).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "P", 0).
shifts_hard_ub(3, "P", 0).
shifts_hard_lb(3, "A", 0).
shifts_soft_lb(3, "A", 0).
shifts_soft_ub(3, "A", 0).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_hard_lb(4, "N", 1).
shifts_hard_lb(4, "J", 1).
shifts_hard_lb(4, "S", 1).
shifts_soft_lb(4, "N", 2).
shifts_soft_lb(4, "J", 2).
shifts_soft_lb(4, "S", 2).
shifts_soft_ub(4, "N", 2).
shifts_soft_ub(4, "J", 2).
shifts_soft_ub(4, "S", 2).
shifts_hard_ub(4, "N", 2).
shifts_hard_ub(4, "J", 2).
shifts_hard_ub(4, "S", 2).
shifts_hard_lb(4, "★", 0).
shifts_hard_lb(4, "☆", 0).
shifts_soft_lb(4, "★", 0).
shifts_soft_lb(4, "☆", 0).
shifts_soft_ub(4, "★", 0).
shifts_soft_ub(4, "☆", 0).
shifts_hard_ub(4, "★", 0).
shifts_hard_ub(4, "☆", 0).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "P", 0).
shifts_hard_ub(4, "P", 0).
shifts_hard_lb(4, "A", 0).
shifts_soft_lb(4, "A", 0).
shifts_soft_ub(4, "A", 0).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_hard_lb(5, "N", 2).
shifts_hard_lb(5, "J", 2).
shifts_hard_lb(5, "S", 2).
shifts_soft_lb(5, "N", 3).
shifts_soft_lb(5, "J", 3).
shifts_soft_lb(5, "S", 3).
shifts_soft_ub(5, "N", 4).
shifts_soft_ub(5, "J", 4).
shifts_soft_ub(5, "S", 4).
shifts_hard_ub(5, "N", 5).
shifts_hard_ub(5, "J", 5).
shifts_hard_ub(5, "S", 5).
shifts_hard_lb(5, "★", 0).
shifts_hard_lb(5, "☆", 0).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_ub(5, "★", 0).
shifts_soft_ub(5, "☆", 0).
shifts_hard_ub(5, "★", 0).
shifts_hard_ub(5, "☆", 0).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "P", 0).
shifts_hard_ub(5, "P", 0).
shifts_hard_lb(5, "A", 0).
shifts_soft_lb(5, "A", 0).
shifts_soft_ub(5, "A", 0).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_hard_lb(6, "N", 2).
shifts_hard_lb(6, "J", 2).
shifts_hard_lb(6, "S", 2).
shifts_soft_lb(6, "N", 2).
shifts_soft_lb(6, "J", 2).
shifts_soft_lb(6, "S", 2).
shifts_soft_ub(6, "N", 2).
shifts_soft_ub(6, "J", 2).
shifts_soft_ub(6, "S", 2).
shifts_hard_ub(6, "N", 3).
shifts_hard_ub(6, "J", 3).
shifts_hard_ub(6, "S", 3).
shifts_hard_lb(6, "★", 0).
shifts_hard_lb(6, "☆", 0).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_ub(6, "★", 0).
shifts_soft_ub(6, "☆", 0).
shifts_hard_ub(6, "★", 0).
shifts_hard_ub(6, "☆", 0).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "P", 0).
shifts_hard_ub(6, "P", 0).
shifts_hard_lb(6, "A", 0).
shifts_soft_lb(6, "A", 0).
shifts_soft_ub(6, "A", 0).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_hard_lb(7, "N", 3).
shifts_hard_lb(7, "J", 3).
shifts_hard_lb(7, "S", 3).
shifts_soft_lb(7, "N", 3).
shifts_soft_lb(7, "J", 3).
shifts_soft_lb(7, "S", 3).
shifts_soft_ub(7, "N", 3).
shifts_soft_ub(7, "J", 3).
shifts_soft_ub(7, "S", 3).
shifts_hard_ub(7, "N", 3).
shifts_hard_ub(7, "J", 3).
shifts_hard_ub(7, "S", 3).
shifts_hard_lb(7, "★", 0).
shifts_hard_lb(7, "☆", 0).
shifts_soft_lb(7, "★", 0).
shifts_soft_lb(7, "☆", 0).
shifts_soft_ub(7, "★", 0).
shifts_soft_ub(7, "☆", 0).
shifts_hard_ub(7, "★", 0).
shifts_hard_ub(7, "☆", 0).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "P", 0).
shifts_hard_ub(7, "P", 0).
shifts_hard_lb(7, "A", 0).
shifts_soft_lb(7, "A", 0).
shifts_soft_ub(7, "A", 0).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_hard_lb(8, "N", 3).
shifts_hard_lb(8, "J", 3).
shifts_hard_lb(8, "S", 3).
shifts_soft_lb(8, "N", 4).
shifts_soft_lb(8, "J", 4).
shifts_soft_lb(8, "S", 4).
shifts_soft_ub(8, "N", 5).
shifts_soft_ub(8, "J", 5).
shifts_soft_ub(8, "S", 5).
shifts_hard_ub(8, "N", 6).
shifts_hard_ub(8, "J", 6).
shifts_hard_ub(8, "S", 6).
shifts_hard_lb(8, "★", 0).
shifts_hard_lb(8, "☆", 0).
shifts_soft_lb(8, "★", 0).
shifts_soft_lb(8, "☆", 0).
shifts_soft_ub(8, "★", 0).
shifts_soft_ub(8, "☆", 0).
shifts_hard_ub(8, "★", 0).
shifts_hard_ub(8, "☆", 0).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "P", 0).
shifts_hard_ub(8, "P", 0).
shifts_hard_lb(8, "A", 0).
shifts_soft_lb(8, "A", 0).
shifts_soft_ub(8, "A", 0).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_hard_lb(9, "N", 3).
shifts_hard_lb(9, "J", 3).
shifts_hard_lb(9, "S", 3).
shifts_soft_lb(9, "N", 4).
shifts_soft_lb(9, "J", 4).
shifts_soft_lb(9, "S", 4).
shifts_soft_ub(9, "N", 5).
shifts_soft_ub(9, "J", 5).
shifts_soft_ub(9, "S", 5).
shifts_hard_ub(9, "N", 6).
shifts_hard_ub(9, "J", 6).
shifts_hard_ub(9, "S", 6).
shifts_hard_lb(9, "★", 0).
shifts_hard_lb(9, "☆", 0).
shifts_soft_lb(9, "★", 0).
shifts_soft_lb(9, "☆", 0).
shifts_soft_ub(9, "★", 0).
shifts_soft_ub(9, "☆", 0).
shifts_hard_ub(9, "★", 0).
shifts_hard_ub(9, "☆", 0).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "P", 0).
shifts_hard_ub(9, "P", 0).
shifts_hard_lb(9, "A", 0).
shifts_soft_lb(9, "A", 0).
shifts_soft_ub(9, "A", 0).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_hard_lb(10, "N", 3).
shifts_hard_lb(10, "J", 3).
shifts_hard_lb(10, "S", 3).
shifts_soft_lb(10, "N", 4).
shifts_soft_lb(10, "J", 4).
shifts_soft_lb(10, "S", 4).
shifts_soft_ub(10, "N", 5).
shifts_soft_ub(10, "J", 5).
shifts_soft_ub(10, "S", 5).
shifts_hard_ub(10, "N", 6).
shifts_hard_ub(10, "J", 6).
shifts_hard_ub(10, "S", 6).
shifts_hard_lb(10, "★", 0).
shifts_hard_lb(10, "☆", 0).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_ub(10, "★", 0).
shifts_soft_ub(10, "☆", 0).
shifts_hard_ub(10, "★", 0).
shifts_hard_ub(10, "☆", 0).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "P", 0).
shifts_hard_ub(10, "P", 0).
shifts_hard_lb(10, "A", 0).
shifts_soft_lb(10, "A", 0).
shifts_soft_ub(10, "A", 0).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_hard_lb(11, "N", 3).
shifts_hard_lb(11, "J", 3).
shifts_hard_lb(11, "S", 3).
shifts_soft_lb(11, "N", 4).
shifts_soft_lb(11, "J", 4).
shifts_soft_lb(11, "S", 4).
shifts_soft_ub(11, "N", 5).
shifts_soft_ub(11, "J", 5).
shifts_soft_ub(11, "S", 5).
shifts_hard_ub(11, "N", 6).
shifts_hard_ub(11, "J", 6).
shifts_hard_ub(11, "S", 6).
shifts_hard_lb(11, "★", 0).
shifts_hard_lb(11, "☆", 0).
shifts_soft_lb(11, "★", 0).
shifts_soft_lb(11, "☆", 0).
shifts_soft_ub(11, "★", 0).
shifts_soft_ub(11, "☆", 0).
shifts_hard_ub(11, "★", 0).
shifts_hard_ub(11, "☆", 0).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "P", 0).
shifts_hard_ub(11, "P", 0).
shifts_hard_lb(11, "A", 0).
shifts_soft_lb(11, "A", 0).
shifts_soft_ub(11, "A", 0).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_hard_lb(12, "N", 3).
shifts_hard_lb(12, "J", 3).
shifts_hard_lb(12, "S", 3).
shifts_soft_lb(12, "N", 4).
shifts_soft_lb(12, "J", 4).
shifts_soft_lb(12, "S", 4).
shifts_soft_ub(12, "N", 5).
shifts_soft_ub(12, "J", 5).
shifts_soft_ub(12, "S", 5).
shifts_hard_ub(12, "N", 6).
shifts_hard_ub(12, "J", 6).
shifts_hard_ub(12, "S", 6).
shifts_hard_lb(12, "★", 0).
shifts_hard_lb(12, "☆", 0).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_ub(12, "★", 0).
shifts_soft_ub(12, "☆", 0).
shifts_hard_ub(12, "★", 0).
shifts_hard_ub(12, "☆", 0).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "P", 0).
shifts_hard_ub(12, "P", 0).
shifts_hard_lb(12, "A", 0).
shifts_soft_lb(12, "A", 0).
shifts_soft_ub(12, "A", 0).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_hard_lb(13, "N", 2).
shifts_hard_lb(13, "J", 2).
shifts_hard_lb(13, "S", 2).
shifts_soft_lb(13, "N", 3).
shifts_soft_lb(13, "J", 3).
shifts_soft_lb(13, "S", 3).
shifts_soft_ub(13, "N", 3).
shifts_soft_ub(13, "J", 3).
shifts_soft_ub(13, "S", 3).
shifts_hard_ub(13, "N", 3).
shifts_hard_ub(13, "J", 3).
shifts_hard_ub(13, "S", 3).
shifts_hard_lb(13, "★", 0).
shifts_hard_lb(13, "☆", 0).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_ub(13, "★", 0).
shifts_soft_ub(13, "☆", 0).
shifts_hard_ub(13, "★", 0).
shifts_hard_ub(13, "☆", 0).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "P", 0).
shifts_hard_ub(13, "P", 0).
shifts_hard_lb(13, "A", 0).
shifts_soft_lb(13, "A", 0).
shifts_soft_ub(13, "A", 0).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_hard_lb(14, "N", 3).
shifts_hard_lb(14, "J", 3).
shifts_hard_lb(14, "S", 3).
shifts_soft_lb(14, "N", 4).
shifts_soft_lb(14, "J", 4).
shifts_soft_lb(14, "S", 4).
shifts_soft_ub(14, "N", 5).
shifts_soft_ub(14, "J", 5).
shifts_soft_ub(14, "S", 5).
shifts_hard_ub(14, "N", 6).
shifts_hard_ub(14, "J", 6).
shifts_hard_ub(14, "S", 6).
shifts_hard_lb(14, "★", 0).
shifts_hard_lb(14, "☆", 0).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_ub(14, "★", 0).
shifts_soft_ub(14, "☆", 0).
shifts_hard_ub(14, "★", 0).
shifts_hard_ub(14, "☆", 0).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "P", 0).
shifts_hard_ub(14, "P", 0).
shifts_hard_lb(14, "A", 0).
shifts_soft_lb(14, "A", 0).
shifts_soft_ub(14, "A", 0).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_hard_lb(15, "N", 3).
shifts_hard_lb(15, "J", 3).
shifts_hard_lb(15, "S", 3).
shifts_soft_lb(15, "N", 4).
shifts_soft_lb(15, "J", 4).
shifts_soft_lb(15, "S", 4).
shifts_soft_ub(15, "N", 5).
shifts_soft_ub(15, "J", 5).
shifts_soft_ub(15, "S", 5).
shifts_hard_ub(15, "N", 6).
shifts_hard_ub(15, "J", 6).
shifts_hard_ub(15, "S", 6).
shifts_hard_lb(15, "★", 0).
shifts_hard_lb(15, "☆", 0).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_ub(15, "★", 0).
shifts_soft_ub(15, "☆", 0).
shifts_hard_ub(15, "★", 0).
shifts_hard_ub(15, "☆", 0).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "P", 0).
shifts_hard_ub(15, "P", 0).
shifts_hard_lb(15, "A", 0).
shifts_soft_lb(15, "A", 0).
shifts_soft_ub(15, "A", 0).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_hard_lb(16, "N", 3).
shifts_hard_lb(16, "J", 3).
shifts_hard_lb(16, "S", 3).
shifts_soft_lb(16, "N", 4).
shifts_soft_lb(16, "J", 4).
shifts_soft_lb(16, "S", 4).
shifts_soft_ub(16, "N", 5).
shifts_soft_ub(16, "J", 5).
shifts_soft_ub(16, "S", 5).
shifts_hard_ub(16, "N", 6).
shifts_hard_ub(16, "J", 6).
shifts_hard_ub(16, "S", 6).
shifts_hard_lb(16, "★", 0).
shifts_hard_lb(16, "☆", 0).
shifts_soft_lb(16, "★", 0).
shifts_soft_lb(16, "☆", 0).
shifts_soft_ub(16, "★", 0).
shifts_soft_ub(16, "☆", 0).
shifts_hard_ub(16, "★", 0).
shifts_hard_ub(16, "☆", 0).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "P", 0).
shifts_hard_ub(16, "P", 0).
shifts_hard_lb(16, "A", 0).
shifts_soft_lb(16, "A", 0).
shifts_soft_ub(16, "A", 0).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_hard_lb(17, "N", 3).
shifts_hard_lb(17, "J", 3).
shifts_hard_lb(17, "S", 3).
shifts_soft_lb(17, "N", 4).
shifts_soft_lb(17, "J", 4).
shifts_soft_lb(17, "S", 4).
shifts_soft_ub(17, "N", 5).
shifts_soft_ub(17, "J", 5).
shifts_soft_ub(17, "S", 5).
shifts_hard_ub(17, "N", 6).
shifts_hard_ub(17, "J", 6).
shifts_hard_ub(17, "S", 6).
shifts_hard_lb(17, "★", 0).
shifts_hard_lb(17, "☆", 0).
shifts_soft_lb(17, "★", 0).
shifts_soft_lb(17, "☆", 0).
shifts_soft_ub(17, "★", 0).
shifts_soft_ub(17, "☆", 0).
shifts_hard_ub(17, "★", 0).
shifts_hard_ub(17, "☆", 0).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "P", 0).
shifts_hard_ub(17, "P", 0).
shifts_hard_lb(17, "A", 0).
shifts_soft_lb(17, "A", 0).
shifts_soft_ub(17, "A", 0).
shifts_hard_ub(17, "A", 0).
% Staff 18:
shifts_hard_lb(18, "N", 1).
shifts_hard_lb(18, "J", 1).
shifts_hard_lb(18, "S", 1).
shifts_soft_lb(18, "N", 2).
shifts_soft_lb(18, "J", 2).
shifts_soft_lb(18, "S", 2).
shifts_soft_ub(18, "N", 2).
shifts_soft_ub(18, "J", 2).
shifts_soft_ub(18, "S", 2).
shifts_hard_ub(18, "N", 2).
shifts_hard_ub(18, "J", 2).
shifts_hard_ub(18, "S", 2).
shifts_hard_lb(18, "★", 0).
shifts_hard_lb(18, "☆", 0).
shifts_soft_lb(18, "★", 0).
shifts_soft_lb(18, "☆", 0).
shifts_soft_ub(18, "★", 0).
shifts_soft_ub(18, "☆", 0).
shifts_hard_ub(18, "★", 0).
shifts_hard_ub(18, "☆", 0).
shifts_hard_lb(18, "P", 0).
shifts_soft_lb(18, "P", 0).
shifts_soft_ub(18, "P", 0).
shifts_hard_ub(18, "P", 0).
shifts_hard_lb(18, "A", 0).
shifts_soft_lb(18, "A", 0).
shifts_soft_ub(18, "A", 0).
shifts_hard_ub(18, "A", 0).
% Staff 19:
shifts_hard_lb(19, "N", 3).
shifts_hard_lb(19, "J", 3).
shifts_hard_lb(19, "S", 3).
shifts_soft_lb(19, "N", 4).
shifts_soft_lb(19, "J", 4).
shifts_soft_lb(19, "S", 4).
shifts_soft_ub(19, "N", 5).
shifts_soft_ub(19, "J", 5).
shifts_soft_ub(19, "S", 5).
shifts_hard_ub(19, "N", 6).
shifts_hard_ub(19, "J", 6).
shifts_hard_ub(19, "S", 6).
shifts_hard_lb(19, "★", 0).
shifts_hard_lb(19, "☆", 0).
shifts_soft_lb(19, "★", 0).
shifts_soft_lb(19, "☆", 0).
shifts_soft_ub(19, "★", 0).
shifts_soft_ub(19, "☆", 0).
shifts_hard_ub(19, "★", 0).
shifts_hard_ub(19, "☆", 0).
shifts_hard_lb(19, "P", 0).
shifts_soft_lb(19, "P", 0).
shifts_soft_ub(19, "P", 0).
shifts_hard_ub(19, "P", 0).
shifts_hard_lb(19, "A", 0).
shifts_soft_lb(19, "A", 0).
shifts_soft_ub(19, "A", 0).
shifts_hard_ub(19, "A", 0).
% Staff 20:
shifts_hard_lb(20, "N", 3).
shifts_hard_lb(20, "J", 3).
shifts_hard_lb(20, "S", 3).
shifts_soft_lb(20, "N", 4).
shifts_soft_lb(20, "J", 4).
shifts_soft_lb(20, "S", 4).
shifts_soft_ub(20, "N", 5).
shifts_soft_ub(20, "J", 5).
shifts_soft_ub(20, "S", 5).
shifts_hard_ub(20, "N", 6).
shifts_hard_ub(20, "J", 6).
shifts_hard_ub(20, "S", 6).
shifts_hard_lb(20, "★", 0).
shifts_hard_lb(20, "☆", 0).
shifts_soft_lb(20, "★", 0).
shifts_soft_lb(20, "☆", 0).
shifts_soft_ub(20, "★", 0).
shifts_soft_ub(20, "☆", 0).
shifts_hard_ub(20, "★", 0).
shifts_hard_ub(20, "☆", 0).
shifts_hard_lb(20, "P", 0).
shifts_soft_lb(20, "P", 0).
shifts_soft_ub(20, "P", 0).
shifts_hard_ub(20, "P", 0).
shifts_hard_lb(20, "A", 0).
shifts_soft_lb(20, "A", 0).
shifts_soft_ub(20, "A", 0).
shifts_hard_ub(20, "A", 0).
% Staff 21:
shifts_hard_lb(21, "N", 3).
shifts_hard_lb(21, "J", 3).
shifts_hard_lb(21, "S", 3).
shifts_soft_lb(21, "N", 4).
shifts_soft_lb(21, "J", 4).
shifts_soft_lb(21, "S", 4).
shifts_soft_ub(21, "N", 5).
shifts_soft_ub(21, "J", 5).
shifts_soft_ub(21, "S", 5).
shifts_hard_ub(21, "N", 6).
shifts_hard_ub(21, "J", 6).
shifts_hard_ub(21, "S", 6).
shifts_hard_lb(21, "★", 0).
shifts_hard_lb(21, "☆", 0).
shifts_soft_lb(21, "★", 0).
shifts_soft_lb(21, "☆", 0).
shifts_soft_ub(21, "★", 0).
shifts_soft_ub(21, "☆", 0).
shifts_hard_ub(21, "★", 0).
shifts_hard_ub(21, "☆", 0).
shifts_hard_lb(21, "P", 0).
shifts_soft_lb(21, "P", 0).
shifts_soft_ub(21, "P", 0).
shifts_hard_ub(21, "P", 0).
shifts_hard_lb(21, "A", 0).
shifts_soft_lb(21, "A", 0).
shifts_soft_ub(21, "A", 0).
shifts_hard_ub(21, "A", 0).
% Staff 22:
shifts_hard_lb(22, "N", 3).
shifts_hard_lb(22, "J", 3).
shifts_hard_lb(22, "S", 3).
shifts_soft_lb(22, "N", 4).
shifts_soft_lb(22, "J", 4).
shifts_soft_lb(22, "S", 4).
shifts_soft_ub(22, "N", 5).
shifts_soft_ub(22, "J", 5).
shifts_soft_ub(22, "S", 5).
shifts_hard_ub(22, "N", 6).
shifts_hard_ub(22, "J", 6).
shifts_hard_ub(22, "S", 6).
shifts_hard_lb(22, "★", 0).
shifts_hard_lb(22, "☆", 0).
shifts_soft_lb(22, "★", 0).
shifts_soft_lb(22, "☆", 0).
shifts_soft_ub(22, "★", 0).
shifts_soft_ub(22, "☆", 0).
shifts_hard_ub(22, "★", 0).
shifts_hard_ub(22, "☆", 0).
shifts_hard_lb(22, "P", 0).
shifts_soft_lb(22, "P", 0).
shifts_soft_ub(22, "P", 0).
shifts_hard_ub(22, "P", 0).
shifts_hard_lb(22, "A", 0).
shifts_soft_lb(22, "A", 0).
shifts_soft_ub(22, "A", 0).
shifts_hard_ub(22, "A", 0).
% Staff 23:
shifts_hard_lb(23, "N", 3).
shifts_hard_lb(23, "J", 3).
shifts_hard_lb(23, "S", 3).
shifts_soft_lb(23, "N", 4).
shifts_soft_lb(23, "J", 4).
shifts_soft_lb(23, "S", 4).
shifts_soft_ub(23, "N", 5).
shifts_soft_ub(23, "J", 5).
shifts_soft_ub(23, "S", 5).
shifts_hard_ub(23, "N", 6).
shifts_hard_ub(23, "J", 6).
shifts_hard_ub(23, "S", 6).
shifts_hard_lb(23, "★", 0).
shifts_hard_lb(23, "☆", 0).
shifts_soft_lb(23, "★", 0).
shifts_soft_lb(23, "☆", 0).
shifts_soft_ub(23, "★", 0).
shifts_soft_ub(23, "☆", 0).
shifts_hard_ub(23, "★", 0).
shifts_hard_ub(23, "☆", 0).
shifts_hard_lb(23, "P", 0).
shifts_soft_lb(23, "P", 0).
shifts_soft_ub(23, "P", 0).
shifts_hard_ub(23, "P", 0).
shifts_hard_lb(23, "A", 0).
shifts_soft_lb(23, "A", 0).
shifts_soft_ub(23, "A", 0).
shifts_hard_ub(23, "A", 0).
% Staff 24:
shifts_hard_lb(24, "N", 1).
shifts_hard_lb(24, "J", 1).
shifts_hard_lb(24, "S", 1).
shifts_soft_lb(24, "N", 2).
shifts_soft_lb(24, "J", 2).
shifts_soft_lb(24, "S", 2).
shifts_soft_ub(24, "N", 3).
shifts_soft_ub(24, "J", 3).
shifts_soft_ub(24, "S", 3).
shifts_hard_ub(24, "N", 3).
shifts_hard_ub(24, "J", 3).
shifts_hard_ub(24, "S", 3).
shifts_hard_lb(24, "★", 0).
shifts_hard_lb(24, "☆", 0).
shifts_soft_lb(24, "★", 0).
shifts_soft_lb(24, "☆", 0).
shifts_soft_ub(24, "★", 0).
shifts_soft_ub(24, "☆", 0).
shifts_hard_ub(24, "★", 0).
shifts_hard_ub(24, "☆", 0).
shifts_hard_lb(24, "P", 0).
shifts_soft_lb(24, "P", 0).
shifts_soft_ub(24, "P", 0).
shifts_hard_ub(24, "P", 0).
shifts_hard_lb(24, "A", 0).
shifts_soft_lb(24, "A", 0).
shifts_soft_ub(24, "A", 0).
shifts_hard_ub(24, "A", 0).
% Staff 25:
shifts_hard_lb(25, "N", 1).
shifts_hard_lb(25, "J", 1).
shifts_hard_lb(25, "S", 1).
shifts_soft_lb(25, "N", 2).
shifts_soft_lb(25, "J", 2).
shifts_soft_lb(25, "S", 2).
shifts_soft_ub(25, "N", 3).
shifts_soft_ub(25, "J", 3).
shifts_soft_ub(25, "S", 3).
shifts_hard_ub(25, "N", 4).
shifts_hard_ub(25, "J", 4).
shifts_hard_ub(25, "S", 4).
shifts_hard_lb(25, "★", 0).
shifts_hard_lb(25, "☆", 0).
shifts_soft_lb(25, "★", 0).
shifts_soft_lb(25, "☆", 0).
shifts_soft_ub(25, "★", 0).
shifts_soft_ub(25, "☆", 0).
shifts_hard_ub(25, "★", 0).
shifts_hard_ub(25, "☆", 0).
shifts_hard_lb(25, "P", 0).
shifts_soft_lb(25, "P", 0).
shifts_soft_ub(25, "P", 0).
shifts_hard_ub(25, "P", 0).
shifts_hard_lb(25, "A", 0).
shifts_soft_lb(25, "A", 0).
shifts_soft_ub(25, "A", 0).
shifts_hard_ub(25, "A", 0).
% Staff 26:
shifts_hard_lb(26, "N", 1).
shifts_hard_lb(26, "J", 1).
shifts_hard_lb(26, "S", 1).
shifts_soft_lb(26, "N", 2).
shifts_soft_lb(26, "J", 2).
shifts_soft_lb(26, "S", 2).
shifts_soft_ub(26, "N", 3).
shifts_soft_ub(26, "J", 3).
shifts_soft_ub(26, "S", 3).
shifts_hard_ub(26, "N", 4).
shifts_hard_ub(26, "J", 4).
shifts_hard_ub(26, "S", 4).
shifts_hard_lb(26, "★", 0).
shifts_hard_lb(26, "☆", 0).
shifts_soft_lb(26, "★", 0).
shifts_soft_lb(26, "☆", 0).
shifts_soft_ub(26, "★", 0).
shifts_soft_ub(26, "☆", 0).
shifts_hard_ub(26, "★", 0).
shifts_hard_ub(26, "☆", 0).
shifts_hard_lb(26, "P", 0).
shifts_soft_lb(26, "P", 0).
shifts_soft_ub(26, "P", 0).
shifts_hard_ub(26, "P", 0).
shifts_hard_lb(26, "A", 0).
shifts_soft_lb(26, "A", 0).
shifts_soft_ub(26, "A", 0).
shifts_hard_ub(26, "A", 0).
% Staff 27:
shifts_hard_lb(27, "N", 1).
shifts_hard_lb(27, "J", 1).
shifts_hard_lb(27, "S", 1).
shifts_soft_lb(27, "N", 2).
shifts_soft_lb(27, "J", 2).
shifts_soft_lb(27, "S", 2).
shifts_soft_ub(27, "N", 3).
shifts_soft_ub(27, "J", 3).
shifts_soft_ub(27, "S", 3).
shifts_hard_ub(27, "N", 4).
shifts_hard_ub(27, "J", 4).
shifts_hard_ub(27, "S", 4).
shifts_hard_lb(27, "★", 0).
shifts_hard_lb(27, "☆", 0).
shifts_soft_lb(27, "★", 0).
shifts_soft_lb(27, "☆", 0).
shifts_soft_ub(27, "★", 0).
shifts_soft_ub(27, "☆", 0).
shifts_hard_ub(27, "★", 0).
shifts_hard_ub(27, "☆", 0).
shifts_hard_lb(27, "P", 0).
shifts_soft_lb(27, "P", 0).
shifts_soft_ub(27, "P", 0).
shifts_hard_ub(27, "P", 0).
shifts_hard_lb(27, "A", 0).
shifts_soft_lb(27, "A", 0).
shifts_soft_ub(27, "A", 0).
shifts_hard_ub(27, "A", 0).
% Staff 28:
shifts_hard_lb(28, "N", 1).
shifts_hard_lb(28, "J", 1).
shifts_hard_lb(28, "S", 1).
shifts_soft_lb(28, "N", 2).
shifts_soft_lb(28, "J", 2).
shifts_soft_lb(28, "S", 2).
shifts_soft_ub(28, "N", 3).
shifts_soft_ub(28, "J", 3).
shifts_soft_ub(28, "S", 3).
shifts_hard_ub(28, "N", 4).
shifts_hard_ub(28, "J", 4).
shifts_hard_ub(28, "S", 4).
shifts_hard_lb(28, "★", 0).
shifts_hard_lb(28, "☆", 0).
shifts_soft_lb(28, "★", 0).
shifts_soft_lb(28, "☆", 0).
shifts_soft_ub(28, "★", 0).
shifts_soft_ub(28, "☆", 0).
shifts_hard_ub(28, "★", 0).
shifts_hard_ub(28, "☆", 0).
shifts_hard_lb(28, "P", 0).
shifts_soft_lb(28, "P", 0).
shifts_soft_ub(28, "P", 0).
shifts_hard_ub(28, "P", 0).
shifts_hard_lb(28, "A", 0).
shifts_soft_lb(28, "A", 0).
shifts_soft_ub(28, "A", 0).
shifts_hard_ub(28, "A", 0).
% Staff 29:
shifts_hard_lb(29, "N", 1).
shifts_hard_lb(29, "J", 1).
shifts_hard_lb(29, "S", 1).
shifts_soft_lb(29, "N", 2).
shifts_soft_lb(29, "J", 2).
shifts_soft_lb(29, "S", 2).
shifts_soft_ub(29, "N", 3).
shifts_soft_ub(29, "J", 3).
shifts_soft_ub(29, "S", 3).
shifts_hard_ub(29, "N", 4).
shifts_hard_ub(29, "J", 4).
shifts_hard_ub(29, "S", 4).
shifts_hard_lb(29, "★", 0).
shifts_hard_lb(29, "☆", 0).
shifts_soft_lb(29, "★", 0).
shifts_soft_lb(29, "☆", 0).
shifts_soft_ub(29, "★", 0).
shifts_soft_ub(29, "☆", 0).
shifts_hard_ub(29, "★", 0).
shifts_hard_ub(29, "☆", 0).
shifts_hard_lb(29, "P", 0).
shifts_soft_lb(29, "P", 0).
shifts_soft_ub(29, "P", 0).
shifts_hard_ub(29, "P", 0).
shifts_hard_lb(29, "A", 0).
shifts_soft_lb(29, "A", 0).
shifts_soft_ub(29, "A", 0).
shifts_hard_ub(29, "A", 0).
% Staff 30:
shifts_hard_lb(30, "N", 1).
shifts_hard_lb(30, "J", 1).
shifts_hard_lb(30, "S", 1).
shifts_soft_lb(30, "N", 2).
shifts_soft_lb(30, "J", 2).
shifts_soft_lb(30, "S", 2).
shifts_soft_ub(30, "N", 3).
shifts_soft_ub(30, "J", 3).
shifts_soft_ub(30, "S", 3).
shifts_hard_ub(30, "N", 4).
shifts_hard_ub(30, "J", 4).
shifts_hard_ub(30, "S", 4).
shifts_hard_lb(30, "★", 0).
shifts_hard_lb(30, "☆", 0).
shifts_soft_lb(30, "★", 0).
shifts_soft_lb(30, "☆", 0).
shifts_soft_ub(30, "★", 0).
shifts_soft_ub(30, "☆", 0).
shifts_hard_ub(30, "★", 0).
shifts_hard_ub(30, "☆", 0).
shifts_hard_lb(30, "P", 0).
shifts_soft_lb(30, "P", 0).
shifts_soft_ub(30, "P", 0).
shifts_hard_ub(30, "P", 0).
shifts_hard_lb(30, "A", 0).
shifts_soft_lb(30, "A", 0).
shifts_soft_ub(30, "A", 0).
shifts_hard_ub(30, "A", 0).
% Staff 31:
shifts_hard_lb(31, "N", 0).
shifts_hard_lb(31, "J", 0).
shifts_hard_lb(31, "S", 0).
shifts_soft_lb(31, "N", 0).
shifts_soft_lb(31, "J", 0).
shifts_soft_lb(31, "S", 0).
shifts_soft_ub(31, "N", 0).
shifts_soft_ub(31, "J", 0).
shifts_soft_ub(31, "S", 0).
shifts_hard_ub(31, "N", 0).
shifts_hard_ub(31, "J", 0).
shifts_hard_ub(31, "S", 0).
shifts_hard_lb(31, "★", 0).
shifts_hard_lb(31, "☆", 0).
shifts_soft_lb(31, "★", 0).
shifts_soft_lb(31, "☆", 0).
shifts_soft_ub(31, "★", 0).
shifts_soft_ub(31, "☆", 0).
shifts_hard_ub(31, "★", 0).
shifts_hard_ub(31, "☆", 0).
shifts_hard_lb(31, "P", 0).
shifts_soft_lb(31, "P", 0).
shifts_soft_ub(31, "P", 0).
shifts_hard_ub(31, "P", 0).
shifts_hard_lb(31, "A", 0).
shifts_soft_lb(31, "A", 0).
shifts_soft_ub(31, "A", 0).
shifts_hard_ub(31, "A", 0).
% Staff 32:
shifts_hard_lb(32, "N", 0).
shifts_hard_lb(32, "J", 0).
shifts_hard_lb(32, "S", 0).
shifts_soft_lb(32, "N", 0).
shifts_soft_lb(32, "J", 0).
shifts_soft_lb(32, "S", 0).
shifts_soft_ub(32, "N", 0).
shifts_soft_ub(32, "J", 0).
shifts_soft_ub(32, "S", 0).
shifts_hard_ub(32, "N", 0).
shifts_hard_ub(32, "J", 0).
shifts_hard_ub(32, "S", 0).
shifts_hard_lb(32, "★", 0).
shifts_hard_lb(32, "☆", 0).
shifts_soft_lb(32, "★", 0).
shifts_soft_lb(32, "☆", 0).
shifts_soft_ub(32, "★", 0).
shifts_soft_ub(32, "☆", 0).
shifts_hard_ub(32, "★", 0).
shifts_hard_ub(32, "☆", 0).
shifts_hard_lb(32, "P", 0).
shifts_soft_lb(32, "P", 0).
shifts_soft_ub(32, "P", 0).
shifts_hard_ub(32, "P", 0).
shifts_hard_lb(32, "A", 0).
shifts_soft_lb(32, "A", 0).
shifts_soft_ub(32, "A", 0).
shifts_hard_ub(32, "A", 0).
% Staff 33:
shifts_hard_lb(33, "N", 0).
shifts_hard_lb(33, "J", 0).
shifts_hard_lb(33, "S", 0).
shifts_soft_lb(33, "N", 0).
shifts_soft_lb(33, "J", 0).
shifts_soft_lb(33, "S", 0).
shifts_soft_ub(33, "N", 0).
shifts_soft_ub(33, "J", 0).
shifts_soft_ub(33, "S", 0).
shifts_hard_ub(33, "N", 0).
shifts_hard_ub(33, "J", 0).
shifts_hard_ub(33, "S", 0).
shifts_hard_lb(33, "★", 0).
shifts_hard_lb(33, "☆", 0).
shifts_soft_lb(33, "★", 0).
shifts_soft_lb(33, "☆", 0).
shifts_soft_ub(33, "★", 0).
shifts_soft_ub(33, "☆", 0).
shifts_hard_ub(33, "★", 0).
shifts_hard_ub(33, "☆", 0).
shifts_hard_lb(33, "P", 0).
shifts_soft_lb(33, "P", 0).
shifts_soft_ub(33, "P", 0).
shifts_hard_ub(33, "P", 0).
shifts_hard_lb(33, "A", 0).
shifts_soft_lb(33, "A", 0).
shifts_soft_ub(33, "A", 0).
shifts_hard_ub(33, "A", 0).
% Staff 34:
shifts_hard_lb(34, "N", 0).
shifts_hard_lb(34, "J", 0).
shifts_hard_lb(34, "S", 0).
shifts_soft_lb(34, "N", 0).
shifts_soft_lb(34, "J", 0).
shifts_soft_lb(34, "S", 0).
shifts_soft_ub(34, "N", 0).
shifts_soft_ub(34, "J", 0).
shifts_soft_ub(34, "S", 0).
shifts_hard_ub(34, "N", 0).
shifts_hard_ub(34, "J", 0).
shifts_hard_ub(34, "S", 0).
shifts_hard_lb(34, "★", 0).
shifts_hard_lb(34, "☆", 0).
shifts_soft_lb(34, "★", 0).
shifts_soft_lb(34, "☆", 0).
shifts_soft_ub(34, "★", 0).
shifts_soft_ub(34, "☆", 0).
shifts_hard_ub(34, "★", 0).
shifts_hard_ub(34, "☆", 0).
shifts_hard_lb(34, "P", 0).
shifts_soft_lb(34, "P", 0).
shifts_soft_ub(34, "P", 0).
shifts_hard_ub(34, "P", 0).
shifts_hard_lb(34, "A", 0).
shifts_soft_lb(34, "A", 0).
shifts_soft_ub(34, "A", 0).
shifts_hard_ub(34, "A", 0).
% Staff 35:
shifts_hard_lb(35, "N", 0).
shifts_hard_lb(35, "J", 0).
shifts_hard_lb(35, "S", 0).
shifts_soft_lb(35, "N", 0).
shifts_soft_lb(35, "J", 0).
shifts_soft_lb(35, "S", 0).
shifts_soft_ub(35, "N", 0).
shifts_soft_ub(35, "J", 0).
shifts_soft_ub(35, "S", 0).
shifts_hard_ub(35, "N", 0).
shifts_hard_ub(35, "J", 0).
shifts_hard_ub(35, "S", 0).
shifts_hard_lb(35, "★", 0).
shifts_hard_lb(35, "☆", 0).
shifts_soft_lb(35, "★", 0).
shifts_soft_lb(35, "☆", 0).
shifts_soft_ub(35, "★", 0).
shifts_soft_ub(35, "☆", 0).
shifts_hard_ub(35, "★", 0).
shifts_hard_ub(35, "☆", 0).
shifts_hard_lb(35, "P", 0).
shifts_soft_lb(35, "P", 0).
shifts_soft_ub(35, "P", 0).
shifts_hard_ub(35, "P", 0).
shifts_hard_lb(35, "A", 0).
shifts_soft_lb(35, "A", 0).
shifts_soft_ub(35, "A", 0).
shifts_hard_ub(35, "A", 0).
% Staff 36:
shifts_hard_lb(36, "N", 0).
shifts_hard_lb(36, "J", 0).
shifts_hard_lb(36, "S", 0).
shifts_soft_lb(36, "N", 0).
shifts_soft_lb(36, "J", 0).
shifts_soft_lb(36, "S", 0).
shifts_soft_ub(36, "N", 0).
shifts_soft_ub(36, "J", 0).
shifts_soft_ub(36, "S", 0).
shifts_hard_ub(36, "N", 0).
shifts_hard_ub(36, "J", 0).
shifts_hard_ub(36, "S", 0).
shifts_hard_lb(36, "★", 0).
shifts_hard_lb(36, "☆", 0).
shifts_soft_lb(36, "★", 0).
shifts_soft_lb(36, "☆", 0).
shifts_soft_ub(36, "★", 0).
shifts_soft_ub(36, "☆", 0).
shifts_hard_ub(36, "★", 0).
shifts_hard_ub(36, "☆", 0).
shifts_hard_lb(36, "P", 0).
shifts_soft_lb(36, "P", 0).
shifts_soft_ub(36, "P", 0).
shifts_hard_ub(36, "P", 0).
shifts_hard_lb(36, "A", 0).
shifts_soft_lb(36, "A", 0).
shifts_soft_ub(36, "A", 0).
shifts_hard_ub(36, "A", 0).
horizontal_constraint_type("日").
horizontal_constraint_type("N").
horizontal_constraint_type("J").
horizontal_constraint_type("S").
horizontal_constraint_type("★").
horizontal_constraint_type("☆").
horizontal_constraint_type("P").
horizontal_constraint_type("A").
forbidden_pair(4,7).
forbidden_pair(8,11).
% Positive shifts ------------------------------------------
% Staff 6:
pos_shift(6, "日", ("N")).
pos_shift(6, "土", ("N")).
pos_shift(6, "祝", ("N")).
% Staff 18:
pos_shift(18, "日", ("S")).
pos_shift(18, "金", ("N")).
pos_shift(18, "土", ("J")).
% Staff 34:
pos_shift(34, "月", ("日")).
pos_shift(34, "火", ("日")).
pos_shift(34, "水", ("日")).
pos_shift(34, "金", ("日")).
% Staff 35:
pos_shift(35, "火", ("日")).
pos_shift(35, "水", ("日")).
pos_shift(35, "木", ("日")).
pos_shift(35, "金", ("日")).
% Staff 36:
pos_shift(36, "月", ("日")).
pos_shift(36, "火", ("日")).
pos_shift(36, "水", ("日")).
pos_shift(36, "木", ("日")).
pos_shift(36, "金", ("日")).
% Negative shifts ------------------------------------------
% Staff 2:
neg_shift(2, "日", ("日";"N";"J";"S")).
neg_shift(2, "土", ("日";"N";"J";"S")).
neg_shift(2, "祝", ("日";"N";"J";"S")).
% Staff 3:
neg_shift(3, "日", ("N")).
neg_shift(3, "土", ("N")).
% Staff 4:
neg_shift(4, "月", ("N")).
neg_shift(4, "火", ("N")).
neg_shift(4, "水", ("N")).
neg_shift(4, "木", ("N")).
neg_shift(4, "金", ("N")).
neg_shift(4, "祝", ("N")).
% Staff 6:
neg_shift(6, "月", ("N")).
neg_shift(6, "火", ("N")).
neg_shift(6, "水", ("N")).
neg_shift(6, "木", ("N")).
neg_shift(6, "金", ("N")).
% Staff 13:
neg_shift(13, "金", ("日";"N")).
% Staff 18:
neg_shift(18, "日", ("日";"N";"J")).
neg_shift(18, "金", ("日";"J";"S")).
neg_shift(18, "土", ("日";"N";"S")).
% Staff 34:
neg_shift(34, "日", ("日";"N";"J";"S";"P")).
neg_shift(34, "土", ("日";"N";"J";"S";"P")).
neg_shift(34, "祝", ("日";"N";"J";"S";"P")).
% Staff 35:
neg_shift(35, "日", ("日";"N";"J";"S";"P")).
neg_shift(35, "土", ("日";"N";"J";"S";"P")).
neg_shift(35, "祝", ("日";"N";"J";"S";"P")).
% Staff 36:
neg_shift(36, "日", ("日";"N";"J";"S";"P")).
neg_shift(36, "土", ("日";"N";"J";"S";"P")).
neg_shift(36, "祝", ("日";"N";"J";"S";"P")).
