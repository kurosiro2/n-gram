% Staffs ---------------------------------------------------
staff(1, "寺島　由美子", "師長", "0102025", 0).
staff(2, "中込　美幸", "副師長", "0058294", 5).
staff(3, "杉本　美貴", "副師長", "0051525", 5).
staff(4, "舘田　安希子", "看護師", "0031188", 5).
staff(5, "小池　友佳梨", "看護師", "0283402", 5).
staff(6, "築野　梨奈", "看護師", "0312840", 5).
staff(7, "古屋　愛有香", "看護師", "0312920", 2).
staff(8, "立川　理奈子", "看護師", "0326607", 2).
staff(9, "原　砂貴子", "看護師", "0341218", 5).
staff(10, "斉藤　麻未", "看護師", "0355465", 2).
staff(11, "森本　柚実", "看護師", "0355705", 3).
staff(12, "大久保　茉耶", "看護師", "0370657", 3).
staff(13, "田邉　愛海", "看護師", "0370817", 3).
staff(14, "中川　菜々子", "看護師", "0384563", 3).
staff(15, "小林　結菜", "看護師", "0418437", 3).
staff(16, "梶原　葉南", "看護師", "0418448", 3).
staff(17, "加賀美　里奈", "看護師", "0418450", 2).
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
group("師長").
group("師長", 1).
group("熟練").
group("熟練", 2).
group("熟練", 3).
group("熟練", 4).
group("熟練", 5).
group("熟練", 6).
group("熟練", 7).
group("熟練", 10).
group("中堅").
group("中堅", 11).
group("中堅", 12).
group("中堅", 13).
group("中堅", 14).
group("新人").
group("新人", 9).
group("新人", 15).
group("新人", 16).
group("新人", 17).
group("夜勤").
group("夜勤", 2).
group("夜勤", 3).
group("夜勤", 4).
group("夜勤", 5).
group("夜勤", 6).
group("夜勤", 7).
group("夜勤", 9).
group("夜勤", 10).
group("夜勤", 11).
group("夜勤", 12).
group("夜勤", 13).
group("夜勤", 14).
group("夜勤", 15).
group("夜勤", 16).
group("夜勤", 17).
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
consecutive_work_ub(5).
% 禁止パターン

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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(追加部ここから)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 妥当な勤務パターン
% valid_pref2_next("J", "S", "○").

shifts("日").
shifts("日","日").
total_shifts_ub("全員","日", 77).
total_shifts_lb("全員","日", 68).
shifts("○").
shifts("○","○").
total_shifts_ub("全員","○", 134).
total_shifts_lb("全員","○", 131).
shifts("☆").
shifts("☆","☆").
total_shifts_ub("全員","☆", 14).
total_shifts_lb("全員","☆", 3).
shifts("★").
shifts("★","★").
total_shifts_ub("全員","★", 14).
total_shifts_lb("全員","★", 3).
shifts("P").
shifts("P","P").
total_shifts_ub("全員","P", 3).
total_shifts_lb("全員","P", 2).
shifts("J").
shifts("J","J").
total_shifts_ub("全員","J", 72).
total_shifts_lb("全員","J", 60).
shifts("S").
shifts("S","S").
total_shifts_ub("全員","S", 72).
total_shifts_lb("全員","S", 60).
shifts("N").
shifts("N","N").
total_shifts_ub("全員","N", 72).
total_shifts_lb("全員","N", 60).
patterns("○○", "○○").
pattern("○○", 2).
pattern("○○", 0, "○").
pattern("○○", 1, "○").
total_patterns_ub("全員","○○", 36).
total_patterns_lb("全員","○○", 32).
patterns("NN", "NN").
pattern("NN", 2).
pattern("NN", 0, "N").
pattern("NN", 1, "N").
total_patterns_ub("全員","NN", 11).
total_patterns_lb("全員","NN", 7).
patterns("○日", "○日").
pattern("○日", 2).
pattern("○日", 0, "○").
pattern("○日", 1, "日").
total_patterns_ub("全員","○日", 32).
total_patterns_lb("全員","○日", 25).
patterns("N○", "N○").
pattern("N○", 2).
pattern("N○", 0, "N").
pattern("N○", 1, "○").
total_patterns_ub("全員","N○", 17).
total_patterns_lb("全員","N○", 14).
patterns("S○", "S○").
pattern("S○", 2).
pattern("S○", 0, "S").
pattern("S○", 1, "○").
total_patterns_ub("全員","S○", 53).
total_patterns_lb("全員","S○", 41).
patterns("JN", "JN").
pattern("JN", 2).
pattern("JN", 0, "J").
pattern("JN", 1, "N").
total_patterns_ub("全員","JN", 0).
total_patterns_lb("全員","JN", 0).
patterns("○S", "○S").
pattern("○S", 2).
pattern("○S", 0, "○").
pattern("○S", 1, "S").
total_patterns_ub("全員","○S", 0).
total_patterns_lb("全員","○S", 0).
patterns("JJ", "JJ").
pattern("JJ", 2).
pattern("JJ", 0, "J").
pattern("JJ", 1, "J").
total_patterns_ub("全員","JJ", 0).
total_patterns_lb("全員","JJ", 0).
patterns("○☆", "○☆").
pattern("○☆", 2).
pattern("○☆", 0, "○").
pattern("○☆", 1, "☆").
total_patterns_ub("全員","○☆", 0).
total_patterns_lb("全員","○☆", 0).
patterns("SN", "SN").
pattern("SN", 2).
pattern("SN", 0, "S").
pattern("SN", 1, "N").
total_patterns_ub("全員","SN", 0).
total_patterns_lb("全員","SN", 0).
patterns("SS", "SS").
pattern("SS", 2).
pattern("SS", 0, "S").
pattern("SS", 1, "S").
total_patterns_ub("全員","SS", 0).
total_patterns_lb("全員","SS", 0).
patterns("NS", "NS").
pattern("NS", 2).
pattern("NS", 0, "N").
pattern("NS", 1, "S").
total_patterns_ub("全員","NS", 0).
total_patterns_lb("全員","NS", 0).
patterns("○N", "○N").
pattern("○N", 2).
pattern("○N", 0, "○").
pattern("○N", 1, "N").
total_patterns_ub("全員","○N", 44).
total_patterns_lb("全員","○N", 35).
patterns("J○", "J○").
pattern("J○", 2).
pattern("J○", 0, "J").
pattern("J○", 1, "○").
total_patterns_ub("全員","J○", 0).
total_patterns_lb("全員","J○", 0).
patterns("JS", "JS").
pattern("JS", 2).
pattern("JS", 0, "J").
pattern("JS", 1, "S").
total_patterns_ub("全員","JS", 66).
total_patterns_lb("全員","JS", 54).
patterns("★○", "★○").
pattern("★○", 2).
pattern("★○", 0, "★").
pattern("★○", 1, "○").
total_patterns_ub("全員","★○", 0).
total_patterns_lb("全員","★○", 0).
% patterns("◎日", "◎日").
% pattern("◎日", 2).
% pattern("◎日", 0, "◎").
% pattern("◎日", 1, "日").
% total_patterns_ub("全員","◎日", 3).
% total_patterns_lb("全員","◎日", 1).
patterns("N★", "N★").
pattern("N★", 2).
pattern("N★", 0, "N").
pattern("N★", 1, "★").
total_patterns_ub("全員","N★", 2).
total_patterns_lb("全員","N★", 1).
patterns("J☆", "J☆").
pattern("J☆", 2).
pattern("J☆", 0, "J").
pattern("J☆", 1, "☆").
total_patterns_ub("全員","J☆", 0).
total_patterns_lb("全員","J☆", 0).
patterns("S日", "S日").
pattern("S日", 2).
pattern("S日", 0, "S").
pattern("S日", 1, "日").
total_patterns_ub("全員","S日", 1).
total_patterns_lb("全員","S日", 0).
patterns("日N", "日N").
pattern("日N", 2).
pattern("日N", 0, "日").
pattern("日N", 1, "N").
total_patterns_ub("全員","日N", 12).
total_patterns_lb("全員","日N", 9).
patterns("N☆", "N☆").
pattern("N☆", 2).
pattern("N☆", 0, "N").
pattern("N☆", 1, "☆").
total_patterns_ub("全員","N☆", 0).
total_patterns_lb("全員","N☆", 0).
patterns("日☆", "日☆").
pattern("日☆", 2).
pattern("日☆", 0, "日").
pattern("日☆", 1, "☆").
total_patterns_ub("全員","日☆", 0).
total_patterns_lb("全員","日☆", 0).
patterns("日★", "日★").
pattern("日★", 2).
pattern("日★", 0, "日").
pattern("日★", 1, "★").
total_patterns_ub("全員","日★", 1).
total_patterns_lb("全員","日★", 1).
patterns("☆日", "☆日").
pattern("☆日", 2).
pattern("☆日", 0, "☆").
pattern("☆日", 1, "日").
total_patterns_ub("全員","☆日", 0).
total_patterns_lb("全員","☆日", 0).
% patterns("○◎", "○◎").
% pattern("○◎", 2).
% pattern("○◎", 0, "○").
% pattern("○◎", 1, "◎").
% total_patterns_ub("全員","○◎", 5).
% total_patterns_lb("全員","○◎", 2).
% patterns("◎○", "◎○").
% pattern("◎○", 2).
% pattern("◎○", 0, "◎").
% pattern("◎○", 1, "○").
% total_patterns_ub("全員","◎○", 5).
% total_patterns_lb("全員","◎○", 2).
patterns("★S", "★S").
pattern("★S", 2).
pattern("★S", 0, "★").
pattern("★S", 1, "S").
total_patterns_ub("全員","★S", 0).
total_patterns_lb("全員","★S", 0).
patterns("★N", "★N").
pattern("★N", 2).
pattern("★N", 0, "★").
pattern("★N", 1, "N").
total_patterns_ub("全員","★N", 0).
total_patterns_lb("全員","★N", 0).
patterns("J★", "J★").
pattern("J★", 2).
pattern("J★", 0, "J").
pattern("J★", 1, "★").
total_patterns_ub("全員","J★", 0).
total_patterns_lb("全員","J★", 0).
patterns("S★", "S★").
pattern("S★", 2).
pattern("S★", 0, "S").
pattern("S★", 1, "★").
total_patterns_ub("全員","S★", 1).
total_patterns_lb("全員","S★", 0).
patterns("☆S", "☆S").
pattern("☆S", 2).
pattern("☆S", 0, "☆").
pattern("☆S", 1, "S").
total_patterns_ub("全員","☆S", 0).
total_patterns_lb("全員","☆S", 0).
patterns("★J", "★J").
pattern("★J", 2).
pattern("★J", 0, "★").
pattern("★J", 1, "J").
total_patterns_ub("全員","★J", 0).
total_patterns_lb("全員","★J", 0).
patterns("★日", "★日").
pattern("★日", 2).
pattern("★日", 0, "★").
pattern("★日", 1, "日").
total_patterns_ub("全員","★日", 0).
total_patterns_lb("全員","★日", 0).
patterns("★★", "★★").
pattern("★★", 2).
pattern("★★", 0, "★").
pattern("★★", 1, "★").
total_patterns_ub("全員","★★", 0).
total_patterns_lb("全員","★★", 0).
patterns("J日", "J日").
pattern("J日", 2).
pattern("J日", 0, "J").
pattern("J日", 1, "日").
total_patterns_ub("全員","J日", 0).
total_patterns_lb("全員","J日", 0).
patterns("☆N", "☆N").
pattern("☆N", 2).
pattern("☆N", 0, "☆").
pattern("☆N", 1, "N").
total_patterns_ub("全員","☆N", 0).
total_patterns_lb("全員","☆N", 0).
patterns("☆☆", "☆☆").
pattern("☆☆", 2).
pattern("☆☆", 0, "☆").
pattern("☆☆", 1, "☆").
total_patterns_ub("全員","☆☆", 0).
total_patterns_lb("全員","☆☆", 0).
% patterns("◎☆", "◎☆").
% pattern("◎☆", 2).
% pattern("◎☆", 0, "◎").
% pattern("◎☆", 1, "☆").
% total_patterns_ub("全員","◎☆", 0).
% total_patterns_lb("全員","◎☆", 0).
% patterns("☆◎", "☆◎").
% pattern("☆◎", 2).
% pattern("☆◎", 0, "☆").
% pattern("☆◎", 1, "◎").
% total_patterns_ub("全員","☆◎", 0).
% total_patterns_lb("全員","☆◎", 0).
% patterns("日◎", "日◎").
% pattern("日◎", 2).
% pattern("日◎", 0, "日").
% pattern("日◎", 1, "◎").
% total_patterns_ub("全員","日◎", 2).
% total_patterns_lb("全員","日◎", 1).
patterns("S☆", "S☆").
pattern("S☆", 2).
pattern("S☆", 0, "S").
pattern("S☆", 1, "☆").
total_patterns_ub("全員","S☆", 0).
total_patterns_lb("全員","S☆", 0).
patterns("N日", "N日").
pattern("N日", 2).
pattern("N日", 0, "N").
pattern("N日", 1, "日").
total_patterns_ub("全員","N日", 7).
total_patterns_lb("全員","N日", 4).
patterns("NJ", "NJ").
pattern("NJ", 2).
pattern("NJ", 0, "N").
pattern("NJ", 1, "J").
total_patterns_ub("全員","NJ", 32).
total_patterns_lb("全員","NJ", 23).
% patterns("★◎", "★◎").
% pattern("★◎", 2).
% pattern("★◎", 0, "★").
% pattern("★◎", 1, "◎").
% total_patterns_ub("全員","★◎", 0).
% total_patterns_lb("全員","★◎", 0).
% patterns("◎N", "◎N").
% pattern("◎N", 2).
% pattern("◎N", 0, "◎").
% pattern("◎N", 1, "N").
% total_patterns_ub("全員","◎N", 3).
% total_patterns_lb("全員","◎N", 2).
% patterns("J◎", "J◎").
% pattern("J◎", 2).
% pattern("J◎", 0, "J").
% pattern("J◎", 1, "◎").
% total_patterns_ub("全員","J◎", 0).
% total_patterns_lb("全員","J◎", 0).
% patterns("◎S", "◎S").
% pattern("◎S", 2).
% pattern("◎S", 0, "◎").
% pattern("◎S", 1, "S").
% total_patterns_ub("全員","◎S", 0).
% total_patterns_lb("全員","◎S", 0).
% patterns("◎P", "◎P").
% pattern("◎P", 2).
% pattern("◎P", 0, "◎").
% pattern("◎P", 1, "P").
% total_patterns_ub("全員","◎P", 2).
% total_patterns_lb("全員","◎P", 1).
patterns("日○", "日○").
pattern("日○", 2).
pattern("日○", 0, "日").
pattern("日○", 1, "○").
total_patterns_ub("全員","日○", 19).
total_patterns_lb("全員","日○", 15).
patterns("○★", "○★").
pattern("○★", 2).
pattern("○★", 0, "○").
pattern("○★", 1, "★").
total_patterns_ub("全員","○★", 5).
total_patterns_lb("全員","○★", 1).
patterns("日日", "日日").
pattern("日日", 2).
pattern("日日", 0, "日").
pattern("日日", 1, "日").
total_patterns_ub("全員","日日", 35).
total_patterns_lb("全員","日日", 25).
patterns("日S", "日S").
pattern("日S", 2).
pattern("日S", 0, "日").
pattern("日S", 1, "S").
total_patterns_ub("全員","日S", 0).
total_patterns_lb("全員","日S", 0).
patterns("S○", "S○").
pattern("S○", 2).
pattern("S○", 0, "S").
pattern("S○", 1, "○").
total_patterns_ub("全員","S○", 52).
total_patterns_lb("全員","S○", 41).
patterns("SP", "SP").
pattern("SP", 2).
pattern("SP", 0, "S").
pattern("SP", 1, "P").
total_patterns_ub("全員","SP", 0).
total_patterns_lb("全員","SP", 0).
patterns("○○○", "○○○").
pattern("○○○", 3).
pattern("○○○", 0, "○").
pattern("○○○", 1, "○").
pattern("○○○", 2, "○").
total_patterns_ub("全員","○○○", 7).
total_patterns_lb("全員","○○○", 5).
patterns("S○★", "S○★").
pattern("S○★", 3).
pattern("S○★", 0, "S").
pattern("S○★", 1, "○").
pattern("S○★", 2, "★").
total_patterns_ub("全員","S○★", 1).
total_patterns_lb("全員","S○★", 0).
patterns("NNN", "NNN").
pattern("NNN", 3).
pattern("NNN", 0, "N").
pattern("NNN", 1, "N").
pattern("NNN", 2, "N").
total_patterns_ub("全員","NNN", 1).
total_patterns_lb("全員","NNN", 0).
patterns("日JS", "日JS").
pattern("日JS", 3).
pattern("日JS", 0, "日").
pattern("日JS", 1, "J").
pattern("日JS", 2, "S").
total_patterns_ub("全員","日JS", 6).
total_patterns_lb("全員","日JS", 4).
patterns("JS日", "JS日").
pattern("JS日", 3).
pattern("JS日", 0, "J").
pattern("JS日", 1, "S").
pattern("JS日", 2, "日").
total_patterns_ub("全員","JS日", 0).
total_patterns_lb("全員","JS日", 0).
patterns("○N○", "○N○").
pattern("○N○", 3).
pattern("○N○", 0, "○").
pattern("○N○", 1, "N").
pattern("○N○", 2, "○").
total_patterns_ub("全員","○N○", 8).
total_patterns_lb("全員","○N○", 6).
patterns("○日○", "○日○").
pattern("○日○", 3).
pattern("○日○", 0, "○").
pattern("○日○", 1, "日").
pattern("○日○", 2, "○").
total_patterns_ub("全員","○日○", 5).
total_patterns_lb("全員","○日○", 3).
patterns("N○○", "N○○").
pattern("N○○", 3).
pattern("N○○", 0, "N").
pattern("N○○", 1, "○").
pattern("N○○", 2, "○").
total_patterns_ub("全員","N○○", 5).
total_patterns_lb("全員","N○○", 3).
patterns("S○N", "S○N").
pattern("S○N", 3).
pattern("S○N", 0, "S").
pattern("S○N", 1, "○").
pattern("S○N", 2, "N").
total_patterns_ub("全員","S○N", 21).
total_patterns_lb("全員","S○N", 13).
patterns("○日日", "○日日").
pattern("○日日", 3).
pattern("○日日", 0, "○").
pattern("○日日", 1, "日").
pattern("○日日", 2, "日").
total_patterns_ub("全員","○日日", 14).
total_patterns_lb("全員","○日日", 10).
patterns("日○日", "日○日").
pattern("日○日", 3).
pattern("日○日", 0, "日").
pattern("日○日", 1, "○").
pattern("日○日", 2, "日").
total_patterns_ub("全員","日○日", 7).
total_patterns_lb("全員","日○日", 4).
patterns("日日日", "日日日").
pattern("日日日", 3).
pattern("日日日", 0, "日").
pattern("日日日", 1, "日").
pattern("日日日", 2, "日").
total_patterns_ub("全員","日日日", 17).
total_patterns_lb("全員","日日日", 11).
patterns("☆○日", "☆○日").
pattern("☆○日", 3).
pattern("☆○日", 0, "☆").
pattern("☆○日", 1, "○").
pattern("☆○日", 2, "日").
total_patterns_ub("全員","☆○日", 1).
total_patterns_lb("全員","☆○日", 0).
patterns("N日S", "N日S").
pattern("N日S", 3).
pattern("N日S", 0, "N").
pattern("N日S", 1, "日").
pattern("N日S", 2, "S").
total_patterns_ub("全員","N日S", 0).
total_patterns_lb("全員","N日S", 0).
patterns("N○日", "N○日").
pattern("N○日", 3).
pattern("N○日", 0, "N").
pattern("N○日", 1, "○").
pattern("N○日", 2, "日").
total_patterns_ub("全員","N○日", 3).
total_patterns_lb("全員","N○日", 1).
patterns("日日N", "日日N").
pattern("日日N", 3).
pattern("日日N", 0, "日").
pattern("日日N", 1, "日").
pattern("日日N", 2, "N").
total_patterns_ub("全員","日日N", 3).
total_patterns_lb("全員","日日N", 2).
patterns("SJP", "SJP").
pattern("SJP", 3).
pattern("SJP", 0, "S").
pattern("SJP", 1, "J").
pattern("SJP", 2, "P").
total_patterns_ub("全員","SJP", 0).
total_patterns_lb("全員","SJP", 0).
% patterns("☆◎N", "☆◎N").
% pattern("☆◎N", 3).
% pattern("☆◎N", 0, "☆").
% pattern("☆◎N", 1, "◎").
% pattern("☆◎N", 2, "N").
% total_patterns_ub("全員","☆◎N", 0).
% total_patterns_lb("全員","☆◎N", 0).
patterns("NNJ", "NNJ").
pattern("NNJ", 3).
pattern("NNJ", 0, "N").
pattern("NNJ", 1, "N").
pattern("NNJ", 2, "J").
total_patterns_ub("全員","NNJ", 5).
total_patterns_lb("全員","NNJ", 2).
patterns("☆○N", "☆○N").
pattern("☆○N", 3).
pattern("☆○N", 0, "☆").
pattern("☆○N", 1, "○").
pattern("☆○N", 2, "N").
total_patterns_ub("全員","☆○N", 1).
total_patterns_lb("全員","☆○N", 0).
patterns("日○N", "日○N").
pattern("日○N", 3).
pattern("日○N", 0, "日").
pattern("日○N", 1, "○").
pattern("日○N", 2, "N").
total_patterns_ub("全員","日○N", 3).
total_patterns_lb("全員","日○N", 2).
patterns("N○N", "N○N").
pattern("N○N", 3).
pattern("N○N", 0, "N").
pattern("N○N", 1, "○").
pattern("N○N", 2, "N").
total_patterns_ub("全員","N○N", 5).
total_patterns_lb("全員","N○N", 3).
patterns("N日日", "N日日").
pattern("N日日", 3).
pattern("N日日", 0, "N").
pattern("N日日", 1, "日").
pattern("N日日", 2, "日").
total_patterns_ub("全員","N日日", 2).
total_patterns_lb("全員","N日日", 1).
patterns("日S○", "日S○").
pattern("日S○", 3).
pattern("日S○", 0, "日").
pattern("日S○", 1, "S").
pattern("日S○", 2, "○").
total_patterns_ub("全員","日S○", 0).
total_patterns_lb("全員","日S○", 0).
patterns("○N☆", "○N☆").
pattern("○N☆", 3).
pattern("○N☆", 0, "○").
pattern("○N☆", 1, "N").
pattern("○N☆", 2, "☆").
total_patterns_ub("全員","○N☆", 0).
total_patterns_lb("全員","○N☆", 0).
patterns("○日S", "○日S").
pattern("○日S", 3).
pattern("○日S", 0, "○").
pattern("○日S", 1, "日").
pattern("○日S", 2, "S").
total_patterns_ub("全員","○日S", 0).
total_patterns_lb("全員","○日S", 0).
patterns("JS☆", "JS☆").
pattern("JS☆", 3).
pattern("JS☆", 0, "J").
pattern("JS☆", 1, "S").
pattern("JS☆", 2, "☆").
total_patterns_ub("全員","JS☆", 0).
total_patterns_lb("全員","JS☆", 0).
patterns("JS○", "JS○").
pattern("JS○", 3).
pattern("JS○", 0, "J").
pattern("JS○", 1, "S").
pattern("JS○", 2, "○").
total_patterns_ub("全員","JS○", 49).
total_patterns_lb("全員","JS○", 38).
patterns("○NJ", "○NJ").
pattern("○NJ", 3).
pattern("○NJ", 0, "○").
pattern("○NJ", 1, "N").
pattern("○NJ", 2, "J").
total_patterns_ub("全員","○NJ", 21).
total_patterns_lb("全員","○NJ", 14).
patterns("○○N", "○○N").
pattern("○○N", 3).
pattern("○○N", 0, "○").
pattern("○○N", 1, "○").
pattern("○○N", 2, "N").
total_patterns_ub("全員","○○N", 12).
total_patterns_lb("全員","○○N", 11).
patterns("○○J", "○○J").
pattern("○○J", 3).
pattern("○○J", 0, "○").
pattern("○○J", 1, "○").
pattern("○○J", 2, "J").
total_patterns_ub("全員","○○J", 4).
total_patterns_lb("全員","○○J", 3).
patterns("日NJ", "日NJ").
pattern("日NJ", 3).
pattern("日NJ", 0, "日").
pattern("日NJ", 1, "N").
pattern("日NJ", 2, "J").
total_patterns_ub("全員","日NJ", 5).
total_patterns_lb("全員","日NJ", 3).
patterns("日N○", "日N○").
pattern("日N○", 3).
pattern("日N○", 0, "日").
pattern("日N○", 1, "N").
pattern("日N○", 2, "○").
total_patterns_ub("全員","日N○", 4).
total_patterns_lb("全員","日N○", 2).

% patterns("○JS○", "○JS○").
% pattern("○JS○", 4).
% pattern("○JS○", 0, "○").
% pattern("○JS○", 1, "J").
% pattern("○JS○", 2, "S").
% pattern("○JS○", 3, "○").
% total_patterns_ub("全員","○JS○", 10).
% total_patterns_lb("全員","○JS○", 6).
% patterns("NJS○", "NJS○").
% pattern("NJS○", 4).
% pattern("NJS○", 0, "N").
% pattern("NJS○", 1, "J").
% pattern("NJS○", 2, "S").
% pattern("NJS○", 3, "○").
% total_patterns_ub("全員","NJS○", 22).
% total_patterns_lb("全員","NJS○", 13).
% patterns("JS○○", "JS○○").
% pattern("JS○○", 4).
% pattern("JS○○", 0, "J").
% pattern("JS○○", 1, "S").
% pattern("JS○○", 2, "○").
% pattern("JS○○", 3, "○").
% total_patterns_ub("全員","JS○○", 13).
% total_patterns_lb("全員","JS○○", 9).
% patterns("JS○J", "JS○J").
% pattern("JS○J", 4).
% pattern("JS○J", 0, "J").
% pattern("JS○J", 1, "S").
% pattern("JS○J", 2, "○").
% pattern("JS○J", 3, "J").
% total_patterns_ub("全員","JS○J", 5).
% total_patterns_lb("全員","JS○J", 3).
% patterns("○日日○", "○日日○").
% pattern("○日日○", 4).
% pattern("○日日○", 0, "○").
% pattern("○日日○", 1, "日").
% pattern("○日日○", 2, "日").
% pattern("○日日○", 3, "○").
% total_patterns_ub("全員","○日日○", 4).
% total_patterns_lb("全員","○日日○", 2).
% patterns("日○日日", "日○日日").
% pattern("日○日日", 4).
% pattern("日○日日", 0, "日").
% pattern("日○日日", 1, "○").
% pattern("日○日日", 2, "日").
% pattern("日○日日", 3, "日").
% total_patterns_ub("全員","日○日日", 3).
% total_patterns_lb("全員","日○日日", 1).
% patterns("日日○日", "日日○日").
% pattern("日日○日", 4).
% pattern("日日○日", 0, "日").
% pattern("日日○日", 1, "日").
% pattern("日日○日", 2, "○").
% pattern("日日○日", 3, "日").
% total_patterns_ub("全員","日日○日", 4).
% total_patterns_lb("全員","日日○日", 1).
% patterns("JS○日", "JS○日").
% pattern("JS○日", 4).
% pattern("JS○日", 0, "J").
% pattern("JS○日", 1, "S").
% pattern("JS○日", 2, "○").
% pattern("JS○日", 3, "日").
% total_patterns_ub("全員","JS○日", 8).
% total_patterns_lb("全員","JS○日", 6).
% patterns("JS○★", "JS○★").
% pattern("JS○★", 4).
% pattern("JS○★", 0, "J").
% pattern("JS○★", 1, "S").
% pattern("JS○★", 2, "○").
% pattern("JS○★", 3, "★").
% total_patterns_ub("全員","JS○★", 1).
% total_patterns_lb("全員","JS○★", 0).
% patterns("S○★☆", "S○★☆").
% pattern("S○★☆", 4).
% pattern("S○★☆", 0, "J").
% pattern("S○★☆", 1, "S").
% pattern("S○★☆", 2, "★").
% pattern("S○★☆", 3, "☆").
% total_patterns_ub("全員","S○★☆", 1).
% total_patterns_lb("全員","S○★☆", 0).
% patterns("★☆○日", "★☆○日").
% pattern("★☆○日", 4).
% pattern("★☆○日", 0, "★").
% pattern("★☆○日", 1, "☆").
% pattern("★☆○日", 2, "○").
% pattern("★☆○日", 3, "日").
% total_patterns_ub("全員","★☆○日", 1).
% total_patterns_lb("全員","★☆○日", 0).
% patterns("日○N○", "日○N○").
% pattern("日○N○", 4).
% pattern("日○N○", 0, "日").
% pattern("日○N○", 1, "○").
% pattern("日○N○", 2, "N").
% pattern("日○N○", 3, "○").
% total_patterns_ub("全員","日○N○", 1).
% total_patterns_lb("全員","日○N○", 0).
% patterns("NJSJ", "NJSJ").
% pattern("NJSJ", 4).
% pattern("NJSJ", 0, "N").
% pattern("NJSJ", 1, "J").
% pattern("NJSJ", 2, "S").
% pattern("NJSJ", 3, "J").
% total_patterns_ub("全員","NJSJ", 5).
% total_patterns_lb("全員","NJSJ", 3).
% patterns("○○日日", "○○日日").
% pattern("○○日日", 4).
% pattern("○○日日", 0, "○").
% pattern("○○日日", 1, "○").
% pattern("○○日日", 2, "日").
% pattern("○○日日", 3, "日").
% total_patterns_ub("全員","○○日日", 6).
% total_patterns_lb("全員","○○日日", 4).
% patterns("日S○○", "日S○○").
% pattern("日S○○", 4).
% pattern("日S○○", 0, "日").
% pattern("日S○○", 1, "S").
% pattern("日S○○", 2, "○").
% pattern("日S○○", 3, "○").
% total_patterns_ub("全員","日S○○", 0).
% total_patterns_lb("全員","日S○○", 0).
% patterns("○NJS○", "○NJS○").
% pattern("○NJS○", 5).
% pattern("○NJS○", 0, "○").
% pattern("○NJS○", 1, "N").
% pattern("○NJS○", 2, "J").
% pattern("○NJS○", 3, "S").
% pattern("○NJS○", 4, "○").
% total_patterns_ub("全員","○NJS○", 14).
% total_patterns_lb("全員","○NJS○", 7).
% patterns("○日JS○", "○日JS○").
% pattern("○日JS○", 5).
% pattern("○日JS○", 0, "○").
% pattern("○日JS○", 1, "日").
% pattern("○日JS○", 2, "J").
% pattern("○日JS○", 3, "S").
% pattern("○日JS○", 4, "○").
% total_patterns_ub("全員","○日JS○", 3).
% total_patterns_lb("全員","○日JS○", 1).
% patterns("JS○JS", "JS○JS").
% pattern("JS○JS", 5).
% pattern("JS○JS", 0, "J").
% pattern("JS○JS", 1, "S").
% pattern("JS○JS", 2, "○").
% pattern("JS○JS", 3, "J").
% pattern("JS○JS", 4, "S").
% total_patterns_ub("全員","JS○JS", 5).
% total_patterns_lb("全員","JS○JS", 3).
% patterns("JSJSJ", "JSJSJ").
% pattern("JSJSJ", 5).
% pattern("JSJSJ", 0, "J").
% pattern("JSJSJ", 1, "S").
% pattern("JSJSJ", 2, "J").
% pattern("JSJSJ", 3, "S").
% pattern("JSJSJ", 4, "J").
% total_patterns_ub("全員","JSJSJ", 0).
% total_patterns_lb("全員","JSJSJ", 0).
% patterns("SJSJS", "SJSJS").
% pattern("SJSJS", 5).
% pattern("SJSJS", 0, "S").
% pattern("SJSJS", 1, "J").
% pattern("SJSJS", 2, "S").
% pattern("SJSJS", 3, "J").
% pattern("SJSJS", 4, "S").
% total_patterns_ub("全員","SJSJS", 0).
% total_patterns_lb("全員","SJSJS", 0).
% patterns("日日○日日", "日日○日日").
% pattern("日日○日日", 5).
% pattern("日日○日日", 0, "日").
% pattern("日日○日日", 1, "日").
% pattern("日日○日日", 2, "○").
% pattern("日日○日日", 3, "日").
% pattern("日日○日日", 4, "日").
% total_patterns_ub("全員","日日○日日", 2).
% total_patterns_lb("全員","日日○日日", 1).
% patterns("○○日日日", "○○日日日").
% pattern("○○日日日", 5).
% pattern("○○日日日", 0, "○").
% pattern("○○日日日", 1, "○").
% pattern("○○日日日", 2, "日").
% pattern("○○日日日", 3, "日").
% pattern("○○日日日", 4, "日").
% total_patterns_ub("全員","○○日日日", 4).
% total_patterns_lb("全員","○○日日日", 2).
% patterns("NNJS○", "NNJS○").
% pattern("NNJS○", 5).
% pattern("NNJS○", 0, "N").
% pattern("NNJS○", 1, "N").
% pattern("NNJS○", 2, "J").
% pattern("NNJS○", 3, "S").
% pattern("NNJS○", 4, "○").
% total_patterns_ub("全員","NNJS○", 3).
% total_patterns_lb("全員","NNJS○", 2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(追加部ここまで)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% 前日・翌日可能シフト
% ============================================================
% pred/succ (copy-paste版)  ★=準夜, ☆=深夜
% 目的:
%   - ★→☆ を許可（準夜→深夜）
%   - 夜勤明け(J/S/★/☆)は休み(○/◎)へ落とせる
%   - J の翌日は S だけに固定しない（○/◎も許可）
%   - S の翌日も J/休み を許可
% ============================================================

% --- 夜勤系の「翌日」(明け) ---
succ_shift("J", "S").
succ_shift("J", "○").
succ_shift("J", "◎").

succ_shift("S", "J").
succ_shift("S", "○").
succ_shift("S", "◎").

% 準夜(★)→深夜(☆)
succ_shift("★", "☆").

% 準夜/深夜の明けは休みへ
succ_shift("★", "○").
succ_shift("★", "◎").
succ_shift("☆", "○").
succ_shift("☆", "◎").


% --- 夜勤系の「前日」(前に置けるもの) ---
% J の前に置けるもの
pred_shift("N", "J").
pred_shift("P", "J").
pred_shift("日", "J").
pred_shift("○", "J").
pred_shift("◎", "J").
pred_shift("S", "J").   % pred_shift("J","S") と同等（Jの前日はSもOK）

% ★ の前に置けるもの（準夜の入り）
pred_shift("N", "★").
pred_shift("P", "★").
pred_shift("日", "★").
pred_shift("○", "★").
pred_shift("◎", "★").

% ☆ の前は ★（準夜→深夜）
pred_shift("★", "☆").


% --- 日勤/通常の前後（元の定義を維持） ---
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

succ_shift("N","N").
pred_shift("N","N").


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
staff_dweek_hard_lb("全員", "日", "平日", 0).
staff_dweek_hard_lb("全員", "日", "土日", 0).
staff_dweek_hard_lb("全員", "日", "祝日", 0).
staff_dweek_soft_lb("全員", "日", "平日", 2).
staff_dweek_soft_lb("全員", "日", "土日", 0).
staff_dweek_soft_lb("全員", "日", "祝日", 0).
staff_dweek_soft_ub("全員", "日", "平日", 7).
staff_dweek_soft_ub("全員", "日", "土日", 0).
staff_dweek_soft_ub("全員", "日", "祝日", 0).
staff_dweek_hard_ub("全員", "日", "平日", 7).
staff_dweek_hard_ub("全員", "日", "土日", 0).
staff_dweek_hard_ub("全員", "日", "祝日", 0).
shifts("N").
shifts("N", "N").
staff_dweek_hard_lb("全員", "N", "平日", 2).
shifts("J").
shifts("J", "J").
staff_dweek_hard_lb("全員", "J", "平日", 2).
shifts("S").
shifts("S", "S").
staff_dweek_hard_lb("全員", "S", "平日", 2).
staff_dweek_hard_lb("全員", "N", "土日", 2).
staff_dweek_hard_lb("全員", "J", "土日", 2).
staff_dweek_hard_lb("全員", "S", "土日", 2).
staff_dweek_hard_lb("全員", "N", "祝日", 2).
staff_dweek_hard_lb("全員", "J", "祝日", 2).
staff_dweek_hard_lb("全員", "S", "祝日", 2).
staff_dweek_soft_lb("全員", "N", "平日", 2).
staff_dweek_soft_lb("全員", "J", "平日", 2).
staff_dweek_soft_lb("全員", "S", "平日", 2).
staff_dweek_soft_lb("全員", "N", "土日", 2).
staff_dweek_soft_lb("全員", "J", "土日", 2).
staff_dweek_soft_lb("全員", "S", "土日", 2).
staff_dweek_soft_lb("全員", "N", "祝日", 2).
staff_dweek_soft_lb("全員", "J", "祝日", 2).
staff_dweek_soft_lb("全員", "S", "祝日", 2).
staff_dweek_soft_ub("全員", "N", "平日", 2).
staff_dweek_soft_ub("全員", "J", "平日", 2).
staff_dweek_soft_ub("全員", "S", "平日", 2).
staff_dweek_soft_ub("全員", "N", "土日", 2).
staff_dweek_soft_ub("全員", "J", "土日", 2).
staff_dweek_soft_ub("全員", "S", "土日", 2).
staff_dweek_soft_ub("全員", "N", "祝日", 2).
staff_dweek_soft_ub("全員", "J", "祝日", 2).
staff_dweek_soft_ub("全員", "S", "祝日", 2).
staff_dweek_hard_ub("全員", "N", "平日", 3).
staff_dweek_hard_ub("全員", "J", "平日", 3).
staff_dweek_hard_ub("全員", "S", "平日", 3).
staff_dweek_hard_ub("全員", "N", "土日", 2).
staff_dweek_hard_ub("全員", "J", "土日", 2).
staff_dweek_hard_ub("全員", "S", "土日", 2).
staff_dweek_hard_ub("全員", "N", "祝日", 2).
staff_dweek_hard_ub("全員", "J", "祝日", 2).
staff_dweek_hard_ub("全員", "S", "祝日", 2).
shifts("★").
shifts("★", "★").
staff_dweek_soft_ub("全員", "★", "平日", 2).
shifts("☆").
shifts("☆", "☆").
staff_dweek_soft_ub("全員", "☆", "平日", 2).
staff_dweek_soft_ub("全員", "★", "土日", 0).
staff_dweek_soft_ub("全員", "☆", "土日", 0).
staff_dweek_soft_ub("全員", "★", "祝日", 0).
staff_dweek_soft_ub("全員", "☆", "祝日", 0).
shifts("P").
shifts("P", "P").
staff_dweek_hard_ub("全員", "P", "平日", 2).
staff_dweek_hard_ub("全員", "P", "土日", 0).
staff_dweek_hard_ub("全員", "P", "祝日", 0).
shifts("N+P+★").
shifts("N+P+★", "N").
shifts("N+P+★", "P").
shifts("N+P+★", "★").
staff_dweek_soft_lb("全員", "N+P+★", "平日", 2).
staff_dweek_soft_lb("全員", "N+P+★", "土日", 2).
staff_dweek_soft_lb("全員", "N+P+★", "祝日", 2).
staff_dweek_hard_lb("熟練", "N", "平日", 0).
staff_dweek_hard_lb("熟練", "J", "平日", 0).
staff_dweek_hard_lb("熟練", "S", "平日", 0).
staff_dweek_hard_lb("熟練", "N", "土日", 0).
staff_dweek_hard_lb("熟練", "J", "土日", 0).
staff_dweek_hard_lb("熟練", "S", "土日", 0).
staff_dweek_hard_lb("熟練", "N", "祝日", 0).
staff_dweek_hard_lb("熟練", "J", "祝日", 0).
staff_dweek_hard_lb("熟練", "S", "祝日", 0).
staff_dweek_soft_lb("熟練", "N", "平日", 1).
staff_dweek_soft_lb("熟練", "J", "平日", 1).
staff_dweek_soft_lb("熟練", "S", "平日", 1).
staff_dweek_soft_lb("熟練", "N", "土日", 1).
staff_dweek_soft_lb("熟練", "J", "土日", 1).
staff_dweek_soft_lb("熟練", "S", "土日", 1).
staff_dweek_soft_lb("熟練", "N", "祝日", 1).
staff_dweek_soft_lb("熟練", "J", "祝日", 1).
staff_dweek_soft_lb("熟練", "S", "祝日", 1).
staff_dweek_soft_ub("熟練", "N", "平日", 2).
staff_dweek_soft_ub("熟練", "J", "平日", 2).
staff_dweek_soft_ub("熟練", "S", "平日", 2).
staff_dweek_soft_ub("熟練", "N", "土日", 2).
staff_dweek_soft_ub("熟練", "J", "土日", 2).
staff_dweek_soft_ub("熟練", "S", "土日", 2).
staff_dweek_soft_ub("熟練", "N", "祝日", 2).
staff_dweek_soft_ub("熟練", "J", "祝日", 2).
staff_dweek_soft_ub("熟練", "S", "祝日", 2).
staff_dweek_hard_ub("熟練", "N", "平日", 3).
staff_dweek_hard_ub("熟練", "J", "平日", 3).
staff_dweek_hard_ub("熟練", "S", "平日", 3).
staff_dweek_hard_ub("熟練", "N", "土日", 3).
staff_dweek_hard_ub("熟練", "J", "土日", 3).
staff_dweek_hard_ub("熟練", "S", "土日", 3).
staff_dweek_hard_ub("熟練", "N", "祝日", 3).
staff_dweek_hard_ub("熟練", "J", "祝日", 3).
staff_dweek_hard_ub("熟練", "S", "祝日", 3).
staff_dweek_soft_lb("師長", "日", "平日", 1).
staff_dweek_soft_lb("師長", "日", "土日", 0).
staff_dweek_soft_lb("師長", "日", "祝日", 0).
staff_dweek_soft_ub("師長", "日", "平日", 1).
staff_dweek_soft_ub("師長", "日", "土日", 0).
staff_dweek_soft_ub("師長", "日", "祝日", 0).
staff_dweek_hard_lb("中堅", "N", "平日", 0).
staff_dweek_hard_lb("中堅", "J", "平日", 0).
staff_dweek_hard_lb("中堅", "S", "平日", 0).
staff_dweek_hard_lb("中堅", "N", "土日", 0).
staff_dweek_hard_lb("中堅", "J", "土日", 0).
staff_dweek_hard_lb("中堅", "S", "土日", 0).
staff_dweek_hard_lb("中堅", "N", "祝日", 0).
staff_dweek_hard_lb("中堅", "J", "祝日", 0).
staff_dweek_hard_lb("中堅", "S", "祝日", 0).
staff_dweek_soft_lb("中堅", "N", "平日", 1).
staff_dweek_soft_lb("中堅", "J", "平日", 1).
staff_dweek_soft_lb("中堅", "S", "平日", 1).
staff_dweek_soft_lb("中堅", "N", "土日", 1).
staff_dweek_soft_lb("中堅", "J", "土日", 1).
staff_dweek_soft_lb("中堅", "S", "土日", 1).
staff_dweek_soft_lb("中堅", "N", "祝日", 1).
staff_dweek_soft_lb("中堅", "J", "祝日", 1).
staff_dweek_soft_lb("中堅", "S", "祝日", 1).
staff_dweek_soft_ub("中堅", "N", "平日", 1).
staff_dweek_soft_ub("中堅", "J", "平日", 1).
staff_dweek_soft_ub("中堅", "S", "平日", 1).
staff_dweek_soft_ub("中堅", "N", "土日", 1).
staff_dweek_soft_ub("中堅", "J", "土日", 1).
staff_dweek_soft_ub("中堅", "S", "土日", 1).
staff_dweek_soft_ub("中堅", "N", "祝日", 1).
staff_dweek_soft_ub("中堅", "J", "祝日", 1).
staff_dweek_soft_ub("中堅", "S", "祝日", 1).
staff_dweek_hard_ub("中堅", "N", "平日", 1).
staff_dweek_hard_ub("中堅", "J", "平日", 1).
staff_dweek_hard_ub("中堅", "S", "平日", 1).
staff_dweek_hard_ub("中堅", "N", "土日", 1).
staff_dweek_hard_ub("中堅", "J", "土日", 1).
staff_dweek_hard_ub("中堅", "S", "土日", 1).
staff_dweek_hard_ub("中堅", "N", "祝日", 1).
staff_dweek_hard_ub("中堅", "J", "祝日", 1).
staff_dweek_hard_ub("中堅", "S", "祝日", 1).
staff_dweek_hard_lb("新人", "J", "平日", 0).
staff_dweek_hard_lb("新人", "S", "平日", 0).
staff_dweek_hard_lb("新人", "J", "土日", 0).
staff_dweek_hard_lb("新人", "S", "土日", 0).
staff_dweek_hard_lb("新人", "J", "祝日", 0).
staff_dweek_hard_lb("新人", "S", "祝日", 0).
staff_dweek_soft_lb("新人", "J", "平日", 0).
staff_dweek_soft_lb("新人", "S", "平日", 0).
staff_dweek_soft_lb("新人", "J", "土日", 0).
staff_dweek_soft_lb("新人", "S", "土日", 0).
staff_dweek_soft_lb("新人", "J", "祝日", 0).
staff_dweek_soft_lb("新人", "S", "祝日", 0).
staff_dweek_soft_ub("新人", "J", "平日", 1).
staff_dweek_soft_ub("新人", "S", "平日", 1).
staff_dweek_soft_ub("新人", "J", "土日", 1).
staff_dweek_soft_ub("新人", "S", "土日", 1).
staff_dweek_soft_ub("新人", "J", "祝日", 1).
staff_dweek_soft_ub("新人", "S", "祝日", 1).
staff_dweek_hard_ub("新人", "J", "平日", 1).
staff_dweek_hard_ub("新人", "S", "平日", 1).
staff_dweek_hard_ub("新人", "J", "土日", 1).
staff_dweek_hard_ub("新人", "S", "土日", 1).
staff_dweek_hard_ub("新人", "J", "祝日", 1).
staff_dweek_hard_ub("新人", "S", "祝日", 1).
vertical_constraint_type("全員","日","人数").
vertical_constraint_type("全員","N","人数").
vertical_constraint_type("全員","J","人数").
vertical_constraint_type("全員","S","人数").
vertical_constraint_type("全員","★","人数").
vertical_constraint_type("全員","☆","人数").
vertical_constraint_type("全員","P","人数").
vertical_constraint_type("全員","N+P+★","人数").
vertical_constraint_type("熟練","N","人数").
vertical_constraint_type("熟練","J","人数").
vertical_constraint_type("熟練","S","人数").
vertical_constraint_type("師長","日","人数").
vertical_constraint_type("中堅","N","人数").
vertical_constraint_type("中堅","J","人数").
vertical_constraint_type("中堅","S","人数").
vertical_constraint_type("新人","J","人数").
vertical_constraint_type("新人","S","人数").
% Horizontal constraints -----------------------------------
% Staff 1:
shifts_soft_lb(1, "N", 0).
shifts_soft_lb(1, "J", 0).
shifts_soft_lb(1, "S", 0).
shifts_soft_ub(1, "N", 0).
shifts_soft_ub(1, "J", 0).
shifts_soft_ub(1, "S", 0).
shifts_hard_ub(1, "N", 0).
shifts_hard_ub(1, "J", 0).
shifts_hard_ub(1, "S", 0).
shifts_hard_lb(1, "★", 0).
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
shifts_hard_ub(1, "A", 0).
% Staff 2:
shifts_soft_lb(2, "N", 2).
shifts_soft_lb(2, "J", 2).
shifts_soft_lb(2, "S", 2).
shifts_soft_ub(2, "N", 2).
shifts_soft_ub(2, "J", 2).
shifts_soft_ub(2, "S", 2).
shifts_hard_lb(2, "★", 0).
shifts_hard_lb(2, "☆", 0).
shifts_soft_lb(2, "★", 0).
shifts_soft_lb(2, "☆", 0).
shifts_soft_ub(2, "★", 2).
shifts_soft_ub(2, "☆", 2).
shifts_hard_ub(2, "★", 2).
shifts_hard_ub(2, "☆", 2).
shifts_hard_lb(2, "P", 0).
shifts_soft_lb(2, "P", 0).
shifts_soft_ub(2, "P", 2).
shifts_hard_ub(2, "P", 2).
shifts_hard_ub(2, "A", 0).
% Staff 3:
shifts_soft_lb(3, "N", 2).
shifts_soft_lb(3, "J", 2).
shifts_soft_lb(3, "S", 2).
shifts_soft_ub(3, "N", 2).
shifts_soft_ub(3, "J", 2).
shifts_soft_ub(3, "S", 2).
shifts_hard_lb(3, "★", 0).
shifts_hard_lb(3, "☆", 0).
shifts_soft_lb(3, "★", 0).
shifts_soft_lb(3, "☆", 0).
shifts_soft_ub(3, "★", 2).
shifts_soft_ub(3, "☆", 2).
shifts_hard_ub(3, "★", 2).
shifts_hard_ub(3, "☆", 2).
shifts_hard_lb(3, "P", 0).
shifts_soft_lb(3, "P", 0).
shifts_soft_ub(3, "P", 2).
shifts_hard_ub(3, "P", 2).
shifts_hard_ub(3, "A", 0).
% Staff 4:
shifts_soft_lb(4, "N", 2).
shifts_soft_lb(4, "J", 2).
shifts_soft_lb(4, "S", 2).
shifts_soft_ub(4, "N", 4).
shifts_soft_ub(4, "J", 4).
shifts_soft_ub(4, "S", 4).
shifts_hard_lb(4, "★", 0).
shifts_hard_lb(4, "☆", 0).
shifts_soft_lb(4, "★", 0).
shifts_soft_lb(4, "☆", 0).
shifts_soft_ub(4, "★", 2).
shifts_soft_ub(4, "☆", 2).
shifts_hard_ub(4, "★", 2).
shifts_hard_ub(4, "☆", 2).
shifts_hard_lb(4, "P", 0).
shifts_soft_lb(4, "P", 0).
shifts_soft_ub(4, "P", 2).
shifts_hard_ub(4, "P", 2).
shifts_hard_ub(4, "A", 0).
% Staff 5:
shifts_soft_lb(5, "N", 0).
shifts_soft_lb(5, "J", 0).
shifts_soft_lb(5, "S", 0).
shifts_soft_ub(5, "N", 0).
shifts_soft_ub(5, "J", 0).
shifts_soft_ub(5, "S", 0).
shifts_hard_lb(5, "★", 0).
shifts_hard_lb(5, "☆", 0).
shifts_soft_lb(5, "★", 0).
shifts_soft_lb(5, "☆", 0).
shifts_soft_ub(5, "★", 2).
shifts_soft_ub(5, "☆", 2).
shifts_hard_ub(5, "★", 2).
shifts_hard_ub(5, "☆", 2).
shifts_hard_lb(5, "P", 0).
shifts_soft_lb(5, "P", 0).
shifts_soft_ub(5, "P", 2).
shifts_hard_ub(5, "P", 2).
shifts_hard_ub(5, "A", 0).
% Staff 6:
shifts_soft_lb(6, "N", 2).
shifts_soft_lb(6, "J", 2).
shifts_soft_lb(6, "S", 2).
shifts_soft_ub(6, "N", 4).
shifts_soft_ub(6, "J", 4).
shifts_soft_ub(6, "S", 4).
shifts_hard_lb(6, "★", 0).
shifts_hard_lb(6, "☆", 0).
shifts_soft_lb(6, "★", 0).
shifts_soft_lb(6, "☆", 0).
shifts_soft_ub(6, "★", 2).
shifts_soft_ub(6, "☆", 2).
shifts_hard_ub(6, "★", 2).
shifts_hard_ub(6, "☆", 2).
shifts_hard_lb(6, "P", 0).
shifts_soft_lb(6, "P", 0).
shifts_soft_ub(6, "P", 2).
shifts_hard_ub(6, "P", 2).
shifts_hard_ub(6, "A", 0).
% Staff 7:
shifts_soft_lb(7, "N", 2).
shifts_soft_lb(7, "J", 2).
shifts_soft_lb(7, "S", 2).
shifts_soft_ub(7, "N", 4).
shifts_soft_ub(7, "J", 4).
shifts_soft_ub(7, "S", 4).
shifts_hard_lb(7, "★", 0).
shifts_hard_lb(7, "☆", 0).
shifts_soft_lb(7, "★", 0).
shifts_soft_lb(7, "☆", 0).
shifts_soft_ub(7, "★", 2).
shifts_soft_ub(7, "☆", 2).
shifts_hard_ub(7, "★", 2).
shifts_hard_ub(7, "☆", 2).
shifts_hard_lb(7, "P", 0).
shifts_soft_lb(7, "P", 0).
shifts_soft_ub(7, "P", 2).
shifts_hard_ub(7, "P", 2).
shifts_hard_ub(7, "A", 0).
% Staff 8:
shifts_soft_lb(8, "N", 0).
shifts_soft_lb(8, "J", 0).
shifts_soft_lb(8, "S", 0).
shifts_soft_ub(8, "N", 0).
shifts_soft_ub(8, "J", 0).
shifts_soft_ub(8, "S", 0).
shifts_hard_lb(8, "★", 0).
shifts_hard_lb(8, "☆", 0).
shifts_soft_lb(8, "★", 0).
shifts_soft_lb(8, "☆", 0).
shifts_soft_ub(8, "★", 2).
shifts_soft_ub(8, "☆", 2).
shifts_hard_ub(8, "★", 2).
shifts_hard_ub(8, "☆", 2).
shifts_hard_lb(8, "P", 0).
shifts_soft_lb(8, "P", 0).
shifts_soft_ub(8, "P", 2).
shifts_hard_ub(8, "P", 2).
shifts_hard_ub(8, "A", 0).
% Staff 9:
shifts_soft_lb(9, "N", 2).
shifts_soft_lb(9, "J", 2).
shifts_soft_lb(9, "S", 2).
shifts_soft_ub(9, "N", 4).
shifts_soft_ub(9, "J", 4).
shifts_soft_ub(9, "S", 4).
shifts_hard_lb(9, "★", 0).
shifts_hard_lb(9, "☆", 0).
shifts_soft_lb(9, "★", 0).
shifts_soft_lb(9, "☆", 0).
shifts_soft_ub(9, "★", 2).
shifts_soft_ub(9, "☆", 2).
shifts_hard_ub(9, "★", 2).
shifts_hard_ub(9, "☆", 2).
shifts_hard_lb(9, "P", 0).
shifts_soft_lb(9, "P", 0).
shifts_soft_ub(9, "P", 2).
shifts_hard_ub(9, "P", 2).
shifts_hard_ub(9, "A", 0).
% Staff 10:
shifts_soft_lb(10, "N", 2).
shifts_soft_lb(10, "J", 2).
shifts_soft_lb(10, "S", 2).
shifts_soft_ub(10, "N", 4).
shifts_soft_ub(10, "J", 4).
shifts_soft_ub(10, "S", 4).
shifts_hard_lb(10, "★", 0).
shifts_hard_lb(10, "☆", 0).
shifts_soft_lb(10, "★", 0).
shifts_soft_lb(10, "☆", 0).
shifts_soft_ub(10, "★", 2).
shifts_soft_ub(10, "☆", 2).
shifts_hard_ub(10, "★", 2).
shifts_hard_ub(10, "☆", 2).
shifts_hard_lb(10, "P", 0).
shifts_soft_lb(10, "P", 0).
shifts_soft_ub(10, "P", 2).
shifts_hard_ub(10, "P", 2).
shifts_hard_ub(10, "A", 0).
% Staff 11:
shifts_soft_lb(11, "N", 2).
shifts_soft_lb(11, "J", 2).
shifts_soft_lb(11, "S", 2).
shifts_soft_ub(11, "N", 4).
shifts_soft_ub(11, "J", 4).
shifts_soft_ub(11, "S", 4).
shifts_hard_lb(11, "★", 0).
shifts_hard_lb(11, "☆", 0).
shifts_soft_lb(11, "★", 0).
shifts_soft_lb(11, "☆", 0).
shifts_soft_ub(11, "★", 2).
shifts_soft_ub(11, "☆", 2).
shifts_hard_ub(11, "★", 2).
shifts_hard_ub(11, "☆", 2).
shifts_hard_lb(11, "P", 0).
shifts_soft_lb(11, "P", 0).
shifts_soft_ub(11, "P", 2).
shifts_hard_ub(11, "P", 2).
shifts_hard_ub(11, "A", 0).
% Staff 12:
shifts_soft_lb(12, "N", 2).
shifts_soft_lb(12, "J", 2).
shifts_soft_lb(12, "S", 2).
shifts_soft_ub(12, "N", 4).
shifts_soft_ub(12, "J", 4).
shifts_soft_ub(12, "S", 4).
shifts_hard_lb(12, "★", 0).
shifts_hard_lb(12, "☆", 0).
shifts_soft_lb(12, "★", 0).
shifts_soft_lb(12, "☆", 0).
shifts_soft_ub(12, "★", 2).
shifts_soft_ub(12, "☆", 2).
shifts_hard_ub(12, "★", 2).
shifts_hard_ub(12, "☆", 2).
shifts_hard_lb(12, "P", 0).
shifts_soft_lb(12, "P", 0).
shifts_soft_ub(12, "P", 2).
shifts_hard_ub(12, "P", 2).
shifts_hard_ub(12, "A", 0).
% Staff 13:
shifts_soft_lb(13, "N", 2).
shifts_soft_lb(13, "J", 2).
shifts_soft_lb(13, "S", 2).
shifts_soft_ub(13, "N", 4).
shifts_soft_ub(13, "J", 4).
shifts_soft_ub(13, "S", 4).
shifts_hard_lb(13, "★", 0).
shifts_hard_lb(13, "☆", 0).
shifts_soft_lb(13, "★", 0).
shifts_soft_lb(13, "☆", 0).
shifts_soft_ub(13, "★", 2).
shifts_soft_ub(13, "☆", 2).
shifts_hard_ub(13, "★", 2).
shifts_hard_ub(13, "☆", 2).
shifts_hard_lb(13, "P", 0).
shifts_soft_lb(13, "P", 0).
shifts_soft_ub(13, "P", 2).
shifts_hard_ub(13, "P", 2).
shifts_hard_ub(13, "A", 0).
% Staff 14:
shifts_soft_lb(14, "N", 2).
shifts_soft_lb(14, "J", 2).
shifts_soft_lb(14, "S", 2).
shifts_soft_ub(14, "N", 4).
shifts_soft_ub(14, "J", 4).
shifts_soft_ub(14, "S", 4).
shifts_hard_lb(14, "★", 0).
shifts_hard_lb(14, "☆", 0).
shifts_soft_lb(14, "★", 0).
shifts_soft_lb(14, "☆", 0).
shifts_soft_ub(14, "★", 2).
shifts_soft_ub(14, "☆", 2).
shifts_hard_ub(14, "★", 2).
shifts_hard_ub(14, "☆", 2).
shifts_hard_lb(14, "P", 0).
shifts_soft_lb(14, "P", 0).
shifts_soft_ub(14, "P", 2).
shifts_hard_ub(14, "P", 2).
shifts_hard_ub(14, "A", 0).
% Staff 15:
shifts_soft_lb(15, "N", 2).
shifts_soft_lb(15, "J", 2).
shifts_soft_lb(15, "S", 2).
shifts_soft_ub(15, "N", 3).
shifts_soft_ub(15, "J", 3).
shifts_soft_ub(15, "S", 3).
shifts_hard_lb(15, "★", 0).
shifts_hard_lb(15, "☆", 0).
shifts_soft_lb(15, "★", 0).
shifts_soft_lb(15, "☆", 0).
shifts_soft_ub(15, "★", 2).
shifts_soft_ub(15, "☆", 2).
shifts_hard_ub(15, "★", 2).
shifts_hard_ub(15, "☆", 2).
shifts_hard_lb(15, "P", 0).
shifts_soft_lb(15, "P", 0).
shifts_soft_ub(15, "P", 2).
shifts_hard_ub(15, "P", 2).
shifts_hard_ub(15, "A", 0).
% Staff 16:
shifts_soft_lb(16, "N", 2).
shifts_soft_lb(16, "J", 2).
shifts_soft_lb(16, "S", 2).
shifts_soft_ub(16, "N", 3).
shifts_soft_ub(16, "J", 3).
shifts_soft_ub(16, "S", 3).
shifts_hard_lb(16, "★", 0).
shifts_hard_lb(16, "☆", 0).
shifts_soft_lb(16, "★", 0).
shifts_soft_lb(16, "☆", 0).
shifts_soft_ub(16, "★", 2).
shifts_soft_ub(16, "☆", 2).
shifts_hard_ub(16, "★", 2).
shifts_hard_ub(16, "☆", 2).
shifts_hard_lb(16, "P", 0).
shifts_soft_lb(16, "P", 0).
shifts_soft_ub(16, "P", 2).
shifts_hard_ub(16, "P", 2).
shifts_hard_ub(16, "A", 0).
% Staff 17:
shifts_soft_lb(17, "N", 2).
shifts_soft_lb(17, "J", 2).
shifts_soft_lb(17, "S", 2).
shifts_soft_ub(17, "N", 3).
shifts_soft_ub(17, "J", 3).
shifts_soft_ub(17, "S", 3).
shifts_hard_lb(17, "★", 0).
shifts_hard_lb(17, "☆", 0).
shifts_soft_lb(17, "★", 0).
shifts_soft_lb(17, "☆", 0).
shifts_soft_ub(17, "★", 2).
shifts_soft_ub(17, "☆", 2).
shifts_hard_ub(17, "★", 2).
shifts_hard_ub(17, "☆", 2).
shifts_hard_lb(17, "P", 0).
shifts_soft_lb(17, "P", 0).
shifts_soft_ub(17, "P", 2).
shifts_hard_ub(17, "P", 2).
shifts_hard_ub(17, "A", 0).
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
