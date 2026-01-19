% 優先度付き巨大近傍探索（探索開始時に初期値として過去の割り当てを引き継ぐ）
%                            if                           [priority lv, modifier] (default priority lv is 0)
#heuristic assigned(N, D, S) : legacy(assigned(N, D, S)). [1, true]

% シフトの固定要求
:- fixed(assigned(N, D, S1)), ext_assigned(N, D, S2), S1 != S2.
%:- fixed(assigned(N, D, S)), not ext_assigned(N, D, S).

