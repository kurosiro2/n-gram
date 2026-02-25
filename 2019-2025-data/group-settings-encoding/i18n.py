import re

TIME_REFORMER_SHIFT_MAP = {
    "日勤"                   : {"ja": "日", "en": "D" }, # Day
    "長日勤1"                : {"ja": "N",  "en": "LD"}, # Long Day
    "長日勤2"                : {"ja": "N",  "en": "LD"}, # Long Day
    "準夜１"                 : {"ja": "★", "en": "E" }, # Evening
    "準夜３"                 : {"ja": "★", "en": "E" }, # Evening
    "深夜１"                 : {"ja": "☆", "en": "N" }, # Night
    "2交代夜勤入り"          : {"ja": "★", "en": "E" }, # Evening
    "2交代夜勤明け"          : {"ja": "☆", "en": "N" }, # Night
    "短準夜1"                : {"ja": "J",  "en": "SE"}, # Short Evening
    "短準夜2"                : {"ja": "J",  "en": "SE"}, # Short Evening
    "短深夜"                 : {"ja": "S",  "en": "SN"}, # Short Night
    "週休"                   : {"ja": "○",  "en": "WR"}, # Weekly Rest
    "祝祭日"                 : {"ja": "◎",  "en": "PH"}, # Public Holiday
    "年次休暇"               : {"ja": "年", "en": "AL"}, # Annual Leave
    "午前年"                 : {"ja": "年", "en": "AL"}, # Annual Leave, 過去にしか現れないので年休扱いとする
    "午後年"                 : {"ja": "年", "en": "AL"}, # Annual Leave, 過去にしか現れないので年休扱いとする
    "Ａ１"                   : {"ja": "A",  "en": "EM"}, # Early Morning
    "Ａ２"                   : {"ja": "A",  "en": "EM"}, # Early Morning
    "Ｐ１"                   : {"ja": "P",  "en": "LM"}, # Late Morning
    "Ｐ２"                   : {"ja": "P",  "en": "LM"}, # Late Morning
    "特別休暇"               : {"ja": "特", "en": "SP"}, # Special Leave
    "特別休暇　コロナ"       : {"ja": "特", "en": "SP"}, # Special Leave
    "健康増進休暇"           : {"ja": "健", "en": "HL"}, # Health Leave
    "結婚"                   : {"ja": "婚", "en": "WL"}, # Wedding Leave
    "産休"                   : {"ja": "産", "en": "ML"}, # Maternity Leave
    "育休"                   : {"ja": "育", "en": "PL"}, # Parental Leave
    "病休"                   : {"ja": "病", "en": "SL"}, # Sick Leave
    "看護休暇"               : {"ja": "看", "en": "NL"}, # Nursing Leave
    "忌引"                   : {"ja": "忌", "en": "BL"}, # Bereavement Leave
    "ボランティア休暇"       : {"ja": "ボ", "en": "VL"}, # Volunteer Leave
    "人間ドック"             : {"ja": "ド", "en": "HC"}, # Health Check
    "出張"                   : {"ja": "出", "en": "BT"}, # Business Trip
    "研修"                   : {"ja": "研", "en": "TR"}, # Training
    "欠勤"                   : {"ja": "欠", "en": "AB"}, # Absence
    "休職"                   : {"ja": "休", "en": "LA"}, # Leave of Absence
    "日勤(申し送り除外なし)" : {"ja": "日", "en": "D" }, # Day
    "Ｆ1(申し送り除外なし)"  : {"ja": "日", "en": "D" }, # Day
    "Ｆ2(申し送り除外なし)"  : {"ja": "日", "en": "D" }, # Day
    "Ｐ３"                   : {"ja": "P",  "en": "LM"}, # Late Morning
    "Ｐ4"                    : {"ja": "P",  "en": "LM"}, # Late Morning
    "管理師長Ｎ1"            : {"ja": "N",  "en": "LD"}, # Long Day
    "管理師長Ｎ1(休日用)"    : {"ja": "N",  "en": "LD"}, # Long Day
    "管理師長Ｎ2"            : {"ja": "N",  "en": "LD"}, # Long Day
    "管理師長Ｎ2(休日用)"    : {"ja": "N",  "en": "LD"}, # Long Day
    "管理師長Ｊ1"            : {"ja": "J",  "en": "SE"}, # Short Evening
    "管理師長Ｊ2"            : {"ja": "J",  "en": "SE"}, # Short Evening
    "管理師長Ｓ"             : {"ja": "S",  "en": "SN"}, # Short Night
    "/"                      : {"ja": "/",  "en": "/"},  # Not Available
}

SHORT_SHIFT_MAP = {}
for key, value in TIME_REFORMER_SHIFT_MAP.items():
    ja_value = value["ja"]
    en_value = value["en"]
    SHORT_SHIFT_MAP[ja_value] = {"ja": ja_value, "en": en_value }

# 未対応のシフト（数字は2019-2024年度の出現回数）
"""
      1 11時間年休
      1 初期救急4時間（20：00-24：00）
      1 初期救急6時間（18：00-24：00）
      3 管理準夜
      3 管理深夜
      9 勤務未定
     36 学生補助者　4時間(休憩なし)14：00～
     44 8：30～看護部長・副看護部長日勤
    195 初期救急3時間（17：30-20：30）
    312 初期救急4時間（19：30-23：30）
    320 初期救急6時間（17：30-23：30）
    360 管理日直（看護部日勤）
    779 復帰支援日勤１
    948 2交代管理夜勤入り
    948 2交代管理夜勤明け
   1216
   1545 午前年
   2999 管理日勤
   3798 8：00～看護部長・副看護部長日勤
   3946 午後年
   6466 復帰支援日勤２
"""
# 現在の懸念点
# * 午前年，午後年はある程度の頻度で出現する
#   - 過去のシフト担当回数に若干の影響あるが，無視できるかも
#   - 気になるのは他シフトとの接続関係（たとえば 午前年-J などがありうる（許されるのか））
# * 未対応のシフトが出現した場合どうするか？

JOB_MAP = {
    "師長"      : {"ja": "師長"    , "en": "Chief Nurse"     },
    "看護師長"  : {"ja": "師長"    , "en": "Chief Nurse"     },
    "副師長"    : {"ja": "副師長"  , "en": "Deputy Chief Nurse" },
    "副看護師長": {"ja": "副師長"  , "en": "Deputy Chief Nurse" },
    "看護師"    : {"ja": "看護師"  , "en": "Nurse"           },
    "助産師"    : {"ja": "助産師"  , "en": "Midwife"         },
}

GROUP_MAP = {
    "全員"        : {"ja": "全員"        , "en": "All"            },
    "師長"        : {"ja": "師長"        , "en": "Heads"          },
    "リーダー"     : {"ja": "リーダー"    , "en": "Leaders"          },
    "熟練"        : {"ja": "熟練"        , "en": "Seniors"        },
    "熟練１"      : {"ja": "熟練１"      , "en": "Seniors1"       },
    "熟練２"      : {"ja": "熟練２"      , "en": "Seniors2"       },
    "中堅"        : {"ja": "中堅"        , "en": "Mid-levels"     },
    "若手"        : {"ja": "若手"        , "en": "Juniors"        },
    "新人"        : {"ja": "新人"        , "en": "Newcomers"      },
    "新人１"      : {"ja": "新人１"      , "en": "Newcomers1"     },
    "新人２"      : {"ja": "新人２"      , "en": "Newcomers2"     },
    "新採用者"    : {"ja": "新採用者"    , "en": "New hires"      },
    "夜勤"        : {"ja": "夜勤"        , "en": "Night"          },
    "土日代行可"  : {"ja": "土日代行可"  , "en": "Weekend Leaders"},
    "日勤リーダー可": {"ja": "日勤リーダー可", "en": "Day Leaders"    },
    "夜勤リーダー可": {"ja": "夜勤リーダー可", "en": "Night Leaders"  },
    "若手以上"    : {"ja": "若手以上"    , "en": "Juniors+"       },
    "新人以上"    : {"ja": "新人以上"    , "en": "Newcomers+"     },
}

DWEEK_MAP = {
    "日": {"ja": "日", "en": "Su"},
    "月": {"ja": "月", "en": "Mo"},
    "火": {"ja": "火", "en": "Tu"},
    "水": {"ja": "水", "en": "We"},
    "木": {"ja": "木", "en": "Th"},
    "金": {"ja": "金", "en": "Fr"},
    "土": {"ja": "土", "en": "Sa"},
    "祝": {"ja": "祝", "en": "Ho"},
}

DWEEK_TYPE_MAP = {
    "平日": {"ja": "平日", "en": "Weekday"},
    "土日": {"ja": "土日", "en": "Weekend"},
    "祝日": {"ja": "祝日", "en": "Holiday"},
}

VERTICAL_TARGET_TYPE_MAP = {
    "人数": {"ja": "人数", "en": "Staffs"},
    "点数": {"ja": "点数", "en": "Points"},
}

def map_time_reformer_shift(shift, locale):
    if shift not in TIME_REFORMER_SHIFT_MAP:
        raise KeyError(f"Shift '{shift}' doesn't exists in the translation mapping")
    return TIME_REFORMER_SHIFT_MAP[shift][locale[:2]]

# グループを翻訳する関数
def map_group(group, locale="ja"):
    """
    グループ名称を指定したロケールに基づいて変換する。

    Parameters:
        group (str): 元のグループ名称。
        locale (str): "jp" または "en"。

    Returns:
        str: 翻訳後のグループ名称。
    """
    if group not in GROUP_MAP:
        raise KeyError(f"Nurse group '{group}' doesn't exists in the translation mapping")
    return GROUP_MAP[group][locale[:2]]

# 職名を翻訳する関数
def map_job(job, locale="ja"):
    """
    職名を指定したロケールに基づいて変換する。

    Parameters:
        group (str): 元の職名
        locale (str): "jp" または "en"。

    Returns:
        str: 翻訳後の職名
    """
    if job not in JOB_MAP:
        raise KeyError(f"Job name '{job}' doesn't exists in the translation mapping")
    return JOB_MAP[job][locale[:2]]

# シフトを翻訳する関数
def map_shift(shift, locale="ja"):
    """
    シフト名称を指定したロケールに基づいて変換する。

    Parameters:
        shift (str): 元のシフト名称。
        locale (str): "jp" または "en"。

    Returns:
        str: 翻訳後のシフト名称。
    """
    if shift not in SHORT_SHIFT_MAP:
        raise KeyError(f"Shift '{shift}' doesn't exists in the translation mapping")
    return SHORT_SHIFT_MAP[shift][locale[:2]]

def map_shift_patterns_name(patterns, locale):
    if locale[:2] == "ja":
        return patterns
    patterns_name = [ map_shift_pattern_name(pattern, locale) for pattern in patterns.split("+")]
    if len(patterns_name) == 1:
        return patterns_name[0]
    return '+'.join([f"({name})" for name in patterns_name])

def map_shift_pattern_name(pattern, locale):
    if locale[:2] == "ja":
        return pattern
    return '-'.join(map_shift(shift_char, locale) for shift_char in pattern)

def map_shift_group_name(shift_group, locale):
    return '+'.join([map_shift(shift, locale) for shift in shift_group.split("+")])

def map_shift_def(shift_def, locale="ja"):
    org_shift_def = shift_def
    org_shift_groups = re.split(r'[,，]', org_shift_def)
    new_shift_groups = []
    for org_shift_group in org_shift_groups:
        org_shifts = re.split(r'[+＋]', org_shift_group)
        new_shifts = [map_shift(s, locale) for s in org_shifts]
        new_shift_groups.append("+".join(new_shifts))
    new_shift_def = ",".join(new_shift_groups)
    return new_shift_def

def map_dweek(dweek, locale="ja"):
    if dweek not in DWEEK_MAP:
        raise KeyError(f"Dweek '{dweek}' doesn't exists in the translation mapping")
    return DWEEK_MAP[dweek][locale[:2]]

def map_dweek_type(dweek_type, locale="ja"):
    if dweek_type not in DWEEK_TYPE_MAP:
        raise KeyError(f"Dweek type '{dweek_type}' doesn't exists in the translation mapping")
    return DWEEK_TYPE_MAP[dweek_type][locale[:2]]

def map_vertical_target_type(target_type, locale="ja"):
    if target_type not in VERTICAL_TARGET_TYPE_MAP:
        raise KeyError(f"Vertical target type '{target_type}' doesn't exists in the translation mapping")
    return VERTICAL_TARGET_TYPE_MAP[target_type][locale[:2]]
