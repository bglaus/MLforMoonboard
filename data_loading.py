import pandas as pd

DATA_PATH = './data/2017.json'

FONT_SCALE = {
    '8B+': 14,     
    '8B' : 13,
    '8A+': 12,
    '8A' : 11,
    '7C+': 10,
    '7C' : 9,
    '7B+': 8,
    '7B' : 7, 
    '7A+': 6,
    '7A' : 5, 
    '6C+': 4,
    '6C' : 3,
    '6B+': 2,
    '6B' : 1,
    '6A+': 0,
}

V_GRADE = {
    'V13': 10,
    'V12': 9,
    'V11': 8,
    'V10': 7,
    'V9':  6,
    'V8':  5,
    'V7':  4,
    'V6':  3,
    'V5':  2,
    'V4':  1,
    'V3':  0,
}

FONT_SCALE_TO_V_GRADE = {
    '8B+': 'V13',     
    '8B' : 'V12',
    '8A+': 'V11',
    '8A' : 'V10',
    '7C+':  'V9',
    '7C' :  'V8',
    '7B+':  'V8',
    '7B' :  'V7', 
    '7A+':  'V6',
    '7A' :  'V5', 
    '6C+':  'V5',
    '6C' :  'V4',
    '6B+':  'V4',
    '6B' :  'V3',
    '6A+':  'V3',
}


def load_dataframe(path=DATA_PATH):   
    return pd.read_json(path)

def add_font_scale_column(df):
    df['font_scale'] = df.Grade.apply(lambda x: FONT_SCALE[x])
    return df

def add_v_grade(df):
    df['v_grade_str'] = df.Grade.apply(lambda x: FONT_SCALE_TO_V_GRADE[x])
    df['v_grade'] = df.v_grade_str.apply(lambda x: V_GRADE[x])
    return df

