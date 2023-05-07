import pandas as pd
from config import BOARD

DATA_PATH = f'./data/{BOARD}.json'

# Numerical representation of the font scale.  
# Numbers don't match to V grade, use FONT_SCALE_TO_V_GRADE to convert grades.
FONT_SCALE = {
    '8B+': 15,     
    '8B' : 14,
    '8A+': 13,
    '8A' : 12,
    '7C+': 11,
    '7C' : 10,
    '7B+': 9,
    '7B' : 8, 
    '7A+': 7,
    '7A' : 6, 
    '6C+': 5,
    '6C' : 4,
    '6B+': 3,
    '6B' : 2,
    '6A+': 1,
    '6A' : 0,
}

# Numerical representation of V grades. 
# Numbers don't match to font scale, use inverse of FONT_SCALE_TO_V_GRADE to convert grades.
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

# Turn Font scale into matching V grade.
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
    '6A' :  'V3'
}


def load_dataframe(path=DATA_PATH):   
    '''Load pandas datafrom from a json file defined at path.'''
    df = pd.read_json(path)
    df = df.rename(columns = {'Grade' : 'font_scale', 'Moves' : 'holds', 'UserRating' : 'user_rating'})
    return df

def add_font_scale(df):
    '''Add an additional column to the dataframe containing a numerical representation of the font scale.'''
    df['font_scale_int'] = df.font_scale.apply(lambda x: FONT_SCALE[x])
    return df

def add_v_grade(df):
    '''Adds two new columns, v_grade (e.g. V4) and numerical representation of the v_grade (e.g. 1)'''
    df['v_grade'] = df.font_scale.apply(lambda x: FONT_SCALE_TO_V_GRADE[x])
    df['v_grade_int'] = df.v_grade.apply(lambda x: V_GRADE[x])
    return df

