import pandas as pd
import json
import re

# Receives a dataframe and returns a dataframe with only the columns specified in the columns parameter
def select_columns(df, columns):
    return df[columns]

# Receives a 1-5 rating and maps it to a 1-3 rating
def map_questionnaire_answer(level, dataset, feature):
    if dataset == 'fbk':
        if feature == 'stress':
            return map_stress_fbk_questionnaire_answer(level)
    elif dataset == 'studentlife':
        if feature == 'stress':
            return map_stress_sl_questionnaire_answer(level)
        elif feature == 'experience':
            return map_experience_sl_questionnaire_answer(level)
        elif feature == 'hours':
            return map_hours_sl_questionnaire_answer(level)
        elif feature == 'sleep_rate':
            return map_sleep_rate_sl_questionnaire_answer(level)


def map_sleep_rate_sl_questionnaire_answer(level):
    level = int(level)
    level -= 5
    return abs(level)

def map_hours_sl_questionnaire_answer(level):
    level = int(level)
    if level > 8:
        return int(3)
    elif level < 5:
        return int(1)
    else:
        return int(2)
    

def map_experience_sl_questionnaire_answer(level):
    level = int(level)
    if level > 3:
        return int(1)
    elif level == 1:
        return int(2)
    else:
        return int(3)

# Maps a 1-5 rating to a 1-3 rating
def map_stress_sl_questionnaire_answer(level):
    level = int(level)
    if level > 3:
        return int(1)
    elif level == 1:
        return int(2)
    elif level == 2 or level == 3:
        return int(3)

# Maps a 1-5 rating to a 1-3 rating
def map_stress_fbk_questionnaire_answer(level):
    level = int(level)
    if level > 3:
        return int(3)
    elif level == 3:
        return int(2)
    elif level < 3:
        return int(1)

# Parses a row of the questionnaire answers and returns the mapped answers 
def parse_fbk_answers(row):
    answers = json.loads(row)
    stress = int(re.search('Rating: (.*) out of 5', answers['what_is_your_stress_level']).group(1))
    effective = re.search('Rating: (.*) out of 5', answers['Effective_Condition']).group(1)
    effort = int(re.search('Rating: (.*) out of 5', answers['It_takes_me_effort']).group(1))
    current_activity = re.search('Rating: (.*) out of 5', answers['my_current_activity']).group(1)
    friendly = re.search('Rating: (.*) out of 5', answers['Friendly_Condition']).group(1)
    angry = re.search('Rating: (.*) out of 5', answers['Angry_Condition']).group(1)
    tense = re.search('Rating: (.*) out of 5', answers['Tense_Condition']).group(1)
    energetic = re.search('Rating: (.*) out of 5', answers['Energetic_Condition']).group(1)
    anxious = re.search('Rating: (.*) out of 5', answers['Anxious_Condition']).group(1)
    cheerfull = re.search('Rating: (.*) out of 5', answers['Cheerfull_Condition']).group(1)
    sad = re.search('Rating: (.*) out of 5', answers['Sad_Condition']).group(1)
    try:
        sleep = int(re.search('Rating: (.*) out of 5', answers['how_did_you_sleep_tonight']).group(1))
    except:
        sleep = float('nan')
    something_good = re.search('Rating: (.*) out of 5', answers['This_is_something_I_am_good_at']).group(1)
    something_else = re.search('Rating: (.*) out of 5', answers['I_would_rather_do_something_else']).group(1)
    return stress, sleep, something_good, something_else, effective, effort, current_activity, energetic

# Receives a dataframe with data specified in 1 second intervals and converts it to minute intervals
def convert_to_minutely(df):
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df['date'] = df.index
    df['day'] = df['date'].dt.floor('T')
    df.set_index('day', inplace=True)
    #df.drop(['date'], axis = 1, inplace=True)
    #df = df[~df.index.duplicated(keep='last')]
    return df

# Receives a dataframe with data specified in 1 minute intervals and converts it to hourly intervals
def convert_to_hourly(df):
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df['date'] = df.index
    df['date'] = df['date'].dt.floor('H')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df

# loads the data and drops the unused columns
def load_sl_data_and_drop_unused(data, additional_columns_to_del=[]):
    data = pd.DataFrame(data)
    data.drop(['null', 'location']+additional_columns_to_del, axis = 1, inplace=True, errors='ignore')
    data.dropna(how='any', inplace=True)
    return data

def sl_set_date_index(data):
    data['date'] = pd.to_datetime(data['resp_time'], unit='s')
    data.set_index('date', inplace=True)
    data = convert_to_hourly(data)
    return data


def get_sl_stress_data(data):
    data = load_sl_data_and_drop_unused(data)
    if 'level' not in data.columns:
        return None
    data = sl_set_date_index(data)
    data['stress'] = data.apply(lambda x: map_stress_sl_questionnaire_answer(x['level']), axis=1)
    data.drop(['level', 'resp_time'], axis = 1, inplace=True, errors='ignore')
    return data

def get_sl_class_data(data):
    data = load_sl_data_and_drop_unused(data, ['course_id', 'due', 'enjoyed_class'])
    if 'hours' not in data.columns or 'experience' not in data.columns:
        return None
    data = sl_set_date_index(data)
    data['hours_worked'] = data['hours']
    data['enjoyed_class'] = data.apply(lambda x: map_questionnaire_answer(x['experience'], 'studentlife', 'experience'), axis=1)

    data.drop(['resp_time', 'due', 'hours', 'experience'], axis = 1, inplace=True, errors='ignore')
    return data

def get_sl_class2_data(data):
    data = load_sl_data_and_drop_unused(data, ['grade'])
    if 'challenge' not in data.columns or 'effort' not in data.columns: # effort y challenges -> 1 = máximo de esfuerzo, 6 = mínimo de esfuerzo
        return None
    data = sl_set_date_index(data)
    data.drop(['resp_time'], axis = 1, inplace=True, errors='ignore')
    return data

def get_sl_lab_data(data):
    data = load_sl_data_and_drop_unused(data)
    if 'duration' not in data.columns or 'enjoy' not in data.columns:
        return None
    data = sl_set_date_index(data)
    data['lab_duration'] = data['duration']
    data['lab_enjoy'] =  data['enjoy'] # inversa
    data.drop(['resp_time'], axis = 1, inplace=True, errors='ignore')
    return data

def get_sl_sleep_data(data):
    data = load_sl_data_and_drop_unused(data, ['social'])
    if 'rate' not in data.columns or 'hour' not in data.columns:
        return None
    data = sl_set_date_index(data)
    data['sleep_hours'] = data['hour']
    data['sleep_rate'] =  data.apply(lambda x: map_questionnaire_answer(x['rate'], 'studentlife', 'sleep_rate'), axis=1) # data['rate'] # inversa
    data.drop(['resp_time', 'hour', 'rate'], axis = 1, inplace=True, errors='ignore')
    return data

def get_sl_social_data(data):
    data = load_sl_data_and_drop_unused(data)
    if 'number' not in data.columns:
        return None
    data = sl_set_date_index(data)
    data['no_persons_contacted'] = data['number']
    data.drop(['resp_time', 'number'], axis = 1, inplace=True, errors='ignore')
    return data


def filter_by_week(data, dataset, strategy='balanced'):
    data['week'] = data.index.isocalendar().week

    if strategy == 'balanced':
        if dataset == 'fbk':
            data = data[(data['week'] >= 45) & (data['week'] <= 51)]
        elif dataset == 'sl':
            data = data[(data['week'] >= 13) & (data['week'] <= 19)]
    elif strategy == 'all':
        pass
    elif strategy == 'reliable':
        if dataset == 'fbk':
            data = data[(data['week'] >= 46) & (data['week'] <= 50)]
        elif dataset == 'sl':
            data = data[(data['week'] >= 13) & (data['week'] <= 16)]
    
    del data['week']
    return data

def get_consecutive_voice(df, threshold=2):
    df1 = df.unstack().rename_axis(('col','idx')).reset_index(name='val')
    m = df1['val'].eq(True)
    g = (df1['val'] != df1.groupby('col')['val'].shift()).cumsum()
    mask = g.groupby(g).transform('count').ge(threshold) & m
    return (df1[mask].groupby([df1['col'], g])['idx']
                    .agg(['first','last'])
                    .reset_index(level=1, drop=True)
                    .reset_index())

def get_voice_data(data):
    data.index = pd.to_datetime(data.index)
    data.drop(['SSID', 'MAC', 'Power', 'Answer'], axis = 1, inplace=True)
    data.dropna(how='all', inplace=True)
    data.sort_index(inplace=True)
    filtered_df = data[['Voice']].pipe(get_consecutive_voice, threshold=2)
    filtered_df['duration'] = (filtered_df['last'] - filtered_df['first']).dt.total_seconds()
    filtered_df.set_index('first', inplace=True)
    data['duration'] = filtered_df['duration']
    return data

def get_sl_voice_data(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['start_timestamp'], unit='s')
    data['duration'] = (data[' end_timestamp'] - data['start_timestamp'])
    return data