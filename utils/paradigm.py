import pandas as pd 

from public_analysis_code.ibc_public.utils_paradigm import make_paradigm

def block_duration(x):
    # Compute duration of grouped trials 
    return (x['onset'].iloc[-1] + x['duration'].iloc[-1]) - x['onset'].iloc[0]


def block_agg(x, type_indx=0):
    # Group trials into blocks
    agg_dict = {
        'onset': x['onset'].iloc[0],
        'duration': block_duration(x),
        'trial_type': x['trial_type'].iloc[type_indx]
    }
    return agg_dict


def create_event_file(event_fp, task):
    # Create event dataframe
    # Get task string from task metadata
    task_str = task.split('-')[1]
    df_events = make_paradigm(event_fp, task_str)
    if task_str == 'HcpWm':
        df_events['trial_type'] = df_events.trial_type.str.replace('0','Zero').str.replace('2','Two')
        df_events['block'] = (df_events.trial_type != df_events.trial_type.shift()).cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x)))
    elif task_str == 'HcpLanguage':
        df_events = df_events.loc[df_events.trial_type != 'dummy'].copy()
    elif task_str == 'HcpGambling':
        df_events['block'] = (df_events.onset.diff() > 4).cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x)))
        df_events['trial_type'] = 'gambling'
    elif task_str == 'HcpMotor':
        df_events['block'] = (df_events.trial_type =='cue').cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x, type_indx=1)))
    elif task_str == 'HcpEmotion':
        df_events['block'] = (df_events.trial_type != df_events.trial_type.shift()).cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x)))
    elif task_str == 'HcpRelational':
        block_ind = df_events.trial_type != df_events.trial_type.shift()
        block_ind += df_events.onset.diff() > 20
        df_events['block'] = block_ind.cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x)))
    elif task_str == 'RSVPLanguage':
        df_events = make_paradigm(event_fp, task_str)
        df_events['block'] = (df_events.trial_type !='probe').cumsum()
        df_events = df_events.groupby('block').apply(lambda x: pd.Series(block_agg(x)))
        

    return df_events




