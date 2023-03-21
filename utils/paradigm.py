import pandas as pd 

from public_analysis_code.ibc_public.utils_paradigm import make_paradigm

preference_dict = {
    'face_too-slow': 'face',
    'food_too-slow': 'food',
    'house_too-slow': 'house',
    'painting_too-slow': 'painting'                   
}

def create_event_file(event_fp, task):
    # Create event dataframe
    # Get task string from task metadata
    task_str = task.split('-')[1]
    if task_str[:10] == 'Preference':
        df_events = pd.read_csv(event_fp, delimiter='\t')
        df_events['trial_type'] = df_events['trial_type'].replace(preference_dict)
    else:
        df_events = make_paradigm(event_fp, task_str)
    
    # Special processing for certain tasks
    if task_str == 'HcpWm':
        replace_dict = {'0': 'Zero', '2': 'Two'}
        df_events['trial_type'] = \
        df_events.trial_type.str.strip().replace(replace_dict, regex=True)
    elif task_str == 'MVEB':
        replace_dict = {'2': 'Two', '4': 'Four', '6': 'Six'}
        df_events['trial_type'] = \
        df_events.trial_type.str.strip().replace(replace_dict, regex=True)
        df_events = df_events.loc[df_events.trial_type != 'p_startup']
    elif task_str == 'MVIS':
        replace_dict = {'2': 'Two', '4': 'Four', '6': 'Six'}
        df_events['trial_type'] = \
        df_events.trial_type.str.strip().replace(replace_dict, regex=True)
        df_events = df_events.loc[df_events.trial_type != 'p_startup']
    elif task_str == 'Audi':
        df_events = df_events.groupby(['onset', 'trial_type']).first().reset_index()

    # append task to trial type string
    df_events['trial_type'] = f'{task_str}_' + df_events.trial_type

    return df_events[['onset', 'duration', 'trial_type']]




