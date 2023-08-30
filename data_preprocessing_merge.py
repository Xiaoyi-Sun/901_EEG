# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:09
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : data_preprocessing_merge.py
# @Software: PyCharm


import pandas as pd
import os


# %% Create functions to import and process files.
def get_raw_file(path):
    raw_file_in = pd.read_csv(path)

    ind = raw_file_in[raw_file_in.iloc[:, 0] == 'Row'].index

    # look at the starting lines
    new_header = raw_file_in.iloc[ind].values[0]

    # remove the extra lines
    raw_file_in = raw_file_in.iloc[(ind + 1)[0]:]

    # set new header
    raw_file_in.columns = new_header
    return raw_file_in


def modify_respondent(respondent):
    if '' in respondent and respondent.count('') >= 2:
        parts = respondent.split('_')
        if len(parts[0]) <= 3 and parts[1].startswith('group'):
            return '_'.join(parts[1:])
    return respondent





def extrac_data_v_old(data):
    # new columns
    try:
        data = data[['Timestamp', 'GSR RAW', 'GSR Resistance CAL', 'GSR Conductance CAL', 'SourceStimuliName',
                     'Phasic Signal', 'Tonic Signal', 'Peak Amplitude', 'Heart Rate PPG ALG']]
    except KeyError:
        return False
    # select only stimuli events
    source = ["K", "N", "C", "F", "V", "B", "A4", "J", "M", "Q", "U", "A", "P", "G", "A1-1", "W", "A3",
              "O", "AA2", "H"]

    data = data.loc[data['SourceStimuliName'].isin(source)]

    # change object to float
    for col_ in data.drop('SourceStimuliName', axis=1).columns:
        data[col_] = data[col_].astype(float)

    # drop missing values
    data = data.dropna()

    # reset index
    data = data.reset_index(drop=True)

    peak_amp = pd.Series(data.groupby(['SourceStimuliName'])['Peak Amplitude'])
    ps_data = pd.Series(data.groupby(['SourceStimuliName'])['Phasic Signal'])
    tn_data = pd.Series(data.groupby(['SourceStimuliName'])['Tonic Signal'])
    time = pd.Series(data.groupby(['SourceStimuliName'])['Timestamp'])
    heart_rate = pd.Series(data.groupby(['SourceStimuliName'])['Heart Rate PPG ALG'])

    return data, ps_data, peak_amp, tn_data, time
# group data
def group_data_v_old(folder_path):
    # Initialize an empty list to store processed DataFrames
    processed_dfs = []

    # Read GSR csv
    for path in os.listdir(folder_path):
        if path.endswith('.csv'):
            raw_file = get_raw_file(os.path.join(folder_path, path))
            if raw_file is not None:
                formated_raw_data, phasic_data, peak_amplitude, tonic_data, time = extrac_data_v_old(raw_file)
                if formated_raw_data is not None:
                    df = formated_raw_data.copy()
                    df['Respondent'] = os.path.splitext(path)[0]  # Extract respondent name from file name
                    processed_dfs.append(df)

    # Concatenate all processed DataFrames into one
    df = pd.concat(processed_dfs, ignore_index=True)
    name_mapping = {
        'AA2': 'A2',
        'A1-1': 'A4'
    }

    # Mapping of original stimulus names to transformed names
    transformed_stimulus_names = {
        'A1': 'A1_LP',
        'A2': 'A2_LP',
        'A3': 'A3_LP',
        'A4': 'A4_LP',
        'A': 'A_HN',
        'B': 'B_HN',
        'C': 'C_LN',
        'F': 'F_HN',
        'G': 'G_HP',
        'H': 'H_HP',
        'J': 'J_Ne',
        'K': 'K_Ne',
        'M': 'M_LN',
        'N': 'N_LN',
        'O': 'O_LN',
        'P': 'P_HP',
        'U': 'U_Ne ',
        'V': 'V_Ne',
        'W': 'W_HN',
        'Q': 'Q_HP'
    }
    # Replace the old names with the new names in the 'SourceStimuliName' column
    df['SourceStimuliName'] = df['SourceStimuliName'].replace(name_mapping)

    df['Respondent'] = df['Respondent'].apply(modify_respondent)
    df['SourceStimuliName'].replace(transformed_stimulus_names, inplace=True)

    return df



def extract_data_v_new(data):
    # new columns
    try:
        data = data[['Timestamp', 'GSR RAW', 'GSR Resistance CAL', 'GSR Conductance CAL', 'SourceStimuliName',
                     'Phasic Signal', 'Tonic Signal', 'Peak Amplitude', 'Heart Rate PPG ALG']]
    except KeyError:
        return False
    # select only stimuli events
    source = ["K", "N", "C", "F", "V", "B", "A4", "J", "M", "Q", "U", "A", "P", "G", "A1", "W", "A3",
              "O", "A2", "H"]

    data = data.loc[data['SourceStimuliName'].isin(source)]

    # change object to float
    for col_ in data.drop('SourceStimuliName', axis=1).columns:
        data[col_] = data[col_].astype(float)

    # drop missing values
    data = data.dropna()

    # reset index
    data = data.reset_index(drop=True)

    peak_amp = pd.Series(data.groupby(['SourceStimuliName'])['Peak Amplitude'])
    ps_data = pd.Series(data.groupby(['SourceStimuliName'])['Phasic Signal'])
    tn_data = pd.Series(data.groupby(['SourceStimuliName'])['Tonic Signal'])
    time = pd.Series(data.groupby(['SourceStimuliName'])['Timestamp'])
    heart_rate = pd.Series(data.groupby(['SourceStimuliName'])['Heart Rate PPG ALG'])
    return data, ps_data, peak_amp, tn_data, time

# group data
def group_data_v_new(folder_path):
    # Initialize an empty list to store processed DataFrames
    processed_dfs = []

    # Read GSR csv
    for path in os.listdir(folder_path):
        if path.endswith('.csv'):
            raw_file = get_raw_file(os.path.join(folder_path, path))
            if raw_file is not None:
                formated_raw_data, phasic_data, peak_amplitude, tonic_data,  time= extract_data_v_new(raw_file)
                if formated_raw_data is not None:
                    df = formated_raw_data.copy()
                    df['Respondent'] = os.path.splitext(path)[0]  # Extract respondent name from file name
                    processed_dfs.append(df)

    # Concatenate all processed DataFrames into one
    df = pd.concat(processed_dfs, ignore_index=True)
    # Mapping of original stimulus names to transformed names
    transformed_stimulus_names = {
        'A1': 'A1_LP',
        'A2': 'A2_LP',
        'A3': 'A3_LP',
        'A4': 'A4_LP',
        'A': 'A_HN',
        'B': 'B_HN',
        'C': 'C_LN',
        'F': 'F_HN',
        'G': 'G_HP',
        'H': 'H_HP',
        'J': 'J_Ne',
        'K': 'K_Ne',
        'M': 'M_LN',
        'N': 'N_LN',
        'O': 'O_LN',
        'P': 'P_HP',
        'U': 'U_Ne ',
        'V': 'V_Ne',
        'W': 'W_HN',
        'Q': 'Q_HP'
    }

    df['Respondent'] = df['Respondent'].apply(modify_respondent)
    df['SourceStimuliName'].replace(transformed_stimulus_names, inplace=True)

    return df


def concatenate_dataframes(data_old, data_new):
    data_new_df = pd.read_csv(data_new)
    data_old_df = pd.read_csv(data_old)
    updated_data_df = pd.concat([data_old_df, data_new_df])
    return updated_data_df


