import os
import pandas as pd
import pickle
from config import get_config_universal
import numpy as np


import pandas as pd

#from config import get_config


def read_csv(filename):
    df = pd.read_csv(filename)
    return df
config = get_config_universal('DtoS')
datatype_list = ['steptime', 'ia']
activity_list = ["C1", "C2", "Normal"]
subject_list = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08", "Subject09", "Subject10", "Subject11", "Subject12", "Subject13", "Subject14"]

dataset_path = './Datasets/'

def read_static_data(subject_list, dataset_path):
    data = {}
    for subject in subject_list:
        if subject not in data:
            data[subject] = {}
        subject_dir_temp = os.path.join(dataset_path, subject)
        subject_dir = os.path.join(subject_dir_temp, os.listdir(subject_dir_temp)[0])
        static_dir = os.path.join(subject_dir, 'static2')
        if os.path.exists(static_dir):
            for root, dirs, files in os.walk(static_dir):
                for name in files:
                    if name.endswith('.csv'):
                        file = os.path.join(root, name)
                        df = read_csv(file)
                        data[subject][name[:-4]] = df
    return data

def read_dynamic_data(subject_list, activity_list, dataset_path):
    data = {}
    for subject in subject_list:
        if subject not in data:
            data[subject] = {}
        subject_dir_temp = os.path.join(dataset_path, subject)
        subject_dir = os.path.join(subject_dir_temp, os.listdir(subject_dir_temp)[0])
        dynamic_dir = os.path.join(subject_dir, 'dynamic')
        for activity in activity_list:
            if activity not in data[subject]:
                data[subject][activity] = {}
            activity_dir = os.path.join(dynamic_dir, activity)
            if os.path.exists(activity_dir):
                for fp_folder in os.listdir(activity_dir):
                    fp_path = os.path.join(activity_dir, fp_folder)
                    if os.path.isdir(fp_path):
                        if fp_folder not in data[subject][activity]:
                            data[subject][activity][fp_folder] = {'ia': [], 'steptime': []}
                        for sub_folder in ['ia', 'steptime']:
                            sub_folder_path = os.path.join(fp_path, sub_folder)
                            if os.path.exists(sub_folder_path):
                                for root, dirs, files in os.walk(sub_folder_path):
                                    for name in files:
                                        if name.endswith('.csv'):
                                            file = os.path.join(root, name)
                                            df = read_csv(file)
                                            data[subject][activity][fp_folder][sub_folder].append(df)
    return data

def get_step_time_from_dynamic_data(dynamic_data):
    step_time_data = {}
    for subject in dynamic_data:
        step_time_data[subject] = {}
        for activity in dynamic_data[subject]:
            step_time_data[subject][activity] = {}
            for fp_folder in dynamic_data[subject][activity]:
                for sub_folder in dynamic_data[subject][activity][fp_folder]:
                    if sub_folder == 'steptime':
                        for df in dynamic_data[subject][activity][fp_folder][sub_folder]:
                            step_time = df.iloc[0, 0]  # Assume the step time is the first value in the DataFrame
                            if fp_folder not in step_time_data[subject][activity]:
                                step_time_data[subject][activity][fp_folder] = {}
                            step_time_data[subject][activity][fp_folder]['steptime'] = step_time
    return step_time_data

def get_step_time_from_dynamic_data(dynamic_data):
    step_time_data = {}
    for subject in dynamic_data:
        step_time_data[subject] = {}
        for activity in dynamic_data[subject]:
            step_time_data[subject][activity] = {}
            for fp_folder in dynamic_data[subject][activity]:
                for df in dynamic_data[subject][activity][fp_folder]['steptime']:
                    step_time = df.iloc[0, 0]  # Assume the step time is the first value in the DataFrame
                    step_time_data[subject][activity][fp_folder] = step_time
    return step_time_data

def structure_dynamic_data(dynamic_data):
    structured_data = {}
    for subject in dynamic_data:
        structured_data[subject] = {}
        for activity in dynamic_data[subject]:
            structured_data[subject][activity] = []
            for fp_folder in dynamic_data[subject][activity]:
                # Read the step time directly from the corresponding CSV file in the steptime folder
                step_time_df = dynamic_data[subject][activity][fp_folder]['steptime']
                if step_time_df:
                    step_time = step_time_df[0].iloc[0, 0]  # Assume the step time is the first value in the DataFrame
                    factor = int(fp_folder[0])  # Extract the factor from fp_folder (e.g., '1fp' -> 1, '2fp' -> 2, etc.)
                    num_frames = int(step_time * 200 * factor)
                    padding = pd.DataFrame([[None] * len(dynamic_data[subject][activity][fp_folder]['ia'][0].columns)] * num_frames, columns=dynamic_data[subject][activity][fp_folder]['ia'][0].columns)
                    
                    for dynamic_df in dynamic_data[subject][activity][fp_folder]['ia']:
                        structured_df = pd.concat([padding, dynamic_df], ignore_index=True)
                        structured_data[subject][activity].append(structured_df)
                else:
                    print(f"Step time data for {subject} {activity} {fp_folder} is missing.")
                    for dynamic_df in dynamic_data[subject][activity][fp_folder]['ia']:
                        structured_data[subject][activity].append(dynamic_df)
    return structured_data

def map_static_to_dynamic(static_data, dynamic_data):
    mapped_data = {}
    for subject in dynamic_data:
        mapped_data[subject] = {}
        for activity in dynamic_data[subject]:
            mapped_data[subject][activity] = []
            for i, dynamic_df in enumerate(dynamic_data[subject][activity]):
                if activity in static_data[subject]:
                    static_df = static_data[subject][activity]
                    # Repeat the static data to match the length of the dynamic data
                    repeated_static_df = pd.concat([static_df] * (len(dynamic_df) // len(static_df) + 1), ignore_index=True)[:len(dynamic_df)]
                    combined_df = pd.concat([repeated_static_df, dynamic_df], axis=1).dropna()
                    mapped_data[subject][activity].append(combined_df)
                else:
                    print(f"Static data for {subject} {activity} is missing.")
                    mapped_data[subject][activity].append(dynamic_df)
    return mapped_data

def extract_data_for_dataset(mapped_data, static_columns, dynamic_columns):
    dataset = {'static': {}, 'dynamic': {}}
    
    for subject in mapped_data:
        dataset['static'][subject] = {}
        dataset['dynamic'][subject] = {}
        
        for activity in mapped_data[subject]:
            dataset['static'][subject][activity] = []
            dataset['dynamic'][subject][activity] = []
            
            for combined_df in mapped_data[subject][activity]:
                dynamic_selected = combined_df[dynamic_columns]
                static_selected = combined_df[static_columns]
                
                dataset['dynamic'][subject][activity].append(dynamic_selected)
                dataset['static'][subject][activity].append(static_selected)
                    
    return dataset

def filter_invalid_data(dataset):
    filtered_dataset = {'static': {}, 'dynamic': {}}
    
    for subject in dataset['dynamic']:
        filtered_dataset['static'][subject] = {}
        filtered_dataset['dynamic'][subject] = {}
        
        for activity in dataset['dynamic'][subject]:
            valid_dynamic_data = []
            valid_static_data = []
            
            for i in range(len(dataset['dynamic'][subject][activity])):
                dynamic_data = dataset['dynamic'][subject][activity][i]
                static_data = dataset['static'][subject][activity][i]
                
                if dynamic_data.shape[0] != 0 and static_data.shape[0] != 0:
                    valid_dynamic_data.append(dynamic_data)
                    valid_static_data.append(static_data)
                else:
                    print(f"Removed invalid data for subject {subject}, activity {activity}, index {i}")
            
            filtered_dataset['dynamic'][subject][activity] = valid_dynamic_data
            filtered_dataset['static'][subject][activity] = valid_static_data
            
    return filtered_dataset 

def filter_invalid_data_and_add_differential(dataset):
    filtered_dataset = {'static': {}, 'dynamic': {}}
    
    for subject in dataset['dynamic']:
        filtered_dataset['static'][subject] = {}
        filtered_dataset['dynamic'][subject] = {}
        
        for activity in dataset['dynamic'][subject]:
            valid_dynamic_data = []
            valid_static_data = []
            
            for i in range(len(dataset['dynamic'][subject][activity])):
                dynamic_data = dataset['dynamic'][subject][activity][i]
                static_data = dataset['static'][subject][activity][i]
                
                if dynamic_data.shape[0] != 0 and static_data.shape[0] != 0:
                    # Compute the differential for APIA to get APRICA
                    differential_data = np.diff(dynamic_data['APIA'].values, axis=0)
                    # Pad the differential to match the original shape
                    differential_data = np.insert(differential_data, 0, 0)
                    # Add differential as a new column named APRICA
                    combined_dynamic_data = dynamic_data.copy()
                    combined_dynamic_data['APRCIA'] = differential_data
                    
                    valid_dynamic_data.append(combined_dynamic_data)
                    valid_static_data.append(static_data)
                else:
                    print(f"Removed invalid data for subject {subject}, activity {activity}, index {i}")
            
            filtered_dataset['dynamic'][subject][activity] = valid_dynamic_data
            filtered_dataset['static'][subject][activity] = valid_static_data
            
    return filtered_dataset

def accumulate_static_values(dataset):
    accumulated_static_values = {'static': {}, 'dynamic': dataset['dynamic']}
    
    for subject in dataset['static']:
        accumulated_static_values['static'][subject] = {}
        
        for activity in dataset['static'][subject]:
            accumulated_values = []
            
            for static_data in dataset['static'][subject][activity]:
                if static_data.shape[0] != 0:
                    accumulated_value = static_data.sum().sum()  # Accumulate the sum of all values for this trial
                    accumulated_values.append(accumulated_value)
            
            accumulated_static_values['static'][subject][activity] = accumulated_values
            
    return accumulated_static_values
# Example usage
static_data = read_static_data(subject_list, dataset_path)
dynamic_data = read_dynamic_data(subject_list, activity_list, dataset_path)
structured_data = structure_dynamic_data(dynamic_data)
mapped_data = map_static_to_dynamic(static_data, structured_data)
training_data = extract_data_for_dataset(mapped_data, ["DCOP"], ["APIA"])
filtered_dataset = filter_invalid_data(training_data)
dataset = filter_invalid_data_and_add_differential(filtered_dataset)
onevalue = accumulate_static_values(dataset)


# Save the mapped data if needed
dl_dataset = './Datasets/'
dataset_file = dl_dataset + 'ALL_DCOP.p'

if os.path.isfile(dataset_file):
    print('file exists')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
else:
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

print("The highest pickle protocol available is:", pickle.HIGHEST_PROTOCOL)
