import os
import pickle
import pandas as pd


class LoadPickleDataSet:
    def __init__(self, config):
        '''
       
        '''
        self.dl_dataset_path = config['dl_dataset_path']
        self.dataset_name = config['dl_dataset']
        self.static_labels = config['static_labels']
        self.dynamic_labels = config['dynamic_labels']
        self.selected_labels =  config['selected_trial_type']
        self.dataset = {}
        self.static = None
        self.dynamic = None

    def load_dataset(self):
        dataset_file = os.path.join(self.dl_dataset_path + self.dataset_name)
        if os.path.isfile(dataset_file):
            print('file exist')
            with open(dataset_file, 'rb') as f:
                self.dataset = pickle.load(f)
                print(f"Loaded dataset keys: {self.dataset.keys()}")
                if 'static' in self.dataset:
                    print(f"static type: {type(self.dataset['static'])}")
                else:
                    print("static data is missing from the dataset.")
                if 'dynamic' in self.dataset:
                    print(f"dynamic data type: {type(self.dataset['dynamic'])}")
                else:
                    print("dynamic data is missing from the dataset.")
        else:
            print('this dataset is not exist: run prepare_dataset.py first')

    def get_static(self):
        static = self.dataset.get('static', None)
        self.static = []
        
    
        for subject, activities in static.items():
            for activity, accumulated_value in activities.items():
                if isinstance(accumulated_value, list):
                    self.static.extend(accumulated_value)  # If accumulated_value is a list of float64 values
                elif isinstance(accumulated_value, float):  # If accumulated_value is a single float64 value
                    self.static.append(accumulated_value)
                else:
                    print(f"Unexpected type for accumulated value: {type(accumulated_value)}")
        return self.static


    def get_dynamic(self):
        dynamic = self.dataset.get('dynamic', None)
        self.dynamic = []
        labels_data = []
        all_columns = set()
        for subject, activities in dynamic.items():
            for activity, df_list in activities.items():
                if isinstance(df_list, list):
                    for df in df_list:
                        if isinstance(df, pd.DataFrame):
                            all_columns.update(df.columns)
                            missing_labels = [label for label in self.dynamic_labels if label not in df.columns]
                            if not missing_labels:
                                self.dynamic.append(df[self.dynamic_labels].values)
                                labels_data.append({'subject': subject, 'activity': activity})
                            else:
                                print(f"Missing labels in activity {activity} for subject {subject}: {missing_labels}")
                        else:
                            print(f"Expected DataFrame, got {type(df)} for activity {activity}")
                else:
                    print(f"Expected list, got {type(df_list)} for activity {activity}")
        self.labels = pd.DataFrame(labels_data)
        return self.dynamic,self.labels
    
    
    def run_get_dataset(self):
        self.load_dataset()
        self.get_dynamic()
        self.get_static()
        selected_y_values = self.static
        selected_x_values = self.dynamic
        selected_labels =  self.labels
        del self.dataset
        return selected_x_values, selected_y_values, selected_labels


