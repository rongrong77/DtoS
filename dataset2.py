import numpy as np
import pandas as pd
from loading.loadpickledataset import LoadPickleDataSet
from preprocessing.filter_imu import FilterIMU
from preprocessing.filter_opensim import FilterOpenSim
from preprocessing.remove_outlier import remove_outlier
from preprocessing.resample import Resample
from preprocessing.segmentation.fixwindowsegmentation import FixWindowSegmentation
import pickle
from sklearn.model_selection import KFold

class DataSet:
    def __init__(self, config, load_dataset=True):
        self.config = config
        self.x = []
        self.y = []
        self.labels = []
        self.selected_trial_type = config['selected_trial_type']
        self.segmentation_method = config['segmentation_method']
        self.resample = config['resample']
        self.n_sample = len(self.y)
        if load_dataset:
            self.load_dataset()
            self.train_subjects = config['train_subjects']
            self.test_subjects = config['test_subjects']
            self.valid_subjects = config['valid_subjects']
            self.train_activity = config['train_activity']
            self.test_activity = config['test_activity']
            self.valid_activity = config['valid_activity']
        self.train_dataset = {}
        self.test_dataset = {}
        self.valid_dataset = {}

    def load_dataset(self):
        getdata_handler = LoadPickleDataSet(self.config)
        self.x, self.y, self.labels = getdata_handler.run_get_dataset()

    def run_segmentation(self, x, y, labels):
        if self.segmentation_method == 'fixedwindow':
            segmentation_handler = FixWindowSegmentation(x, y, labels, max_length=self.config['target_padding_length'])
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

            # Ensure y values remain single values
            self.y = [np.mean(segment) for segment in self.y]

        return self.x, self.y, self.labels

    def concatenate_data(self):
        self.labels = pd.concat(self.labels, axis=0, ignore_index=True)
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.array(self.y)

    def run_dataset_split(self):
        if set(self.test_subjects).issubset(self.train_subjects):
            train_labels = self.labels[~self.labels['subject'].isin(self.test_subjects)]
            test_labels = self.labels[self.labels['subject'].isin(self.test_subjects)]
        else:
            train_labels = self.labels[self.labels['subject'].isin(self.train_subjects)]
            test_labels = self.labels[self.labels['subject'].isin(self.test_subjects)]

        train_index = train_labels.index.values
        test_index = test_labels.index.values

        train_x = [self.x[i] for i in train_index]
        train_y = [self.y[i] for i in train_index]
        self.train_dataset['x'] = train_x
        self.train_dataset['y'] = train_y
        self.train_dataset['labels'] = train_labels.reset_index(drop=True)

        test_x = [self.x[i] for i in test_index]
        test_y = [self.y[i] for i in test_index]
        self.test_dataset['x'] = test_x
        self.test_dataset['y'] = test_y
        self.test_dataset['labels'] = test_labels.reset_index(drop=True)

        del train_labels, test_labels, train_x, train_y, test_x, test_y
        return self.train_dataset, self.test_dataset

    def run_k_fold_split(self, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(self.train_dataset['labels']):
            train_x = [self.train_dataset['x'][i] for i in train_index]
            train_y = [self.train_dataset['y'][i] for i in train_index]
            train_labels = self.train_dataset['labels'].iloc[train_index]

            val_x = [self.train_dataset['x'][i] for i in val_index]
            val_y = [self.train_dataset['y'][i] for i in val_index]
            val_labels = self.train_dataset['labels'].iloc[val_index]

            train_dataset = {
                'x': train_x,
                'y': train_y,
                'labels': train_labels.reset_index(drop=True)
            }
            val_dataset = {
                'x': val_x,
                'y': val_y,
                'labels': val_labels.reset_index(drop=True)
            }


        return train_dataset,val_dataset
