import os
import pickle
import numpy as np
import pandas as pd


class FixWindowSegmentation:
    def __init__(self, imu_signal, ia_signal, labels,max_length):

        self.imu_signal = imu_signal
        self.ia_signal = ia_signal
        self.general_labels = labels
        self.updated_general_labels = []
        self.max_length = max_length
        self.x = []
        self.y = []

    
    def run_ia_segmentation(self):
        self.y = np.array(self.ia_signal)

    def run_imu_segmentation(self):
        x_signals_segmented = []
        for data in self.imu_signal:
            padded_data = self.pad_along_axis(data, self.max_length, axis=0)
            x_signals_segmented.append(padded_data)
        self.x = np.array(x_signals_segmented)


    def run_label_segmentation(self):
        labels_segmented = []
        for i, row in self.general_labels.iterrows():
            label_value = row.values
            labels_segmented.append(label_value)
        self.updated_general_labels = np.array(labels_segmented)

    def _run_segmentation(self):
        self.run_imu_segmentation()
        self.run_ia_segmentation()
        self.run_label_segmentation()
        return self.x, self.y, self.updated_general_labels

    def pad_along_axis(self, array, target_length, axis=0):
        pad_size = target_length - array.shape[axis]
        if pad_size < 0:
            raise ValueError(f"Target length {target_length} is smaller than array length {array.shape[axis]}")
        npad = [(0, 0) for _ in range(len(array.shape))]
        npad[axis] = (0, pad_size)
        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)




