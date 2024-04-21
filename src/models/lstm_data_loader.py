import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import pdb


class MoodDataset(Dataset):
    def __init__(self, csv_file, sequence_length=1, mode='regression'):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            sequence_length (int): Number of days in each sequence.
            mode (string): Mode of operation, 'regression' or 'classification'.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.mode = mode

        # Exclude non-feature columns
        features_columns = self.data_frame.drop(columns=['id', 'day', 'time', 'mood', 'appCat.unknown', 'appCat.other', 'appCat.entertainment'])
        self.features = torch.tensor(features_columns.values, dtype=torch.float32)

        if mode == 'classification':
            # For classification, round and convert mood to long integers
            self.labels = torch.tensor(self.data_frame['mood'].round().values, dtype=torch.long)
        elif mode == 'regression':
            # For regression, use mood as float
            self.labels = torch.tensor(self.data_frame['mood'].values, dtype=torch.float32)

        self.feature_sequences, self.label_sequences = self._create_sequences()

    def _create_sequences(self):
        """
        Generates sequences of features and corresponding labels from sorted and grouped data
        """
        feature_sequences, label_sequences = [], []
        grouped = self.data_frame.groupby('id')
        for _, group in grouped:
            group_features = torch.tensor(group.drop(columns=['id', 'day', 'time', 'mood', 'appCat.unknown', 'appCat.other', 'appCat.entertainment']).values, dtype=torch.float32)
            group_labels = torch.tensor(group['mood'].values, dtype=torch.long if self.mode == 'classification' else torch.float32)
            num_sequences = len(group) - self.sequence_length + 1
            for i in range(num_sequences):
                feature_sequences.append(group_features[i:i + self.sequence_length])
                label_sequences.append(group_labels[i + self.sequence_length - 1])
        return feature_sequences, label_sequences

    def __len__(self):
        return len(self.feature_sequences)

    def __getitem__(self, idx):
        return self.feature_sequences[idx], self.label_sequences[idx]

    def get_num_features(self):
        return self.features.shape[1]
