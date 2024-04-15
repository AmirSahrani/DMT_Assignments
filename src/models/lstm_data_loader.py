from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import pdb

class MoodDataset(Dataset):
    def __init__(self, csv_file, included_columns, sequence_length):
        self.data = pd.read_csv(csv_file)

        # Ensure that the 'mood' column is in the included columns for labels
        included_columns.append('mood')
        
        # Filter the DataFrame to only include specified columns
        self.data = self.data[included_columns]

        # Extracting labels ('mood' column) and converting them to long type for classification
        self.labels = self.data['mood'].astype(int).values
        
        # Drop the 'mood' column from features
        self.features = self.data.drop(columns=['mood'], errors='ignore').values
        
        # Create sequences
        self.feature_sequences, self.label_sequences = self._create_sequences(sequence_length)

    def _create_sequences(self, sequence_length):
        # Create sequences of features and corresponding labels
        feature_sequences, label_sequences = [], []
        for i in range(len(self.features) - sequence_length + 1):
            feature_sequences.append(self.features[i:i + sequence_length])
            label_sequences.append(self.labels[i + sequence_length - 1])  # Label for the last item in the sequence
        return np.array(feature_sequences), np.array(label_sequences)

    def __len__(self):
        return len(self.feature_sequences)

    def __getitem__(self, index):
        # Return a single sequence of features and the corresponding label
        return torch.tensor(self.feature_sequences[index], dtype=torch.float), torch.tensor(self.label_sequences[index], dtype=torch.long)


