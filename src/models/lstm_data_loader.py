from torch.utils.data import Dataset
import pandas as pd
import torch

class MoodDataset(Dataset):
    def __init__(self, csv_file, included_columns, sequence_length):
        # Load the data
        self.data = pd.read_csv(csv_file)

        # Filter the DataFrame to only include specified columns
        self.data = self.data[included_columns + ['mood']]  # Include 'mood' explicitly

        # Extracting labels ('mood' column) and converting them to long type for classification
        self.labels = torch.tensor(self.data['mood'].values).long()

        # Drop the 'mood' column from features to ensure it's not included in the input features
        self.features = self.data.drop(columns=['mood']).values
        
        # Create sequences of features and labels
        self.feature_sequences, self.label_sequences = self._create_sequences(sequence_length)

    def _create_sequences(self, sequence_length):
        # Initialize lists to hold sequences of features and labels
        feature_sequences, label_sequences = [], []
        for i in range(len(self.features) - sequence_length):
            feature_sequences.append(self.features[i:i + sequence_length])
            label_sequences.append(self.labels[i + sequence_length])
        return feature_sequences, label_sequences

    def __len__(self):
        # Return the number of sequences available
        return len(self.feature_sequences)

    def __getitem__(self, index):
        # Return a tuple of feature sequence and corresponding label for the given index
        return torch.tensor(self.feature_sequences[index], dtype=torch.float), self.label_sequences[index]

