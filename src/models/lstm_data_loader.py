import pandas as pd
from torch.utils.data import Dataset
import torch
import pdb

class MoodDataset(Dataset):
    def __init__(self, csv_file, mode='classification'):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            mode (string): Mode of operation, 'classification' or 'regression'.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.mode = mode

        self.features = self.data_frame.drop(columns=['id', 'day', 'time', 'mood', 'appCat.unknown', 'appCat.other', 'appCat.communication', 'sms', 'appCat.utilities', 'appCat.game'])
        self.features = torch.tensor(self.features.values, dtype=torch.float32)

        if mode == 'classification':
            self.targets = self.data_frame['mood']
            self.targets = torch.tensor(self.targets.values, dtype=torch.long)
        elif mode == 'regression':
            # For regression, 'mood' might need to be normalized or scaled depending on your specific needs
            self.targets = self.data_frame['mood'].astype(float)  # Ensure the mood is treated as a float
            self.targets = torch.tensor(self.targets.values, dtype=torch.float32)  # Use float32 for regression targets

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_num_features(self):
        return self.features.shape[1]

