import pandas as pd
from torch.utils.data import Dataset
import torch
import pdb

class MoodDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with data.
        """
        self.data_frame = pd.read_csv(csv_file)

        self.features = self.data_frame.drop(columns=['id', 'day', 'time', 'mood', 'appCat.unknown', 'appCat.other', 'appCat.communication', 'sms', 'appCat.utilities', 'appCat.game'])
        self.targets = self.data_frame['mood']

        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.targets = torch.tensor(self.targets.values, dtype=torch.long)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_num_features(self):
        return self.features.shape[1]
        
