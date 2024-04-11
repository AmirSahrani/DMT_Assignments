from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

class MoodDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values  # exclude last column because it is the target
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float), torch.tensor(self.labels[index], dtype=torch.long)
