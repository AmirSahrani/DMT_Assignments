from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import pprint


def get_user_data(df, row, window_user, window_other):
    '''
    Get the rows before the current row for every user, the context length for the current user and the other users can be specified.
    WARNING: This function assumes that the data is sorted by time, and is meant to be used in a pandas apply function.

    :param df: The dataframe containing the dataframe
    :param row: The current row
    :param window_user: The context length for the current user
    :param window_other: The context length for the other users

    :return: The features and labels for the current user and the other users
    '''
    features = []
    user = row['id']
    time = row['time']

    user_data = df[df['id'] == user]
    user_data = user_data[user_data['time'] <= time]
    others_data = df[df['id'] != user]
    others_data = others_data[others_data['time'] < time]

    if len(user_data) < window_user or len(others_data) < window_other:
        return np.array([]), np.array([])

    else:
        user_data = user_data.iloc[-window_user:]
        others_data = others_data.iloc[-window_other:]

        user_features = (user_data.values.flatten())
        others_features = others_data.values.flatten()

        features = np.concatenate([user_features, others_features])

        labels = user_data.iloc[-1, -1]

        return np.array(features), np.array(labels)


def test_user_data():
    df = pd.DataFrame({
        'id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'time': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'label': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })

    features, labels = zip(*df.apply(lambda x: get_user_data(df, x, 2, 2), axis=1))
    pprint.pprint(features)
    pprint.pprint(labels)


class MoodDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        features = self.data.apply(lambda x: get_user_data(self.data, x, 5, 5), axis=1)
        self.features, self.labels = zip(*features)

        # Drop any empty rows
        indices = [i for i, x in enumerate(self.features) if x.size > 0 and self.labels[i].size > 0]
        self.features = np.array([self.features[i] for i in indices])
        self.labels = np.array([self.labels[i] for i in indices])
        print(self.features.shape, self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float), torch.tensor(self.labels[index], dtype=torch.long)


if __name__ == "__main__":
    MoodDataset('data/preprocessed/toy_data.csv')
    # test_user_data()
