import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd


def train_lstm(lstm, data):
    # load Data
    data = pd.read_csv(data)
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    # split data
    train_data = data.iloc[:int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)):]


def main(data, loss_func, **kwargs):
    lstm = t.nn.LSTM(**kwargs)
    print(lstm)

    train_lstm(lstm, loss_func, data)


if __name__ == '__main__':
    main(data='data/preprocessed/toy_data.csv', input_size=10, hidden_size=20, num_layers=2, output_size=1, loss_func=F.mse_loss)
    main(data='data/preprocessed/toy_data.csv', input_size=10, hidden_size=20, num_layers=2, output_size=10, loss_func=F.cross_entropy)
