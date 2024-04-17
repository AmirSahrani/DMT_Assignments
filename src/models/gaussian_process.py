from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def data_numeric(data):
    non_numeric_columns = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype.name == 'category']
    return data.drop(columns=non_numeric_columns)  # Drop non-numeric columns

    # Perform one-hot encoding and concatenate all at once


def shift_moods(df):
    # Shift the 'mood' column within each group defined by 'id'
    df['next_mood'] = df.groupby('id')['mood'].shift(-1)
    # Drop the last entry of each group because its 'next_mood' will be NaN
    df = df.dropna(subset=['next_mood'])

    # make subset dtype int inplace
    df['next_mood'] = df['next_mood'].astype(int)
    return df


def feature_selection(X, y):
    # Create the Gaussian Process Regressor
    gp = GaussianProcessRegressor()

    sfs = SequentialFeatureSelector(gp, n_features_to_select=5, direction='backward')
    sfs.fit(X, y)
    columns = X.columns[sfs.get_support()]
    return columns


def main():
    time_columns = ['time', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos']
    # Load and preprocess data
    data = pd.read_csv('../../data/preprocessed/train_final.csv')
    # data = data.drop(columns=time_columns, axis=1)
    data = shift_moods(data)
    data_numerical = data_numeric(data)

    test_data = pd.read_csv('../../data/preprocessed/test_final.csv')
    # test_data = test_data.drop(columns=time_columns, axis=1)
    test_data = shift_moods(test_data)
    test_data_numerical = data_numeric(test_data)
    print(len(data_numerical.columns))

    # Define features and target
    y_train = data_numerical.pop('next_mood').values
    X_train = data_numerical.values
    y_test = test_data_numerical.pop('next_mood').values
    X_test = test_data_numerical.values


    # output shapes of the split data
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    print(f'''
    X_train shape {X_train.shape},
    X_test.shape {X_test.shape},
    y_train.shape {y_train.shape},
    y_test.shape {y_test.shape}''')

    print(f'''
    -------------------
    min y_train: {min(y_train)}
    max y_train: {max(y_train)}
    mean y_train: {np.mean(y_train):.2f}

    min y_test: {min(y_test)}
    max y_test: {max(y_test)}
    mean y_test: {np.mean(y_test):.2f}
    -------------------
    ''')

    # Train Gaussian Process Regressor on selected features
    # gp = GaussianProcessRegressor()
    # Using ridge regression
    gp = Ridge()
    gp.fit(X_train, y_train)

    print(f'Train Score: {gp.score(X_train, y_train):.2f}')

    y_pred = gp.predict(X_test)
    y_pred = np.round(y_pred)

    print(f'''
    -------------------
    min y_pred: {min(y_pred)}
    max y_pred: {max(y_pred)}
    mean y_pred: {np.mean(y_pred):.2f}
    -------------------
    ''')

    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')

    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted')
    plt.plot(np.arange(len(y_test)), y_test, label='True')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
