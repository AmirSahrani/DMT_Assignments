from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv('../../data/preprocessed/engineered_data.csv')
    mood = data['mood']
    print(len(data))
    # change id to a categorical variable
    data['id'] = data['id'].astype('category')

    # drop columns that are not numeric
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data_numerical = data[num_cols]

    data_numerical = data_numerical.drop(columns=['mood'])
    X_train, X_test, y_train, y_test = train_test_split(data_numerical.values, mood.values, test_size=0.2, random_state=52)

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

    gp = GaussianProcessRegressor()
    gp.fit(X_train, y_train)
    print(f'Train Score: {gp.score(X_train, y_train):.2f}')
    print(f'Test Score: {gp.score(X_test, y_test):.2f}')

    y_pred, y_err = gp.predict(X_test, return_std=True)

    print(f'''
    -------------------
    min y_pred: {min(y_pred)}
    max y_pred: {max(y_pred)}
    mean y_pred: {np.mean(y_pred):.2f}
    -------------------
    ''')

    plt.errorbar(np.arange(len(y_test)), y_pred, yerr=y_err, fmt='-', label='Predicted')
    plt.plot(np.arange(len(y_test)), y_test, label='True')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
