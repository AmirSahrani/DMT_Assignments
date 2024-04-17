from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def data_numeric(data):
    non_numeric_columns = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype.name == 'category']
    return data.drop(columns=non_numeric_columns)  # Drop non-numeric columns

    # Perform one-hot encoding and concatenate all at once


def feature_selection(X, y):
    # Create the Gaussian Process Regressor
    gp = GaussianProcessRegressor()

    sfs = SequentialFeatureSelector(gp, n_features_to_select=5, direction='backward')
    sfs.fit(X, y)
    columns = X.columns[sfs.get_support()]
    return columns


def main():
    # Load and preprocess data
    data = pd.read_csv('../../data/preprocessed/train_final.csv')
    # data = data.drop(columns=time_columns, axis=1)
    data_numerical = data_numeric(data)

    test_data = pd.read_csv('../../data/preprocessed/test_final.csv')
    # test_data = test_data.drop(columns=time_columns, axis=1)
    test_data_numerical = data_numeric(test_data)
    print(len(data_numerical.columns))

    # Define features and target
    y_train = data_numerical.pop('mood_lag_5').values
    X_train = data_numerical.values
    y_test = test_data_numerical.pop('mood_lag_5').values
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
    # gp = GaussianProcessRegressor(alpha=1e-2, n_restarts_optimizer=10, normalize_y=True, random_state=42)
    # Using ridge regression
    gp = KernelRidge()
    methods = ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    param_grid = {'kernel': methods}

    # Perform grid search with cross-validation
    folds = 5
    grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, cv=folds)
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_method = best_params['kernel']

    print(f"Best Kernel Method: {best_method}")
    print(f"Best Model Parameters: {best_params}")
    print(f"Best Model Train Score: {best_model.score(X_train, y_train):.2f}")
    print(f"Best Model Test Score: {best_model.score(X_test, y_test):.2f}")

    gp = KernelRidge(kernel=best_method).fit(X_train, y_train)

    y_pred_unround = gp.predict(X_test)
    y_pred = np.round(y_pred_unround)

    print(f'''
    -------------------
    min y_pred: {min(y_pred)}
    max y_pred: {max(y_pred)}
    mean y_pred: {np.mean(y_pred):.2f}
    -------------------
    ''')

    mse = np.mean((y_pred_unround - y_test) ** 2)
    mae = np.mean(np.abs(y_pred_unround - y_test))

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Accuracy: {sum(y_pred == y_test)/len(y_test)}%')

    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted')
    plt.plot(np.arange(len(y_test)), y_test, label='True')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
