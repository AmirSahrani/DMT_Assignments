from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.format": "png",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "0.8",
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})


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
    methods = ['linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    alphas = np.linspace(0.2, 1.3, num=7)  # Example range of alpha values
    methods = ['linear']
    alphas = [0.07]
    param_grid = {
        'kernel': methods,
        'alpha': alphas
    }

    # Perform grid search with cross-validation
    folds = 5
    grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, cv=folds, verbose=False, scoring='neg_mean_squared_error', error_score=True)
    grid_search.fit(X_train, y_train)
    df = pd.DataFrame(grid_search.cv_results_)
    df = df.drop('params', axis=1)
    df = df.dropna(axis=1)
    df = df.sort_values(by='rank_test_score')
    print(grid_search.cv_results_.keys())
    df = df[['mean_test_score', 'std_test_score', 'rank_test_score']]
    # remove underscores from column naames and capitalize
    df.columns = [col.replace('_', ' ').capitalize() for col in df.columns]
    print(df.to_latex())

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_method = best_params['kernel']

    print(f"Best Kernel Method: {best_method}")
    print(f"Best Model Parameters: {best_params}")
    print(f"Best Model Penalty: {best_params['alpha']:.2f}")
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

    mse_rounded = np.mean((y_pred - y_test) ** 2)
    mae_rounded = np.mean(np.abs(y_pred - y_test))

    # create latex table for mse
    df = pd.DataFrame({
        'MSE': [mse, mse_rounded],
        'MAE': [mae, mae_rounded]
    }, index=['Unrounded', 'Rounded'])
    print(df.to_latex(float_format="%.2f"))

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Accuracy: {sum(y_pred == y_test)/len(y_test)}%')

    counts = {i: 0 for i in range(11)}
    counts_pred = {i: 0 for i in range(11)}
    for i in range(len(y_test)):
        counts[y_test[i]] += 1
        counts_pred[y_pred[i]] += 1

    x = np.arange(0, 11)
    width = 0.4  # width of each bar

    plt.bar(x - width / 2, counts.values(), width=width, label='True Mood')
    plt.bar(x + width / 2, counts_pred.values(), width=width, label='Predicted Mood')
    plt.title('True vs Predicted Mood')
    plt.xlabel('Mood')
    plt.ylabel('Frequency')

    # Set the x-axis ticks to be in the middle>

    plt.legend()
    plt.savefig('../../figures/mood_pred_ridge.png')
    plt.show()


if __name__ == '__main__':
    main()
