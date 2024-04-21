import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, make_scorer, mean_squared_error, mean_absolute_error, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt

import os
if isinstance(os.path, str):
    pass
else:
    os._path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    print('File is lstm_eval, loc is:', os._path)


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


def main(visualize=True):

    train_data = pd.read_csv('../../data/preprocessed/train_final.csv')
    test_data = pd.read_csv('../../data/preprocessed/test_final.csv')

    cols = ['id', 'activity', 'call', 'circumplex.arousal', 'circumplex.valence']
    cols += [f'mood_lag_{i}' for i in range(1, 5)]

    X_train = train_data[cols]
    X_train['id'] = X_train['id'].str[-2:]

    X_test = test_data[cols]
    X_test['id'] = X_test['id'].str[-2:]

    y_train = train_data['mood'].round()
    y_test = test_data['mood'].round()

    # Identify unique labels in the training set
    train_labels = np.unique(y_train)
    # shift = min(train_labels)

    # Identify unique labels in the training set
    train_labels = np.unique(y_train)
    shift = min(train_labels)

    # Create a mapping from train labels to consecutive integers
    label_mapping = {label: i for i, label in enumerate(train_labels)}

    # Map labels to consecutive integers
    y_train = np.array([label_mapping[label] for label in y_train])
    y_test = np.array([label_mapping[label] if label in label_mapping else -1 for label in y_test])

    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(1, 20),
    }
    rf = RandomForestClassifier(random_state=42)

    # Cross-entropy loss as the evaluation metric
    log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    f1_scorer = make_scorer(f1_score, average='weighted')
    rand_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=2, scoring=f1_scorer, random_state=42)
    rand_search.fit(X_train, y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    best_params = rand_search.best_params_
    best_rf = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
    best_rf.fit(X_train, y_train)

    # y_pred_probability = best_rf.predict_proba(X_test)
    # y_pred = [train_labels[i] for i in y_pred_probability.argmax(axis=1)]

    # # Filter out test samples with missing labels
    # valid_indices = np.where(y_test != -1)
    # y_test_filtered = y_test[valid_indices]
    # y_pred_probability_filtered = y_pred_probability[valid_indices]

    y_pred = best_rf.predict(X_test)

    # logloss = log_loss(y_test_filtered, y_pred_probability_filtered[:, :len(train_labels)], labels=train_labels)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # y_pred = y_pred_probability.argmax(axis=1) + shift
    # y_test = y_test + shift
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score', f1)
    print('MSE', np.mean((y_test - y_pred) ** 2))
    print('MAE', np.mean(np.abs(y_test - y_pred)))

    # plt.plot(range(len(y_test)), y_test, color='red', label='Actual Mood', marker='o', linestyle='--', markersize=3)
    # plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Mood', marker='o', linestyle='--', markersize=3)

    counts = {i: 0 for i in range(10)}
    counts_pred = {i: 0 for i in range(10)}
    for i in range(len(y_test)):
        counts[y_test[i]] += 1
        counts_pred[y_pred[i]] += 1

    if visualize:
        x = np.arange(2, 12)
        width = 0.4  # width of each bar
        plt.bar(x - width / 2, counts.values(), width=width, label='True Mood')
        plt.bar(x + width / 2, counts_pred.values(), width=width, label='Classified Mood')
        plt.title('True vs Classified Mood')
        plt.xlabel('Mood')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return y_test, y_pred


if __name__ == '__main__':
    main()
