import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import shift_moods

train_data = shift_moods(pd.read_csv('../../data/preprocessed/train_final.csv'))
test_data = shift_moods(pd.read_csv('../../data/preprocessed/test_final.csv'))

X_train = train_data[['id', 'activity', 'call', 'circumplex.arousal', 'circumplex.valence']]
X_train['id'] = X_train['id'].str[-2:]

X_test = test_data[['id', 'activity', 'call', 'circumplex.arousal', 'circumplex.valence']]
X_test['id'] = X_test['id'].str[-2:]

y_train = train_data['next_mood'].round()
y_test = test_data['next_mood'].round()

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
rand_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=2, scoring=log_loss_scorer, random_state=42)
rand_search.fit(X_train, y_train)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
best_params = rand_search.best_params_
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

y_pred_probability = best_rf.predict_proba(X_test)
y_pred = [train_labels[i] for i in y_pred_probability.argmax(axis=1)]

# Filter out test samples with missing labels
valid_indices = np.where(y_test != -1)
y_test_filtered = y_test[valid_indices]
y_pred_probability_filtered = y_pred_probability[valid_indices]

logloss = log_loss(y_test_filtered, y_pred_probability_filtered[:, :len(train_labels)], labels=train_labels)
y_pred = y_pred_probability.argmax(axis=1) + shift
y_test = y_test + shift
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Cross-Entropy Loss:', logloss)
print('MSE', np.mean((y_test - y_pred) ** 2))
print('MAE', np.mean(np.abs(y_test - y_pred)))

# plt.plot(range(len(y_test)), y_test, color='red', label='Actual Mood', marker='o', linestyle='--', markersize=3)
# plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Mood', marker='o', linestyle='--', markersize=3)
bin_edges = np.linspace(min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred)), 11)
print(bin_edges)
plt.hist(y_test, bins=11, label='True Mood')
plt.hist(y_pred, bins=11, label='Classified Mood')
plt.legend()
plt.xlabel('Mood')
plt.ylabel("Frequency")
plt.title('Predicted vs. Actual Mood')
plt.show()
