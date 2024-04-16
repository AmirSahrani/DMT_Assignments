import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt

"""From Before Feature Engineering
#Loading in the data - currently just the toy data from Amir
data = pd.read_csv('../../data/preprocessed/toy_data.csv')
data['mood'] = data['mood'].round()  # Can't be continuous
data['time'] = (pd.to_datetime(data['time']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

# Separating the feature data and the target data
X = data.drop(columns=['mood'])
y = data['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""

train_data = pd.read_csv('../../data/preprocessed/train_set.csv')
train_data['mood'] = train_data['mood'].round()
train_data['id'] = train_data['id'].str[-2:]

test_data = pd.read_csv('../../data/preprocessed/test_set.csv')
test_data['mood'] = test_data['mood'].round()
test_data['id'] = test_data['id'].str[-2:]

X_train = train_data.drop(columns=['mood', 'day'])
X_test = test_data.drop(columns=['mood', 'day'])

y_train = train_data['mood']
y_test = test_data['mood']

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20),
}

rf = RandomForestClassifier(random_state=42)

# Cross-entropy loss as the evaluation metric
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
rand_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=2, scoring=log_loss_scorer, random_state=42)
rand_search.fit(X_train, y_train)

idx_3 = np.where(y_train == 3)[0][0]
print('idx_3:', idx_3)
print(y_train.iloc[idx_3])
X_test.iloc[-1] = X_train.iloc[idx_3]
y_test.iloc[-1] = y_train.iloc[idx_3]


best_params = rand_search.best_params_
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

y_pred_probability = best_rf.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_probability)
y_pred = y_pred_probability.argmax(axis=1)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Cross-Entropy Loss:', logloss)

plt.plot(range(len(y_test)), y_test, color='red', label='Actual Mood', marker='o', linestyle='--', markersize=3)
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Mood', marker='o', linestyle='--', markersize=3)
plt.legend()
plt.title('Predicted vs. Actual Mood')
plt.show()

# print("Cross-Entropy Loss:", logloss)
