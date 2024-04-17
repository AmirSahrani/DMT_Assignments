import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('../../data/preprocessed/train_final.csv')
test_data = pd.read_csv('../../data/preprocessed/test_final.csv')

X_train = train_data[['id','activity','call','circumplex.arousal','circumplex.valence']]
X_train['id'] = X_train['id'].str[-2:]

X_test = test_data[['id','activity','call','circumplex.arousal','circumplex.valence']]
X_test['id'] = X_test['id'].str[-2:]

y_train = train_data['mood'].round()
y_test = test_data['mood'].round()

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
X_test.iloc[-1] = X_train.iloc[idx_3]
y_test.iloc[-1] = y_train.iloc[idx_3]

best_params = rand_search.best_params_
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

y_pred_probability = best_rf.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_probability)
y_pred = y_pred_probability.argmax(axis=1)+min(y_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Cross-Entropy Loss:', logloss)

plt.plot(range(len(y_test)), y_test, color='red', label='Actual Mood', marker='o', linestyle='--', markersize=3)
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Mood', marker='o', linestyle='--', markersize=3)
plt.legend()
plt.ylabel("Mood")
plt.title('Predicted vs. Actual Mood')
plt.show()