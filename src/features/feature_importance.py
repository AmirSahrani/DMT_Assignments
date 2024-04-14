import joblib
import pandas as pd
import matplotlib.pyplot as plt

rf_classifier = joblib.load('../../data/models/rf_trained_model.joblib')

train_set = pd.read_csv('../../data/preprocessed/train_set.csv')
feature_names = train_set.drop(['id', 'mood'], axis=1).columns

importances = rf_classifier.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(importances)), feature_importances['Importance'], color='b', align='center')
plt.yticks(range(len(importances)), feature_importances['Feature'])
plt.gca().invert_yaxis()
plt.xlabel('Relative Importance')
plt.show()
