import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Loading in the data - currently just the toy data from Amir
data = pd.read_csv('/Users/sophieengels/DMT_Assignments/data/preprocessed/toy_data.csv')
data['mood'] = data['mood'].round() #Can't be continuous

#Separating the feature data and the target data
X = data.drop(columns=['id', 'time', 'mood'])  
y = data['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the model and training it - number of estimators and random_state can be tuned later when we have the real data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

#Trying out the test set
y_pred = rf_classifier.predict(X_test)

#Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

