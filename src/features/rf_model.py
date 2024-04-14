import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime
import pdb

def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

train_set = pd.read_csv('../../data/preprocessed/train_set.csv')

# Convert 'day' to day of the week (0=Monday, 6=Sunday)
train_set['day_of_week'] = pd.to_datetime(train_set['day']).dt.dayofweek

# Convert 'time' to part of the day (morning, afternoon, evening, night)
train_set['part_of_day'] = pd.to_datetime(train_set['time']).dt.hour.apply(get_part_of_day)

train_set = train_set.drop(['day', 'time'], axis=1)

# Encode the categorical 'part_of_day' column
label_encoder = LabelEncoder()
train_set['part_of_day'] = label_encoder.fit_transform(train_set['part_of_day'])
# Round mood values to discrete classes (1-10)
train_set['mood_class'] = train_set['mood'].round().astype(int)

features = train_set.drop(['id', 'mood', 'mood_class'], axis=1)
target = train_set['mood_class']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(features, target)
joblib.dump(rf_classifier, '../../data/models/rf_trained_model.joblib')
