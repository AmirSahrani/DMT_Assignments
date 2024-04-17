import pandas as pd
import numpy as np
import pdb

filepath = '../../data/preprocessed/day_data.csv'  
data = pd.read_csv(filepath)

# Round the mood values to the nearest integer (discrete classes of 1-10)
data['mood'] = data['mood'].round().astype(int)

# Convert 'day' and 'time' columns to datetime
data['day'] = pd.to_datetime(data['day'])

# Temporal features to capture the cyclical nature of human behavior 
# to learn mood fluctuations depending on the time of day (morning vs. night) or day of the week (weekday vs. weekend)
data['day_of_week'] = data['day'].dt.dayofweek
data['day_of_month'] = data['day'].dt.day
data['month'] = data['day'].dt.month

# Sort data by 'id' and 'day'
data_sorted = data.sort_values(by=['id', 'day'])

# Create lagged features for the past 5 days for each individual
feature_columns = [ 'activity','circumplex.arousal','circumplex.valence', 'mood']
for feature in feature_columns:
    for lag in range(1, 6):
        data_sorted[f'{feature}_lag_{lag}'] = data_sorted.groupby('id')[feature].shift(lag)

# Fill NaN values with zeros resulting from lagging
lag_range = range(1, 6)
fillna_dict = {f'{feature}_lag_{lag}': 0 for feature in feature_columns for lag in lag_range}
data_cleaned = data_sorted.fillna(fillna_dict)

if 'mood' in data_cleaned.columns:
    cols = [col for col in data_cleaned.columns if col != 'mood']
    cols.append('mood')

    data_cleaned = data_cleaned[cols]
else:
    print("The 'mood' column does not exist in the DataFrame.")

engineered_data_path = '../../data/preprocessed/new_engineered_data.csv' 
data_cleaned.to_csv(engineered_data_path, index=False)

print(f'Engineered data saved to {engineered_data_path}')
