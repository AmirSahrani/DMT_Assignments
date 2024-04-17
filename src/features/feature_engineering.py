import pandas as pd
import numpy as np

filepath = '../../data/preprocessed/day_data.csv'  
data = pd.read_csv(filepath)

# Round the mood values to the nearest integer (discrete classes of 1-10)
data['mood'] = data['mood'].round().astype(int)

# Convert 'day' and 'time' columns to datetime
data['day'] = pd.to_datetime(data['day'])
data['time'] = pd.to_datetime(data['time'])

# Temporal features to capture the cyclical nature of human behavior 
# to learn mood fluctuations depending on the time of day (morning vs. night) or day of the week (weekday vs. weekend)
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek
data['day_of_month'] = data['time'].dt.day
data['month'] = data['time'].dt.month

# Convert hour into cyclic features (e.g., hour 23 is close to hour 0)
data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)

# Sort data by 'id' and 'day'
data_sorted = data.sort_values(by=['id', 'day'])

# Create lagged features for the past 5 days for each individual
feature_columns = [ 'activity','circumplex.arousal','circumplex.valence']
for feature in feature_columns:
    for lag in range(1, 6):
        data_sorted[f'{feature}_lag_{lag}'] = data_sorted.groupby('id')[feature].shift(lag)

# Drop rows with NaN values resulting from lagging
data_cleaned = data_sorted.dropna(subset=[f'{feature}_lag_5' for feature in feature_columns])

# Calculate the average mood for each day per individual
daily_mood = data_cleaned.groupby(['id', 'day'])['mood'].mean().reset_index(name='mood_avg')

# Merge the average mood with the cleaned data
data_with_avg_mood = data_cleaned.merge(daily_mood, on=['id', 'day'])

# Save the engineered data to a CSV file
engineered_data_path = '../../data/preprocessed/new_engineered_data.csv' 
data_with_avg_mood.to_csv(engineered_data_path, index=False)

print(f'Engineered data saved to {engineered_data_path}')
