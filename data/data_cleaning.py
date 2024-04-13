import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import missingno as mo

data = pd.read_csv("../data/raw/dataset_mood_smartphone.csv", index_col=0)
data['time'] = pd.to_datetime(data['time'])

# Convert time column to datetime format
data['time'] = pd.to_datetime(data['time'])
data['time'] = data['time'].dt.round('H')

# Pivot the data to create separate columns for each variable
data_pivot = data.pivot_table(index=['id', 'time'], columns='variable', values='value')

# Reset the index to flatten the column hierarchy
data_pivot.reset_index(inplace=True)


"""Removing Outliers Based on Visual Inspection"""

# Negative value from 'builtin' 
negative_builtin_indices = np.where(data_pivot['appCat.builtin'] < 0)
data_pivot = data_pivot.drop(negative_builtin_indices[0])

#Extreme positive value from 'entertainment' 
positive_ent_indices = np.where(data_pivot['appCat.builtin'] > 30000)
data_pivot = data_pivot.drop(positive_ent_indices[0])

#Extreme positive value from 'office' 
positive_office_indices = np.where(data_pivot['appCat.office'] > 30000)
data_pivot = data_pivot.drop(positive_office_indices[0])

#Extreme positive value from 'social' 
positive_social_indices = np.where(data_pivot['appCat.social'] > 30000)
data_pivot = data_pivot.drop(positive_social_indices[0])


"""Missing Values"""

# fill in the screen time for the entire day, sum the number of sms and call for the entire day
data_pivot['day'] = data_pivot['time'].dt.floor('D')
data_pivot['screen'] = data_pivot['screen'].groupby([data_pivot['id'], data_pivot['day']]).transform('sum')
data_pivot['sms'] = data_pivot['sms'].groupby([data_pivot['id'], data_pivot['day']]).transform('sum')
data_pivot['call'] = data_pivot['call'].groupby([data_pivot['id'], data_pivot['day']]).transform('sum')

for column in data_pivot.columns:
    for user in data_pivot['id'].unique():
        # window sliding fill
        if 0 < data_pivot[column].isnull().sum() < 500:
            data_pivot[column][data_pivot['id'] == user] = data_pivot[column][data_pivot['id'] == user].rolling(4, min_periods=1).mean()

day = data_pivot['time'].dt.floor('D')
data_pivot['day'] = day

day_data = data_pivot.groupby(['id', 'day']).mean()
day_data.reset_index(inplace=True)
# Sort data by id and day
day_data.sort_values(['id', 'day'], inplace=True)

for column in day_data.columns:
    if 'appCat.' in column:
        day_data[column].fillna(0, inplace=True)
        
day_data.dropna(subset=['mood'], inplace=True)

for column in ['activity', 'circumplex.arousal', 'circumplex.valence']:
    print(f"Processing column: {column}")
    for user in day_data['id'].unique():
        user_data = day_data[day_data['id'] == user][column]
        
        # Perform linear interpolation
        interpolated_data = user_data.interpolate(method='linear')
        
        # Check if there are still NaNs and attempt to fill them
        if interpolated_data.isna().any():
            # Fill NaNs at the beginning and end by forward and backward filling
            interpolated_data.fillna(method='ffill', inplace=True)
            interpolated_data.fillna(method='bfill', inplace=True)

        # Assign back the interpolated (and potentially forward/backward filled) data
        day_data.loc[day_data['id'] == user, column] = interpolated_data


# Save the cleaned data
print(f'''
Data cleaning complete.
Saving cleaned data to ../data/preprocessed/day_data.csv
    Total number of rows: {len(day_data)}
    Total number of columns: {len(day_data.columns)}
    Orginal data shape: {data_pivot.shape}

      ''')
day_data.to_csv("../data/preprocessed/day_data.csv", index=False)

train_split = day_data[np.random.choice(day_data['id'], int(0.8 * len(day_data['id'])))]
test_split = day_data[~day_data['id'].isin(train_split['id'])]

train_split.to_csv("../data/preprocessed/train_split.csv", index=False)
test_split.to_csv("../data/preprocessed/test_split.csv", index=False)
