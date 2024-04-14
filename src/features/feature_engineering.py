import pandas as pd
import numpy as np

data = pd.read_csv('../../data/preprocessed/day_data.csv')

# Round the mood values to the nearest integer (discrete classes of 1-10)
data['mood'] = data['mood'].round().astype(int)

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

# Define the number of days to look back for lagged features
n_days = 5
feature_columns = [column for column in data.columns if column.startswith('appCat.')]

# Generate lagged features for the specified number of days
for column in feature_columns:
    for lag in range(1, n_days + 1):
        data[f'{column}_lag{lag}'] = data.groupby('id')[column].shift(lag)

# Generate rolling window features for the specified number of days to understand the trends and volatility in the data
# For instance, a sudden increase in app usage or activity levels might be indicative of mood changes
for column in feature_columns:
    data[f'{column}_rolling_mean'] = data.groupby('id')[column].rolling(window=n_days, min_periods=1).mean().reset_index(level=0, drop=True)
    data[f'{column}_rolling_std'] = data.groupby('id')[column].rolling(window=n_days, min_periods=1).std().reset_index(level=0, drop=True)

# Change features: day over day differences to detect significant shifts in behavior or usage patterns
for column in feature_columns:
    data[f'{column}_daily_change'] = data.groupby('id')[column].diff().fillna(0)

data.drop(['time'], axis=1, inplace=True)

# Drop rows with NaN values that were created by shifting for lagged features
data.dropna(inplace=True)

data.to_csv('../../data/preprocessed/engineered_data.csv', index=False)
