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