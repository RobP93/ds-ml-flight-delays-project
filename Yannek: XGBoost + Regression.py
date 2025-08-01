# SESSION SETUP ================================================================
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
flights_import = pd.read_csv('./data/flights_train_cleaned.csv')


# DATA PRE-PROCESSING ==========================================================
# Copy data and remove id column
flights = flights_import.copy()
flights.drop('Unnamed: 0', axis=1)

# Fix format of date columns
flights['datop'] = pd.to_datetime(flights['datop'])
flights['std'] = pd.to_datetime(flights['std'])
flights['sta'] = pd.to_datetime(flights['sta'])

# Fix format of categorical columns
flights['id'] = flights['id'].astype('category')
flights['fltid'] = flights['fltid'].astype('category')
flights['depstn'] = flights['depstn'].astype('category')
flights['arrstn'] = flights['arrstn'].astype('category')
flights['status'] = flights['status'].astype('category')
flights['ac'] = flights['ac'].astype('category')