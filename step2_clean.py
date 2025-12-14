import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('datacenter_data.csv')
print(f"Original dataset shape: {df.shape}")

# Convert Time to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Handle missing values with median
print("\nHandling missing values...")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Remove outliers using Z-score > 3
print("Removing outliers (Z-score > 3)...")
numeric_columns = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[numeric_columns]))
outliers_mask = (z_scores > 3).any(axis=1)
df_clean = df[~outliers_mask].copy()

# Add time-based features
print("Adding time-based features...")
df_clean['Hour'] = df_clean['Time'].dt.hour
df_clean['Day_of_Week'] = df_clean['Time'].dt.dayofweek
df_clean['Month'] = df_clean['Time'].dt.month

# Add Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_clean['Season'] = df_clean['Month'].apply(get_season)

# Save cleaned dataset
df_clean.to_csv('cleaned_data.csv', index=False)

print("\n[SUCCESS] Data cleaning completed!")
print(f"\nOriginal shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
print(f"\nCleaned data saved as 'cleaned_data.csv'")
print(f"\nNew columns added: Hour, Day_of_Week, Month, Season")
