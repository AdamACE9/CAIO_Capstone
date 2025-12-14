import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
print("Creating synthetic data center dataset...")

# Generate 500 hourly timestamps starting Jan 1, 2023
start_date = datetime(2023, 1, 1, 0, 0, 0)
timestamps = [start_date + timedelta(hours=i) for i in range(500)]

# Generate random data for each column
data = {
    'Time': timestamps,
    'Energy_Consumption_kWh': np.random.uniform(300, 500, 500),
    'Server_Load_percent': np.random.uniform(50, 100, 500),
    'Ambient_Temperature_C': np.random.uniform(18, 30, 500),
    'Cooling_Efficiency_percent': np.random.uniform(70, 95, 500)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('datacenter_data.csv', index=False)

print("\n[SUCCESS] Dataset created successfully!")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset saved as 'datacenter_data.csv'")
