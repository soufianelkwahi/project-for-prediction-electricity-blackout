# Task 1: Load real PVGIS data from merged CSV (all cities)
import pandas as pd

# Load the merged real data file created from PVGIS CSVs
print("Loading real data from all_cities_real_data.csv...")
df = pd.read_csv("all_cities_real_data.csv")
df['date'] = pd.to_datetime(df['date'])

print(f"Data loaded successfully!")
print(f"Number of rows: {len(df)}")
print(f"Cities: {sorted(df['city'].unique())}")
print(df.head())

# Save a backup if needed
df.to_csv("backup_real_data.csv", index=False)
print("Backup saved as backup_real_data.csv")