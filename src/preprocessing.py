"""
Step 1 Preprocessing module for the Predictive Maintenance project.

This script handles:
- Data loading
- Data cleaning
"""

import pandas as pd
import numpy as np

print("Starting file repair process...")
try:
    with open('sensor.csv', 'r') as file:
        lines = file.readlines()

    with open('sensor_clean.csv', 'w') as out_file:
        for i, line in enumerate(lines):
            line_fixed = line.replace(';', ',')
            line_fixed = line_fixed.rstrip().rstrip(',')
            out_file.write(line_fixed + '\n')

    print("File 'sensor.csv' successfully repaired and saved as 'sensor_clean.csv'.")

except FileNotFoundError:
    print("\n🚨 ERROR: The file 'sensor.csv' was not found. Please make sure it has been uploaded in the Colab 'Files' panel.")
    raise

# Load the cleaned file and create the DataFrame
df = pd.read_csv('sensor_clean.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
df = df.set_index('timestamp')
cols_to_drop = ['Unnamed: 0', 'sensor_15']
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

print("--- Loading et cleaning Completed Successfully ---")

# --- STEP 2: PROBLEM FORMALIZATION (ASSURANCE) ---
import matplotlib.pyplot as plt
import seaborn as sns

# Fill missing values using forward and backward fill
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Create binary target variable
df['target'] = df['machine_status'].apply(lambda x: 0 if x == 'NORMAL' else 1)

print("DataFrame and target variable are ready for visualization.")
