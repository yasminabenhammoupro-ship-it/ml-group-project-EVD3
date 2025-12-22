# --- STEP 2: PROBLEM FORMALIZATION (ASSURANCE) ---
import matplotlib.pyplot as plt
import seaborn as sns

# Fill missing values using forward and backward fill
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Create binary target variable
df['target'] = df['machine_status'].apply(lambda x: 0 if x == 'NORMAL' else 1)

print("DataFrame and target variable are ready for visualization.")
