# --- STEP 4: DATA SPLIT & BASELINE MODELING ---

# Import libraries (Safety)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Separate Features (X) and Target (y)
X = df.drop(['machine_status', 'target'], axis=1)
y = df['target']

# 2. Split data (Time-based splitting)
train_size = int(len(df) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"\nTraining set size: {X_train.shape[0]} ({round(X_train.shape[0]/len(df)*100)}%)")
print(f"Testing set size: {X_test.shape[0]} ({round(X_test.shape[0]/len(df)*100)}%)")

# 3. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Implement Baseline Model: Logistic Regression
print("\n--- Training Baseline Model (Logistic Regression) ---")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 5. Prediction and Evaluation
y_pred = log_reg.predict(X_test_scaled)

# --- Model Evaluation ---
print("\n--- Baseline Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report (Key element for analysis)
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=['NORMAL (0)', 'ANOMALY (1)']
))

# Confusion Matrix (Key Visualization)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal (0)', 'Anomaly (1)'],
    yticklabels=['Normal (0)', 'Anomaly (1)']
)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
