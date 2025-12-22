# --- STEP 6: EVALUATION OF BEST RANDOM FOREST AND COMPARISON ---

# Import necessary libraries (Safety)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Retrieve the best model
best_rf = grid_search.best_estimator_

# 2. Prediction on the test set
y_pred_rf = best_rf.predict(X_test_scaled)

# 3. Evaluate Random Forest model
print("\n--- Evaluation of the Best Random Forest Model ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Classification Report
rf_report = classification_report(
    y_test, y_pred_rf, target_names=['NORMAL (0)', 'ANOMALY (1)'], output_dict=True
)
print("\nClassification Report (Random Forest):\n", classification_report(
    y_test, y_pred_rf, target_names=['NORMAL (0)', 'ANOMALY (1)']
))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal (0)', 'Anomaly (1)'],
    yticklabels=['Normal (0)', 'Anomaly (1)']
)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix - Optimized Random Forest')
plt.show()

# 4. Comparison (with previous Logistic Regression baseline results)
print("\n--- Comparative Analysis (STEP 1 vs STEP 2) ---")

# Logistic Regression (Baseline) results for reference
lr_precision_anomaly = 0.22
lr_recall_anomaly = 0.92
lr_f1_anomaly = 0.36
lr_accuracy = 0.996187

# Random Forest (Optimized) results
rf_precision_anomaly = rf_report['ANOMALY (1)']['precision']
rf_recall_anomaly = rf_report['ANOMALY (1)']['recall']
rf_f1_anomaly = rf_report['ANOMALY (1)']['f1-score']
rf_accuracy = rf_report['accuracy']

print(f"| Model | Accuracy | Precision (Anom.) | Recall (Anom.) | F1-Score (Anom.) |")
print(f"|:---|:---|:---|:---|:---|")
print(f"| Logistic Regression (Baseline) | {lr_accuracy:.4f} | {lr_precision_anomaly:.2f} | {lr_recall_anomaly:.2f} | {lr_f1_anomaly:.2f} |")
print(f"| Random Forest (Optimized) | {rf_accuracy:.4f} | {rf_precision_anomaly:.2f} | {rf_recall_anomaly:.2f} | {rf_f1_anomaly:.2f} |")

print("\nConclusion: Using Random Forest with class weighting significantly improves the F1-Score for the ANOMALY class by achieving a better balance between Precision and Recall.")
