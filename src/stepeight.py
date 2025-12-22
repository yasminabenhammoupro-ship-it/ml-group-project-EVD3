# --- STEP 8: EVALUATION OF BEST XGBOOST AND FINAL COMPARISON ---

# Import necessary libraries (Safety)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Retrieve the best model
best_xgb = xgb_grid_search.best_estimator_

# 2. Prediction on the test set
y_pred_xgb = best_xgb.predict(X_test_scaled)

# 3. Evaluate XGBoost model
print("\n--- Evaluation of the Best XGBoost Model ---")
xgb_report = classification_report(
    y_test, y_pred_xgb, target_names=['NORMAL (0)', 'ANOMALY (1)'], output_dict=True
)
print("\nClassification Report (XGBoost):\n", classification_report(
    y_test, y_pred_xgb, target_names=['NORMAL (0)', 'ANOMALY (1)']
))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm_xgb,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal (0)', 'Anomaly (1)'],
    yticklabels=['Normal (0)', 'Anomaly (1)']
)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix - Optimized XGBoost')
plt.show()

# 4. Prepare Final Comparison (Add RF results)
# Retrieve Random Forest results for comparison
rf_precision_anomaly = rf_report['ANOMALY (1)']['precision']
rf_recall_anomaly = rf_report['ANOMALY (1)']['recall']
rf_f1_anomaly = rf_report['ANOMALY (1)']['f1-score']
rf_accuracy = rf_report['accuracy']

# Logistic Regression (Baseline) results for reference
lr_precision_anomaly = 0.22
lr_recall_anomaly = 0.92
lr_f1_anomaly = 0.36
lr_accuracy = 0.996187

# XGBoost results
xgb_precision_anomaly = xgb_report['ANOMALY (1)']['precision']
xgb_recall_anomaly = xgb_report['ANOMALY (1)']['recall']
xgb_f1_anomaly = xgb_report['ANOMALY (1)']['f1-score']
xgb_accuracy = xgb_report['accuracy']

print("\n--- FINAL COMPARISON TABLE (Baseline vs RF vs XGBoost) ---")
print("| Model | Accuracy | Precision (Anom.) | Recall (Anom.) | F1-Score (Anom.) |")
print("|:---|:---|:---|:---|:---|")
print(f"| Logistic Regression (Baseline) | {lr_accuracy:.4f} | {lr_precision_anomaly:.2f} | {lr_recall_anomaly:.2f} | {lr_f1_anomaly:.2f} |")
print(f"| Random Forest (Optimized) | {rf_accuracy:.4f} | {rf_precision_anomaly:.2f} | {rf_recall_anomaly:.2f} | {rf_f1_anomaly:.2f} |")
print(f"| XGBoost (Optimized) | {xgb_accuracy:.4f} | {xgb_precision_anomaly:.2f} | {xgb_recall_anomaly:.2f} | {xgb_f1_anomaly:.2f} |")

# Conclusion
best_f1 = max(lr_f1_anomaly, rf_f1_anomaly, xgb_f1_anomaly)
if best_f1 == xgb_f1_anomaly:
    print("\nFinal Conclusion: The optimized XGBoost model achieved the highest F1-Score, making it the most effective solution for anomaly detection on this imbalanced dataset.")
elif best_f1 == rf_f1_anomaly:
    print("\nFinal Conclusion: The optimized Random Forest model achieved the highest F1-Score, demonstrating its superiority over the linear model and XGBoost in this configuration.")
else:
    print("\nFinal Conclusion: Despite optimization, advanced models did not outperform the baseline model in terms of F1-Score.")
