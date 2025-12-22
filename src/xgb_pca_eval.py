# --- STEP 10: RE-EVALUATION OF XGBOOST ON PCA DATA ---

# Import libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preparation for XGBoost (Reusing the weighting ratio)
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight = count_neg / count_pos

# 2. Retrieve best hyperparameters found in STEP 7 for XGBoost
best_xgb_params = xgb_grid_search.best_params_

# 3. Train XGBoost model on PCA-transformed data
print("\n--- Training XGBoost on PCA Data ---")
xgb_pca_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    **best_xgb_params  # Use best hyperparameters
)

xgb_pca_model.fit(X_train_pca, y_train)

# 4. Prediction and Evaluation
y_pred_xgb_pca = xgb_pca_model.predict(X_test_pca)

xgb_pca_report = classification_report(
    y_test, y_pred_xgb_pca, target_names=['NORMAL (0)', 'ANOMALY (1)'], output_dict=True
)

# 5. Display results
print("\n--- Evaluation of XGBoost Model on PCA Data ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_pca):.4f}")
print("\nClassification Report (XGBoost + PCA):\n", classification_report(
    y_test, y_pred_xgb_pca, target_names=['NORMAL (0)', 'ANOMALY (1)']
))

# Confusion Matrix
cm_xgb_pca = confusion_matrix(y_test, y_pred_xgb_pca)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm_xgb_pca,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal (0)', 'Anomaly (1)'],
    yticklabels=['Normal (0)', 'Anomaly (1)']
)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix - XGBoost on PCA Data')
plt.show()

# 6. Final Comparison Table (Integrating PCA results)

# XGBoost FULL DATA results (from STEP 8)
xgb_full_precision = xgb_report['ANOMALY (1)']['precision']
xgb_full_recall = xgb_report['ANOMALY (1)']['recall']
xgb_full_f1 = xgb_report['ANOMALY (1)']['f1-score']
xgb_full_accuracy = xgb_report['accuracy']

# XGBoost PCA results
xgb_pca_precision = xgb_pca_report['ANOMALY (1)']['precision']
xgb_pca_recall = xgb_pca_report['ANOMALY (1)']['recall']
xgb_pca_f1 = xgb_pca_report['ANOMALY (1)']['f1-score']
xgb_pca_accuracy = xgb_pca_report['accuracy']

print("\n--- FINAL COMPARISON TABLE (XGBoost: Full Data vs PCA) ---")
print(f"Number of features used: {X_train_scaled.shape[1]} (Full Data) vs {NUM_COMPONENTS_PCA} (PCA)")
print("| Model | Accuracy | Precision (Anom.) | Recall (Anom.) | F1-Score (Anom.) |")
print("|:---|:---|:---|:---|:---|")
print(f"| XGBoost (Full Data) | {xgb_full_accuracy:.4f} | {xgb_full_precision:.2f} | {xgb_full_recall:.2f} | {xgb_full_f1:.2f} |")
print(f"| XGBoost (PCA) | {xgb_pca_accuracy:.4f} | {xgb_pca_precision:.2f} | {xgb_pca_recall:.2f} | {xgb_pca_f1:.2f} |")

# Final Conclusion on the relevance of PCA
if xgb_pca_f1 >= xgb_full_f1 * 0.95:  # If performance is maintained at 95%
    print(f"\nFinal Conclusion: PCA reduced the number of features from {X_train_scaled.shape[1]} to {NUM_COMPONENTS_PCA} "
          f"(approx. {100 * (1 - NUM_COMPONENTS_PCA / X_train_scaled.shape[1]):.0f}% reduction) while maintaining similar performance "
          f"(F1-Score preserved). This is excellent for model industrialization (training and inference speed).")
else:
    print("\nFinal Conclusion: PCA reduced the F1-Score performance (significant loss). We should keep all features for the final production model.")
