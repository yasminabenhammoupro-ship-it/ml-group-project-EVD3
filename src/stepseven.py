# --- STEP 7: XGBOOST + CLASS WEIGHTING + GRIDSEARCH ---

# Import necessary libraries
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import classification_report

# 1. Preparation for XGBoost
# XGBoost uses the 'scale_pos_weight' parameter for class imbalance handling.
# It is calculated as the ratio of negative samples to positive samples.
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight = count_neg / count_pos

print(f"Class ratio (0:Normal / 1:Anomaly) in the Training Set: {count_neg}:{count_pos}")
print(f"Calculated scale_pos_weight parameter: {scale_pos_weight:.2f}")

# 2. Define Base Model with Imbalance Handling
# 'scale_pos_weight' replaces 'class_weight' for XGBoost.
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    use_label_encoder=False,
    eval_metric='logloss'  # Log-loss metric for binary classification
)

# 3. Define Hyperparameter Grid (Smaller than RF for runtime reasons)
xgb_param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [3, 5, 7],      # Maximum depth
    'learning_rate': [0.1]       # Learning rate
}

# 4. Choose Optimization Metric (Macro F1-Score)
f1_macro_scorer = make_scorer(f1_score, average='macro')

# 5. Initialize and Run GridSearchCV
print("\nStarting GridSearch for XGBoost (this may take a few minutes)...")

xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring=f1_macro_scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Fit on training data (warning: can be long)
xgb_grid_search.fit(X_train_scaled, y_train)

print("\n--- XGBoost GridSearch Completed ---")
print("Best Hyperparameters:", xgb_grid_search.best_params_)
print("Best Macro F1-Score (Cross-Validation):", xgb_grid_search.best_score_)
