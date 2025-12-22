# --- STEP 5: RANDOM FOREST + CLASS WEIGHTING + GRIDSEARCH ---

# Import necessary libraries (Safety)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Data is already prepared from STEP 4: X_train_scaled, y_train, X_test_scaled, y_test

# 1. Define Base Model with Class Imbalance Handling
# The option 'class_weight="balanced"' is crucial for this problem (Lab 4)
# It gives more weight to errors on the minority class (ANOMALY=1)
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
print("Random Forest model initialized with class_weight='balanced'.")

# 2. Define Hyperparameter Grid (GridSearch - Lab 6)
# We choose a reasonable grid to avoid excessive RAM usage or long runtime
param_grid = {
    'n_estimators': [100, 200],       # Number of trees in the forest
    'max_depth': [5, 10, 15],         # Maximum depth of the trees
    'min_samples_split': [5, 10]      # Minimum number of samples required to split a node
}

# 3. Choose Optimization Metric (Scoring)
# F1-score is the best compromise between Precision and Recall
# Macro F1-score is ideal as it treats both classes equally,
# which is essential for an imbalanced problem
f1_macro_scorer = make_scorer(f1_score, average='macro')

# 4. Initialize and Run GridSearchCV
# Using 3-fold cross-validation (cv=3) for faster execution
print("Starting GridSearch (this may take a few minutes)...")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=f1_macro_scorer,  # Optimize for macro F1-score
    cv=3,
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Fit on training data
grid_search.fit(X_train_scaled, y_train)

print("\n--- GridSearch Completed ---")
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Macro F1-Score (Cross-Validation):", grid_search.best_score_)
