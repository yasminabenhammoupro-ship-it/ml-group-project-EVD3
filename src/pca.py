# --- STEP 9: PRINCIPAL COMPONENT ANALYSIS (PCA) ---

# Import libraries
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# The data X_train_scaled and X_test_scaled are used

# 1. Apply PCA on the standardized training data
# We specify the number of components to retain 95% of the variance
pca = PCA(n_components=0.95, random_state=42)

# Fit PCA on the training data
X_train_pca = pca.fit_transform(X_train_scaled)

# Apply the transformation on the test data
X_test_pca = pca.transform(X_test_scaled)

print("\n--- PCA Completed ---")
print(f"Initial number of features: {X_train_scaled.shape[1]}")
print(f"Number of principal components retained (for 95% variance): {pca.n_components_}")

# Explained variance analysis (Plot)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axvline(x=pca.n_components_, color='r', linestyle='--', label='95% Variance Retained')
plt.title('Cumulative Explained Variance by Principal Component')
plt.legend()
plt.grid()
plt.show()

# Store PCA results for the report
NUM_COMPONENTS_PCA = pca.n_components_
