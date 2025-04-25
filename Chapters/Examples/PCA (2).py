# PCA using Python
# Import tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.stats import bartlett
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca.fit(X_scaled)

# Explained Variance and Scree Plot
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot Scree Plot
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='g', label='Individual explained variance')
plt.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', color='b', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend()
plt.show()

# Principal Components (Scores) Plot (Biplot)
# Projecting the data onto the first two principal components
X_pca = pca.transform(X_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris.target_names[y], palette='Set1', s=100)
plt.title('PCA - 2D Plot (First Two Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.show()

# Correlation Matrix (Check for Linear Relationships)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(X_scaled).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# Bartlett's Test of Sphericity (Test for correlation)
chi2, p_value = bartlett(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], X_scaled[:, 3])
print(f'Bartlett\'s Test: Chi-Squared = {chi2:.2f}, p-value = {p_value:.4f}')

# Bartlett’s test p-value < 0.05 suggests that PCA is appropriate (reject the null hypothesis of no correlation).

# Visualizing PCA Loading Plot (How each feature contributes to each principal component)

plt.figure(figsize=(8, 6))
sns.heatmap(pca.components_, cmap='coolwarm', annot=True, xticklabels=iris.feature_names, yticklabels=[f'PC{i+1}' for i in range(len(pca.components_))])
plt.title('PCA Component Loadings')
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.show()

# Identify Outliers Using Distance from Mean
# Calculate Mahalanobis distance to detect outliers
mean = np.mean(X_scaled, axis=0)
cov_matrix = np.cov(X_scaled.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)
mahal_dist = np.array([np.sqrt((x - mean).T @ inv_cov_matrix @ (x - mean)) for x in X_scaled])

# Plot Mahalanobis distances
plt.figure(figsize=(8, 6))
plt.scatter(range(len(mahal_dist)), mahal_dist, color='r', s=50)
plt.axhline(y=2.5, color='b', linestyle='--', label='Threshold for Outliers')
plt.xlabel('Data Point')
plt.ylabel('Mahalanobis Distance')
plt.title('Mahalanobis Distance to Identify Outliers')
plt.legend()
plt.show()

# Outliers are typically those with a Mahalanobis distance greater than a threshold (e.g., 2.5).
