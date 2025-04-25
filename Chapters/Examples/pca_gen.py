# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------
# Step 1: Simulation Parameters
# ----------------------------------------
n_individuals = 150     # Number of individuals (samples)
n_snps = 300            # Number of SNPs (features)
n_populations = 3       # Number of population groups

# ----------------------------------------
# Step 2: Simulate Allele Frequencies for Each Population
# ----------------------------------------
# Allele frequencies are between 0.05 and 0.95 to avoid rare or fixed SNPs
allele_freqs = np.random.uniform(0.05, 0.95, size=(n_populations, n_snps))

# ----------------------------------------
# Step 3: Assign Each Individual to a Population
# ----------------------------------------
pop_labels = np.random.choice(n_populations, size=n_individuals)

# ----------------------------------------
# Step 4: Generate Genotype Data (0, 1, 2) Based on Hardy-Weinberg Equilibrium
# ----------------------------------------
genotype_matrix = np.zeros((n_individuals, n_snps))
for i in range(n_individuals):
    pop = pop_labels[i]
    probs = allele_freqs[pop]
    genotype_matrix[i] = np.random.binomial(2, probs)

# ----------------------------------------
# Step 5: Prepare DataFrame for Analysis
# ----------------------------------------
genotype_df = pd.DataFrame(genotype_matrix.astype(int))
genotype_df['Population'] = pop_labels

# ----------------------------------------
# Step 6: Standardize the Genotype Data
# ----------------------------------------
X = genotype_df.drop('Population', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# Step 7: Principal Component Analysis (PCA)
# ----------------------------------------
pca = PCA(n_components=10)  # Calculate more PCs for scree plot
pca_result = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_ * 100

# ----------------------------------------
# Step 8: Scree Plot – Visualizing Explained Variance
# ----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), explained_var[:10], marker='o', linestyle='-', color='darkblue')
plt.title('Scree Plot: Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# Step 9: PCA Scatter Plot – Visualizing First Two PCs
# ----------------------------------------
pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
pca_df['Population'] = pop_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Population', palette='Set2', s=80)
plt.title('PCA of Simulated SNP Genotype Data')
plt.xlabel(f'PC1 ({explained_var[0]:.1f}% Variance)')
plt.ylabel(f'PC2 ({explained_var[1]:.1f}% Variance)')
plt.legend(title='Population Group')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# Step 10: Basic Summary Statistics Check
# ----------------------------------------
# These help us understand the standardized data distribution
summary_stats = pd.DataFrame({
    'Mean': np.mean(X_scaled, axis=0),
    'Std Dev': np.std(X_scaled, axis=0),
    'Min': np.min(X_scaled, axis=0),
    'Max': np.max(X_scaled, axis=0)
})

print("Summary statistics for standardized features (first 5 SNPs):")
print(summary_stats.head())
