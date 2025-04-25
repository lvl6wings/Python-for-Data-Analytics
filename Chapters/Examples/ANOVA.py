import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the 'iris' dataset from seaborn
iris = sns.load_dataset('iris')

# Get the petal_length for each species
setosa = iris[iris['species'] == 'setosa']['petal_length']
versicolor = iris[iris['species'] == 'versicolor']['petal_length']
virginica = iris[iris['species'] == 'virginica']['petal_length']

# Check normality using Shapiro-Wilk test
shapiro_setosa = stats.shapiro(setosa)
shapiro_versicolor = stats.shapiro(versicolor)
shapiro_virginica = stats.shapiro(virginica)

print("Shapiro-Wilk Test for Normality:")
print(f"Setosa p-value: {shapiro_setosa.pvalue}")
print(f"Versicolor p-value: {shapiro_versicolor.pvalue}")
print(f"Virginica p-value: {shapiro_virginica.pvalue}")

# If p-value < 0.05, we reject the null hypothesis of normality
normal_setosa = shapiro_setosa.pvalue > 0.05
normal_versicolor = shapiro_versicolor.pvalue > 0.05
normal_virginica = shapiro_virginica.pvalue > 0.05

# Check homogeneity of variance using Levene's test
levene_stat, levene_p = stats.levene(setosa, versicolor, virginica)
print("\nLevene's Test for Homogeneity of Variance:")
print(f"Levene's test p-value: {levene_p}")

# Perform One-Way ANOVA
anova_model = ols('petal_length ~ species', data=iris).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA Results:")
print(anova_table)

# Decision about the null hypothesis
alpha = 0.05
if anova_table['PR(>F)'][0] < alpha:
    print("\nWe reject the null hypothesis, indicating a significant difference in petal lengths among species.")
else:
    print("\nWe fail to reject the null hypothesis, indicating no significant difference in petal lengths among species.")

# Post-hoc test: Tukey's HSD test for pairwise comparisons
if anova_table['PR(>F)'][0] < alpha:
    print("\nPerforming Tukey's HSD post-hoc test for pairwise comparisons:")
    tukey = pairwise_tukeyhsd(endog=iris['petal_length'], groups=iris['species'], alpha=0.05)
    print(tukey)

# Visualizing the F-statistic on the sampling distribution
# Create an F-distribution for plotting
dfn = len(np.unique(iris['species'])) - 1  # Between-groups degrees of freedom
dfd = len(iris) - len(np.unique(iris['species']))  # Within-groups degrees of freedom
x = np.linspace(0, 5, 1000)
y = stats.f.pdf(x, dfn, dfd)

# Plot the sampling distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="F-distribution", color='b')
plt.axvline(anova_table['F'][0], color='r', linestyle='--', label=f"F-statistic: {anova_table['F'][0]:.2f}")

# Marking the rejection region (alpha = 0.05)
plt.fill_between(x, y, where=(x > stats.f.ppf(1-alpha, dfn, dfd)), color='red', alpha=0.3, label="Rejection region (alpha=0.05)")

# Labels and legend
plt.title("F-statistic on Sampling Distribution")
plt.xlabel("F-statistic values")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
