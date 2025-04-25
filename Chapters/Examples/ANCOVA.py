import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm

# Generate synthetic data for ANCOVA
np.random.seed(42)
n = 200

# Generate groups A and B, continuous covariate (age), and dependent variable (income)
group = np.random.choice(['A', 'B'], size=n)  # Two groups
age = np.random.normal(30, 10, size=n)  # Covariate: Age
group_effect = np.where(group == 'A', 10, 0)  # Group A has an effect of 10 on income
income = 30000 + 500 * age + group_effect + np.random.normal(0, 1000, size=n)  # Income with some noise

# Create a DataFrame
df = pd.DataFrame({'group': group, 'age': age, 'income': income})

# Check normality of residuals
model = ols('income ~ group + age', data=df).fit()
residuals = model.resid
shapiro_residuals = stats.shapiro(residuals)

print("Shapiro-Wilk Test for Residuals Normality:")
print(f"p-value: {shapiro_residuals.pvalue}")

# Check homogeneity of variance using Levene's test
levene_stat, levene_p = stats.levene(df[df['group'] == 'A']['income'], df[df['group'] == 'B']['income'])
print("\nLevene's Test for Homogeneity of Variance:")
print(f"Levene's test p-value: {levene_p}")

# Check linearity and homogeneity of regression slopes
# Create interaction term (group * age) for testing homogeneity of regression slopes
interaction_model = ols('income ~ group * age', data=df).fit()
interaction_anova = anova_lm(interaction_model)
print("\nHomogeneity of Regression Slopes Test (Interaction Term):")
print(interaction_anova)

# Perform ANCOVA (without interaction term)
ancova_model = ols('income ~ group + age', data=df).fit()
ancova_results = anova_lm(ancova_model)
print("\nANCOVA Results:")
print(ancova_results)

# Decision about the null hypothesis
alpha = 0.05
if ancova_results['PR(>F)'][0] < alpha:
    print("\nWe reject the null hypothesis, indicating a significant effect of the group on income after controlling for age.")
else:
    print("\nWe fail to reject the null hypothesis, indicating no significant effect of the group on income after controlling for age.")

# Visualizing the F-statistic on the sampling distribution
# Assuming the distribution of the F-statistic for the ANCOVA
dfn = len(df['group'].unique()) - 1  # Between-groups degrees of freedom
dfd = len(df) - len(df['group'].unique()) - 1  # Residual degrees of freedom
x = np.linspace(0, 5, 1000)
y = stats.f.pdf(x, dfn, dfd)

# Plot the sampling distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="F-distribution", color='b')
plt.axvline(ancova_results['F'][0], color='r', linestyle='--', label=f"F-statistic: {ancova_results['F'][0]:.2f}")

# Marking the rejection region (alpha = 0.05)
plt.fill_between(x, y, where=(x > stats.f.ppf(1-alpha, dfn, dfd)), color='red', alpha=0.3, label="Rejection region (alpha=0.05)")

# Labels and legend
plt.title("F-statistic on Sampling Distribution for ANCOVA")
plt.xlabel("F-statistic values")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
