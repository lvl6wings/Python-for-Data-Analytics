import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the 'tips' dataset from seaborn
tips = sns.load_dataset('tips')

# Separate the data into two groups based on 'sex' (Male and Female)
male_tips = tips[tips['sex'] == 'Male']['tip']
female_tips = tips[tips['sex'] == 'Female']['tip']

# Check normality using Shapiro-Wilk test
shapiro_male = stats.shapiro(male_tips)
shapiro_female = stats.shapiro(female_tips)

print("Shapiro-Wilk Test for Normality:")
print(f"Male group p-value: {shapiro_male.pvalue}")
print(f"Female group p-value: {shapiro_female.pvalue}")

# If p-value < 0.05, we reject the null hypothesis of normality
normal_male = shapiro_male.pvalue > 0.05
normal_female = shapiro_female.pvalue > 0.05

# Check homogeneity of variance using Levene's test
levene_stat, levene_p = stats.levene(male_tips, female_tips)
print("\nLevene's Test for Homogeneity of Variance:")
print(f"Levene's test p-value: {levene_p}")

# Perform Independent Samples t-test
t_stat, p_value = stats.ttest_ind(male_tips, female_tips, equal_var=(levene_p > 0.05))

print("\nIndependent Samples T-test:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Decision about the null hypothesis
alpha = 0.05
if p_value < alpha:
    print("\nWe reject the null hypothesis, indicating a significant difference in tips between male and female.")
else:
    print("\nWe fail to reject the null hypothesis, indicating no significant difference in tips between male and female.")

# Visualizing the t-statistic on the sampling distribution
# Assuming normality for the sampling distribution of the t-statistic

# Define degrees of freedom
df = len(male_tips) + len(female_tips) - 2

# Create a t-distribution for plotting
x = np.linspace(-5, 5, 1000)
y = stats.t.pdf(x, df)

# Plot the sampling distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="t-distribution", color='b')
plt.axvline(t_stat, color='r', linestyle='--', label=f"t-statistic: {t_stat:.2f}")

# Marking the rejection region (alpha = 0.05)
plt.fill_between(x, y, where=(x > stats.t.ppf(1-alpha/2, df)) | (x < stats.t.ppf(alpha/2, df)), 
                 color='red', alpha=0.3, label="Rejection region (alpha=0.05)")

# Labels and legend
plt.title("T-Statistic on Sampling Distribution")
plt.xlabel("t-statistic values")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
