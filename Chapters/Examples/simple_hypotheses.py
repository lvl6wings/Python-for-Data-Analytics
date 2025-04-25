import numpy as np
import scipy.stats as stats

# Sample data
np.random.seed(42)
data1 = np.random.normal(10, 5, 100)  # Data from normal distribution (mean = 10, std = 5)
data2 = np.random.normal(10, 5, 100)  # Another sample, same distribution

# Hypothesis 1: Correlation Test (Pearson)
# H0: No correlation, H1: There is a correlation
correlation, p_value_correlation = stats.pearsonr(data1, data2)
print("Pearson Correlation Test:")
print(f"Correlation: {correlation}, p-value: {p_value_correlation}\n")
if p_value_correlation < 0.05:
    print("Reject null hypothesis: There is a significant correlation.")
else:
    print("Fail to reject null hypothesis: No significant correlation.\n")

# Hypothesis 2: Z-Test (One-sample Z-test)
# H0: Sample mean = population mean, H1: Sample mean != population mean
# Population parameters: population mean = 10, population std = 5
population_mean = 10
population_std = 5
sample_mean = np.mean(data1)
sample_size = len(data1)
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

# Calculate the p-value for the two-tailed Z-test
p_value_z = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

print("Z-Test (One-sample Z-test):")
print(f"Z-score: {z_score}, p-value: {p_value_z}\n")
if p_value_z < 0.05:
    print("Reject null hypothesis: The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject null hypothesis: No significant difference between the sample mean and population mean.\n")

# Hypothesis 3: One-sample t-test
# H0: Sample mean = population mean, H1: Sample mean != population mean
# Population mean = 10
t_stat, p_value_t = stats.ttest_1samp(data1, population_mean)
print("One-sample t-Test:")
print(f"T-statistic: {t_stat}, p-value: {p_value_t}\n")
if p_value_t < 0.05:
    print("Reject null hypothesis: The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject null hypothesis: No significant difference between the sample mean and population mean.\n")
