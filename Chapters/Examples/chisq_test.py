import seaborn as sns
import pandas as pd
import scipy.stats as stats

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Create a contingency table for Gender and Survived
contingency_table = pd.crosstab(titanic['sex'], titanic['survived'])

# Display the contingency table
print("Contingency Table:")
print(contingency_table)

# Perform the Chi-Square Test for Independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print results
print("\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2_stat}")
print(f"p-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies Table: \n{expected}")

# Interpretation of results
if p_value < 0.05:
    print("\nReject null hypothesis: There is a significant association between Gender and Survival.")
else:
    print("\nFail to reject null hypothesis: There is no significant association between Gender and Survival.")
