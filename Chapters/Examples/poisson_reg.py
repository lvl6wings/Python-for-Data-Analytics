import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# Set random seed for reproducibility
np.random.seed(42)

# --------------------------
# Step 1: Simulated Study Data
# --------------------------

# Sample size
n_patients = 200

# Exposure time: months enrolled in health plan
months_enrolled = np.random.uniform(6, 24, size=n_patients)

# Predictor 1: Patient age (in years)
age = np.random.normal(loc=50, scale=10, size=n_patients)

# Predictor 2: Insurance status (1 = has insurance, 0 = no insurance)
has_insurance = np.random.binomial(n=1, p=0.7, size=n_patients)

# Outcome: Number of hospital visits per year (count data)
# We'll simulate this using a Poisson distribution with a log link function
# True underlying relationship: log(visits) = 0.2*age - 0.6*insurance + log(months_enrolled)
log_lambda = 0.5 + 0.02 * age - 0.6 * has_insurance + np.log(months_enrolled)
expected_visits = np.exp(log_lambda)
hospital_visits = np.random.poisson(expected_visits)

# Create DataFrame
df = pd.DataFrame({
    'hospital_visits': hospital_visits,
    'age': age,
    'has_insurance': has_insurance,
    'months_enrolled': months_enrolled
})

# --------------------------
# Step 2: Fit Poisson Regression Model
# --------------------------

# Fit the model using a log link and offset for months_enrolled
model = smf.glm(formula='hospital_visits ~ age + has_insurance',
                data=df,
                family=sm.families.Poisson(),
                offset=np.log(df['months_enrolled'])).fit()

# --------------------------
# Step 3: Add Residuals and Fitted Values
# --------------------------

df['predicted_visits'] = model.fittedvalues
df['residuals'] = df['hospital_visits'] - df['predicted_visits']
df['pearson_residuals'] = model.resid_pearson

# --------------------------
# Step 4: Check for Overdispersion
# --------------------------

pearson_chi2 = sum(df['pearson_residuals']**2)
overdispersion_ratio = pearson_chi2 / model.df_resid

# --------------------------
# Step 5: Visualize Residuals
# --------------------------

plt.figure(figsize=(12, 5))

# Residuals vs Fitted
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['predicted_visits'], y=df['residuals'])
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Hospital Visits')
plt.xlabel('Predicted Number of Visits')
plt.ylabel('Raw Residuals')

# Distribution of Pearson Residuals
plt.subplot(1, 2, 2)
sns.histplot(df['pearson_residuals'], kde=True, bins=30, color='teal')
plt.title('Distribution of Pearson Residuals')
plt.xlabel('Pearson Residuals')

plt.tight_layout()
plt.show()

# --------------------------
# Step 6: Print Model Summary
# --------------------------

print("\nPoisson Regression Model Summary:\n")
print(model.summary())

# --------------------------
# Step 7: Model Diagnostics
# --------------------------

print("\nOverdispersion Check:")
print(f"Pearson Chi-Square / Degrees of Freedom = {overdispersion_ratio:.2f}")
if overdispersion_ratio > 1.5:
    print("→ Evidence of overdispersion detected. Consider using a Negative Binomial model.")
else:
    print("→ No strong evidence of overdispersion.")

# Pseudo R-squared (McFadden's)
ll_null = model.null_deviance / -2
ll_model = model.deviance / -2
pseudo_r2 = 1 - (ll_model / ll_null)

print(f"\nModel AIC: {model.aic:.2f}")
print(f"Model Deviance: {model.deviance:.2f}")
print(f"Null Deviance: {model.null_deviance:.2f}")
print(f"McFadden's Pseudo R-squared: {pseudo_r2:.3f}")

# Likelihood Ratio Test vs Null Model
null_model = smf.glm(formula='hospital_visits ~ 1',
                     data=df,
                     family=sm.families.Poisson(),
                     offset=np.log(df['months_enrolled'])).fit()
lr_stat = 2 * (model.llf - null_model.llf)
lr_pvalue = chi2.sf(lr_stat, df=model.df_model)

print(f"\nLikelihood Ratio Test Statistic: {lr_stat:.2f}")
print(f"p-value: {lr_pvalue:.4f}")
if lr_pvalue < 0.05:
    print("→ Model is statistically significant compared to the null model.")
else:
    print("→ Model is NOT statistically significant.")
