# Logistic regression in python
# Import tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant

# Load data (Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Binarize the target variable for logistic regression (e.g., class 0 vs. class 1 and 2)
y_binary = (y == 0).astype(int)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for Extreme Outliers using Z-scores (absolute Z > 3 is considered extreme)
z_scores = np.abs(stats.zscore(X_train_scaled))
outliers = (z_scores > 3).all(axis=1)  # Consider any row with Z > 3 in any feature as an outlier
print(f"Outliers in training data: {np.where(outliers)[0]}")

# 3. Box-Tidwell Test for Linearity (check if the relationship between continuous features and logit is linear)
# This test involves adding interaction terms to check for non-linearity in the relationship
def box_tidwell_test(X_train_scaled, y_train):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train_scaled)
    X_poly_with_const = add_constant(X_poly)
    
    # Fit a logistic regression model with the interaction terms (non-linear features)
    model = LogisticRegression()
    model.fit(X_poly_with_const, y_train)
    return model

# Running Box-Tidwell test
box_tidwell_model = box_tidwell_test(X_train_scaled, y_train)
print(f"Box-Tidwell Test Model Coefficients: {box_tidwell_model.coef_}")

# Build and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Predictions and model evaluation
y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


