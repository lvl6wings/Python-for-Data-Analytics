# Data cleaning and reshaping with Python
# Examples with a generated dataset

import pandas as pd
import numpy as np

# Step 1: Creating a sample dataset with missing values
data = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', np.nan],
    'Age': [25, 30, np.nan, 22, 28],
    'Department': ['HR', 'IT', np.nan, 'Finance', 'Marketing'],
    'Salary': [50000, 60000, 55000, np.nan, 65000]
}

# Create a DataFrame
df = pd.DataFrame(data)

print("Original DataFrame with Missing Values:")
print(df)

# Step 2: Identifying missing values
print("\nMissing Values in the DataFrame:")
print(df.isnull().sum())

# Step 3: Handling Missing Data
# Option 1: Drop rows with any missing values
df_cleaned_dropna = df.dropna()

# Option 2: Fill missing values with a placeholder or method
df_cleaned_fillna = df.fillna({
    'Name': 'Unknown',         # Fill missing names with 'Unknown'
    'Age': df['Age'].mean(),   # Fill missing age with the mean age
    'Department': 'Not Assigned',  # Fill missing departments
    'Salary': df['Salary'].median()  # Fill missing salary with median salary
})

# Step 4: Reshaping the DataFrame
# Example: Melting the DataFrame (unpivot)
df_melted = pd.melt(df_cleaned_fillna, id_vars=['ID', 'Name'], value_vars=['Age', 'Department', 'Salary'],
                    var_name='Attribute', value_name='Value')

# Example: Pivoting the DataFrame (restructuring)
df_pivoted = df_melted.pivot(index='ID', columns='Attribute', values='Value')

# Step 5: Viewing the results
print("\nCleaned DataFrame (with filled missing values):")
print(df_cleaned_fillna)

print("\nMelted DataFrame:")
print(df_melted)

print("\nPivoted DataFrame:")
print(df_pivoted)
