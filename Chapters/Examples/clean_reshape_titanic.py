# Data cleaning and reshaping in Python
# Example with the built-in Titanic dataset

import pandas as pd
import seaborn as sns

# Step 1: Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

print("Original Titanic Dataset:")
print(df.head())

# Step 2: Identify missing values
print("\nMissing Values in Titanic Dataset:")
print(df.isnull().sum())

# Step 3: Handle missing values
# Option 1: Drop rows with any missing values
df_dropped = df.dropna(subset=['age', 'embarked'])  # Dropping rows with missing 'age' or 'embarked'

# Option 2: Fill missing values with a method
# Fill missing 'age' with the median age and 'embarked' with the most frequent value
df_filled = df.fillna({
    'age': df['age'].median(),      # Fill missing 'age' with the median age
    'embarked': df['embarked'].mode()[0],  # Fill missing 'embarked' with the most frequent port
    'embark_town': df['embark_town'].mode()[0],  # Fill missing 'embark_town' with the most frequent town
})

# Step 4: Reshape the DataFrame
# Example: Melting the DataFrame to long format (unpivot)
df_melted = pd.melt(df_filled, id_vars=['sex', 'class', 'survived'], 
                    value_vars=['age', 'fare'], var_name='Attribute', value_name='Value')

# Example: Pivoting the DataFrame (restructuring)
df_pivoted = df_melted.pivot_table(index=['sex', 'class', 'survived'], columns='Attribute', values='Value', aggfunc='mean')

# Step 5: View the results
print("\nCleaned Titanic Dataset (filled missing values):")
print(df_filled.head())

print("\nMelted Titanic DataFrame:")
print(df_melted.head())

print("\nPivoted Titanic DataFrame:")
print(df_pivoted.head())
