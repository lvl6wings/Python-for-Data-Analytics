# Data visualization example using seaborn

# Import necessary modules
import seaborn as sns
import matplotlib.pyplot as plt

# Load the flights dataset, it comes along with the seaborn module
flights = sns.load_dataset('flights')

# 1. Dataset info like metadata, variable types, etc
# (not all print commands have to be submitted, this just makes visualization less cluttered)
print("Dataset Info:")
print(flights.info())
print("\n")

# 2. Descriptive statistics
print("Descriptive Statistics:")
print(flights.describe())
print("\n")

# 3. Check for missing data
print("Missing Values:")
print(flights.isnull().sum())
print("\n")

# 4. Univariate analysis
## a. Distribution of passengers
plt.figure(figsize=(8, 6))
sns.histplot(flights['passengers'], kde=True, color='blue')
plt.title('Distribution of Passengers')
plt.xlabel('Number of Passengers (in thousands)')
plt.ylabel('Frequency')
plt.show()

# 5. Bivariate analysis
## a. Relationship between year and passengers (line plot)
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='passengers', data=flights)
plt.title('Passengers over Time')
plt.xlabel('Year')
plt.ylabel('Number of Passengers (in thousands)')
plt.show()

## b. Passengers per month across different years (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='passengers', data=flights)
plt.title('Distribution of Passengers across Different Months')
plt.xlabel('Month')
plt.ylabel('Number of Passengers (in thousands)')
plt.show()

# 6. Multivariate analysis
## a. Heatmap for passengers by month and year (pivot table)
flights_pivot = flights.pivot_table('passengers', index='month', columns='year')

plt.figure(figsize=(12, 6))
sns.heatmap(flights_pivot, annot=True, fmt='d', cmap="YlGnBu", cbar_kws={'label': 'Number of Passengers (in thousands)'})
plt.title('Flight Passengers (in thousands) by Month and Year')
plt.show()

# 7. Seasonal trends - pairplot to visualize relationships between year and passengers across months
sns.pairplot(flights, hue='month', palette='Set2')
plt.suptitle('Pairplot of Passengers across Months and Years', y=1.02)
plt.show()

# 8. Correlation Heatmap (though we only have year and passengers, it will highlight relationships if extended to other datasets)
# (a heatmap is a useful way to cisualize a correlation matrix, but it doesn't have to only be used for correlation)
corr = flights.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

