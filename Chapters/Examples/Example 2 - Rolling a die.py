# Modules 2 - Statistics & Probability using Scipy and Numpy
# Example 2 - Throwing a fair die

# Import necessary modules
import numpy as np
from scipy import stats

# A fair six-sided die is rolled, what are the odds a 4 will be observed?
# Define the nuber of outcomes (sides) and the target outcome
sides = 6
target_number = 4

# Since it is fair, all outcomes are equally likely
# Therefore, P(Rolling 4) = Number of target outcomes / Total number of outcomes
prob_target_number = 1 / sides
print(f"Probability of rolling a {target_number}: {prob_target_number}")

# Switching to statistics, the die throws get sampled
# Simulate rolling the die 10000 times
# Set the seed to a specific number for RNG reproduction
np.random.seed(0)
# The range(start, stop) method generates integers from start to stop - 1
rolls = np.random.choice(range(1, sides + 1), size = 10000)

# Calculate the proportion of times the target outcome 4 is rolled
simulated_prob = np.sum(rolls == target_number) / 10000
print(f"Simulated probability of rolling a {target_number}: {simulated_prob}")

# Comparing this with the theoretical probability
print(f"Difference between simulated and theoretical probability: {abs(simulated_prob - prob_target_number)}")


