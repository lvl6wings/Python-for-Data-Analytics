# Modules 2 - Statistics & Probability using Scipy and Numpy
# Example 1 - Drawing cards

# Import necessary modules
import numpy as np
from scipy import stats

# A standard deck of cards has 26 red and 26 black ones
# What is the probability of drawing a red card?
# Define the total number of cards and number of red cards
total_cards = 52
red_cards = 26

# Probability theory defines different stochastic models which give the probability of events
# From basic probability theory, P(Drawing a red card) = Number of red cards / Total number of cards
prob_red_card = red_cards / total_cards
print(f"Probability of drawing a red card: {prob_red_card}")
# Probability theory is a mathematical discipline and the conclusions drawn are deductive

# Statistical theory defines different methods to study data patterns using stochastic models
# They often involve sampling a known population, recording the findings, and calculating frequencies
# Simulating the drawing of 10000 cards and counting how many are red
# Seeting the random seed allows for RNG reproducibility
np.random.seed(0)
# Use the array data strucure to create the deck
cards = np.array(['red']*red_cards + ['black']*(total_cards - red_cards))
# Use the Numpy shuffle method to shuffle the deck
np.random.shuffle(cards)

# Draw 10000 cards
drawn_cards = np.random.choice(cards, size=10000)

# Calculate the proportion of red cards drawn
simulated_prob = np.sum(drawn_cards == 'red') / 10000
print(f"Simulated probability of drawing a red card: {simulated_prob}")

# Statistics involves empirical sampling and the conclusions drawn are inductive

# Comparing this with the theoretical probability
print(f"Difference between simulated and theoretical probability: {abs(simulated_prob - prob_red_card)}")

# This is a fundamental difference between probability and statistics:
# One creates theoretical models which explain events
# The other gathers data to test such models