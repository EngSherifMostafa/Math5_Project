import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
p = 0.3  # Probability of success

# Generate Random Variable
x_values = np.array([0, 1])  # Possible values for a Bernoulli random variable
rv_bernoulli = stats.bernoulli(p)
random_variable = rv_bernoulli.rvs(size=1000)

# Plot PMF
plt.subplot(2, 2, 1)
pmf = rv_bernoulli.pmf(x_values)
plt.stem(x_values, pmf, basefmt='b', markerfmt='bo', linefmt='b-', label='Bernoulli pmf')
plt.title("Bernoulli Distribution PMF")
plt.xlabel("Random Variable")
plt.ylabel("Probability Mass")

# Plot CDF
plt.subplot(2, 2, 2)
cdf = rv_bernoulli.cdf(x_values)
plt.step(x_values, cdf, where='post', label='CDF')
plt.title("Bernoulli Distribution CDF")
plt.xlabel("Random Variable")
plt.ylabel("Cumulative Probability")

# Compute Expectation and Variance
expectation = rv_bernoulli.mean()
variance = rv_bernoulli.var()
print("Expectation:", expectation)
print("Variance:", variance)

# Generate 1000 Random Variables
random_sample = rv_bernoulli.rvs(size=1000)

# Plot Histogram of the Random Sample
plt.subplot(2, 2, 3)
plt.hist(random_sample, bins=[-0.5, 0.5, 1.5], color='skyblue', alpha=0.7, density=True, label='Random Sample')
plt.title("Histogram of Bernoulli Random Sample")
plt.xlabel("Random Variable")
plt.ylabel("Frequency")

# Show Plots
plt.tight_layout()
plt.show()
