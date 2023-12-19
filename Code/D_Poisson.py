import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
lambd = 3  # Adjust lambda (mean parameter) according to your specific case

# Generate Random Variable
x = np.arange(0, 11)
rv_poisson = stats.poisson(lambd)
f_poisson = rv_poisson.pmf(x)

# Plot PMF
plt.plot(x, f_poisson, 'bo', ms=8, label='poisson pmf')
plt.vlines(x, 0, f_poisson, colors='b', lw=5, alpha=0.5)
plt.legend()
plt.title("Poisson Distribution PMF")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.show()

# Generate Random Sample of 1000 observations
random_sample = rv_poisson.rvs(size=1000)

# Plot CDF
cdf = rv_poisson.cdf(x)
plt.step(x, cdf, where='post', label='CDF')
plt.title("Poisson CDF")
plt.xlabel("Number of Events")
plt.ylabel("Cumulative Probability")

# Compute Expectation and Variance
expectation = rv_poisson.mean()
variance = rv_poisson.var()
print("Expectation:", expectation)
print("Variance:", variance)

# Show Plots
plt.tight_layout()
plt.show()

# Plot Histogram of the Random Sample
plt.hist(random_sample, bins=np.arange(0, 12) - 0.5, color='skyblue', alpha=0.7, density=True, label='Random Sample')
plt.title("Histogram of Poisson Random Sample")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.legend()
plt.show()
