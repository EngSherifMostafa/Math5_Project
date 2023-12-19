import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
p = 0.3  # Adjust probability parameter 'p' according to your specific case

# Generate Random Variable
x = np.arange(1, 11)
rv_geometric = stats.geom(p)
pmf = rv_geometric.pmf(x)  # Calculate PMF

# Plot PMF
plt.plot(x, pmf, 'bo', ms=8, label='geom pmf')
plt.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
plt.legend()
plt.title("Geometric Distribution PMF")
plt.xlabel("Number of Trials")
plt.ylabel("Probability")
plt.show()

# Generate Random Sample of 1000 observations
random_sample = rv_geometric.rvs(size=1000)

# Plot Histogram of the Random Sample
plt.hist(random_sample, bins=np.arange(1, 12) - 0.5, color='skyblue', alpha=0.7, density=True, label='Random Sample')
plt.title("Histogram of Geometric Random Sample")
plt.xlabel("Number of Trials")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot CDF
cdf = rv_geometric.cdf(x)
plt.step(x, cdf, where='post', label='CDF')
plt.title("Geometric CDF")
plt.xlabel("Number of Trials")
plt.ylabel("Cumulative Probability")

# Compute Expectation and Variance
expectation = rv_geometric.mean()
variance = rv_geometric.var()
print("Expectation:", expectation)
print("Variance:", variance)

# Show Plots
plt.tight_layout()
plt.show()
