# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


x = np.linspace(-10, 10, 1000)
mu = 0
sigma = 1
f = stats.norm.pdf(x, mu, sigma)
F = stats.norm.cdf(x, mu, sigma)

# compute Expectation
# compute Variance
# Generate 1000 random variables
random_vars = np.random.normal(loc=mu, scale=sigma, size=1000)
print(f'Expectation = {np.mean(random_vars)}')
print(f'Variance = {np.var(random_vars)}')
print(f'Random variable\n{random_vars}')

# plot the Probability Density Function of the exponential distribution
plt.title('Gaussian' + '\n' + 'PDF')
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.plot(x, f)
plt.show()

# plot the Cumulative Distribution Function of the exponential distribution
plt.title('Gaussian' + '\n' + 'CDF')
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.plot(x, F)
plt.show()

# show histogram
plt.title('Histogram' + '\n' + 'Random variable 1000 samples')
plt.xlabel('Random variable')
plt.ylabel('Frequency')
plt.hist(random_vars, bins='auto')
plt.show()
