import numpy as np
from scipy.stats import randint
import matplotlib.pyplot as plt


a, b = 1, 6
# Mean and variance
low, high = 1, 7
mean, var = randint.stats(low, high)
print("Mean = ", mean)
print("Variance = ", round(var, 5))

# PMF
x = np.arange(a, b+1)
discrete_uniform_distribution = randint(a, b+1)
discrete_uniform_pmf = discrete_uniform_distribution.pmf(x)
plt.plot(x, discrete_uniform_pmf, 'bo', ms=8)
plt.vlines(x, 0, discrete_uniform_pmf, colors='b', lw=5, alpha=0.5)
plt.title('Uniform' + '\n' + 'PMF')
plt.xlabel('State')
plt.ylabel('Probability')
plt.show()

# CDF
x = np.arange(a, b+1)
discrete_uniform_distribution = randint(a, b+1)
discrete_uniform_cdf = discrete_uniform_distribution.cdf(x)
plt.plot(x, discrete_uniform_cdf, 'bo', ms=8)
plt.title('Uniform' + '\n' + 'CDF')
plt.xlabel('State')
plt.ylabel('Probability')
plt.show()

# Plotting Random Variable
R = randint .rvs(a, b, size=1000)
plt.title('Histogram' + '\n' + 'Random variable 5000 samples')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.hist(R)
plt.show()
