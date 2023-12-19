# import libraries
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# define the rate parameter (inverse of the scale parameter)
lambda_1 = 0.5

# generate an array of x values
x = np.linspace(0, 10, 100)

# compute Expectation
# compute Variance
# Generate 1000 random variables
random_vars = np.random.exponential(scale=1/lambda_1, size=1000)
print(f'Expectation = {np.mean(random_vars)}')
print(f'Variance = {np.var(random_vars)}')
print(f'Random variable\n{random_vars}')

# plot the Probability Density Function of the exponential distribution
plt.plot(x, stats.expon.pdf(x, scale=1/lambda_1))
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.title('Exponential' + '\n' + 'PDF')
plt.show()

# plot the Cumulative Distribution Function of the exponential distribution
plt.plot(x, stats.expon.cdf(x, scale=1/lambda_1))
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.title('Exponential' + '\n' + 'CDF')
plt.show()

# show histogram
plt.title('Histogram' + '\n' + 'Random variable 1000 samples')
plt.xlabel('Random variable')
plt.ylabel('Frequency')
plt.hist(random_vars, bins='auto')
plt.show()
