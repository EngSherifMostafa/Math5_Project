# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Parameters for the uniform distribution
start = 0
width = 1

# Create a uniform distribution
rv = stats.uniform(loc=start, scale=width)

# Create an array of x values
x = np.linspace(-0.1, 1.1, 1000)

# Generate the PDF and CDF
pdf = rv.pdf(x)
cdf = rv.cdf(x)

# compute Expectation
# compute Variance
# Generate 1000 random variables
random_vars = np.random.uniform(low=-0.1, high=0.1, size=1000)
print(f'Expectation = {np.mean(random_vars)}')
print(f'Variance = {np.var(random_vars)}')
print(f'Random variable\n{random_vars}')

# plot the Probability Density Function of the uniform distribution
plt.plot(x, pdf)
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.title('Uniform' + '\n' + 'PDF')
plt.show()

# plot the Cumulative Distribution Function of the uniform distribution
plt.plot(x, cdf)
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.title('Uniform' + '\n' + 'CDF')
plt.show()

# show histogram
plt.title('Histogram' + '\n' + 'Random variable 1000 samples')
plt.xlabel('Random variable')
plt.ylabel('Frequency')
plt.hist(random_vars, bins='auto')
plt.show()
