# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters for the continuous uniform distribution
a = 0  # lower bound
b = 1  # upper bound

# Generate values for x
x_uniform = np.linspace(a, b, 1000)

# Probability Density Function (PDF) for the continuous uniform distribution
f_uniform = stats.uniform.pdf(x_uniform, a, b - a)

# Cumulative Distribution Function (CDF) for the continuous uniform distribution
F_uniform = stats.uniform.cdf(x_uniform, a, b - a)

# Generate 1000 random variables from the uniform distribution
random_vars_uniform = np.random.uniform(low=a, high=b, size=1000)
print(f'Expectation = {np.mean(random_vars_uniform)}')
print(f'Variance = {np.var(random_vars_uniform)}')
print(f'Random variable\n{random_vars_uniform}')

# Plot the Probability Density Function of the continuous uniform distribution
plt.title('Continuous Uniform' + '\n' + 'PDF')
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.plot(x_uniform, f_uniform)
plt.show()

# Plot the Cumulative Distribution Function of the continuous uniform distribution
plt.title('Continuous Uniform' + '\n' + 'CDF')
plt.xlabel('Stats')
plt.ylabel('Probability')
plt.plot(x_uniform, F_uniform)
plt.show()

# Show histogram for the random variables from the uniform distribution
plt.title('Histogram' + '\n' + 'Random variable 1000 samples (Uniform)')
plt.xlabel('Random variable')
plt.ylabel('Frequency')
plt.hist(random_vars_uniform, bins='auto')
plt.show()

