import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Code to Generate binomial PMF
p = 0.5
n = 10
rv = stats.binom(n,p)
x = np.arange(11)
f = rv.pmf(x)
plt.plot(x, f, 'bo', ms=10);
plt.vlines(x, 0, f, colors='b', lw=5, alpha=0.5)
plt.title('Binomial' + '\n' + 'PMF')
plt.xlabel('State')
plt.ylabel('Probability')
plt.show()

# Code to generate 5000 Binomial random variables
p = 0.5
n = 10
X = np.random.binomial(n,p,size=5000)
plt.hist(X,bins='auto');
plt.title('Binomial' + '\n' + 'RV')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Code to compute the Mean and Variance of a Binomial random variable
p = 0.5
n = 10
rv = stats.binom(n,p)
M, V = rv.stats(moments='mv')
print("Mean:",M)
print("Variance:",V)

# Code to Generate binomial CDF
p = 0.5
n = 10
rv = stats.binom(n,p)
x = np.arange(11)
F = rv.cdf(x)
plt.plot(x, F, 'bo', ms=10);
plt.vlines(x, 0, F, colors='b', lw=5, alpha=0.5)
plt.title('Binomial' + '\n' + 'CDF')
plt.xlabel('State')
plt.ylabel('Probability')
plt.show()