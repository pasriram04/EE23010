import numpy as np 
import matplotlib.pyplot as plt

def generate(m, n):
    # x is an even distribution of 1s and 7s, which is a distribution with mean 4 and standard deviation 9. 
    # m samples of x each of size n are generated
    x = np.random.choice([1, 7], size=(m,n))
    y = np.zeros(m)

    # The 100 values of y are defined, to calculate expected value
    y = np.array([np.sum(x[i])/n for i in range(m)])
    randvar = ((y - 4)**2)/n

    return np.sum(randvar)/m

# The value of m is kept low as we have to use a loop, so the computational time has to be reduced.
m = 100
x = np.linspace(100, 10000, 100, dtype = 'int') 
expectation_theo = 9/(x**2)
expectation_sim = [generate(m, i) for i in x]

# Plot
plt.plot(x, expectation_theo, label='Theoretical Expectation')
plt.plot(x, expectation_sim, label='Simulated Expectation')
plt.xlabel('Value of n')
plt.ylabel('Expectation')
plt.yscale('log')
plt.title('Theoretical vs Simulated Expectation')
plt.legend()
plt.show()
