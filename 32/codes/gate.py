import numpy as np
import matplotlib.pyplot as plt

# Load data from the .dat file
data = np.loadtxt('output.dat')

# Extracting columns
x_values = data[:, 0]
expectation_theo = data[:, 1]
expectation_sim = data[:, 2]

# Plotting
plt.plot(x_values, expectation_theo, label='Theoretical Expectation')
plt.plot(x_values, expectation_sim, label='Simulated Expectation')
plt.yscale('log')  # Log scale for better visualization
plt.xlabel('Sample Size (n)')
plt.ylabel('Expectation')
plt.title('Theoretical vs Simulated Expectation')
plt.legend()
plt.show()
