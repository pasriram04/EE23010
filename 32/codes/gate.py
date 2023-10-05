import numpy as np 

n = 10000

# x is an even distribution of 1s and 7s, which is a distribution with mean 4 and standard deviation 9. 
# 100 samples of x each of size n are generated
x = np.random.choice([1, 7], size=(100,n))
y = np.zeros(100)

# The 100 values of y are defined, to calculate expected value
for i in range(100):
    y[i] = np.sum(x[i])/n

randvar = ((y - 4)**2)/n
print("Expected Value = ", np.round(np.sum(randvar)/100,3))