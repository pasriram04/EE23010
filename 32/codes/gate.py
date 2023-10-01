import numpy as np 

n = 100000
y = np.random.normal(4.0, 3.0/sqrt(n), n)
randvar = ((y - 4)**2)/n
print("Expected Value = ", np.round(np.sum(randvar)/n,3))
