import random
import numpy as np

# Numbers and weights
nums = [1, 2, 3, 4, 5, 6]
pr = [1/44, 4/44, 9/44, 8/44, 10/44, 12/44]

# Generating 10000 random numbers with given probability weights
random_numbers = random.choices(nums, weights = pr, k = 10000)

# Expeected values
ex = sum(random_numbers)/10000
e3x2 = sum(3*np.multiply(random_numbers,random_numbers))/10000
pgeq4 = len(list(filter(lambda x: x >= 4, random_numbers)))/10000

# Output
print("E(X) =      ",ex)
print("E(3X^2) =   ",e3x2)
print("P(X >= 4) = ",pgeq4)