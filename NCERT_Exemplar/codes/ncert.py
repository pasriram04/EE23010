import numpy as np
import matplotlib.pyplot as plt

# Generate 10000 dice rolls
def dice_roll():
        dice = np.array(np.random.randint(1, 7, size=10000))         
        return dice

# Dice rolls generation - 2 sets for Apoorv and 1 set for Peehu                                                 
apoorv_1 = dice_roll()                                  
apoorv_2 = dice_roll()                                  
peehu = dice_roll()                                                                                             
count_apoorv = [0]*37                                    
count_peehu = [0]*37

# Calculating products and squares and counting them  
prod = np.multiply(apoorv_1, apoorv_2)
sq = np.multiply(peehu, peehu)
count_apoorv = np.unique(prod, return_counts=True)
count_peehu = np.unique(sq, return_counts=True)

print("Apoorv: ")
print("Values:    ", *count_apoorv[0], sep='\t')
print("Frequency: ", *count_apoorv[1], sep='\t')
print("Peehu: ")
print("Values:    ", *count_peehu[0], sep='\t')
print("Frequency: ", *count_peehu[1], sep='\t')

# Plotting bar graphs using random variables
plt.bar(count_apoorv[0], count_apoorv[1])
plt.xlabel('Product of dice values')
plt.ylabel('Probability - Apoorv')
plt.title('Probability Distribution')
plt.show()

plt.bar(count_peehu[0], count_peehu[1])
plt.xlabel('Square of dice value')
plt.ylabel('Probability - Peehu')
plt.title('Probability Distribution')
plt.show()

# Cumulative Distribution Functions
cdf_apoorv = np.cumsum(count_apoorv[1])
cdf_peehu = np.cumsum(count_peehu[1])

print("Cumulative Distribution Functions: ")
print("Apoorv: ")
print("Values:    ", *count_apoorv[0], sep='\t')
print("Frequency: ", *cdf_apoorv, sep='\t')
print("Peehu: ")
print("Values:    ", *count_peehu[0], sep='\t')
print("Frequency: ", *cdf_peehu, sep='\t')

plt.bar(count_apoorv[0], cdf_apoorv)
plt.xlabel('Product of dice values')
plt.ylabel('CDF - Apoorv')
plt.title('Probability Distribution')
plt.show()

plt.bar(count_peehu[0], cdf_peehu)
plt.xlabel('Square of dice value')
plt.ylabel('CDF - Peehu')
plt.title('Probability Distribution')
plt.show()