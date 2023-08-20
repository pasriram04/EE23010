import random
import matplotlib.pyplot as plt

# Generate 10000 dice rolls
def dice_roll():
        dice = []
        for i in range(0,10000):
                dice.append(random.randint(1,6))                    
        return dice

# Dice rolls generation - 2 sets for Apoorv and 1 set for Peehu                                                 
apoorv_1 = dice_roll()                                  
apoorv_2 = dice_roll()                                  
peehu = dice_roll()                                                                                             
count_apoorv = [0]*37                                    
count_peehu = [0]*37

# Calculating products and squares and counting them    
for i in range(0,10000):                                      
        prod = apoorv_1[i]*apoorv_2[i]
        sq = peehu[i]*peehu[i]
        count_apoorv[prod] = count_apoorv[prod] + 1
        count_peehu[sq] = count_peehu[sq] + 1

for i in range(0,37):
        print("Probability of Apoorv getting a ",i," as a product is ", (count_apoorv[i]/10000.0))                             
        
for i in range(0,37):
        print("Probability of Peehu getting a ",i," as a square is ", (count_peehu[i]/10000.0))

# Initializing Arrays to store horizontal data and probability vertically, and also CDFs
x = [0]*37
cdf_apoorv = [0]*37
cdf_peehu = [0]*37
for i in range(0,37):
        x[i] = i
        count_apoorv[i] = count_apoorv[i]/10000.0
        count_peehu[i] = count_peehu[i]/10000.0
        if(i>0):
                cdf_apoorv[i] = cdf_apoorv[i-1] + count_apoorv[i]
                cdf_peehu[i] = cdf_peehu[i-1] + count_peehu[i]


# Plotting bar graphs using random variables
plt.bar(x, count_apoorv)
plt.xlabel('Product of dice values')
plt.ylabel('Probability - Apoorv')
plt.title('Probability Distribution')
plt.show()


plt.bar(x, count_peehu)
plt.xlabel('Square of dice value')
plt.ylabel('Probability - Peehu')
plt.title('Probability Distribution')
plt.show()


plt.bar(x, cdf_apoorv)
plt.xlabel('Product of dice values')
plt.ylabel('CDF - Apoorv ')
plt.title('Probability Distribution')
plt.show()


plt.bar(x, cdf_peehu)
plt.xlabel('Square of dice value')
plt.ylabel('CDF - Peehu')
plt.title('Probability Distribution')
plt.show()
