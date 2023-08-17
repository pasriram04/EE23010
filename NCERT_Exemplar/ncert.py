from scipy.stats import bernoulli                       
import random

# Generate 1000 dice rolls
def dice_roll():
        dice = []
        for i in range(0,1000):
                dice.append(random.randint(1,6))                    
        return dice

# Dice rolls generation - 2 sets for Apoorv and 1 set for Peehu                                                 
apoorv_1 = dice_roll()                                  
apoorv_2 = dice_roll()                                  
peehu = dice_roll()                                                                                             
count_apoorv = 0                                        
count_peehu = 0

# Calculating products and squares and counting them    
for i in range(0,100):                                      
        if(apoorv_1[i]apoorv_2[i] == 36):                          
                count_apoorv = count_apoorv + 1                     
        if(peehu[i]*2 == 36):                                      
                count_peehu = count_peehu + 1

print("Probability of Apoorv getting a 36 as a product is ", (count_apoorv/1000.0))                             
print("Probability of Peehu getting a 36 as a square is ", (count_peehu/1000.0))