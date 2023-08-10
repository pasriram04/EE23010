import sys
sys.path.insert(0,'/home/sriram/Downloads/A/EE23010/codes/CoordGeo')  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

# Verifying circumcentre condition using BC
def verify_circumcentre_condition(A, B, C, x):
    # Convert to np vectors
    vector_A = np.array(A)
    vector_B = np.array(B)
    vector_C = np.array(C)
    vector_x = np.array(x)
    
    # Calculate midpoint of B and C
    midpoint_BC = (vector_B + vector_C) / 2
    
    # Calculate LHS
    LHS = np.dot((vector_x - ((vector_B + vector_C) / 2)), vector_B - vector_C)
    
    # Check if the dot product is zero
    #print(LHS)
    if np.isclose(LHS,0,1.0e-12):
        return True
    else:
        return False

# Triangle sides
A = np.array([1.0, -1.0])
B = np.array([-4.0, 6.0])
C = np.array([-3.0, -5.0])
out = ccircle(A,B,C)
O = out[0]
print("Circumcentre: ",O)
M = (B+C)/2.0

result = verify_circumcentre_condition(A, B, C, O)

if result:
    print("The condition is satisfied.")
else:
    print("The condition is not satisfied.")

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_MO = line_gen(O,M)

#Generating the circumcircle
[O,R] = ccircle(A,B,C)
x_circ= circ_gen(O,R)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_MO[0,:],x_MO[1,:],label='$bisector$')

#Plotting the circumcircle
plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
M = M.reshape(-1,1)

tri_coords = np.block([A,B,C,O])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.savefig('/home/sriram/Downloads/A/EE23010/figs/plot.png')
plt.show()
