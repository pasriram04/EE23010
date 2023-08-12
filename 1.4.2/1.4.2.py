import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# line_gen from CoordGeo
def line_gen(A,B):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

# Vertices
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])

# Orthogonal Matrix
omat = np.array([[0,1],[-1,0]]) 

# Direction Vector
def dir_vec(P,Q):
  return Q-P
  
# Normal Vector 
def norm_vec(P,Q):
  return omat@dir_vec(P,Q)

# Unit Direction Vectors
uAB = norm_vec(A,B)/(np.linalg.norm(norm_vec(A,B)))
uCA = norm_vec(C,A)/(np.linalg.norm(norm_vec(C,A)))

# Unit Normal Vectors
nAB = omat@uAB
nCA = omat@uCA

# Midpoints
mAB = (A + B)/2.0
mCA = (A + C)/2.0

# Intersection of two lines
def find_intersection(point1, direction1, point2, direction2):
    P = np.block([[direction1], [direction2]])
    Q = np.array([point1@direction1, point2@direction2])
    intersection_point = np.linalg.inv(P)@Q
    return intersection_point

# Orthocentre Calculation
O = find_intersection(mAB,nAB,mCA,nCA)
print("Orthocentre: ",O)

# Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OAB = line_gen(O,mAB)
x_OAC = line_gen(O,mCA)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OAB[0,:],x_OAB[1,:],label='$Bisector of AB$')
plt.plot(x_OAC[0,:],x_OAC[1,:],label='$Bisector of AC$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)

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
plt.savefig('/home/sriram/Downloads/B/EE23010/EE23010/1.4.2/figs/plot.png')
plt.show()
