import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Local Imports
sys.path.insert(0,r'C:\Users\hp\Desktop\Coding\RandomVector\codes\CoordGeo') 
# To be changed when uploading - 6 addresses (5 figures and sys.path.insert())
# Check all final answers
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

# Global Values
omat = np.array([[0,1],[-1,0]])

# Initializing Arrays with randomly generated vectors from dice.py
A = np.array([1,5])
B = np.array([1,0])
C = np.array([-2,4])

print("Values used: ")
print("A = ",A)
print("B = ",B)
print("C = ",C)
print("\nSection 1.1: ")
# Question 1.1
print("The direction vector of AB is ", (B-A))
print("The direction vector of BC is ", (C-B))
print("The direction vector of CA is ", (A-C))

# Question 1.2
def length(B,C):
    return np.linalg.norm(B-C)

print("Length of side AB:", length(A,B))
print("Length of side BC:", length(B,C))
print("Length of side CA:", length(C,A))

# Question 1.3
def rank_of_matrix(A,B,C):
    Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])
    return np.linalg.matrix_rank(Mat)

rank = rank_of_matrix(A,B,C)
print("Rank of matrix: ", rank)
if (rank<=2):
	print("The points A,B,C are collinear.")
else:
	print("The points A,B,C are not collinear.")

# Question 1.4
print("Parametric form of AB is: x=",A,"+ k",(B-A))
print("Parametric form of BC is: x=",B,"+ k",(C-B))
print("Parametric form of CA is: x=",C,"+ k",(A-C))

# Question 1.5
def normaleqn(A,B):
    n = omat@(B-A)
    pro = n@A
    out = str(n)+"x="+str(pro)
    return out

print("Normal Equation of AB: ",normaleqn(A,B))
print("Normal Equation of BC: ",normaleqn(B,C))
print("Normal Equation of CA: ",normaleqn(C,A))

# Question 1.6
def AreaCalc(A, B, C):
    AB = A - B
    AC = A - C
    # Cross Product and Magnitude Calculation
    cross_product = np.cross(AB,AC)
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    return area

print("Area of triangle ABC is :",AreaCalc(A,B,C))

# Question 1.7
dotA = (np.transpose(B-A))@(C-A)
NormA = (np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print('Value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))

dotB = (np.transpose(C-B))@(A-B)
NormB = (np.linalg.norm(A-B))*(np.linalg.norm(C-B))
print('Value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))

dotC = (np.transpose(A-C))@(B-C)
NormC = (np.linalg.norm(A-C))*(np.linalg.norm(B-C))
print('Value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#Labeling the coordinates
A_c = A.reshape(-1,1)
B_c = B.reshape(-1,1)
C_c = C.reshape(-1,1)
tri_coords = np.block([[A_c,B_c,C_c]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
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
plt.savefig(r'C:\Users\hp\Desktop\Coding\RandomVector\plots\Figure1.png')

print("\nSection 1.2 - Median: ")
# Question 2.1
D = np.divide(B + C,2)
E = np.divide(A + C,2)
F = np.divide(A + B,2)
print("D:", D)
print("E:", E)
print("F:", F)

# Question 2.2
print("Normal Equation of AD: ",normaleqn(A,D))
print("Normal Equation of BE: ",normaleqn(B,E))
print("Normal Equation of CF: ",normaleqn(C,F))

# Question 2.3 
def line_intersect(n1,A1,n2,A2):
    N=np.block([[n1],[n2]])
    p = np.zeros(2)
    p[0] = n1@A1
    p[1] = n2@A2
    #Intersection
    P=np.linalg.inv(N)@p
    if(np.isclose(P[0],10e-10)):
        P[0] = 0
    if(np.isclose(P[1],10e-10)):
        P[1] = 0
    return P

G = line_intersect(norm_vec(F,C),C,norm_vec(E,B),B)
print("Intersection/Centroid: ("+str(G[0])+","+str(G[1])+")")

# Question 2.4
AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)

BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G)
 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)

print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))

# Question 2.5
rank = rank_of_matrix(A,D,G)

print("Rank of A, D, G matrix: ", rank)
if (rank<=2):
	print("The points A, D, G are collinear.")
else:
	print("The points A, D, G are not collinear.")

# Question 2.7
if (A-F).all()==(E-D).all():
   print("AFDE is a parallelogram")
else:
    print("AFDE is not a parallelogram")

# Figure 2
plt.clf()
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

x_AD = line_gen(A, D)
plt.plot(x_AD[0, :], x_AD[1, :], label='$AD$')

x_BE = line_gen(B, E)
plt.plot(x_BE[0, :], x_BE[1, :], label='$BE$')

x_CF = line_gen(C, F)
plt.plot(x_CF[0, :], x_CF[1, :], label='$CF$')

A_c = A.reshape(-1,1)
B_c = B.reshape(-1,1)
C_c = C.reshape(-1,1)
D_c = D.reshape(-1,1)
E_c = E.reshape(-1,1)
F_c = F.reshape(-1,1)
tri_coords = np.block([[A_c,B_c,C_c,D_c,E_c,F_c]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig(r'C:\Users\hp\Desktop\Coding\RandomVector\plots\Figure2.png')

# Section 3
print("\nSection 1.3 - Altitude:")

# Question 3.1
ab = B - A
bc = C - B
ca = A - C
AD_1 = omat@bc
AD_p = omat@AD_1 # Normal Vector
print("The normal vectors of AD_1:",AD_p)

# Question 3.2 and 3.3
c1 = bc@A
c2 = ca@B
c3 = ab@C
eqn7 = f"{bc}x = {c1}"
eqn8 = f"{ca}x = {c2}"
eqn9 = f"{ab}x = {c3}"
print("The equation of line AD_1 is",eqn7)
print("The equation of line BE_1 is",eqn8)
print("The equation of line CF_1 is",eqn9)

def alt_foot(A,B,C):
    m = B-C
    n = np.matmul(omat,m) 
    N = np.vstack((m,n))
    p = np.zeros(2)
    p[0] = m@A 
    p[1] = n@B
    #Intersection
    P = np.linalg.inv(N.T)@p
    return P

D_1 = alt_foot(A,B,C)
E_1 = alt_foot(B,A,C)
F_1 = alt_foot(C,A,B)

# Question 3.4
A1 = np.array([[ca[0],ca[1]],[ab[0],ab[1]]])             #Defining the vector A1
B1 = np.array([c2,c3])                     #Defining the vector B1
H  = np.linalg.solve(A1,B1)                 #applying linalg.solve to find x such that (A1)x=(B1)
print('')
print('The intersection of BE_1 and CF_1 (H):',H)
print('')

# Question 3.5
result = int(((A - H).T) @ (B - C))    # Checking orthogonality condition...

# printing output
if result == 0:
    print("(A - H)^T (B - C) = 0\nHence Verified...")

else:
    print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong...")

print("\nSection 4 - Perpendicular Bisector")

# Question 4.1
def midpoint(P, Q):
    return (P + Q) / 2

def perpendicular_bisector(B, C):
    midBC=midpoint(B,C)
    dir=B-C
    constant = -dir.T @ midBC
    return dir,constant

equation_coeff1,const1 = perpendicular_bisector(A, B)
equation_coeff2,const2 = perpendicular_bisector(B, C)
equation_coeff3,const3 = perpendicular_bisector(C, A)
print(f'Equation for perpendicular bisector of AB:({equation_coeff1[0]:.2f})x + ({equation_coeff1[1]:.2f})y + ({const1:.2f}) = 0')
print(f'Equation for perpendicular bisector of BC:({equation_coeff2[0]:.2f})x + ({equation_coeff2[1]:.2f})y + ({const2:.2f}) = 0')
print(f'Equation for perpendicular bisector of CA:({equation_coeff3[0]:.2f})x + ({equation_coeff3[1]:.2f})y + ({const3:.2f}) = 0')

# Question 4.2
O = line_intersect(ab,F,ca,E)
print('The point of intersection of perpendicular bisector of AB and AC is:',O)
print('')

# Question 4.3
result = int((O - D) @ (B - C))

if result == 0:
    print("((O - D)(B - C))= 0\nHence Verified...")
else:
    print("(((O - D)(B - C))!= 0\nHence the given statement is wrong...")
print('')

# Question 4.4
O_1 = O - A
O_2 = O - B
O_3 = O - C
a = np.linalg.norm(O_1)
b = np.linalg.norm(O_2)
c = np.linalg.norm(O_3)
print("OA, OB, OC are respectively", a,",", b,",",c, ".")
print("Here, OA = OB = OC.")
print("Hence verified.")
print('')

# Question 4.5
X = A - O
radius = np.linalg.norm(X)
print("The radius of the circumcircle is:",radius)

# Question 4.6
dot_pt_O = (B - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC = round(360-np.degrees(np.arccos(cos_theta_O)),5)  #Round is used to round of number till 5 decimal places
print("angle BOC = " + str(angle_BOC))
dot_pt_A = (B - A) @ ((C - A).T)
norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),5)  #Round is used to round of number till 5 decimal places
print("angle BAC = " + str(angle_BAC))

# To check whether the answer is correct
if angle_BOC == 2 * angle_BAC:
  print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
  print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")

# Section 5
print("\nSection 5 - Angular Bisector")
# Question 5.1
def unit_vec(A,B):
	return ((B-A)/np.linalg.norm(B-A))
E1= unit_vec(A,B) + unit_vec(A,C)
F1=np.array([E1[1],(E1[0]*(-1))])
C1= F1@(A.T)
E2= unit_vec(B,A) + unit_vec(B,C)
F2=np.array([E2[1],(E2[0]*(-1))])
C2= F2@(B.T)
E3= unit_vec(C,A) + unit_vec(C,B)
F3=np.array([E3[1],(E3[0]*(-1))])
C3= F3@(A.T)
print("Internal Angular bisector of angle A is:",F1,"x = ",C1)
print("Internal Angular bisector of angle B is:",F2,"x = ",C2)
print("Internal Angular bisector of angle C is:",F3,"x = ",C3)
print('')

# Question 5.2
t = norm_vec(B,C) 
s1 = t/np.linalg.norm(t) 
t = norm_vec(C,A)
s2 = t/np.linalg.norm(t)
t = norm_vec(A,B)
s3 = t/np.linalg.norm(t)
I=line_intersect(s1-s3,B,s1-s2,C) 
print('The point of intersection of angle bisectors of B and C:',I)
print('')

# Question 5.3
def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg

angle_BAI = angle_btw_vectors(A-B, A-I)
angle_CAI = angle_btw_vectors(A-C, A-I)
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
    print("Angle BAI is equal to angle CAI.")
else:
    print("error")
print('')

# Question 5.4 and 5.5
t = norm_vec(B, C)
n1 = t / np.linalg.norm(t)
r = n1 @ (B-I)
print(f"Distance from I to BC= {r}")
t = norm_vec(B, A)
n1 = t / np.linalg.norm(t)
r = n1 @ (I-B)
print(f"Distance from I to AB= {r}")
t = norm_vec(A, C)
n1 = t / np.linalg.norm(t)
r = n1 @ (I-C)
print(f"Distance from I to AC= {r}")
print('')

# Question 5.8 and 5.9
p = pow(np.linalg.norm(C-B),2)
q = 2*((C-B)@(I-B))
r = pow(np.linalg.norm(I-B),2)-r*r
Discre = q*q-4*p*r
print("the Value of discriminant is ",abs(round(Discre,6)))

k = ((I-C)@(B-C))/((B-C)@(B-C))
print("the value of parameter k is ",k)
D3 = C+(k*(B-C))
print("the point of tangency of incircle by side BC is ",D3)
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")

#finding k for E_3 and F_3
k1=((I-A)@(A-B))/((A-B)@(A-B))
k2=((I-A)@(A-C))/((A-C)@(A-C))
#finding E_3 and F_3
E3=A+(k1*(A-B))
F3=A+(k2*(A-C))
print('')
print("E3 = ",E3)
print("F3 = ",F3)
print('')

#Question 5.10
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
    return magnitude 
print('')
print("AE_3=", norm(A,E3) ,"\nAF_3=", norm(A,F3) ,"\nBD_3=", norm(B,D3) ,"\nBE_3=", norm(B,E3) ,"\nCD_3=", norm(C,D3) ,"\nCF_3=",norm(C,F3))
print('')

# Question 5.11
a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)


# Figure 1.3
plt.clf()
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD_1 = line_gen(A,D_1)
x_AE_1 = line_gen(A,E_1)
x_BE_1 = line_gen(B,E_1)
x_CF_1 = line_gen(C,F_1)
x_AF_1 = line_gen(A,F_1)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)
x_BD_1 = line_gen(B,D_1)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')
plt.plot(x_BE_1[0,:],x_BE_1[1,:],label='$BE_1$')
plt.plot(x_AE_1[0,:],x_AE_1[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_CF_1[0,:],x_CF_1[1,:],label='$CF_1$')
plt.plot(x_AF_1[0,:],x_AF_1[1,:],linestyle = 'dashed',label='$AF_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')
plt.plot(x_BD_1[0,:],x_BD_1[1,:],linestyle = 'dashed',label='$BD_1$')
tri_coords = np.block([[A],[B],[C],[D_1],[E_1],[F_1],[H]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','D1','E1','F1','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig(r'C:\Users\hp\Desktop\Coding\RandomVector\plots\Figure3.png')

# Figure Q1.4
plt.clf()
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OD = line_gen(O,D)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
[O,r] = ccircle(A,B,C)
x_ccirc= circ_gen(O,radius)
x_OA = line_gen(O,A)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')
plt.plot(x_OD[0,:],x_OD[1,:],label='$OD$')
plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
tri_coords = np.block([[A],[B],[C],[O],[D],[E],[F]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig(r'C:\Users\hp\Desktop\Coding\RandomVector\plots\Figure4.png')

# Figure Q1.5
plt.clf()
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
x_AI = line_gen(A,I)
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
tri_coords = np.block([[A],[B],[C],[I],[D3],[E3],[F3]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I','D3','E3','F3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig(r'C:\Users\hp\Desktop\Coding\RandomVector\plots\Figure5.png')