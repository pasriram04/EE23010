import numpy as np

# Vertices
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])

# Defining required vectors
D = (B+C)/2
E = (C+A)/2
F = (A+B)/2
G = (A+B+C)/3

# Calculating the distances
BG = np.linalg.norm(G-B)
GE = np.linalg.norm(E-G)

CG = np.linalg.norm(G-C)
GF = np.linalg.norm(F-G)

AG = np.linalg.norm(G-A)
GD = np.linalg.norm(D-G)

# Printing the ratios
print("BG/GE = ",(BG/GE))
print("CG/GF = ",(CG/GF))
print("AG/GD = ",(AG/GD))
