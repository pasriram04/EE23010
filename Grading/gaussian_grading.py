import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import os

# Parameter k
k = 0.586785

# Code begins here
data = pd.read_excel("marks.xlsx")      #reading input
df1 = data.loc[:,"Marks"]    #storing marks column
x = np.array(df1).reshape(-1, 1)

mu = x.mean()
var = x.var()
z = (x - mu)/np.sqrt(var)
N = x.size                              

# Plot the histogram with intervals of 3 marks
bin_width = 3
bins = np.arange(6, 72, bin_width)
plt.hist(x, bins=bins, density=True, alpha=0.6, color='b', label='Scores')

# Fit a Gaussian distribution to the data
mu, std = norm.fit(x)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_axis, mu, std)  # Scale the Gaussian
plt.plot(x_axis, p, 'k', linewidth=2, label='Gaussian Fit\n$\mu={:.2f}$, $\sigma={:.2f}$'.format(mu, std))

plt.title('Histogram with Scaled Gaussian Fit')
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.legend()
fig.show()

grades = []
s = ""
#Attach grades
for j in range(N):
    if (z[j] >= 2.5*k): s = 'A'
    elif (z[j] >= 1.5*k): s = 'A-'
    elif (z[j] >= 0.5*k): s = 'B'
    elif (z[j] >= -0.5*k): s = 'B-'
    elif (z[j] >= -1.5*k): s = 'C'
    elif (z[j] >= -2.5*k): s = 'C-'
    else: s = 'D'
    grades.append(s)

data["Grade"] = grades
data.to_excel("marks.xlsx",index = False)  #writing to file
fig = data['Grade'].value_counts().sort_index(ascending=False).plot.bar().get_figure()
ax = fig.gca()
ax.set_xlabel('Grade')
ax.set_ylabel('Number of Students')
ax.grid()
fig.tight_layout()
fig.show()