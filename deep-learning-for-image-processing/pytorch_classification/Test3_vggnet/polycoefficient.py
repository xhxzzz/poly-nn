from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import numpy as np

x = np.arange(-10, 11, 1)
y = np.maximum(x, 0)
poly = lagrange(x, y)
coe = Polynomial(poly).coef
print(coe)
testX = np.arange(-10, 10, 1)
polyRes = []
for i in range(len(testX)):
    temp = 0
    tempx = testX[i]
    for j in range(len(coe)):
        temp += coe[j] * tempx**(len(coe) - 1 - j)
    print(f'({tempx}, {temp})')