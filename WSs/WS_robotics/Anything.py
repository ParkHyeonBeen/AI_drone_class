import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from drawnow import *
import scipy.integrate as integrate
import matplotlib.animation as animation

A = np.arange(0, 8).reshape(2, 2, 2)
B = np.ones((2, 4))
D = np.matrix('-4 -5 6; -10 20 30')
Lst = [5, 6, 8]
zer = np.zeros((2,4))

#C = np.dot(A,D)

sgn = np.sign(D)

print(sgn)
print(A,type(A))
print(A[0, 0, 1],type(A))
print(Lst,type(Lst))
print(np.array(Lst),type(np.array(Lst)))
print(D,type(D))
print(zer,type(zer))



# A[:, :, 0] = np.transpose(np.arange(5,9).reshape(2,2))

print(C,type(C))
