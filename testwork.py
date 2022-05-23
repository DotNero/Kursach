import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import sympy
import math
import re
#Описание классического метода Рунге-Кутты IV-го порядка для двух уравнений
n = 10
left = 0
right = 2
h = (right-left)/n
t = 0
y1 = np.zeros(n)
y2 = np.zeros(n)
y1[0] = 1
y2[0] = 0


def y1dt(y1,y2,t):
    return(y1/(2+2*t) - 2 * t * y2)
def y2dt(y1,y2,t):
    return(y2/(2+2*t) + 2*t*y1)

ky1 = np.zeros(5)
ky2 = np.zeros(5)
t = np.zeros(n)


for i in range(n-1):
    ky1[1] = y1dt(y1[i], y2[i], t[i])
    ky2[1] = y2dt(y1[i], y2[i], t[i])

    ky1[2] = y1dt(y1[i] + (h / 2) * ky1[1], y2[i] + (h / 2) * (ky2[1]), t[i] + h / 2)
    ky2[2] = y2dt(y1[i] + (h / 2) * ky1[1], y2[i] + (h / 2) * (ky2[1]), t[i] + h / 2)

    ky1[3] = y1dt(y1[i] + (h / 2) * ky1[2], y2[i] + (h / 2) * (ky2[2]), t[i] + h / 2)
    ky2[3] = y2dt(y1[i] + (h / 2) * ky1[2], y2[i] + (h / 2) * (ky2[2]), t[i] + h / 2)

    ky1[4] = y1dt(y1[i] + h * ky1[3], y2[i] + h * ky2[3], t[i] + h)
    ky2[4] = y2dt(y1[i] + h * ky1[3], y2[i] + h * ky2[3], t[i] + h)

    y1[i + 1] = y1[i] + (h / 6) * (ky1[1] + 2*ky1[2] + 2*ky1[3] + ky1[4])
    y2[i + 1] = y2[i] + (h / 6) * (ky2[1] + 2*ky2[2] + 2*ky2[3] + ky2[4])
    t[i+1] = t[i]+h

print(y1)
print(y2)
plt.plot(t,y1)
plt.plot(t,y2)
#График погрешности измерений