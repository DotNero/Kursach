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


# for i in range(n-1):
#     ky1[1] = y1dt(y1[i], y2[i], t[i])
#     ky2[1] = y2dt(y1[i], y2[i], t[i])
#
#     ky1[2] = y1dt(y1[i] + (h / 2) * ky1[1], y2[i] + (h / 2) * (ky2[1]), t[i] + h / 2)
#     ky2[2] = y2dt(y1[i] + (h / 2) * ky1[1], y2[i] + (h / 2) * (ky2[1]), t[i] + h / 2)
#
#     ky1[3] = y1dt(y1[i] + (h / 2) * ky1[2], y2[i] + (h / 2) * (ky2[2]), t[i] + h / 2)
#     ky2[3] = y2dt(y1[i] + (h / 2) * ky1[2], y2[i] + (h / 2) * (ky2[2]), t[i] + h / 2)
#
#     ky1[4] = y1dt(y1[i] + h * ky1[3], y2[i] + h * ky2[3], t[i] + h)
#     ky2[4] = y2dt(y1[i] + h * ky1[3], y2[i] + h * ky2[3], t[i] + h)
#
#     y1[i + 1] = y1[i] + (h / 6) * (ky1[1] + 2*ky1[2] + 2*ky1[3] + ky1[4])
#     y2[i + 1] = y2[i] + (h / 6) * (ky2[1] + 2*ky2[2] + 2*ky2[3] + ky2[4])
#     t[i+1] = t[i]+h

# print(y1)
# print(y2)
# plt.plot(t,y1)
# plt.plot(t,y2)
#Обобщённый метод Рунге-Кутта для n-количества передаваемых функций
#y - это вдумерный массив. Первый индекс - номер точки, второй - номер переменной.
#L - это массив функций.
#h - это шаг
#t - это текущее знаение, на котором мы вычисляем
#n - это количество разбиений
#y0 - это вектор начальных значений функций
#нужен, ли метод, в котором мы создаем массив функций? Думаю, что нет, задам их вручную
#Count - количество уравнений

#Для решения моей задачи
#t = s
#y = [phi,r,Z]
#alpha = const
alpha = 0.39*10**(-2)
#Функции для передачи в массив функций L(здесь пока не хватает константы alpha)
def Func1(t, y:np):
    z = -math.sin(y[0])/y[1]-y[1]/alpha**2
    return(z)
def Func2(t, y:np):
    z = math.cos(y[0])
    return(z)
def Func3(t,y:np):
    z = math.sin(y[0])
#Обощённый метод Рунге-Кутта для n-го количества уравнений
def RungeKutta(count:int, L, y:np, h:float, n:int, t:float, y0:np):
    for i in range(count):
        y[0,i] = y0[i]
    k1 = np.zeros(count)
    k2 = np.zeros(count)
    k3 = np.zeros(count)
    k4 = np.zeros(count)
    for i in range(n-1):
        for j in range(count):
                k1[j] = L[j](t[i], y[i,:])
            #newY = np.zeros(count)
           # newY = y + h/2*k1
            #for j in range(count):
                #newY[j] = y[i][j] + h/2 * k1[j]
        for j in range(count):
            k2[j] = L[j](t[i]+h/2, y[i,:] + h/2*k1)
        for j in range(count):
            k3[j] = L[j](t[i]+h/2,y[i,:]+h*k2/2)
        for j in range(count):
            k4[j] = L[j](t[i]+h, y[i,:]+h*k3)
        for j in range(count):
            y[i+1,j] = y[i,j] + h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6
    return(y)

#Задача параметров для метода Рунге-Кутты
#y0 = [phi,r,Z]
y0 = np.array([0,0,0])

left = 0
right = 2

count = 3

n = 100

t = np.zeros(n)
for j in range(1,n):
    t[j] = left
    left += (right-left)/n
y = np.zeros((n,count))

L = [Func1,Func2,Func3]
O = np.zeros((n, count))

O = RungeKutta(count = count, L = L, y = y, h= h, n = n, t = t, y0 = y0 )
print(O)






