import numpy as np
import matplotlib.pyplot as plt
r0 = 0.01
z0 = -5.5
phi0 = 0
alpha = 0.39
""""
y1=phi,y2=r,y3=Z
"""
def f1(t,y1,y2,y3):return ((-np.sin(y1)/y2)-y3/(alpha*alpha))
def f2(t,y1,y2,y3):return np.cos(y2)
def f3(t,y1,y2,y3):
 return -np.sin(y1)
def rk(fun1,fun2,fun3,y0,t0,tn,h):
 n = int((tn-t0)/h)
 t = np.array([t0+h*i for i in range(n+1)])
 y1 = np.zeros(shape = (n+1))
 y2 = np.zeros(shape = (n+1))
 y3 = np.zeros(shape = (n+1))
 y1[0]=y0[0]
 y2[0]=y0[1]
 y3[0]=y0[2]
 for i in range(1,n+1):
     k1y1 = fun1(t[i-1],y1[i-1],y2[i-1],y3[i-1])
     k1y2 = fun2(t[i-1],y1[i-1],y2[i-1],y3[i-1])
     k1y3 = fun3(t[i-1],y1[i-1],y2[i-1],y3[i-1])
     k2y1 = fun1(t[i-1]+h/2,y1[i-1]+k1y1*h/2,y2[i-1]+h/2*k1y2,y3[i-1]+h/2*k1y3)
     k2y2 = fun2(t[i-1]+h/2,y1[i-1]+k1y1*h/2,y2[i-1]+h/2*k1y2,y3[i-1]+h/2*k1y3)
     k2y3 = fun3(t[i-1]+h/2,y1[i-1]+k1y1*h/2,y2[i-1]+h/2*k1y2,y3[i-1]+h/2*k1y3)
 y1[i] = y1[i-1]+h*(k1y1+k2y1+k1y3)
 y2[i] = y2[i-1]+h*(k1y1+k2y1+k1y3)
 y3[i] = y3[i-1]+h*(k1y1+k2y1+k1y3)
 return t,y1,y2,y3
t,y1,y2,y3 = rk(f1,f2,f3,[phi0,r0,z0],0,1,0.1)
print('t:',t)
print('phi:',y1)
print('r:',y2)
print('z:',y3)
plt.plot(y2,y3)
plt.xlabel('r')
plt.ylabel('Z')
plt.show()
plt.plot(t,y2)
plt.xlabel('s')
plt.ylabel('r')
plt.show()
plt.plot(t,y3)
plt.xlabel('s')
plt.ylabel('Z')
plt.show()
plt.plot(t,y1)
plt.xlabel('s')
plt.ylabel('phi')
plt.show()