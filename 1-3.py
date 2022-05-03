import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
lmbd = 10
def f(t, u):
  return -lmbd * u
def rk_2(a, b, n, u0):
  h = (b - a) / n
  t = a
  exact_solution = ex_sol(u0)
  u = np.array([exact_solution(t) for t in [(a + h * i) for i in range(n +
 1)]], dtype=np.longdouble)
  y = np.array([0 for i in range(n + 1)], dtype=np.longdouble)
  y[0] = u0
  for i in range(n):
    k1 = f(t, y[i])
    k2 = f(t + h/2, y[i] + (h*k1)/2)
    y[i + 1] = y[i] + h * k2
    t = t+h
  d = max([abs(u[i] - y[i]) for i in range(n + 1)])
  return y, d
def ex_sol(u0):
  def es(t):
    return u0 * np.exp(-lmbd * t)
  return es
def solution1(a, b, n, u0):
  h = (b - a) / n
  t = np.array([a + h * i for i in range(n + 1)])
  es = ex_sol(u0)
  u = np.array([es(ti) for ti in t])
  y = rk_2(a, b, n, u0)[0]
  plt.plot(t, u, label="approximate solution", color='green')
  plt.plot(t, y, label="exact solution", linestyle="dotted",
 linewidth=2.5,color='purple')
  plt.show()
solution1(0, 5, 100, 1)
def f1(t, y1, y2): return ((-np.sin(t))/np.sqrt(1+np.exp(2*t))) + y1*(y1*y1 + y2*y2 - 1)
def f2(t, y1, y2): return ((np.cos(t))/np.sqrt(1+np.exp(2*t))) + y2*(y1*y1 + y2*y2 - 1)
def y1_es(t): return np.cos(t) / np.sqrt(1+np.exp(2*t))
def y2_es(t): return np.sin(t) / np.sqrt(1+np.exp(2*t))
def rk_2(y1, y2, t, h, f1, f2):
  k1 = f1(t, y1, y2)
  l1 = f2(t, y1, y2)
  k2 = f1(t + h / 2, y1 + (h * k1) / 2, y2 + (h * l1) / 2)
  l2 = f2(t + h / 2, y1 + (h * k1) / 2, y2 + (h * l1) / 2)
  y1_rk = y1 + h * k2
  y2_rk = y2 + h * l2
  return y1_rk, y2_rk
def rk(a, b, n, y1_0, y2_0):
  h = (b - a) / n
  t = a
  y1 = np.array([0 for i in range(n+1)], dtype=np.longdouble)
  y2 = np.array([0 for i in range(n+1)], dtype=np.longdouble)
  y1[0] = y1_0
  y2[0] = y2_0
  for i in range(n):
    y1_tmp, y2_tmp = rk_2(y1[i], y2[i], t, h, f1, f2)
    y1[i + 1] = y1_tmp
    y2[i + 1] = y2_tmp
    t = t + h
  return y1, y2
def rk_test(a, b, n):
  #exact solution for y1, y2
  y1_e = np.array([y1_es(t) for t in np.linspace(a, b, n + 1)],
 dtype=np.longdouble)
  y2_e = np.array([y2_es(t) for t in np.linspace(a, b, n + 1)],
 dtype=np.longdouble)
  #rk for y1,y2
  y1, y2 = rk(a, b, n, y1_e[0], y2_e[0])
  plt.plot(y1_e, label='exact solution for y1',linestyle="dotted",
 linewidth=2.5,
  color='purple')
  plt.plot(y1, label='approximate solution for y1',color='green')
  plt.legend()
  plt.show()
  plt.plot(y2_e, label='exact solution for y2',linestyle="dotted",
 linewidth=2.5,
  color='purple')
  plt.plot(y2, label='approximate solution for y2',color='green')
  plt.legend()
  plt.show()
def err_plt(a, b):
   n = np.array([1000 + 500*i for i in range(10)])
   h = (b - a) / n
   y1_errs = []
   y2_errs = []
   for i in range(len(n)):
     y1_exs = np.array([y1_es(t) for t in np.linspace(a, b, n[i] + 1)],
    dtype=np.longdouble)
     y2_exs = np.array([y2_es(t) for t in np.linspace(a, b, n[i] + 1)],
    dtype=np.longdouble)
     y1, y2 = rk(a, b, n[i], y1_exs[0], y2_exs[0])
     y1_errs.append(max(abs(y1_exs - y1)))
     y2_errs.append(max(abs(y2_exs - y2)))
   y1_errs = np.asarray(y1_errs)
   y2_errs = np.asarray(y2_errs)
   plt.plot(np.log(h), np.log(y1_errs), label='max errors for y1')
   plt.legend()
   plt.show()
   plt.plot(np.log(h), np.log(y2_errs), label='max errors for y2')
   plt.legend()
   plt.show()
   plt.plot(h, [y1_errs[i] / (h[i] ** 2) for i in range(len(h))],
  label='e/h^2 for y1')
   plt.legend()
   plt.show()
   plt.plot(h, [y2_errs[i] / (h[i] ** 2) for i in
  range(len(h))],label='e/h^2 for y2')
   plt.legend()
   plt.show()
n = 5000
rk_test(0, 5, n)
err_plt(0, 5)
