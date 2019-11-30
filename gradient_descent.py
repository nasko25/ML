#!/usr/bin/env python

#import matplotlib.pyplot as plt
import numpy as np
from sympy import *

# ys = [6, 6, 10, 6, 10]
# ys = [5, 5, 6, 7, 10]
# ys = [5, 5, 5, 11, 9]
# plug in the y numbers here
ys = [6, 4, 10, 8, 12]

# def func(a, x, y):
#     return((a*x-y)**2)*(math.log((a*x-y)**2+1))

x = Symbol("x")
y = Symbol("y")
a = Symbol("a")

a_val = 0
steps = 0.001
accuracy = 0.1

func = ((a*x-y)**2)*(log((a*x-y)**2+1))
f = lambdify([a,x,y], diff(func, a))
sum = 0

sign = 1

# find the best 'a' that minimizes the error
print("Optimizing for a...")
for i in range(5):
    while sign != 0:
        for x_i in range(1, 6):
            sum += f(a_val, x_i, ys[x_i-1])
            sign = np.sign(sum)

        if sign == -1:
            a_val += steps
        else:
            a_val -= steps

        if sum < accuracy and sum > -accuracy:
            break

        sum = 0
    steps/=100
    accuracy/=10
    print("a =", a_val)

# print(func)
print("Found the best a =", a_val)
func = lambdify([a,x,y], func, 'numpy')
err = 0
for x_i in range(1, 6):
    err += func(a_val, x_i, ys[x_i-1])

print("Minimal error:",err)
# x_plot = np.linspace(-15,15,100)
# y_plot = []
# for i in x_plot:
#     y_plot.append(func(a_val, i, 1))
#
# plt.plot(x_plot,y_plot)
# plt.show()
