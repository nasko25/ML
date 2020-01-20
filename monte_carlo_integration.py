#!/usr/bin/env python

# The script calculates an integral with 5 variables of the form
# f(x) = integral(f1(x1) + integral(f2(x2) + integral(f3(x3) + integral(f4(x4) + integral(f5(x5) dx5) dx4) dx3) dx2) dx1)
# for the set of numbers between 0 and 1
# using the Monte Carlo algorithm. There is a condition for x4 and x5 - x_4^2 + x_5^2 <= 1

import random
import math

def cond(x4, x5):
    return x4**2+x5**2 <= 1
#
# def func(x, y):
#     return x*y
#
# x = random.uniform(0,1)
# y = random.uniform(0,1)
# points = []
#
# sum = 0
# N = 100000
#
# for i in range(N):
#     while (not cond(x, y) or [x, y] in points):
#         x = random.uniform(0,1)
#         y = random.uniform(0,1)
#     points.append([x, y])
#     sum += func(x, y)
#
# sum = (math.pi/4)*(1/N)*sum
# print(sum)

def f1(x1):
    #return x1**(4/11)
    return x1**(8/11)
def f2(x2):
    #return x2**(7/11)
    return x2**(10/11)

def f3(x3):
    #return x3**(4/11)
    return x3**(8/11)

def f4(x4):
    #return x4**(7/11)
    return x4**(9/11)
def f5(x5):
    #return math.sin(x5)
    return math.sin(x5)

N = 10000000
sum = 0
x1 = random.uniform(0,1)
x2 = random.uniform(0,1)
x3 = random.uniform(0,1)
x4 = random.uniform(0,1)
x5 = random.uniform(0,1)
for i in range(N):
    while (not cond(x4, x5)):
        x4 = random.uniform(0,1)
        x5 = random.uniform(0,1)
    sum += f1(x1) + f2(x2) + f3(x3) + f4(x4) + f5(x5)
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    x3 = random.uniform(0,1)
    x4 = random.uniform(0,1)
    x5 = random.uniform(0,1)

print(sum*math.pi/4*1/N)
