# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:26:43 2020

@author: 사용자
"""
import numpy as np
import random

f = open("../data/Cy3 em spectrum.txt", "r")
cy3str = f.readlines()

f = open("../data/Cy5 em spectrum.txt", "r")
cy5str = f.readlines()

cy3flt = [[0] * 2 for i in range(301)]
cy5flt = [[0] * 2 for i in range(301)]

# string to float
for i in range(301):
    for j in range(2):
        cy3flt[i][j] = float(cy3str[i].split("\t")[j])
        cy5flt[i][j] = float(cy5str[i].split("\t")[j])

# Linear combination of Cy3 and Cy5
xsum_cy = [[0] for i in range(301)]
ysum_cy = [[0] for i in range(301)]
ysum_cy3 = [[0] for i in range(301)]
ysum_cy5 = [[0] for i in range(301)]

for i in range(301):
    xsum_cy[i] = cy3flt[i][0]
for i in range(301):
    ysum_cy3[i] = cy3flt[i][1]
for i in range(301):
    ysum_cy5[i] = cy5flt[i][1]

n1 = input("The number of random numbers : ")
n = int(n1)

results = [[0] * 302 for i in range(n)]

for i in range(n):

    r = random.random()
    s = 1 - r

    for j in range(301):
        ysum_cy[j] = np.multiply(cy3flt[j][1], r) + np.multiply(cy5flt[j][1], s)

    maxi = max(ysum_cy)
    norm_cy = np.divide(ysum_cy, maxi)
    result = np.insert(norm_cy, 0, r)

    results[i] = result

a = 1
