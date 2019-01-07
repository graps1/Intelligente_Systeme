#!/usr/bin/env python3

import numpy as np
from random import random

# given dataset
bank_data = [
    [25, 40000, 'N'],
    [35, 60000, 'N'],
    [45, 80000, 'N'],
    [20, 20000, 'N'],
    [35, 120000, 'N'],
    [52, 18000, 'N'],
    [23, 95000, 'Y'],
    [40, 62000, 'Y'],
    [60, 100000, 'Y'],
    [48, 220000, 'Y'],
    [33, 150000, 'Y']
]

# Task 1.a.)
# Write a program for a 3NN classifier to decide if expect to encounter
# problems handing a loan of 142000 to someone at the age of 48

def classify_3nn(data, classes, target):
    # map to distance
    distances = [np.linalg.norm(np.array(d[:-1]) - np.array(target)) for d in data]
    distances_s = sorted(distances)

    cls_ctr = [0]*len(classes)
    for i in range(3):
        idx = distances.index(distances_s[i])
        cls = data[idx][2]
        cls_ctr[ classes.index(cls) ] += 1
    
    return classes[ cls_ctr.index(max(cls_ctr)) ]

example = [23, 95000]
print("classification of data:", classify_3nn(bank_data, ['Y', 'N'], example))

# Task 1.b.)
# Daten normalisieren
maxs = [max(el) for el in list(zip(*bank_data))[:-1]]
bank_data_norm = [[el[0]/maxs[0], el[1]/maxs[1], el[2]] for el in bank_data]
example_norm = [example[i]/maxs[i] for i in range(len(example))]

print("classification of normalized data:", classify_3nn(bank_data_norm, ['Y', 'N'], example_norm))

# Es gibt einen Unterschied: die Eukldische Distanz ist unterschiedlich bei beiden Methoden,
# weil die einzelnen Werte bei der Normalisierung alle gleich gewichtet werden!

# Task 2.a.)
N = 100
varX, varY = .5, .5
covar = 0
cov = np.array([[varX, covar], [covar, varY]])
r_centers = [[-2,2],[2,2],[0,-2]]

clusters = []
for rctr in r_centers:
    clusters += [np.random.multivariate_normal(np.array(rctr), cov, N)]

examples = np.concatenate((clusters[0], clusters[1], clusters[2])) 

# generate 3 random centers
ctrs = [[random(), random()] for i in range(3)]
convergence = False
while not convergence:
    pts = [[] for i in range(3)]
    for d in examples:
        nearest_ctr = min(ctrs, key=lambda ctr: [np.linalg.norm( d - np.array(ctr) )])
        nearest_ctr_idx = ctrs.index(nearest_ctr)
        pts[nearest_ctr_idx].append(d)

    n_ctrs = [list(np.mean(pt, axis=0)) for pt in pts]
    convergence = any([True if abs(np.linalg.norm(np.array(ctrs[i]) - np.array(n_ctrs[i]))) <= 0.01 else False for i in range(3)])
    ctrs = n_ctrs
    print([ [int(v*100+0.5)/100 for v in ctr] for ctr in ctrs ])

print("convergence reached: generated cluster centers %s vs. real cluster centers %s" % ([ [int(v*100+0.5)/100 for v in ctr] for ctr in ctrs ],r_centers))