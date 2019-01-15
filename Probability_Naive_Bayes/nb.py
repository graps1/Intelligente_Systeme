#!/usr/bin/env python3

import numpy as np


# we want do write a naive bayes classifier
# given the following dataset:

dataset = '''Id Colour Type Origin Stolen
1 red sports domestic yes
2 red sports domestic no
3 red sports domestic yes
4 yellow sports domestic no
5 yellow sports imported yes
6 yellow SUV imported no
7 yellow SUV imported yes
8 yellow SUV domestic no
9 red SUV imported no
10 red sports imported yes'''
print("training data:\n-------\n{}\n-------\n".format(dataset))


dataset = dataset.split("\n")
for idx,ex in enumerate(dataset):
    # split at ' ' and remove the first id (as it doesn't contain any meaning)
    dataset[idx] = ex.split(' ')[1:]

labels = dataset[0]
dataset = dataset[1:]

print("'0' is not stolen, '1' is stolen.")
classification = [1 if x[-1] == 'yes' else 0 for x in dataset]
for idx, ex in enumerate(dataset):
    dataset[idx] = ex[:-1]

attributes = set({})

for attr_idx, attr in enumerate( zip(*dataset) ):
    for a in attr:
        attributes.add( (attr_idx, a ))

# Naive Bayes:
# p(X|Y) = p(Y|X) p(X)/p(Y) = µ p(Y|X)p(X)
# p('stolen'='yes' | attrs) = µ p(attrs | 'stolen'='yes')p('stolen'='yes')
#                           = µ p(a_1 | 'stolen'='yes')p(a_2 | 'stolen'='yes')...
# p('stolen'='no'  | attrs) = µ p(attrs | 'stolen'='no')p('stolen'='no')

# P is the probability matrix which contains the information
# of a vehicle being stolen given the attributes

mapping = {}
p_stolen = sum(classification)/len(classification)
p_not_stolen = 1 - p_stolen

# calculate the probability of the attributes given that the vehicle is stolen
for ex_idx, ex in enumerate(dataset):
    for i,a in enumerate(ex):
        key = i,a,classification[ex_idx]
        if key not in mapping:
            mapping[key] = 0
        mapping[key] += 1

# calculates the probability of each attribute given the classification
def p_attrs(target_cls):
    ret = {}
    all_elmts_ct = sum([1 if t==target_cls else 0 for t in classification])
    for attr in attributes:
        key = attr[0],attr[1],target_cls
        p = 0
        if key in mapping:
            p = mapping[key]/all_elmts_ct
        ret[attr] = p
    return ret


def classify(target):
    res = []
    for cls in set(classification):
        p_attrs_given_cls = p_attrs(cls)
        p = classification.count(cls)/len(classification)
        for idx,t in enumerate(target):
            p *= p_attrs_given_cls[(idx,t)]
        res += [ (cls, p) ]
    return max(res, key=lambda v: v[1])


# we want to know the chance that the target is stolen
target = ['red', 'SUV', 'imported']
res = classify(target)
print("classification for {}: {}".format(target, res[0]))
