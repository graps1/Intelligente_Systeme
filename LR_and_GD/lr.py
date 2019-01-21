#!/usr/bin/env python3

# generate dataset
s = [0, 30, 50, 80, 100, 130, 180]
c = [0, 3.5, 5, 6.8, 7.4, 8, 12]

# plot dataset
import matplotlib.pyplot as plt

N = len(s)
w_1 = N*sum([s[i] * c[i] for i in range(N)]) - sum(s)*sum(c)
w_1 = w_1/(N*sum([x**2 for x in s]) - sum(s)**2)
w_0 = 1/N*(sum(c)-w_1*sum(s))

def f(x):
    return w_0 + w_1*x

print("f(x) = {} + {}x".format(w_0, w_1))

plt.plot([i*20 for i in range(10)], [f(i*20) for i in range(10)])
plt.scatter(s,c)
plt.show()





