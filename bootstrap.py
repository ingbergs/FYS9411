import numpy as np
from math import exp, sqrt
from random import random, seed, choice
import matplotlib.pyplot as plt

N = 1000 #number of bootstraps

f = open('alpha.txt', 'r')
lines = f.readlines()
f.close()

alphas = []
for line in lines:
    alphas.append(float(line.split()[0]))

bootstrap = []
N_alpha = len(alphas)
for i in range(N):
    b = 0
    for j in range(N_alpha):
        b += choice(alphas)
    bootstrap.append(b/N_alpha)

plt.hist(bootstrap, bins=100)
plt.show()
