import numpy as np
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
import time

def Psi(r, alpha):
    product = 1
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        product*=exp(-alpha*r2)
    return product

def local_energy_analytical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        print('a', DIMENSION*alpha-2*alpha**2*r2)
        E+=(DIMENSION*alpha-2*alpha**2*r2+0.5*r2)
    return E/NUMBER_OF_PARTICLES
    #psi = Psi(r, alpha)
    #deriv = (-2*alpha+4*alpha**2*r[0,0]**2)*psi
    #return -0.5*(1/psi)*deriv+0.5*r[0,0]**2

def local_energy_numerical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        print('n', -0.5*deriv2(r, alpha, DX))
        E+=0.5*(-deriv2(r, alpha, DX)+r2)
    return E/NUMBER_OF_PARTICLES
    #psi = Psi(r, alpha)
    #deriv = deriv2(r, alpha, 0.01)
    #return -0.5*(1/psi)*deriv+0.5*r[0,0]**2

def deriv2(r, alpha, dx):
    return (Psi(r-dx, alpha)-2*Psi(r, alpha)+Psi(r+dx, alpha))/(dx**2*Psi(r, alpha))

def local_energy_general(r, alpha, num=False):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        E+=0.5*r2
        if not num:
            E+=DIMENSION*alpha-2*alpha**2*r2
        else:
            E-=0.5*deriv2(r, alpha, DX)
    return E/NUMBER_OF_PARTICLES

def test(n, d):
    seed()
    step_size = 1.0
    r = np.zeros((n, d), np.double)
    for i in range(n):
        for j in range(d):
            r[i, j] = step_size*(random()-0.5)
    psi = Psi(r, 0.5)
    E_a = local_energy_analytical(r, 0.5)
    E_n = local_energy_numerical(r, 0.5)

    print('E_a=', E_a, ', E_n=', E_n)
    print(local_energy_general(r, 0.5), local_energy_general(r, 0.5, num=True))

NUMBER_OF_PARTICLES = 6
DIMENSION = 1
DX = 0.1
test(NUMBER_OF_PARTICLES, DIMENSION)
