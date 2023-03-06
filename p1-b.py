import numpy as np
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt

def Psi(r, alpha):
    product = 1
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        product*=exp(-alpha*r2)
    return product
    #return exp(-alpha*r[0,0]**2)

def local_energy_analytical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        E+=(DIMENSION*alpha-2*alpha**2*r2+0.5*r2)
    return E/NUMBER_OF_PARTICLES
    #psi = Psi(r, alpha)
    #deriv = (-2*alpha+4*alpha**2*r[0,0]**2)*psi
    #return -0.5*(1/psi)*deriv+0.5*r[0,0]**2

def deriv2(r, alpha, dx):
    return (Psi(r-dx, alpha)-2*Psi(r, alpha)+Psi(r+dx, alpha))/(dx**2*Psi(r, alpha))

def local_energy_numerical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2

        E+=(-0.5*deriv2(r, alpha, 0.001)+0.5*r2)
    return E/NUMBER_OF_PARTICLES
    #psi = Psi(r, alpha)
    #deriv = deriv2(r, alpha, 0.01)
    #return -0.5*(1/psi)*deriv+0.5*r[0,0]**2

def MonteCarlo(Energies, E_L):
    accept_rate=0
    n_MCC=100000
    step_size = 1.0 #4.0
    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    a = 0.2
    for ia in range(max_var):
        a += 0.05
        alpha[ia] = a
        energy = 0
        energy2 = 0


        # place the particles randomly
        for i in range(NUMBER_OF_PARTICLES):
            for j in range(DIMENSION):
                pos_old[i, j] = step_size*(random()-0.5)
        psi_old = Psi(pos_old, a)
        deltaE = E_L(pos_old, a)
        #print(deltaE)

        teller=0
        for MCC in range(n_MCC):
            # create a trial for new position
            for i in range(NUMBER_OF_PARTICLES):
                for j in range(DIMENSION):
                    pos_new[i, j] = pos_old[i, j] + step_size*(random()-0.5)
            psi_new = Psi(pos_new, a)

            #Metropolis test
            if random() < psi_new**2/psi_old**2:
                pos_old = pos_new.copy()
                psi_old = psi_new
                deltaE = E_L(pos_old, a)
                teller+=1
            energy += deltaE
            energy2 += deltaE**2

        energy /= n_MCC
        energy2 /= n_MCC
        variance = energy2 - energy**2
        error = np.sqrt(variance/n_MCC)
        Energies[ia] = energy
        Variances[ia] = variance
        accept_rate+=teller/n_MCC
    return Energies, alpha, accept_rate/max_var



NUMBER_OF_PARTICLES = 1
DIMENSION = 1
max_var = 10
alpha = np.zeros(max_var)
Energies_a = np.zeros(max_var)
Energies_n = np.zeros(max_var)
Variances = np.zeros(max_var)


Energies_a, alpha, accept_rate = MonteCarlo(Energies_a, local_energy_analytical)
print(accept_rate) #burde være ca 0.5-0.6, økes med mindre steglengde
Energies_n, alpha, accept_rate = MonteCarlo(Energies_n, local_energy_numerical)
print(accept_rate)


plt.plot(alpha, Energies_a, label='Analytical')
plt.plot(alpha, Energies_n, label='Numerical')
plt.legend()
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.show()
"""
plt.plot(alpha, Variances)
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.show()
"""
## finn ut hvor man skal variere posisionen, hvor skal man putte inn indeks i????
