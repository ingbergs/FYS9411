import numpy as np
from math import exp, sqrt
from random import random, seed, normalvariate
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
    d2 = 0
    dx2 = dx**2
    psi = Psi(r, alpha)
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r[i,j] += dx
            psi1 = Psi(r, alpha)
            r[i,j] -= 2*dx
            psi2 = Psi(r, alpha)
            r[i,j] += dx
            d2 += (psi1-2*psi+psi2)/dx2
    return d2/psi

def local_energy_numerical(r, alpha):
    E = 0
    r2 = 0
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r2+=r[i,j]**2
    E+=0.5*(-deriv2(r, alpha, 1e-6)+r2)
    return E/NUMBER_OF_PARTICLES
    #psi = Psi(r, alpha)
    #deriv = deriv2(r, alpha, 0.01)
    #return -0.5*(1/psi)*deriv+0.5*r[0,0]**2

def QuantumForce(r,alpha):
    qforce = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    for i in range(NUMBER_OF_PARTICLES):
        qforce[i] = -4*alpha*r[i] ## ERLEND: 2 i stedet for 4
    return qforce

def derivative(r):
    r_sum = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        r_sum += r2
    return -r_sum

def MonteCarlo(Energies, E_L):
    accept_rate=0
    n_MCC=100
    step_size = 1.0 #4.0
    dt=0.5 #2.6
    D=0.5
    gamma = 0.025 #0.01


    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    a = 0.3
    for ia in range(max_var):
        #a += 0.025
        alpha[ia] = a
        energy = 0
        energy2 = 0
        Psi_deriv = 0
        PsiE_deriv = 0


        # place the particles randomly
        for i in range(NUMBER_OF_PARTICLES):
            for j in range(DIMENSION):
                #pos_old[i, j] = step_size*(random()-0.5)
                pos_old[i, j] = normalvariate(0.0, 1.0)*np.sqrt(dt)
        psi_old = Psi(pos_old, a)
        qf_old = QuantumForce(pos_old,a)
        deltaE = E_L(pos_old, a)
        #print(deltaE)

        teller=0
        for MCC in range(n_MCC):
            # create a trial for new position
            for i in range(NUMBER_OF_PARTICLES):
                for j in range(DIMENSION):
                    #pos_new[i, j] = pos_old[i, j] + step_size*(random()-0.5)
                    pos_new[i,j] = pos_old[i,j]+normalvariate(0.0,1.0)*sqrt(dt)+qf_old[i,j]*dt*D
                psi_new = Psi(pos_new, a)
                qf_new = QuantumForce(pos_new, a)
                GreensFunction = 0.0
            #for i in range(NUMBER_OF_PARTICLES):
                for j in range(DIMENSION):
                    GreensFunction += 0.5*(qf_old[i,j]+qf_new[i,j])*(D*dt*0.5*(qf_old[i,j]-qf_new[i,j])-pos_new[i,j]+pos_old[i,j])
                GreensFunction = exp(GreensFunction)
                prob_rate = GreensFunction*psi_new**2/psi_old**2
            #Metropolis test
                if random() < prob_rate:
                    for j in range(DIMENSION):
                        pos_old[i,j] = pos_new[i,j]
                        qf_old[i,j] = qf_new[i,j]
                    psi_old = psi_new
                    teller+=1
            deltaE = E_L(pos_old, a)

            energy += deltaE
            energy2 += deltaE**2

            deltaPsi = derivative(pos_old)
            Psi_deriv += deltaPsi
            PsiE_deriv += deltaPsi*deltaE


        energy /= n_MCC
        energy2 /= n_MCC
        variance = energy2 - energy**2
        error = np.sqrt(abs(variance)/n_MCC)
        Energies[ia] = energy
        Variances[ia] = variance
        accept_rate+=teller/n_MCC

        Psi_deriv /= n_MCC
        PsiE_deriv /= n_MCC
        gradient = 2*(PsiE_deriv - Psi_deriv*energy)

        a -= gamma*gradient
    return Energies, alpha, accept_rate/max_var



NUMBER_OF_PARTICLES = 100
DIMENSION = 3
max_var = 25
alpha = np.zeros(max_var)
Energies_a = np.zeros(max_var)
Energies_n = np.zeros(max_var)
Variances = np.zeros(max_var)

f = open('energy-100P-3D.txt', 'w')
for i in range(2**9): #2^9=512
    Energies_n, alpha, accept_rate = MonteCarlo(Energies_n, local_energy_numerical)
    f.write(str(Energies_n[-1])+'\n')
f.close()
print('Optimized alpha:', alpha[-1], '      E =', Energies_n[-1])
