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
    return (Psi(r-dx, alpha)-2*Psi(r, alpha)+Psi(r+dx, alpha))/(dx**2*Psi(r, alpha)) ## ERLEND: deler ikke på Psi

def local_energy_numerical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2

        E+=-0.5*(deriv2(r, alpha, 0.001)-r2)
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
    n_MCC=10000
    step_size = 1.0 #4.0
    dt=0.5 #2.6
    D=0.5
    gamma = 0.01


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
            for j in range(DIMENSION):
                GreensFunction += 0.5*(qf_old[i,j]+qf_new[i,j])*(D*dt*0.5*(qf_old[i,j]-qf_new[i,j])-pos_new[i,j]+pos_old[i,j])
            GreensFunction = exp(GreensFunction)
            prob_rate = GreensFunction*psi_new**2/psi_old**2
            #Metropolis test
            if random() < prob_rate:
                for j in range(DIMENSION):
                    pos_old[i,j] = pos_new[i,j].copy()
                    qf_old[i,j] = qf_new[i,j]
                psi_old = psi_new
                deltaE = E_L(pos_old, a)
                teller+=1
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



NUMBER_OF_PARTICLES = 1
DIMENSION = 3
max_var = 40
alpha = np.zeros(max_var)
Energies_a = np.zeros(max_var)
#Energies_n = np.zeros(max_var)
Variances = np.zeros(max_var)

start_time = time.time()

Energies_a, alpha, accept_rate = MonteCarlo(Energies_a, local_energy_analytical)
print(time.time()-start_time, 's')
print(accept_rate) #burde være ca 0.5-0.6, økes med mindre steglengde
#Energies_n, alpha, accept_rate = MonteCarlo(Energies_n, local_energy_numerical)
#print(accept_rate)

print('Optimized alpha:', alpha[-1], '      E =', Energies_a[-1])
plt.plot(alpha, Energies_a, '-o', label='Analytical')
#plt.plot(alpha, Energies_n, label='Numerical')
plt.legend()
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.show()

plt.plot(alpha, Variances)
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.show()

## finn ut hvor man skal variere posisionen, hvor skal man putte inn indeks i????
