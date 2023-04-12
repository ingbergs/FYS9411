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
        E+=(1-4*alpha**2)*r2
    return DIMENSION*alpha+0.5*E/NUMBER_OF_PARTICLES

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
    r2 = 0
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r2+=r[i,j]**2
    E=0.5*(-deriv2(r, alpha, 1e-6)+r2)
    return E/NUMBER_OF_PARTICLES


def QuantumForce(r,alpha):
    qforce = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            qforce[i,j] = -4*alpha*r[i,j] ## ERLEND: 2 i stedet for 4
    return qforce

def MonteCarlo(Energies, Variances, E_L):
    accept_rate=0
    n_MCC=int(NUMBER_OF_MONTE_CARLO_CYCLES)
    #step_size = STEP_SIZE #4.0
    D=0.5

    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    a = ALPHA_INIT
    for ia in range(MAX_VAR):
        a += ALPHA_STEP
        alpha[ia] = a
        energy = 0
        energy2 = 0


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

        energy /= n_MCC
        energy2 /= n_MCC
        variance = energy2 - energy**2
        error = np.sqrt(abs(variance)/NUMBER_OF_MONTE_CARLO_CYCLES)
        Energies[ia] = energy
        Variances[ia] = variance
        accept_rate+=teller/(NUMBER_OF_MONTE_CARLO_CYCLES*NUMBER_OF_PARTICLES)
    return Energies, alpha, Variances, accept_rate/MAX_VAR



# simulation parameters
NUMBER_OF_PARTICLES = 10
DIMENSION = 3
MAX_VAR = 10
NUMBER_OF_MONTE_CARLO_CYCLES = 1e3
ALPHA_INIT = 0.35
ALPHA_STEP = 0.025
#STEP_SIZE = 1.0
dt=5e-3 #0.1 #2.6
#
file_suff = 'dt5e-3'

alpha = np.zeros(MAX_VAR)
Energies_a = np.zeros(MAX_VAR)
Energies_n = np.zeros(MAX_VAR)
Variances_a = np.zeros(MAX_VAR)
Variances_n = np.zeros(MAX_VAR)

"""
#old
start_time = time.time()

Energies_a, alpha, accept_rate = MonteCarlo(Energies_a, local_energy_analytical)
print(time.time()-start_time, 's')
print(accept_rate) #burde være ca 0.5-0.6, økes med mindre steglengde
Energies_n, alpha, accept_rate = MonteCarlo(Energies_n, local_energy_numerical)
#print(accept_rate)
"""
# calculate numerically
start_time = time.time()
Energies_n, alpha, Variances_n, accept_rate_n = MonteCarlo(Energies_n, Variances_n, local_energy_numerical)
time_n = time.time()-start_time
time_n_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Numerical:', time_n_out, ', acceptance rate:', accept_rate_n)
print()


# calculate analytically
start_time = time.time()
Energies_a, alpha, Variances_a, accept_rate_a = MonteCarlo(Energies_a, Variances_a, local_energy_analytical)
time_a = time.time()-start_time
time_a_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Analytical:', time_a_out, ', acceptance rate:', accept_rate_a)
print()


# write to file
f = open('c-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D_'+file_suff+'.txt', 'w')
f.write(str(time_a)+'    '+str(time_n)+'\n')
f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
f.write('alpha      E_a         E_n         v_a         v_n\n')
for i in range(len(alpha)):
    f.write(str(alpha[i])+'    '+str(Energies_a[i])+'    '+str(Energies_n[i])+'    '+str(Variances_a[i])+'    '+str(Variances_n[i])+'\n')


# plot
plt.plot(alpha, Energies_a, label='Analytical')
plt.plot(alpha, Energies_n, label='Numerical')
plt.legend()
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.show()

plt.plot(alpha, Variances_a)
plt.plot(alpha, Variances_n)
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.show()

## finn ut hvor man skal variere posisionen, hvor skal man putte inn indeks i????
