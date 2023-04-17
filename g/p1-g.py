import numpy as np
from math import exp, sqrt
from random import random, seed, normalvariate
import matplotlib.pyplot as plt
import time

def Psi(r, alpha):
    r2 = 0
    product = 1
    for i in range(NUMBER_OF_PARTICLES):
        r2 += r[i,0]**2 + r[i,1]**2 + beta*r[i,2]**2
        #product*=exp(-alpha*r2)
    product = exp(-alpha*r2)
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(i+1, NUMBER_OF_PARTICLES):
            rirj = sqrt(abs((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2))
            if rirj <= a:
                return 0
            else:
                product*=(1-a/rirj)
    return product

def local_energy(r, alpha, analytical):
    E1=0
    E2=0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
        E1+= r2
        for j in range(i+1, NUMBER_OF_PARTICLES):
            rirj = sqrt(abs((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2))
            if rirj <= a:
                E2+=1e9
            #else add 0
    if analytical:
        E1 -= second_derivative(r, alpha)
    else:
        E1 -= deriv2(r, alpha, 1e-6)
    return 0.5*E1+E2

def second_derivative(r, alpha):
    for i in range(NUMBER_OF_PARTICLES):
        t1 = 1-0.5*beta-alpha*(r[i,0]**2+r[i,1]**2+beta**2*r[i,2]**2)
        t2 = 0
        t3 = 0
        t4 = 0
        for j in range(NUMBER_OF_PARTICLES):
            if j != i:
                rij = sqrt((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2)
                t2 += (r[i,0]**2-r[i,0]*r[j,0]+r[i,1]**2-r[i,1]*r[j,1]+beta*r[i,2]**2-beta*r[i,0]*r[j,0])*u1(rij)/rij
                for k in range(NUMBER_OF_PARTICLES):
                    if k != i:
                        rik = sqrt((r[i,0]-r[k,0])**2+(r[i,1]-r[k,1])**2+(r[i,2]-r[k,2])**2)
                        t3 += (rij*rik)**(-1)*((r[i,0]-r[j,0])*(r[i,0]-r[k,0])+(r[i,1]-r[j,1])*(r[i,1]-r[k,1])+(r[i,2]-r[j,2])*(r[i,2]-r[k,2]))*u1(rij)*u1(rik)
                t4 += u2(rij)+2*u1(rij)/rij
        return -4*alpha*(t1+t2)+t3+t4


def u1(r):
    if r <= a:
        return 0
    return a/(r**2-a*r)

def u2(r):
    if r <= a:
        return 0
    return -a*(1/(r**2-2*a*r-a**2)+2/(r**3-a*r**2))
"""
def local_energy_analytical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
        E+=(DIMENSION*alpha-2*alpha**2*r2+0.5*r2)
    return E/NUMBER_OF_PARTICLES
"""
def deriv2(r, alpha, dx):
    d2 = 0
    dx2 = dx**2
    psi = Psi(r, alpha)
    if psi == 0:
        return 0
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r[i,j] += dx
            psi1 = Psi(r, alpha)
            r[i,j] -= 2*dx
            psi2 = Psi(r, alpha)
            r[i,j] += dx
            d2 += (psi1-2*psi+psi2)/dx2
    return d2/psi
"""
def local_energy_numerical(r, alpha):
    E = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
    E+=0.5*(-deriv2(r, alpha, 1e-6)+r2)
    return E/NUMBER_OF_PARTICLES
"""
def QuantumForce(r,alpha):
    qforce = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    for i in range(NUMBER_OF_PARTICLES):
        qforce[i] = -4*alpha*r[i]
    return qforce

def derivative(r):
    r_sum = 0
    for i in range(NUMBER_OF_PARTICLES):
        r2 = r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
        r_sum += r2
    return -r_sum

def MonteCarlo(alphas, Energies, Variances, analytical = False):
    accept_rate=0
    n_MCC=int(NUMBER_OF_MONTE_CARLO_CYCLES)
    D=0.5
    gamma = GAMMA

    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    alpha = ALPHA_INIT
    for ia in range(MAX_VAR):
        alphas[ia] = alpha
        energy = 0
        energy2 = 0
        Psi_deriv = 0
        PsiE_deriv = 0


        # place the particles randomly
        for i in range(NUMBER_OF_PARTICLES):
            for j in range(DIMENSION):
                pos_old[i, j] = normalvariate(0.0, 1.0)*np.sqrt(dt)
        psi_old = Psi(pos_old, alpha)
        qf_old = QuantumForce(pos_old,alpha)
        deltaE = local_energy(pos_old, alpha, analytical)

        teller=0
        for MCC in range(n_MCC):
            # create a trial for new position
            for i in range(NUMBER_OF_PARTICLES):
                for j in range(DIMENSION):
                    pos_new[i,j] = pos_old[i,j]+normalvariate(0.0,1.0)*sqrt(dt)+qf_old[i,j]*dt*D
                psi_new = Psi(pos_new, alpha)
                qf_new = QuantumForce(pos_new, alpha)
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
            deltaE = local_energy(pos_old, alpha, analytical)

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
        accept_rate+=teller/(n_MCC*NUMBER_OF_PARTICLES)

        Psi_deriv /= n_MCC
        PsiE_deriv /= n_MCC
        gradient = 2*(PsiE_deriv - Psi_deriv*energy)

        alpha -= gamma*gradient
    return Energies, alphas, Variances, accept_rate/MAX_VAR


NUMBER_OF_PARTICLES = 10
DIMENSION = 3
MAX_VAR = 25
NUMBER_OF_MONTE_CARLO_CYCLES = 1e3 #1e3
ALPHA_INIT = 1 #0.3
dt=0.05 #0.5 #2.6
GAMMA = 0.01 #0.075 #0.025 #0.01
beta = 2.82843
a = 0.0043

alpha_n = np.zeros(MAX_VAR)
alpha_a = np.zeros(MAX_VAR)
Energies_a = np.zeros(MAX_VAR)
Energies_n = np.zeros(MAX_VAR)
Variances_a = np.zeros(MAX_VAR)
Variances_n = np.zeros(MAX_VAR)


# calculate numerically
start_time = time.time()
Energies_n, alpha_n, Variances_n, accept_rate_n = MonteCarlo(alpha_n, Energies_n, Variances_n)
time_n = time.time()-start_time
time_n_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Numerical:', time_n_out, ', acceptance rate:', accept_rate_n, ', final alpha:', alpha_n[-1])
print()


# calculate analytically
start_time = time.time()
Energies_a, alpha_a, Variances_a, accept_rate_a = MonteCarlo(alpha_a, Energies_a, Variances_a, analytical = True)
time_a = time.time()-start_time
time_a_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Analytical:', time_a_out, ', acceptance rate:', accept_rate_a, ', final alpha:', alpha_a[-1])
print()

"""
# write to file
f = open('g-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
f.write(str(time_a)+'    '+str(time_n)+'\n')
f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
f.write('alpha_a    alpha_n      E_a         E_n         v_a         v_n\n')
for i in range(len(alpha)):
    f.write(str(alpha_a[i])+'    '+alpha_n[i])+'    '+str(Energies_a[i])+'    '+str(Energies_n[i])+'    '+str(Variances_a[i])+'    '+str(Variances_n[i])+'\n')
"""


# plot
plt.plot(alpha_a, Energies_a, '-o', label='Analytical')
plt.plot(alpha_n, Energies_n, '-o', label='Numerical')
plt.legend()
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.show()

plt.plot(alpha_a, Variances_a, '-o')
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.show()
