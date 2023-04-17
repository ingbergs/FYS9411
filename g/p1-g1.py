import numpy as np
from math import exp, sqrt
from random import random, seed, normalvariate
import matplotlib.pyplot as plt
import time
"""
def Psi(r, alpha):
    product = 1
    for i in range(NUMBER_OF_PARTICLES):
        r2 = 0
        for j in range(DIMENSION):
            r2+=r[i,j]**2
        product*=exp(-alpha*r2)
    return product
    #return exp(-alpha*r[0,0]**2)
"""
def Psi(r, alpha):
    r2 = 0
    product = 1
    for i in range(NUMBER_OF_PARTICLES):
        r2 += r[i,0]**2 + r[i,1]**2 + beta*r[i,2]**2
        for j in range(i+1, NUMBER_OF_PARTICLES):
            rirj = sqrt(abs((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2))
            if rirj <= a:
                return 0
            else:
                product*=(1-a/rirj)
    product *= exp(-alpha*r2)
    return product

def local_energy_numerical(r, alpha):
    E1 = -0.5*deriv2(r, alpha, 1e-6)
    E2 = 0
    E3 = 0
    for i in range(NUMBER_OF_PARTICLES):
        E2 += r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
        for j in range(i+1, NUMBER_OF_PARTICLES):
            rij = sqrt((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2)
            E3 += V_int(rij)
    return (E1 + 0.5*E2 + E3)/NUMBER_OF_PARTICLES

def V_int(r):
    if r > a:
        return 0
    return 1e8

def local_energy_analytical(r, alpha):
    E1 = 0
    E2 = 0
    E3 = 0
    for i in range(NUMBER_OF_PARTICLES):
        E1 += deriv_i(r, alpha, i)
        E2 += r[i,0]**2 + r[i,1]**2 + beta**2*r[i,2]**2
        for j in range(i+1, NUMBER_OF_PARTICLES):
            rij = sqrt((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2+(r[i,2]-r[j,2])**2)
            E3 += V_int(rij)
    return (0.5*(E2-E1) + E3)/NUMBER_OF_PARTICLES

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

def deriv_i(r, alpha, i):
    ri=r[i]
    t1 = 4*alpha*(alpha*(ri[0]**2+ri[1]**2+beta**2*ri[2]**2)-1-0.5*beta)
    t2 = 0
    t3 = 0
    t4 = 0
    for j in range(NUMBER_OF_PARTICLES):
        rj = r[j]
        rij = sqrt((ri[0]-rj[0])**2+(ri[1]-rj[1])**2+(ri[2]-rj[2])**2)
        if j != i:
            t2 -= 4*alpha*2*a/(rij**4-a*rij**3)*(ri[0]**3-2*ri[0]**2*rj[0]+ri[0]*rj[0]**2 + ri[1]**3-2*ri[1]**2*rj[1]+ri[1]*rj[1]**2 + beta*(ri[2]**3-2*ri[2]**2*rj[2]+ri[2]*rj[2]**2))

            for k in range(NUMBER_OF_PARTICLES):
                rk = r[k]
                rik = sqrt((ri[0]-rk[0])**2+(ri[1]-rk[1])**2+(ri[2]-rk[2])**2)
                if k != i:
                    t3 += 4*a**2/((rij**4-a*rij**3)*(rik**4-a*rik**3))*(ri[0]**2-ri[0]*rk[0]-ri[0]*rj[0]+rj[0]*rk[0] + ri[1]**2-ri[1]*rk[1]-ri[1]*rj[1]+rj[1]*rk[1] + ri[2]**2-ri[2]*rk[2]-ri[2]*rj[2]+rj[2]*rk[2])**2

            t4 += a/(rij**3-a*rij**2)*(a/rij-1)

    #t2 *= 4*alpha
    #t3 *= 4*a**2

    return (t1+t2+t3+t4)



"""
def local_energy_numerical(r, alpha):
    E = 0
    r2 = 0
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r2+=r[i,j]**2
    E+=0.5*(-deriv2(r, alpha, 1e-6)+r2)
    return E/NUMBER_OF_PARTICLES
"""

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

def MonteCarlo(alpha, Energies, Variances, E_L):
    accept_rate=0
    n_MCC=int(NUMBER_OF_MONTE_CARLO_CYCLES)
    D=0.5
    gamma = GAMMA #0.01

    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    a = ALPHA_INIT
    for ia in range(MAX_VAR):
        #a += 0.025
        alpha[ia] = a
        print(a)
        energy = 0
        energy2 = 0
        Psi_deriv = 0
        PsiE_deriv = 0


        # place the particles randomly
        for i in range(NUMBER_OF_PARTICLES):
            for j in range(DIMENSION):
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
        accept_rate+=teller/(n_MCC*NUMBER_OF_PARTICLES)

        Psi_deriv /= n_MCC
        PsiE_deriv /= n_MCC
        gradient = 2*(PsiE_deriv - Psi_deriv*energy)

        a -= gamma*gradient
    return Energies, alpha, Variances, accept_rate/MAX_VAR


NUMBER_OF_PARTICLES = 100
DIMENSION = 3
MAX_VAR = 25
NUMBER_OF_MONTE_CARLO_CYCLES = 1e2
ALPHA_INIT = 0.3
dt=0.05 #0.5 #2.6
GAMMA = 0.01 #0.025
    # 1P:   0.025
    # 10P:
    # 100P:
    # 500P:
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
Energies_n, alpha_n, Variances_n, accept_rate_n = MonteCarlo(alpha_n, Energies_n, Variances_n, local_energy_numerical)
time_n = time.time()-start_time
time_n_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Numerical:', time_n_out, ', acceptance rate:', accept_rate_n, ', final alpha:', alpha_n[-1], 'final E_L:', Energies_n[-1])
print()


# calculate analytically
start_time = time.time()
Energies_a, alpha_a, Variances_a, accept_rate_a = MonteCarlo(alpha_a, Energies_a, Variances_a, local_energy_analytical)
time_a = time.time()-start_time
time_a_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Analytical:', time_a_out, ', acceptance rate:', accept_rate_a, ', final alpha:', alpha_a[-1], 'final E_L:', Energies_a[-1])
print()


# write to file
f = open('d-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
f.write(str(time_a)+'    '+str(time_n)+'\n')
f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
f.write('alpha_a    alpha_n      E_a         E_n         v_a         v_n\n')
for i in range(len(alpha_a)):
    f.write(str(alpha_a[i])+'    '+str(alpha_n[i])+'    '+str(Energies_a[i])+'    '+str(Energies_n[i])+'    '+str(Variances_a[i])+'    '+str(Variances_n[i])+'\n')



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

"""
start_time = time.time()

Energies_a, alpha, accept_rate = MonteCarlo(Energies_a, local_energy_analytical)
print(time.time()-start_time, 's')
print(accept_rate) #burde være ca 0.5-0.6, økes med mindre steglengde
Energies_n, alpha, accept_rate = MonteCarlo(Energies_n, local_energy_numerical)
#print(accept_rate)

print('Optimized alpha:', alpha[-1], '      E =', Energies_a[-1])
plt.plot(alpha, Energies_a, '-o', label='Analytical')
plt.plot(alpha, Energies_n, '-o', label='Numerical')
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
"""
