import numpy as np
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
import time
import multiprocessing
import sys

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

    #return (Psi(r-dx, alpha)-2*Psi(r, alpha)+Psi(r+dx, alpha))/(dx**2*Psi(r, alpha)*NUMBER_OF_PARTICLES)

def local_energy_numerical(r, alpha):
    r2 = 0
    for i in range(NUMBER_OF_PARTICLES):
        for j in range(DIMENSION):
            r2+=r[i,j]**2
    E=0.5*(-deriv2(r, alpha, 1e-6)+r2)
    return E/NUMBER_OF_PARTICLES


def MonteCarlo(Energies, Variances, E_L):
    accept_rate=0
    n_MCC=int(NUMBER_OF_MONTE_CARLO_CYCLES)
    step_size = STEP_SIZE

    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    a = ALPHA_INIT
    for ia in range(MAX_VAR):
        a += ALPHA_STEP
        print(f'Alpha {ia}:', a)
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
            #* TODO: find out if this (*...*) should be in the for i loop, the examples say it should, the results says it shouldnt
            # here it is outside: https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/vmc/ipynb/vmc.ipynb
            # here it is inside: https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week2/ipynb/week2.ipynb
            psi_new = Psi(pos_new, a)

            #Metropolis test
            if random() < psi_new**2/psi_old**2:
                pos_old = pos_new.copy()
                psi_old = psi_new
                teller+=1
            #*
            deltaE = E_L(pos_old, a)
            energy += deltaE
            energy2 += deltaE**2

        energy /= n_MCC
        energy2 /= n_MCC
        variance = energy2 - energy**2
        error = np.sqrt(abs(variance)/n_MCC)
        Energies[ia] = energy
        Variances[ia] = variance
        accept_rate+=teller/(n_MCC)
    return Energies, alpha, Variances, accept_rate/MAX_VAR, error


# simulation parameters
NUMBER_OF_PARTICLES = 100
DIMENSION = 3
MAX_VAR = 10
NUMBER_OF_MONTE_CARLO_CYCLES = 2e4
ALPHA_INIT = 0.3
ALPHA_STEP = 0.05
STEP_SIZE = 0.15
    # 1P    1D: 4.0
    # 10P   1D: 1.0
    # 100P  1D: 0.3
    # 500P  1D: 0.13    (/0.569)

    # 1P    2D: 2.5
    # 10P   2D: 0.75    (0.513/0.518)
    # 100P  2D: 0.2     (0.579/0.583)
    # 500P  2D: 0.1     (/0.536) trenger 2 (eller 2.5???) e4 MCC for a fa min pa riktig sted

    # 1P    3D: 2.0     (0.526/0.521)
    # 10P   3D: 0.5     (0.593/0.594)
    # 100P  3D: 0.15    (0.612/0.612)
    # 500P  3D: 0.08    (/0.539) 5e4 (minimum neeesten pa 0.5)


alpha = np.zeros(MAX_VAR)
Energies_a = np.zeros(MAX_VAR)
Energies_n = np.zeros(MAX_VAR)
Variances_a = np.zeros(MAX_VAR)
Variances_n = np.zeros(MAX_VAR)


def maining(Energies, Variances, solver, procnum , return_dict):
    Solution = MonteCarlo(Energies, Variances, solver)
    return_dict[procnum] = Solution
def meanResults(results):
    EnergyL = []
    VarL = []
    
    
    for i in range(len(results[0][0])):
        Energy = 0 
        Var = 0
        for j in range(len(results)):
            Energy += results[j][0][i]
            Var += results[j][2][i]
        EnergyL.append(Energy/len(results))
        VarL.append(Var/len(results))
    return(EnergyL, VarL)

if __name__ == "__main__":
    
     
    start_time = time.time() 
    
    manager = multiprocessing.Manager()
    return_dict_analytical = manager.dict()
    return_dict_numerical = manager.dict()
    jobs_analytical = []
    jobs_numerical = []
    
    
    
    for i in range(6):
     
        an = multiprocessing.Process(target=maining, args=(Energies_a, Variances_a, local_energy_analytical, i, return_dict_analytical))
        time_a = time.time()-start_time
        nu = multiprocessing.Process(target=maining, args=(Energies_n, Variances_n, local_energy_numerical, i, return_dict_numerical))
        time_n = time.time()-time_a
        jobs_analytical.append(an)
        jobs_numerical.append(nu)
        an.start()
        nu.start()
        
        
        
    for proc in jobs_analytical:
        proc.join()
    for proc in jobs_numerical:
        proc.join()
    print("for")
    EV_a = meanResults(return_dict_analytical)
    EV_n = meanResults(return_dict_numerical)
    alphaList = return_dict_analytical[0][1]
    accept_rate_a = return_dict_analytical[0][3]
    accept_rate_n = return_dict_numerical[0][3]
    print("etter")
    
    # write to file
    f = open('b-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
    f.write(str(time_a)+'    '+str(time_n)+'\n')
    f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
    f.write('alpha      E_a         E_n         v_a         v_n        error_a         error_n\n')
    for i in range(len(alpha)):
        f.write(str(alphaList[i])+'    '+str(EV_a[0][i])+'    '+str(EV_n[0][i])+'    '+str(EV_a[1][i])+'    '+str(EV_n[1][i])+'     '+str(EV_a[-1][i])+'    '+str(EV_n[-1][i])+'\n')
    """
    plt.plot(alphaList,EV_a[0])
    plt.plot(alphaList,EV_n[0])
    plt.show()    
    plt.plot(alphaList,EV_a[1])
    plt.plot(alphaList,EV_n[1])
    
    plt.show()
    """
    
"""
      
# calculate numerically
start_time = time.time()
Energies_n, alpha, Variances_n, accept_rate_n = MonteCarlo(Energies_n, Variances_n, local_energy_numerical)

time_n_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
#print('Numerical:', time_n_out, ', acceptance rate:', accept_rate_n)
print()


# calculate analytically
start_time = time.time()
Energies_a, alpha, Variances_a, accept_rate_a = MonteCarlo(Energies_a, Variances_a, local_energy_analytical)
time_a = time.time()-start_time
time_a_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
#print('Analytical:', time_a_out, ', acceptance rate:', accept_rate_a)
print()


"""
"""
# plot
plt.plot(alpha, Energies_a, label='Analytical')
plt.plot(alpha, Energies_n, label='Numerical')
plt.legend()
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.show()

plt.plot(alpha, Variances_a)
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.show()
"""