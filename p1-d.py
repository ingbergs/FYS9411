import numpy as np
from math import exp, sqrt
from random import random, seed, normalvariate
import matplotlib.pyplot as plt
import time
import multiprocessing


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

def MonteCarlo(alphas, Energies, Variances, E_L):
    accept_rate=0
    n_MCC=int(NUMBER_OF_MONTE_CARLO_CYCLES)
    D=0.5
    gamma = GAMMA #0.01

    pos_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    pos_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_old = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)
    qf_new = np.zeros((NUMBER_OF_PARTICLES, DIMENSION), np.double)

    seed()

    alpha = ALPHA_INIT
    for ia in range(MAX_VAR):
        #a += 0.025
        #if Finished:
        #    break;
        alphas[ia] = alpha
        energy = 0
        energy2 = 0
        Psi_deriv = 0
        PsiE_deriv = 0
        #Iterations += 1

        # place the particles randomly
        for i in range(NUMBER_OF_PARTICLES):
            for j in range(DIMENSION):
                pos_old[i, j] = normalvariate(0.0, 1.0)*np.sqrt(dt)
        psi_old = Psi(pos_old, alpha)
        qf_old = QuantumForce(pos_old,alpha)
        deltaE = E_L(pos_old, alpha)
        #print(deltaE)

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
            deltaE = E_L(pos_old, alpha)

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


        alpha-= gamma*gradient
    return Energies, alphas, Variances, accept_rate/MAX_VAR


NUMBER_OF_PARTICLES = 1
DIMENSION = 3
MAX_VAR = 25
NUMBER_OF_MONTE_CARLO_CYCLES = 1e3
ALPHA_INIT = 0.3
dt=0.5 #2.6
GAMMA = 0.075 #0.025 #0.01

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
print('Numerical:', time_n_out, ', acceptance rate:', accept_rate_n, ', final alpha:', alpha_n[-1])
print()


# calculate analytically
start_time = time.time()
Energies_a, alpha_a, Variances_a, accept_rate_a = MonteCarlo(alpha_a, Energies_a, Variances_a, local_energy_analytical)
time_a = time.time()-start_time
time_a_out = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
print('Analytical:', time_a_out, ', acceptance rate:', accept_rate_a, ', final alpha:', alpha_a[-1])
print()


# write to file
f = open('d1-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
f.write(str(time_a)+'    '+str(time_n)+'\n')
f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
f.write('alpha_a    alpha_n      E_a         E_n         v_a         v_n\n')
for i in range(MAX_VAR):
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

def maining(Energies, Variances, solver, procnum , return_dict):
    Solution = MonteCarlo(Energies, Variances, solver)
    return_dict[procnum] = Solution
def meanResults(results):
    iterations = 0
    EnergyL = (results[0][0])
    alphaL = (results[0][1])
    VarL = (results[0][2])
    for i in range(len(results)):

        if(float(results[i][3]) != 0):
            iterations += float(results[i][3])
    #print(iterations/len(results))

    return(EnergyL, alphaL, VarL, iterations)

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

    #d

    EV_a = meanResults(return_dict_analytical)
    EV_n = meanResults(return_dict_numerical)
    alphaList_a = return_dict_analytical[0][1]
    alphaList_n = return_dict_numerical[0][1]
    accept_rate_a = return_dict_analytical[0][3]
    accept_rate_n = return_dict_numerical[0][3]

    plt.plot(EV_a[1], EV_a[0], 'x-')
    plt.plot(EV_n[1], EV_n[0], 'x-')
    plt.show()


    # write to file
    f = open('d-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
    f.write(str(time_a)+'    '+str(time_n)+'\n')
    f.write(str(accept_rate_a)+'    '+str(accept_rate_n)+'\n')
    f.write('alphaList_a     alphaList_n    E_a         E_n         v_a         v_n \n')
    for i in range(len(EV_a[0])):
        f.write(str(EV_a[1][i])+'    '+str(EV_n[1][i])+'   '+str(EV_a[0][i])+'    '+str(EV_n[0][i])+'    '+str(EV_a[2][i])+'    '+str(EV_n[2][i])+'\n')


    f= open('e-'+str(NUMBER_OF_PARTICLES)+'P-'+str(DIMENSION)+'D.txt', 'w')
    f.write('alphaList_a     alphaList_n    E_a         E_n         v_a         v_n \n')
    for i in range(len(return_dict_analytical)):
        f.write(str(return_dict_analytical[i][1])+'   '+str(return_dict_numerical[i][1])+'   '+str(return_dict_analytical[i][0])+'   '+str(return_dict_numerical[i][0]))



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
