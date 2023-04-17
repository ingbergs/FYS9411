import numpy as np
from numpy.random import randint, randn
import matplotlib.pyplot as plt
from random import seed, random, normalvariate, choices
import time  
from scipy.stats import norm
from mpi4py import MPI
import sys


#analytical expression for the wavefunction
def waveFunction(alpha, betha, r):
    wF = 1
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[0])):
            if(j == 2):
                r2 += (betha*r[i,j])**2
            else:
                r2 += r[i,j]**2
        wF *= np.exp(-alpha*r2)
    return(wF)    
    
#analytical expression for the local energy of one boson in 1D     
def localEnergy(alpha, betha, r):
    E = 0
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[i])):
            if(j == 2):
                r2 += (betha*r[i,j])**2
            else:
                r2 += r[i,j]**2
        E+=(len(r[0])*alpha-2*alpha**2*r2+0.5*r2)    
    return(E/len(r))        

#calculating the second derivative of the wavefunction for the numerical solution
def d2 (alpha, betha, r, dx):
    div2 = 0
    dx2 = dx**2
    wF = waveFunction(alpha, betha, r)
    for i in range(len(r)):
        for j in range(len(r[i])):
            r[i,j] += dx
            wF1 = waveFunction(alpha, betha, r)
            r[i,j] -=2*dx
            wF2 = waveFunction(alpha, betha, r)
            r[i,j] += dx
            div2 += (wF1-2*wF+wF2)/dx2
            
    return(div2/wF)

#calculating the local energy numericaly
def localEnergy_num(alpha, betha, r):
    E = 0
    r2 = 0
    for i in range(len(r)):
          
        for j in range(len(r[i])):
            if(j == 2):
                r2 += (betha*r[i,j])**2
            else:
                r2 += r[i,j]**2
            
    E+=0.5*(-d2(alpha, betha, r, 0.00001)+r2)
    return E/len(r)  
  

#calculating the quantum force for the general case without interaction   
def qForce(alpha, betha, r):
    
    qF = np.zeros((len(r),len(r[0])), np.double)
    for i in range(len(r)):
        for j in range(len(r[i])    ):
        
            qF[i,j] = r[i,j]*alpha
    return(-4*qF)

#calculating the wafeFuntion Derivative
def wFD (r):
    wfDr = 0
    
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[i])):
            if(j == 2):
                r2 += (betha*r[i,j])**2
            else:
                r2 += r[i,j]**2
        wfDr += (r2)
    return -wfDr        

#checking if the change in alpha succeeds the threshold value, eps.
def checkAlpha(a_old, a, eps):
    finished = False
    checkList = []
    
    if(abs(a_old-a) <= eps and a_old != a):
        finished = True
        
    
    return finished
        
    
def stat(data):
    return np.mean(data)
    
def bootstrap(data, N):
    bootAlpha=[]
    with open(data) as f:
        lines = f.readlines()
    
    for i in lines:
    
        bootAlpha.append(float(i.split('\n')[0]))
    
    print(bootAlpha)
    statList = []
    for i in range(N):
        outTake = choices(bootAlpha,k=N)
        for j in outTake:
            statList.append(j)
        
    return statList   
    
    
    
    
    
def cutArray(array):
    for i in range(0, len(array)):
        if(array[i] == 0.0):
            cut = i
            break;
    newArray = array[0:i].copy()
    return newArray

#montecarlo cycling starts here, using metropolis-algorithm without importance sampling 
def monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, solver):
    
    D = 0.5
    TS = 0.1   
    lRate = 0.05     
    eps = 1E-5  
    acceptEps = 0.05
    f = maxVar
    aRateChange = .01/N
    
    alphaList = np.zeros(maxVar)
    idealAccept = 0.5
    Energies = np.zeros(maxVar)
    Vars = np.zeros(maxVar)
    posOld = np.zeros((nParticle, dim), np.double)
    posNew = np.zeros((nParticle, dim), np.double)    
    
    qFOld = np.zeros((nParticle,dim), np.double)
    qFNew = np.zeros((nParticle, dim), np.double)
    finished = False
    seed()
    
    for i in range(maxVar):
        #alpha += 0.025
        if(finished):
            break;
            
        alphaList[i] = alpha 
        step = 1.
        wFDeriv = 0
        wFEDeriv = 0
        for j in range(1):
            energy = energy2 = 0.0
            dE = 0.0
            
            for ii in range(nParticle):
                for jj in range(dim):
                    
                    posOld[ii, jj] = normalvariate(0.0,1.0)*np.sqrt(TS)
                    #posOld[ii, jj] = step*(random()-0.5) 
            wfOld = waveFunction(alpha, betha, posOld)
            qFOld = qForce(alpha,betha,posOld)
            dE = localEnergy(alpha, betha, posOld)
            
            acceptCount = 0
            
            for k in range(1,int(N)):
                for particle in range(nParticle):
                    for dimension in range(dim):
                        
                        posNew[particle, dimension] = posOld[particle,dimension]+normalvariate(0.0,1.0)*np.sqrt(TS)+qFOld[particle,dimension]*TS*D
                    
                    wfNew = waveFunction(alpha, betha, posNew)
                    qFNew = qForce(alpha, betha, posNew)
                    GF = 0.0
                    for dimension2 in range(dim):
                        GF += 0.5*(qFOld[particle,dimension2]+qFNew[particle,dimension2])*(D*TS*0.5*(qFOld[particle,dimension2]-qFNew[particle,dimension2])-posNew[particle,dimension2]+posOld[particle,dimension2])
                    GF = np.exp(GF)  
                    ProbRat = GF*wfNew**2/wfOld**2    
                    if(random() <= ProbRat):
                        acceptCount += 1
                        for dimension3 in range(dim):
                        
                            posOld[particle,dimension3] = posNew[particle,dimension3].copy()
                            qFOld[particle,dimension3] = qFNew[particle,dimension3]
                        wfOld = wfNew
                
                if(solver == "Nummerical"):
                    dE = localEnergy_num(alpha,betha, posOld)
                if(solver == "Analytical"):
                    dE = localEnergy(alpha,betha, posOld)
                energy += dE
                energy2 += dE**2
                
                dWf = wFD(posOld)
                wFDeriv += dWf
                wFEDeriv += dWf*dE
                
                """
                acceptanceRate = acceptCount/(k*nParticle)
                acceptanceCheck = acceptanceRate-idealAccept
                
                if(acceptanceCheck < -acceptEps):
                    TS -= aRateChange
                if(acceptanceCheck > acceptEps):
                    TS += aRateChange
                """         
            
           
            
            energy /= N
            energy2 /= N
            
            wFDeriv /= N
            wFEDeriv /= N
            grad = 2*(wFEDeriv-wFDeriv*energy)
            alpha_old = alpha
            #alpha -= lRate*grad
            
            
            var = energy2-energy**2
            Energies[i] = energy    
            Vars[i] = var
            
            if(checkAlpha(alpha_old, alpha, eps)):
                finished = True
                f = i
                
                break;
            
       
    
    #print(acceptanceRate, alpha)
        
    meanEnergy = stat(Energies)
    Vars = stat(Vars)
        
    return(meanEnergy, alphaList, Vars, f)

  
#main

N = 1E3
alpha = 0.4
betha = 1
maxVar = 5
nParticle  = 20
dim = 3
start_time = time.time()    
#Energies, alphaList, Vars, plotter = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Analytical")
"""
dataFile = 'AlphaBootstrap.txt'
f = open(dataFile, 'w')
f.close()
f = open(dataFile, 'a')
"""
#MPI
comm = MPI.COMM_WORLD
numProc = comm.Get_size()

if(numProc > 1):

    rank = comm.rank

    if (rank == 0):
        data = N
    else: 
        data = None

    nMPI_cycles = 10
    data = comm.bcast(data)

    totalEnergy = totalEnergySquared = localProcessEnergy = localProcessEnergy2 = 0 
    
    varPar = np.zeros([4,nMPI_cycles])
    
    Energies_num = np.zeros(nMPI_cycles)
    alphaList_MPI = np.zeros(nMPI_cycles)
    Vars_num = np.zeros(nMPI_cycles)
    
    
    
    
    for i in range(nMPI_cycles):
        alpha += 0.025
        energy, alphaList, Var, plotter = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Analytical")
        energy_num, alphaList_num, Var_num, plotter_num = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Nummerical")
        
        varPar[0,i] = energy
        varPar[1,i] = Var
        varPar[2,i] = energy_num
        varPar[3,i] = Var_num
        
        alphaList_MPI[i] = alpha
        
    
    #print('rank',rank,data)
    
    #comm.reduce([localProcessEnergy2, totalEnergySquared], MPI.SUM, 0)
    #totalVars = comm.reduce(Vars_num,MPI.SUM, 0)/numProc
    varPar = comm.reduce(varPar, MPI.SUM, 0)/numProc 
    

    print("Total number of cycles =", maxVar*nMPI_cycles*numProc*N)  
#print(Energies_num, alphaList_MPI)    

    MPI.Finalize()
        
    print(print("--- %s seconds ---" % (time.time() - start_time)))
    plt.plot(alphaList_MPI, varPar[0], 'o-')
    plt.plot(alphaList_MPI, varPar[2], 'o-')
    plt.legend(['Analytical', 'Numerical'])
    plt.show();plt.close()
    plt.plot(alphaList_MPI, varPar[1], 'o-')
    plt.plot(alphaList_MPI, varPar[3], 'o-') 
    plt.legend(['Analytical', 'Numerical'])    
    plt.show();plt.close()
    
"""
for i in range(1):

    Energies_num, alphaList_num, Vars_num, plotter_num = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Nummerical")
    alphaList_num = cutArray(alphaList_num)
    f.write(str(alphaList_num[-1]) + '\n')
f.close()    
"""


"""
t = (bootstrap(dataFile, 100))
plt.hist(t, bins = 100)
plt.show()
plt.close()


#alphaList = cutArray(alphaList)
#alphaList_num = cutArray(alphaList_num)


#t = bootstrap(alphaList, stat, int(N)) 
#t_num = bootstrap(alphaList_num, stat, int(N))
#print("Optimalized value of alpha = " + str(alphaList[plotter]) + " and it was found in " + str(plotter) + " steps")
#b = 80
#n, binsboot, patches = plt.hist(t, bins = b)
#n_num,binsboot_num, patches_num = plt.hist(t_num, bins = b, alpha=0.5)

#t_compare = t-t_num

#plt.legend(['Analytical', 'Nummerical'])

plt.show();plt.close()
plt.hist(t_compare, bins = b)
plt.show();plt.close()    

y = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, y, 'r--', linewidth=1)
plt.show()
plt.close()

energyMean = []
varMean = []
energyMean_num = []
varMean_num = []
 
nList = np.linspace(1, len(alphaList), len(alphaList))


for i in range(len(Energies)):
    eSum = 0
    varSum = 0
    eSum_num = 0
    varSum_num = 0
    for j in range(len(Energies[i])):
        eSum += Energies[i][j]
        varSum += Vars[i][j]
        eSum_num += Energies_num[i][j]
        varSum_num += Vars_num[i][j]
    energyMean.append(eSum/len(Energies[i]))
    varMean.append(varSum/len(Vars[i]))
    energyMean_num.append(eSum_num/len(Energies_num[i]))
    varMean_num.append(varSum_num/len(Vars_num[i]))

plt.plot(alphaList[0:plotter],Energies[0:plotter], 'x-')
plt.plot(alphaList_num[0:plotter], Energies_num[0:plotter], 'x-')
plt.legend(['Analytical', 'Nummerical'])
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()
plt.close()
plt.plot(alphaList[0:plotter],Vars[0:plotter])
plt.plot(alphaList_num[0:plotter],Vars_num[0:plotter])
#plt.plot(alphaList, varMean)
plt.xlabel('Alpha (AU)')
plt.ylabel('Variance (AU)')


#plt.plot(nList[0:plotter],alphaList[0:plotter], 'x-')
plt.show()
"""