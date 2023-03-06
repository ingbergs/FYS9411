import numpy as np
import matplotlib.pyplot as plt
from random import seed, random

#analytical expression for the wavefunction
def waveFunction(alpha, betha, r):
    
    if(len(r[0]) == 1):
        x = 0
        for i in range(len(r)):
            x += r[i][0]**2
        return(np.exp(-alpha*x))
    
    if(len(r[0]) == 2):
        
        x = r[0][0]**2
        y = r[0][1]**2
        return(np.exp(-alpha*(x+y)))
        
    
    if(len(r[0]) == 3):
        return(np.exp(-alpha*(r[0][0]**2+r[0][1]**2+r[0][2]**2)))

#analytical expression for the local energy of one boson in 1D     
def localEnergy(alpha, betha, r):
    if(len(r[0]) == 1):
        x = 0
        for i in range(len(r)):
            x += r[i][0]**2
        return(len(r[0])*alpha+0.5*(1-4*alpha**2)*x)
    
    if(len(r[0]) == 2):
          
        x = r[0][0]**2
        y = r[0][1]**2
        return((2*alpha-4*alpha*alpha*(x+y)+0.5*(x+y))/len(r[0]))

#montecarlo cycling starts here, using metropolis-algorithm without importance sampling 
def monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim):
    
    alphaList = np.zeros(maxVar)
    step = 1.0 
    Energies = np.zeros((maxVar,maxVar))
    Vars = np.zeros((maxVar,maxVar))
    posOld = np.zeros((nParticle, dim), np.double)
    posNew = posOld    
   
    seed()
    
    for i in range(maxVar):
        alpha += 0.025
        alphaList[i] = alpha 
        
        for j in range(maxVar):
            energy = energy2 = 0.0
            dE = 0.0
            
            for ii in range(nParticle):
                for jj in range(dim):
                    
                    posOld[ii, jj] = step*(random() - 0.5)
               
            wfOld = waveFunction(alpha, betha, posOld)
            
            for k in range(int(N)):
                for iii in range(nParticle):
                    for jjj in range(dim):
                        posNew[iii, jjj] = posOld[iii,jjj] + step*(random()-0.5)
                wfNew = waveFunction(alpha, betha, posNew)
                
                if(random() < wfNew**2/wfOld**2):
                    posOld = posNew.copy()
                    wfOld = wfNew
                    dE = localEnergy(alpha,betha, posOld)
                energy += dE
                energy2 += dE**2
                
            energy /= N
            energy2 /= N
            var = energy2-energy**2
            Energies[i,j] = energy    
            Vars[i,j] = var    
    return(Energies, alphaList, Vars)

    
#main

N = 1E4
alpha = 0.6
betha = 1.0
maxVar = 20
nParticle = 1
dim = 2 

Energies, alphaList, Vars = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim)

plt.plot(alphaList,Energies)
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()
plt.close()
plt.plot(alphaList,Vars)
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()