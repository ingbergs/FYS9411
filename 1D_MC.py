import numpy as np
import matplotlib.pyplot as plt
from random import seed, random

#analytical expression for the wavefunction
def waveFunction(alpha, betha, r):
    wF = 1
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[0])):
            r2 += r[i,j]**2
        wF *= np.exp(-alpha*r2)
    return(wF)    
    """
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
    """
#analytical expression for the local energy of one boson in 1D     
def localEnergy(alpha, betha, r):
    E = 0
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[0])):
            r2 += r[i,j]**2
        E+=(len(r[0])*alpha-2*alpha**2*r2+0.5*r2)    
    return(E/len(r))        



def d2 (alpha, betha, r, dx):
    return((waveFunction(alpha,betha,r+dx) - 2*waveFunction(alpha, betha, r) + waveFunction(alpha, betha, r-dx))/(dx**2))

def localEnergy_num(alpha, betha, r):
    E = 0
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[0])):
            r2+=r[i,j]**2
        
        E+=0.5*(-d2(alpha, betha, r, 0.001)+r2)
    return E/len(r)  
        
#montecarlo cycling starts here, using metropolis-algorithm without importance sampling 
def monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim):
    
    alphaList = np.zeros(maxVar)
    step = .1
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
alpha = 0.3
betha = 1.0
maxVar = 10
nParticle = 50
dim = 1

Energies, alphaList, Vars = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim)

plt.plot(alphaList,Energies)
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()
plt.close()
plt.plot(alphaList,Vars)
plt.xlabel('Alpha (AU)')
plt.ylabel('Variance (AU)')
plt.show()