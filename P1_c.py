import numpy as np
import matplotlib.pyplot as plt
from random import seed, random, normalvariate
import time


#analytical expression for the wavefunction
def waveFunction(alpha, betha, r):
    wF = 1
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[0])):
            r2 += r[i,j]**2
        wF *= np.exp(-alpha*r2)
    return(wF)    
    
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
        
        E+=0.5*(-d2(alpha, betha, r, 0.0001)+r2)
    return E/len(r)  
  
def qForce(alpha, betha, r):
    
    qF = np.zeros((len(r),len(r[0])), np.double)
    qF[0] = -2*r[0]*alpha
    return(qF)
  
#montecarlo cycling starts here, using metropolis-algorithm without importance sampling 
def monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim):
    
    D = 0.5
    TS = 0.001
    alphaList = np.zeros(maxVar)
    idealAccept = 0.5
    Energies = np.zeros((maxVar,maxVar))
    Vars = np.zeros((maxVar,maxVar))
    posOld = np.zeros((nParticle, dim), np.double)
    posNew = np.zeros((nParticle, dim), np.double)    
    
    qFOld = np.zeros((nParticle,dim), np.double)
    qFNew = np.zeros((nParticle, dim), np.double)
   
    seed()
    
    for i in range(maxVar):
        alpha += 0.025
        alphaList[i] = alpha 
        step = 1.
        for j in range(maxVar):
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
                for iii in range(nParticle):
                    for jjj in range(dim):
                        
                        posNew[iii, jjj] = posOld[iii,jjj]+normalvariate(0.0,1.0)*np.sqrt(TS)+qFOld[iii,jjj]*TS*D
                    
                    wfNew = waveFunction(alpha, betha, posNew)
                    qFNew = qForce(alpha, betha, posNew)
                    GF = 0.0
                    for HU in range(dim):
                        GF += 0.5*(qFOld[iii,HU]+qFNew[iii,HU])*(D*TS*0.5*(qFOld[iii,HU]-qFNew[iii,HU])-posNew[iii,HU]+posOld[iii,HU])
                    GF = np.exp(GF)  
                    ProbRat = GF*wfNew**2/wfOld**2    
                    if(random() <= ProbRat):
                        acceptCount += 1
                        for jok in range(dim):
                        
                            posOld[iii,jok] = posNew[iii,jok].copy()
                            qFOld[iii,jok] = qFNew[iii,jok]
                        wfOld = wfNew
                dE = localEnergy(alpha,betha, posOld)
                energy += dE
                energy2 += dE**2
                acceptanceRate = acceptCount/(k*nParticle)
                
                if(acceptanceRate < idealAccept):
                    TS -= .01/N
                if(acceptanceRate > idealAccept):
                    TS += .01/N
                
                    
                
        
            
            energy /= N
            energy2 /= N
            var = energy2-energy**2
            Energies[i,j] = energy    
            Vars[i,j] = var    
        print(acceptanceRate)
    return(Energies, alphaList, Vars)

  
#main

N = 1E4
alpha = 0.3
betha = 1.0
maxVar = 20
nParticle = 10
dim = 1

start_time = time.time()    
Energies, alphaList, Vars = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim)
print(print("--- %s seconds ---" % (time.time() - start_time)))



plt.plot(alphaList,Energies)
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()
plt.close()
plt.plot(alphaList,Vars)
plt.xlabel('Alpha (AU)')
plt.ylabel('Variance (AU)')
plt.show()