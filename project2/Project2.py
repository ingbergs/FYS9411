import numpy as np
import random
import matplotlib.pyplot as plt

#import necessary modules

def waveFunction(r):
    #using the wavefunction from P1 for now
    
    wF = 1
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[i])):
            r2 += r[i,j]**2
        wF *= np.exp(r2)
    return(wF)  

def localEnergy(r):
    E = 0
    
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[i])):
            r2 += r[i,j]**2
        E+=(len(r[0])-2*r2+0.5*r2)    
    return(E/len(r))     

def monteCarlo(a,b,w):
    #using the brute force Monte Carlo without learning parameters for now
    
    nMC = int(1E4)
    step = 0.1
    #initialize starting positions
    oldPos = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    newPos = np.zeros((nParticles, nDimensions), np.double)
    wfOld = waveFunction(oldPos)
    
    
    random.seed()
    energy = 0.0
    deltaE = 0.0
    
    oldWF = waveFunction(oldPos)
    
    for im in range(nMC):
        for ix in range(nParticles):
            for jx in range(nDimensions):
                newPos[ix,jx] = oldPos[ix,jx] + step*(random.random()-0.5)
            
            wfNew = waveFunction(newPos)
                 
            probability = wfNew**2/wfOld**2
            
            if(random.random() > probability):
                oldPos = newPos.copy()
                wfOld = wfNew.copy()
        deltaE = localEnergy(oldPos) 
        energy += deltaE
               
                
        
    energy /= nMC        
    return(energy,oldPos)
    
    




#main

#defining system parameters
nParticles = 10
nDimensions = 3

nHidden = 2

#initializing learning parameters
a = np.random.normal(loc = 0.0, scale = 1.0, size = nParticles*nDimensions)
b = np.random.normal(loc = 0.0, scale = 1.0, size = nHidden)
w = np.random.normal(loc = 0.0, scale = 1.0, size = nParticles*nDimensions*nHidden)
eta = 0.1

maxSamples = 10
energy = 0
eList = np.zeros(maxSamples)
costDerivative = [np.copy(a),np.copy(b),np.copy(w)]

#starting iterations
for i in range(maxSamples):
    energy, costDerivative = monteCarlo(a,b,w)
    """
    aGrad = costDerivative[0]
    bGrad = costDerivative[1]
    wGrad = costDerivative[2]
    
    a -= eta*aGrad
    b -= eta*bGrad
    w -= eta*wGrad
    """
    eList[i] = energy


print(eList, costDerivative)

pList = [1,2,3,4,5,6,7,8,9,10]
fig = plt.figure()
ax = plt.axes(projection = '3d')


plt.plot(costDerivative, 'o')
plt.show()