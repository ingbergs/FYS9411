import numpy as np
import random

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

def monteCarlo(a,b,w):
    #using the brute force Monte Carlo without learning parameters for now
    
    nMC = 1E4
    
    #initialize starting positions
    oldPos = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    
    random.seed()
    energ = 0.0
    deltaE = 0.0
    
    oldWF = waveFunction(oldPos)
    
    for im in range(nMC):
        
    
    
    
    return(0,[0,0,0])
    
    




#main

#defining system parameters
nParticles = 2
nDimensions = 2

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
    aGrad = costDerivative[0]
    bGrad = costDerivative[1]
    wGrad = costDerivative[2]
    
    a -= eta*aGrad
    b -= eta*bGrad
    w -= eta*wGrad
    
    eList[i] = energy


print(costDerivative)