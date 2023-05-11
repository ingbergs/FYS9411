import numpy as np
import random

#import necessary modules





#main

#defining system parameters
nParticles = 1
nDimensions = 1
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
for i in range(0,maxSamples):
    energy, costDerivative = monteCarlo(a,b,w)
    aGrad = costDerivative[0]
    bGrad = costDerivative[1]
    wGrad = costDerivative[2]
    
    a -= eta*aGrad
    b -= eta*bGrad
    w -= eta*wGrad
    
    eList[i] = energy


print(costDerivative)