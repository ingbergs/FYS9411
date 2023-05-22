import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing


#import necessary modules

def waveFunction(r, a, b, w):
    #using the wavefunction for two electrons
    
    
    Psi1 = 0.0
    Psi2 = 1.0
    Q = Qfac(r,b,w)
    
    for iq in range(nParticles):
        for ix in range(nDimensions):
            Psi1 += (r[iq,ix]-a[iq,ix])**2
            
    for ih in range(nHidden):
        Psi2 *= (1.0 + np.exp(Q[ih]))
        
    Psi1 = np.exp(-Psi1/(2))

    return Psi1*Psi2

def localEnergy(r, a, b, w):
    
    locenergy = 0.0
    
    Q = Qfac(r,b,w)

    for iq in range(nParticles):
        for ix in range(nDimensions):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(nHidden):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(Q[ih]) / (1.0 + np.exp(Q[ih]))**2
    
            dlnpsi1 = -(r[iq,ix] - a[iq,ix])  + sum1
            dlnpsi2 = -1 + sum2
            locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)
            
    if(interaction==True):
        for iq1 in range(nParticles):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(nDimensions):
                    distance += (r[iq1,ix] - r[iq2,ix])**2
                    
                locenergy += 1/sqrt(distance)
                
    return locenergy     

def DerivativeWFansatz(r,a,b,w):
    
    
    
    Q = Qfac(r,b,w)
    
    WfDer = np.empty((3,),dtype=object)
    WfDer = [np.copy(a),np.copy(b),np.copy(w)]
    
    WfDer[0] = (r-a)
    WfDer[1] = 1 / (1 + np.exp(-Q))
    
    for ih in range(nHidden):
        WfDer[2][:,:,ih] = w[:,:,ih] / (1+np.exp(-Q[ih]))
            
    return  WfDer

def qForce(alpha, betha, r):
    
    qF = np.zeros((len(r),len(r[0])), np.double)
    for i in range(len(r)):
        for j in range(len(r[i])    ):
        
            qF[i,j] = r[i,j]*alpha
    return(-2*qF)

def Qfac(r, b,w):
    Q = np.zeros((nHidden), np.double)
    temp = np.zeros((nHidden), np.double)
    
    for ih in range(nHidden):
        temp[ih] = (r*w[:,:,ih]).sum()
        
    Q = b + temp
    
    return Q

def monteCarlo(a,b,w):
    #using the brute force Monte Carlo without learning parameters for now
    
    nMC = int(5E4)
    step = 0.0001
    #initialize starting positions
    oldPos = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    newPos = np.zeros((nParticles, nDimensions), np.double)
    wfOld = waveFunction(oldPos, a, b, w)
    
    
    random.seed()
    energy = 0.0
    deltaE = 0.0
    
    EnergyDer = np.empty((3,),dtype=object)
    DeltaPsi = np.empty((3,),dtype=object)
    DerivativePsiE = np.empty((3,),dtype=object)
    EnergyDer = [np.copy(a),np.copy(b),np.copy(w)]
    DeltaPsi = [np.copy(a),np.copy(b),np.copy(w)]
    DerivativePsiE = [np.copy(a),np.copy(b),np.copy(w)]
    for i in range(3): EnergyDer[i].fill(0.0)
    for i in range(3): DeltaPsi[i].fill(0.0)
    for i in range(3): DerivativePsiE[i].fill(0.0)

    oldWF = waveFunction(oldPos, a, b, w)
    
    for im in range(nMC):
        for ix in range(nParticles):
            for jx in range(nDimensions):
                newPos[ix,jx] = oldPos[ix,jx] + step*(random.random()-0.5)
            
            wfNew = waveFunction(newPos, a, b, w)
                 
            probability = wfNew**2/wfOld**2
            
            if(random.random() > probability):
                oldPos = newPos.copy()
                wfOld = wfNew.copy()
        DeltaE = localEnergy(oldPos,a,b,w)
        DerPsi = DerivativeWFansatz(oldPos,a,b,w)
        
        DeltaPsi[0] += DerPsi[0]
        DeltaPsi[1] += DerPsi[1]
        DeltaPsi[2] += DerPsi[2]
        
        energy += DeltaE

        DerivativePsiE[0] += DerPsi[0]*DeltaE
        DerivativePsiE[1] += DerPsi[1]*DeltaE
        DerivativePsiE[2] += DerPsi[2]*DeltaE
            
    # We calculate mean values
    energy /= nMC
    DerivativePsiE[0] /= nMC
    DerivativePsiE[1] /= nMC
    DerivativePsiE[2] /= nMC
    DeltaPsi[0] /= nMC
    DeltaPsi[1] /= nMC
    DeltaPsi[2] /= nMC
    EnergyDer[0]  = 2*(DerivativePsiE[0]-DeltaPsi[0]*energy)
    EnergyDer[1]  = 2*(DerivativePsiE[1]-DeltaPsi[1]*energy)
    EnergyDer[2]  = 2*(DerivativePsiE[2]-DeltaPsi[2]*energy)
    return energy, EnergyDer
    
    




#main

#defining system parameters
nParticles = 1
nDimensions = 1

nHidden = 2

maxSamples = 20

#change to match amount of logical processors available 
nProcessors = 6

interaction = False
#paralellization

def maining(proc, return_dict):
    #initializing learning parameters
    
    a = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    b = np.random.normal(loc = 0.0, scale = 1.0, size = nHidden)
    w = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions,nHidden))
    eta = 0.1
    energy = 0
    aList = []
    bList = []
    wList = []
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
        
   

        
        #print(eList, costDerivative)
    aList.append(costDerivative[0])
    bList.append(costDerivative[1])
    wList.append(costDerivative[2])
    returnList = [eList, aList, bList, wList]    
    return_dict[proc] = returnList
    
def collect(energyList):
    energy = []
    a = []
    #collecting energies
    for i in range(len(energyList[0][0])):
        E = 0
        for j in range(len(energyList)):
            E += energyList[j][0][i]
        E  /= len(energyList)
        energy.append(E)
    return(energy)
    
if __name__ == "__main__":
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    
    for proc in range(nProcessors):
        
        solution = multiprocessing.Process(target = maining, args=(proc, return_dict))
        jobs.append(solution)
        solution.start()
    
    for job in jobs:
        job.join()
    
    
    
    pList = np.linspace(0,maxSamples-1, maxSamples)
    eList = collect(return_dict)
    print('alpha = ' + str(return_dict[0][1]) + '\n betha = ' + str(return_dict[0][2]) + '\n weight = ' + str(return_dict[0][3]))
    plt.plot(pList, eList)
    plt.show()        
"""

fig = plt.figure()
ax = plt.axes(projection = '3d')

#recording data
f = open('Project2_data.txt', 'w')
f.write('Energy Position(x) Position(y) Position(z)\n')
for w in range(len(eList)):
    f.write(str(eList[w])+'\n'  )
f.close()


for particle in costDerivative:
    
    plt.plot(particle[0],particle[1],particle[2], 'o')
plt.show()
"""