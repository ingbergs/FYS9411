import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing
import sys

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

def QuantumForce(r,a,b,w):
    
    qforce = np.zeros((nParticles,nDimensions), np.double)
    sum1 = np.zeros((nParticles,nDimensions), np.double)
    
    Q = Qfac(r,b,w)
    
    for ih in range(nHidden):
        sum1 += w[:,:,ih]/(1+np.exp(-Q[ih]))
    
    qforce = 2*(-(r-a) + sum1)
    
    return qforce


def Qfac(r, b,w):
    Q = np.zeros((nHidden), np.double)
    temp = np.zeros((nHidden), np.double)
    
    for ih in range(nHidden):
        temp[ih] = (r*w[:,:,ih]).sum()
        
    Q = b + temp
    
    return Q

def monteCarlo(a,b,w):
    #using the brute force Monte Carlo without learning parameters for now
    
    nMC = int(2E2)
    step = 0.1
    D = 0.5
    TimeStep = float(sys.argv[2])
    
    #initialize starting positions
    oldPos = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    newPos = np.zeros((nParticles, nDimensions), np.double)
    wfOld = waveFunction(oldPos, a, b, w)
    
    # Quantum force
    QuantumForceOld = np.zeros((nParticles,nDimensions), np.double)
    QuantumForceNew = np.zeros((nParticles,nDimensions), np.double)
    
    
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

    
    
    for im in range(nMC):
        for ix in range(nParticles):
            for jx in range(nDimensions):
                    if(importance):
                        newPos[ix,jx] = oldPos[ix,jx]+random.normalvariate(0.0,1.0)*np.sqrt(TimeStep)+QuantumForceOld[ix,jx]*TimeStep*D
                    else:
                        newPos[ix,jx] = oldPos[ix,jx] + step*(random.random()-0.5)
            
            wfNew = waveFunction(newPos, a, b, w)
            QuantumForceNew = QuantumForce(newPos,a,b,w)
            if(importance):
                
                QuantumForceOld = QuantumForce(oldPos,a,b,w)
                GreensFunction = 0.0
                for j in range(nDimensions):
                    GreensFunction += 0.5*(QuantumForceOld[ix,jx]+QuantumForceNew[ix,jx])*\
                                      (D*TimeStep*0.5*(QuantumForceOld[ix,jx]-QuantumForceNew[ix,jx])-\
                                      newPos[ix,jx]+oldPos[ix,jx])
      
                GreensFunction = np.exp(GreensFunction)
                probability = GreensFunction*wfNew**2/wfOld**2
                
            else:
                probability = wfNew**2/wfOld**2
                #print('som for')
            
            if(random.random() < probability):
                for j in range(nDimensions):
                    oldPos[ix,j] = newPos[ix,j]
                    QuantumForceOld[ix, j] = QuantumForceNew[ix,j]
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

maxSamples = 30

#change to match amount of logical processors available 
nProcessors = 6

importance = False
interaction = False

#paralellization
gamma = 1E-5
def maining(proc, return_dict):
    #initializing learning parameters
    
    a = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions))
    b = np.random.normal(loc = 0.0, scale = 1.0, size = nHidden)
    w = np.random.normal(loc = 0.0, scale = 1.0, size = (nParticles,nDimensions,nHidden))
    eta = float(sys.argv[1])
    energy = 0
    aList = []
    bList = []
    wList = []
    eList = np.zeros(maxSamples)
    
    costDerivative = np.empty((3,),dtype=object)
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
        
        
        #aList.append(a[-1])
        #bList.append(b)
        #wList.append(w)
        
        
   

        
        #print(eList, costDerivative)
    #aList.append(costDerivative[0])
    #bList.append(costDerivative[1])
    #wList.append(costDerivative[2])
    
    returnList = [eList, aList, bList, wList]    
    return_dict[proc] = returnList
    
def collect(energyList):
    energy = []
    
    #a = np.zeros((len(energyList[0][1]),len(energyList[0][1][0])),np.double)
    
    
    #collecting energies
    for i in range(len(energyList[0][0])):
        E = 0
        for j in range(len(energyList)):
            E += energyList[j][0][i]
        E  /= len(energyList)
        energy.append(E)
    #for i in range(len(energyList[0][1])):
    """
    for ia in range(len(energyList)):
        for ja in range(len(a)):
            
            for ka in range(len(a[ja])):
                
                a[ja,ka] += energyList[ia][1][ja][ka]
                
    a /= len(energyList) 
    """
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
    
    
    
    pList = np.linspace(0,maxSamples, maxSamples+1)
    eList = collect(return_dict)
    #print('Energy = ' + str(eList[-1]) + '\nalpha = ' + str(return_dict[0][1]) + '\nbetha = ' + str(return_dict[0][2]) + '\nweight = ' + str(return_dict[0][3]))
    print('Energy = ' + str(eList[-1]) + ' with eta = ' + str(sys.argv[1]) + ' and TS = ' + str(sys.argv[2]))
    f = open('ETAandTSfor1P1D_Brute.txt', 'a')
    f.write(str(eList[-1]) + ' ' + str(sys.argv[1]) + '\n')
"""
    for i in range(len(aList)):
        #print(aList)    
        for j in range(len(aList[i])):
            plt.plot(pList[i],aList[i][j],'x', color= 'blue')
    plt.show()
"""
    
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