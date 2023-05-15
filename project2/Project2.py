import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing


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

def qForce(alpha, betha, r):
    
    qF = np.zeros((len(r),len(r[0])), np.double)
    for i in range(len(r)):
        for j in range(len(r[i])    ):
        
            qF[i,j] = r[i,j]*alpha
    return(-2*qF)

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
nParticles = 2
nDimensions = 3

nHidden = 2

maxSamples = 10

#change to match amount of logical processors available 
nProcessors = 6


#paralellization

def maining(proc, return_dict):
    #initializing learning parameters
    
    a = np.random.normal(loc = 0.0, scale = 1.0, size = nParticles*nDimensions)
    b = np.random.normal(loc = 0.0, scale = 1.0, size = nHidden)
    w = np.random.normal(loc = 0.0, scale = 1.0, size = nParticles*nDimensions*nHidden)
    eta = 0.1
    energy = 0
    eList = np.zeros(maxSamples)
    posList = np.zeros(maxSamples)
    costDerivative = [np.copy(a),np.copy(b),np.copy(w)]
    
    #starting iterations
    for i in range(maxSamples):
        
        print(proc)
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
   


        #print(eList, costDerivative)
    return_dict[proc] = eList
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
    
    
    print(return_dict)    



        

pList = [1,2,3,4,5,6,7,8,9,10]
fig = plt.figure()
ax = plt.axes(projection = '3d')

"""
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