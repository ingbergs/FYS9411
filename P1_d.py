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

#calculating the second derivative of the wavefunction for the nummerical solution
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

#calculating the local energy nummericaly
def localEnergy_num(alpha, betha, r):
    E = 0
    r2 = 0
    for i in range(len(r)):
        
        
        for j in range(len(r[0])):
            r2+=r[i,j]**2
            
    E+=0.5*(-d2(alpha, betha, r, 0.00001)+r2)
    return E/len(r)  
  

#calculating the quantum force for the general case without interaction   
def qForce(alpha, betha, r):
    
    qF = np.zeros((len(r),len(r[0])), np.double)
    for i in range(len(r)):
        for j in range(len(r[i])    ):
        
            qF[i,j] = -2*r[i,j]*alpha
    return(qF)

#calculating the wafeFuntion Derivative
def wFD (r):
    wfDr = 0
    
    for i in range(len(r)):
        r2 = 0
        for j in range(len(r[i])):
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
        
    
    
    
#montecarlo cycling starts here, using metropolis-algorithm without importance sampling 
def monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, solver):
    
    D = 0.5
    TS = 0.1    
    lRate = 0.05
    eps = 1E-5
    acceptEps = 0.1
    f = maxVar
    aRateChange = .01/N
    
    alphaList = np.zeros(maxVar)
    idealAccept = 0.5
    Energies = np.zeros((maxVar,maxVar))
    Vars = np.zeros((maxVar,maxVar))
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
                
                if(solver == "Nummerical"):
                    dE = localEnergy_num(alpha,betha, posOld)
                if(solver == "Analytical"):
                    dE = localEnergy(alpha,betha, posOld)
                energy += dE
                energy2 += dE**2
                
                dWf = wFD(posOld)
                wFDeriv += dWf
                wFEDeriv += dWf*dE
                
                
                acceptanceRate = acceptCount/(k*nParticle)
                acceptanceCheck = acceptanceRate-idealAccept
                
                if(acceptanceCheck < -acceptEps):
                    TS -= aRateChange
                if(acceptanceCheck > acceptEps):
                    TS += aRateChange
                           
             
           
            
            energy /= N
            energy2 /= N
            
            wFDeriv /= N
            wFEDeriv /= N
            grad = 2*(wFEDeriv-wFDeriv*energy)
            alpha_old = alpha
            alpha -= lRate*grad
            
            
            var = energy2-energy**2
            Energies[i,j] = energy    
            Vars[i,j] = var
            
            if(checkAlpha(alpha_old, alpha, eps)):
                finished = True
                f = i
                
                break;
                
        print(acceptanceRate, alpha)
        
    return(Energies, alphaList, Vars, f)

  
#main

N = 5E3
alpha = 0.3
betha = 1.0
maxVar = 40
nParticle  = 10
dim = 3

start_time = time.time()    
Energies, alphaList, Vars, plotter = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Analytical")
Energies_num, alphaList_num, Vars_num, plotter_num = monteCarlo_metropolis(N, alpha, betha, maxVar, nParticle, dim, "Nummerical")
print(print("--- %s seconds ---" % (time.time() - start_time)))

#print("Optimalized value of alpha = " + str(alphaList[plotter]) + " and it was found in " + str(plotter) + " steps")

    
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


plt.plot(alphaList[0:plotter],energyMean[0:plotter], 'x-')
plt.plot(alphaList_num[0:plotter], energyMean_num[0:plotter], 'x-')
plt.legend(['Analytical', 'Nummerical'])
plt.xlabel('Alpha (AU)')
plt.ylabel('Energy (hw)')
plt.show()
plt.close()
#plt.plot(alphaList[0:plotter],varMean[0:plotter])
plt.plot(alphaList, varMean)
plt.xlabel('Alpha (AU)')
plt.ylabel('Variance (AU)')
plt.close()

plt.plot(nList[0:plotter],alphaList[0:plotter], 'x-')
plt.show()