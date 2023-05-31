import matplotlib.pyplot as plt
import numpy as np

eta = []
E = []

ix = 0
ixB = False
etaList = []



with open('ETAandTSfor1P1D_Brute.txt') as f:
    for readline in f:
        line = readline.split('\n')
        
        line = line[0].split(' ')
        
        
        
        if(line[0] != 'nan' and abs(float(line[0])) <= 1 and float(line[0]) > 0):
            E.append(float(line[0]))
            eta.append(float(line[1]))
                
        if(float(line[1]) not in etaList):
            etaList.append(float(line[1]))
        

        
        if(len(eta)> 1 and eta[-1] != eta[-2] and ixB == False):
            ixB = True
            
        if(ixB == False):
            ix += 1
            

eList = []
meanList = []
etaList2 = []
for i in range(1,len(E)):
    if(eta[i] != eta[i-1]):
        etaList2.append(eta[i-1])
        eList.append(np.mean(meanList))
        meanList = []
      
    meanList.append(E[i])
       
print(eList)
print(etaList2)    
        
plt.plot(etaList2,eList)
plt.show()
    
    
    



    
    
    



