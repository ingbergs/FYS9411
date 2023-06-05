import matplotlib.pyplot as plt
import numpy as np

eta = []
E = []

ix = 0
ixB = False

totEtaList = []
totEList = []
eList = []
etaList = []


with open('ETAandTSfor1P1D_Brute_copy.txt') as f:
    for readline in f:
        line = readline.split('\n')
        
        line = line[0].split(' ')
        if(line[0] == 'Energies' and len(eList) > 0):
            totEtaList.append(etaList[-1])
            totEList.append(eList[-1])
            etaList = []
            eList = []
            
        
        if(line[0] != 'Energies'):
            eList.append(line[0])
            etaList.append(line[-1])
        


eList = []
etaList = [] 
c = 1     
print(totEtaList[-1])
E = 0
for i in range(c, len(totEtaList)):
    E += float(totEList[i])
    
    if(totEtaList[i] != totEtaList[i-1]):
        eList.append(E/(float(i-c)))
        etaList.append(totEtaList[i-1])
        E = 0
        c = i

    
       
for i in range(len(eList)):
    print(eList[i], etaList[i])
    
        
plt.plot(etaList,eList, 'x')
plt.show()
    
    
    



    
    
    



