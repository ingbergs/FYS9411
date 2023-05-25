import matplotlib.pyplot as plt
import numpy as np

eta = []
E = []
ix = 0
ixB = False

with open('ETA_test_S001.txt') as f:
    for readline in f:
        line = readline.split(' ')
        E.append(float(line[2]))
        eta.append(float(line[-1]))
        
        if(len(eta)> 1 and eta[-1] != eta[-2] and ixB == False):
            ixB = True
        if(ixB == False):
            ix += 1
pList = np.linspace(1,ix,ix)
legend = []            
for i in range(ix):
    Energy = E[(i*ix):(i+1)*ix]
    legend.append(str(eta[i*ix-1]))
    plt.hist(Energy, bins = 10)
   
    
    

            
plt.legend(legend)
plt.show()


