import os

TS = [4.0,2.0,1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.000001]
f = open('ETAandTSfor1P1D_Brute.txt', 'w')
eta = 4
for i in range(100):
    #os.system('cmd /c "python Project2.py" ' +str(eta))
       
    
    for k in range(16):
        os.system('cmd /c "python Project2.py" ' + str(eta) + ' ' + str(0.01))
    eta *= 0.9        
    

