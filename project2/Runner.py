import os
eta = .01
for i in range(1):
    #os.system('cmd /c "python Project2.py" ' +str(eta))
    f = open('Elist_eta(' + str(eta) + ')_brute.txt', 'w')
    for j in range(512):
        os.system('cmd /c "python Project2.py" ' +str(eta))
        
    eta *= 0.5

