import os
eta = 5
for i in range(10):
    
    
    for j in range(10):
        os.system('cmd /c "python Project2.py" ' +str(eta))
    eta *= 0.5

