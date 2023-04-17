import os


os.system('cmd /c "mpiexec -n 6 python P1_e.py" ' +str(1))
os.system('cmd /c "mpiexec -n 6 python P1_e.py" ' +str(10))
os.system('cmd /c "mpiexec -n 6 python P1_e.py" ' +str(100))
os.system('cmd /c "mpiexec -n 6 python P1_e.py" ' +str(500))
os.system('cmd /c "mpiexec -n 6 python P1_b_brute.py" ' +str(1))
os.system('cmd /c "mpiexec -n 6 python P1_b_brute.py" ' +str(10))
os.system('cmd /c "mpiexec -n 6 python P1_b_brute.py" ' +str(100))
os.system('cmd /c "mpiexec -n 6 python P1_b_brute.py" ' +str(500))
