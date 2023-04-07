import matplotlib.pyplot as plt
from pylab import rcParams

D = 1
P = 1
oppg = 'b'

def plot(x1, x2, y1, y2, y3, y4):
    rcParams.update({"text.usetex": True,"font.family": "serif","font.sans-serif": ["Computer Modern Roman"]})

    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.plot(x2, y2, '-o', label='Numerical', color='r')
    plt.plot(x1, y1, '-o', label='Analytical', color='g')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$E_L$')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.plot(x2, y4, '-o', label='Numerical', color='r')
    plt.plot(x1, y3, '-o', label='Analytical', color='g')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\sigma$')
    plt.grid(True)
    plt.show()

def read_file(filename):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    f = open(filename)
    f.readline()
    l = f.readline()
    v1 = float(l.split()[0])
    v2 = float(l.split()[1])

    two_x = False
    if len(f.readline().split()) > 5:
        two_x = True
    lines = f.readlines()
    f.close()

    for line in lines:
        l = line.split()
        i=0
        x1.append(float(l[i]))
        if two_x:
            i=1
            print(i)
        x2.append(float(l[i]))
        y1.append(float(l[i+1]))
        y2.append(float(l[i+2]))
        y3.append(float(l[i+3]))
        y4.append(float(l[i+4]))
    return x1, x2, y1, y2, y3, y4, v1, v2

alpha_a, alpha_n, E_a, E_n, v_a, v_n, t_a, t_n = read_file(oppg+'-'+str(P)+'P-'+str(D)+'D.txt')
plot(alpha_a, alpha_n, E_a, E_n, v_a, v_n)
