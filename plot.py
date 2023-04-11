import matplotlib.pyplot as plt
from pylab import rcParams

D = 3
P = 500
oppg = 'd'

def plot(x1, x2, y1, y2, y3, y4):
    rcParams.update({"text.usetex": True,"font.family": "serif","font.sans-serif": ["Computer Modern Roman"]})

    x1 = list(filter(lambda num: num != 0, x1))
    x2 = list(filter(lambda num: num != 0, x2))
    y1 = list(filter(lambda num: num != 0, y1))
    y2 = list(filter(lambda num: num != 0, y2))
    y3 = list(filter(lambda num: num != 0, y3))
    y4 = list(filter(lambda num: num != 0, y4))

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.6))
    plt.tight_layout(pad=2.5)
    ax[0].plot(x2, y2, '-^', label='Numerical', color='r')
    ax[0].plot(x1, y1, '-o', label='Analytical', color='g')
    ax[0].legend()
    ax[0].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], ['',0.5, '',0.7, '',0.9, '', 1.1])
    ax[0].set_xlabel(r'$\alpha$ [Å$^{-2}$]')
    ax[0].set_ylabel(r'$E_L$')
    ax[0].grid(True)


    ax[1].plot(x2, y4, '-^', label='Numerical', color='r')
    ax[1].plot(x1, y3, '-o', label='Analytical', color='g')
    ax[1].legend()
    #ax[1].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.3, '',0.5, '',0.7, '',0.9])
    ax[1].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], ['',0.5, '',0.7, '',0.9, '', 1.1])
    ax[1].set_xlabel(r'$\alpha$ [Å$^{-2}$]')
    ax[1].set_ylabel(r'$\sigma^2$')
    ax[1].grid(True)
    plt.savefig(oppg+'-'+str(P)+'P-'+str(D)+'D.pdf')
    plt.show()

    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.tight_layout(pad=3.08)
    plt.plot(x2, y2, '-^', label='Numerical', color='r')
    plt.plot(x1, y1, '-o', label='Analytical', color='g')
    plt.legend()
    plt.xlabel(r'$\alpha$ [Å$^{-2}$]')
    plt.ylabel(r'$E_L$')
    plt.grid(True)
    plt.savefig(oppg+'-'+str(P)+'P-'+str(D)+'D.pdf')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.tight_layout(pad=3.08)
    plt.plot(x2, y4, '-^', label='Numerical', color='r')
    plt.plot(x1, y3, '-o', label='Analytical', color='g')
    plt.legend()
    plt.xlabel(r'$\alpha$ [Å$^{-2}$]')
    plt.ylabel(r'$\sigma$')
    plt.grid(True)
    plt.savefig(oppg+'-'+str(P)+'P-'+str(D)+'D-var.pdf')
    plt.show()
    """

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
    f.readline()

    two_x = False
    if oppg == 'd':
        two_x = True
    lines = f.readlines()
    f.close()

    for line in lines:
        l = line.split()
        i=0
        x1.append(float(l[i]))
        if two_x:
            i=1
        x2.append(float(l[i]))
        y1.append(float(l[i+1]))
        y2.append(float(l[i+2]))
        y3.append(float(l[i+3]))
        y4.append(float(l[i+4]))
    return x1, x2, y1, y2, y3, y4, v1, v2

alpha_a, alpha_n, E_a, E_n, v_a, v_n, t_a, t_n = read_file(oppg+'-'+str(P)+'P-'+str(D)+'D.txt')
plot(alpha_a, alpha_n, E_a, E_n, v_a, v_n)
