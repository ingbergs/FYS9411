import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

D = 3
P = 10
oppg = 'c'

def main():
    filenames = ['c-10P-3D_dt5e-3.txt', 'c-10P-3D_dt1e-2.txt', 'c-10P-3D_dt5e-2.txt', 'c-10P-3D_dt1e-1.txt', 'c-10P-3D_dt5e-1.txt', 'c-10P-3D_dt1e0.txt']
    alpha, E_a, E_n, v_a, v_n = read_file(filenames)
    plot(alpha, E_a, E_n, v_a, v_n)



def plot(x1, y1, y2, y3, y4):
    rcParams.update({"text.usetex": True,"font.family": "serif","font.sans-serif": ["Computer Modern Roman"]})

    labels = ['$dt=0.005$', '$dt=0.01$', '$dt=0.05$', '$dt=0.1$', '$dt=0.5$', '$dt=1.0$']
    color1 = "#FFC0CB"
    color2 = "#E50000"
    color3 = '#90FF90'
    color4 = '#15B01A'
    num_points = len(y1)
    colors1=get_color_gradient(color1, color2, num_points)
    colors2=get_color_gradient(color3, color4, num_points)

    fig, ax = plt.subplots(1, 2, figsize=(6, 4.5))
    plt.tight_layout(pad=2.5)
    for i in range(num_points):
        ax[0].plot(x1, y2[i], '-^', label=labels[i], color=colors1[i])
    #for i in range(num_points):
        #ax[0].plot(x1, y1[i], '-o', label='Analytical', color=colors2[i])
    ax[0].legend()
    #ax[0].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], ['',0.5, '',0.7, '',0.9, '', 1.1])
    ax[0].set_xlabel(r'$\alpha$ [Å$^{-2}$]')
    ax[0].set_ylabel(r'$E_L$')
    ax[0].grid(True)

    for i in range(num_points):
        ax[1].plot(x1, y4[i], '-^', label=labels[i], color=colors1[i])
    #for i in range(num_points):
        #ax[1].plot(x1, y3[i], '-o', label='Analytical', color=colors2[i])
    ax[1].legend()
    #ax[1].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.3, '',0.5, '',0.7, '',0.9])
    #ax[1].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], ['',0.5, '',0.7, '',0.9, '', 1.1])
    ax[1].set_xlabel(r'$\alpha$ [Å$^{-2}$]')
    ax[1].set_ylabel(r'$\sigma^2$')
    ax[1].grid(True)
    plt.savefig(oppg+'-'+str(P)+'P-'+str(D)+'D_dt.pdf')
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

def read_file(filenames):
    alpha = []
    E_a = []
    E_n = []
    var_a = []
    var_n = []
    for filename in filenames:
        x1=[]
        y1=[]
        y2=[]
        y3=[]
        y4=[]
        f = open(filename)
        f.readline()
        l = f.readline()
        #v1 = float(l.split()[0])
        #v2 = float(l.split()[1])
        f.readline()
        lines = f.readlines()
        f.close()

        for line in lines:
            l = line.split()
            i=0
            x1.append(float(l[i]))
            y1.append(float(l[i+1]))
            y2.append(float(l[i+2]))
            y3.append(float(l[i+3]))
            y4.append(float(l[i+4]))
        E_a.append(y1)
        E_n.append(y2)
        var_a.append(y3)
        var_n.append(y4)
    return x1, E_a, E_n, var_a, var_n




def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]
def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
main()
