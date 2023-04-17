import numpy as np
from math import exp, sqrt
from random import random, seed, choice
import matplotlib.pyplot as plt
from scipy.stats import norm
from pylab import rcParams


N = int(1e4) #number of bootstraps

f = open('energy-100P-3D.txt', 'r')
lines = f.readlines()
f.close()

alphas = []
for line in lines:
    alphas.append(float(line.split()[0]))

bootstrap = []
N_alpha = len(alphas)
for i in range(N):
    b = 0
    for j in range(N_alpha):
        b += choice(alphas)
    bootstrap.append(b/N_alpha)

#n, bins, patches = plt.hist(bootstrap, bins=100)
avg = np.mean(bootstrap)
var = np.var(bootstrap)
print('avg:', avg, 'var:', var)
# From that, we know the shape of the fitted Gaussian.
pdf_x = np.linspace(np.min(bootstrap),np.max(bootstrap),100)
pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)

# Then we plot :
rcParams.update({"text.usetex": True,"font.family": "serif","font.sans-serif": ["Computer Modern Roman"]})
fig, ax = plt.subplots(figsize=(5, 3))
plt.tight_layout(pad=2.5)
#plt.figure()
plt.hist(bootstrap,bins=100,density=True,color='g',label='Bootstrap')
plt.plot(pdf_x,pdf_y,'r--',label='Gaussian fit')
plt.legend()
#plt.show()

plt.xlabel(r'$E_L$')
plt.ylabel('Counts')
#plt.text(1.50035, 4000, r'$\langle E_L\rangle$='+str(round(avg,5))+'\n'+r'$\sigma$='+str(round(np.sqrt(var),8)))
#plt.xlim(1.3003, 1.5012)
plt.savefig('bootstrap-100.pdf')
plt.show()

teller_std2 = 0
teller_std3 = 0
std = np.sqrt(var)
min2 = avg-2*std
max2 = avg+2*std
min3 = avg-3*std
max3 = avg+3*std
for i in range(len(bootstrap)):
    if min3 < bootstrap[i] and bootstrap[i] < max3:
        teller_std3+=1
        if min2 < bootstrap[i] and bootstrap[i] < max2:
            teller_std2+=1
print(r'Within 2*sigma:', 100*teller_std2/len(bootstrap), '%')
print(r'Within 3*sigma:', 100*teller_std3/len(bootstrap), '%')

"""
hist, bin_edges = np.histogram(bootstrap, bins=100)
#hist=hist/sum(hist)
n = len(hist)
x_hist=np.zeros(n)
for i in range(n):
    x_hist[i]=(bin_edges[i+1]+bin_edges[i])/2

y_hist=hist

mean = sum(x_hist*y_hist)/sum(y_hist)
sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist)

print(mean, sigma)

plt.plot(x_hist, norm.pdf(x_hist, mean, sigma))





# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
y = norm.pdf(bins, mu, sigma)
#pyplot.plot(x, norm.pdf(x, meanAverage, standardDeviation))
l = plt.plot(bins, y, 'r--', linewidth=2)
"""
