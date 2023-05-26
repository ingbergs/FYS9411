import os, sys

x = []
fileName = str(sys.argv[1])
print(fileName + ' hello')
with open(fileName) as f:
    for readline in f:
        line = readline.split('\n')
        if(line[0] != 'nan'):
            x.append(line[0])
        
w = open(fileName + '_clean.txt', 'w')
for i in range(len(x)):
    if(abs(float(x[i])) < 2):
        w.write(x[i] + '\n')
    

#print(x)