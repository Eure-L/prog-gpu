#!/usr/bin/env python
from genericpath import isfile
import os
import subprocess
import csv
import sys
import argparse


LABEL =None
KERNEL="dgemm"
VERSION="seq"
datafile = None

CRANGE= []
c = 0
C = 1

p=1

### Initialisation des arguments
argv = str(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('-c',"--cercles",help="nombre de cercles à générer")
parser.add_argument('-k',"--kernel",help="kernels a faire tourner")
parser.add_argument('-lb',"--label",help="label a ajouter")

args = parser.parse_args()
print(args)

if(args.cercles != None):
    c = args.cercles
    print(args.cercles)
    if c.isdigit():
        c=int(c)
        C=c+1
        p=100
    else:
        c=c.split(':')
        C=int(c[1])
        p=int(c[2])
        c=int(c[0])
        print(c)
        print(C)
        print(p)

if(args.kernel != None):
    KERNEL=(args.kernel)
kernels = KERNEL.split(",")

if(args.label != None):
    LABEL=(args.label)


FILE="cercles.csv"
### Initialisation des fichiers / repertoires

if (not(os.path.isfile(FILE))):
    datafile = open(FILE, 'a')
    print("kernel;cercles;temps;label;",file=datafile)

else:
    datafile = open(FILE, 'a')

##########
def run(k,cercles):
    exexLine = "./"+ str(k) + " "+ str(cercles)
    print(exexLine)
    out = subprocess.check_output(exexLine, shell=True)

    out = str(out)
    out = out.split(' ')

    perf = (str(out[0])).split("'")
    print(perf)
    perf = perf[1].replace("\\n",'')
    
    print(perf)

    data_line = k+';'+str(cercles)+';'+str(perf)+";"+str(LABEL)
    print(data_line,file=datafile)

#########
print("ok\n")
print(kernels)
range(c,C,p)
tmp = c
print("RUNNING")

for k in kernels:
    c = tmp
    for c in range(c,C,p):
        print(c)
        print(C)
        run(k,c)


datafile.close()
