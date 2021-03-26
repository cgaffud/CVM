"""
reader.py - reads data outputed to .csv files 
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
# If bondIt currently is trying to run something (phase/min), then this import will also run that code first
from bondIt import getAllXValues


def get_mB_mC(filename):
    filename = filename[:-4]
    filesplit = filename.split('_')
    mB, mC = 0,0
    for header in filesplit:
        if header[:2] == 'mB':
            mB = header[2]+'.'+header[3:]
        if header[:2] == 'mC':
            mC = header[2]+'.'+header[3:]
    return mB,mC
                

def read_phase(filename):
    """Reads output from ternary 2D bond approx phase calculations into graphs"""
    csvfile = open(filename)
    csvreader = csv.reader(csvfile, delimiter=',')
    mA_samp, Tc, ys = [], [], []    
    for row in csvreader:
        if (row == []):
            continue
        mA_samp.append(float(row[0]))
        Tc.append(float(row[1]))

        #convert from string back to list
        yrows = row[2][1:-1].split("], [")
        yactual = []
        for i in range(3):
            yrows[i] = (yrows[i][1:] if i == 0 else (yrows[i][:-1] if i == 2 else yrows[i])) 

            yrsplit = yrows[i].split(", ")
            yactual.append([float(yrsplit[i]) for i in range(3)])
        ys.append(yactual)

    mB, mC = get_mB_mC(filename)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Transition Temp v. composition & chem potential with mB='+mB+' and mC='+mC)
    
    gs = fig.add_gridspec(3,2)

    newplt = fig.add_subplot(gs[2,0])
    newplt.plot(mA_samp, Tc)
    newplt.set_xlabel('mA')
    newplt.set_ylabel('Transition Temperature')

    Ae, Be, Ce, Ao, Bo, Co, At, Bt, Ct = getAllXValues(ys)

    newplt = fig.add_subplot(gs[1,:2])
    newplt.set_xlabel('mA')
    newplt.set_ylabel('Composition')
    newplt.set_ylim(-0.05,1.05)

    newplt.plot(mA_samp,Ae,label='Ae', alpha=0.6)
    newplt.plot(mA_samp,Be,label='Be', alpha=0.6)
    newplt.plot(mA_samp,Ce,label='Ce', alpha=0.6)

    newplt.plot(mA_samp,Ao,label='Ao', alpha=0.6)
    newplt.plot(mA_samp,Bo,label='Bo', alpha=0.6)
    newplt.plot(mA_samp,Co,label='Co', alpha=0.6)
    newplt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    newplt = fig.add_subplot(gs[2,1])
    newplt.set_xlabel('mA')
    newplt.set_ylabel('Composition')
    newplt.set_ylim(-0.05,1.05)
    newplt.plot(mA_samp,At,label='At', alpha=0.6)
    newplt.plot(mA_samp,Bt,label='Bt', alpha=0.6)
    newplt.plot(mA_samp,Ct,label='Ct', alpha=0.6)
    newplt.legend()

    plt.show()
       


read_phase("Tc_Comp_v_mA_mB10_mC10.csv")