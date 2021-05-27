import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy
import time

#Computations are done in terms of y_ij, which is an asymmetric square matrix; the first index denotes the component on the first (even) sublattice , and the second on the second (odd) sublattice
#helper functions are written to accept square matrices of any size, so can be used for an n-ary composition of arbitrary n>1.

#helper functions and variables
MAX_ITER    = 2048
Y_PRECISION = 1e-15
T_PRECISION = 1e-9
xA_TOL      = 1e-9

z=4

cA=[[2,1,1],[1,0,0],[1,0,0]]
cB=[[0,1,0],[1,2,1],[0,1,0]]
cC=[[0,0,1],[0,0,1],[1,1,2]]
c=[cA,cB,cC]

def xlx(x):
    if x<0: return 1
    if x==0: return 0
    return x*math.log(x)

def dot(M, N):
    total=0
    rowsize = len(M[0])
    for i in range(len(M)):
        for j in range(rowsize):
            total+= M[i][j]*N[i][j]
    return total

def norm(M):
    return math.sqrt(dot(M,M))

def abstot(M):
    total=0
    for mi in M:
        for mij in mi:
            total+=abs(mij)
    return total            

# These functions also require M to be a well-formed matrix (rowsize)
def diff(M,N):
    rowsize = len(M[0])
    return [[M[i][j] - N[i][j] for j in range(rowsize)] for i in range(len(M))]

def add(M,N):
    rowsize = len(M[0])
    return [[M[i][j] + N[i][j] for j in range(rowsize)] for i in range(len(M))]

def transpose(M):
    return [list(row) for row in zip(*M)]

def Yt(y):
    return add(y, transpose(y))

def Xe(y):
    xe=[]
    for yi in y:
        xei=0
        for yij in yi:
            xei+=yij
        xe.append(xei)
    return xe    

def Xo(y):
    return Xe(transpose(y))

def Xt(y):
    xe=Xe(y)
    xo=Xo(y)
    return [xe[i] + xo[i] for i in range(len(xe))]

#thermodynamic functions
def H(E,y):
    return z*dot(E,y)

def S(y):
    Sxlx, Syly=0,0
    x=Xe(y)+Xo(y)

    for xi in x:
        Sxlx+=xlx(2*xi)

    for yi in y:
        for yij in yi:
            Syly+=xlx(2*yij)

    return (z-1)/2*Sxlx-z/2*Syly

def F(E,y,T):
    if T==0: return H(E,y)
    return H(E,y)-T*S(y)


#minimization
def ytilde(ycur, Eb, T, m):
    rowsize = len(ycur[0])
    yres = [[0 for j in range(rowsize)] for i in range(len(ycur))]

    Xecur = Xe(ycur)
    Xocur = Xo(ycur)

    for i in range(len(ycur)):
        for j in range(rowsize):
            yres[i][j]=math.exp(-Eb[i][j]/T)
            yres[i][j]*=((Xecur[i]*Xocur[j])**((z-1)/z))
            #apply lagrange multipliers
            for k in range(len(m)):
                #TEMPORARY MEASURE
                try:
                    yres[i][j]*=math.exp(m[k]*c[k][i][j]/(z*T))
                except:
                    #This Error should only occur 
                    print("Fixed composition too low for computation")
               
  
    return normalize(yres)

def normalize(ytil):
    total = 0
    rowsize = len(ytil[0])
    for yi in ytil:
        for yij in yi:
            total+=yij
    return [[ytil[i][j]/(2 * total) for j in range(rowsize)] for i in range(len(ytil))]   


def search_y(y, Eb, T, m, xTarget, counter, debug=False):
    mixm = True
    if (xTarget == None): 
        mixm = False 

    counter += counter * mixm
    change = 1
    exited = False
    while change>Y_PRECISION:
            yold=y
            y=ytilde(yold, Eb, T, m)
            counter-=1
            if(counter<1):
                if debug:
                    print('  Max Iterations Reached')
                exited = True
                break
            if mixm:
                m = mix_mus(xTarget, Xt(y), m)
            change=norm(diff(y,yold))
    
    if debug and (not exited):
        print('    Iterations taken:'+str(MAX_ITER-counter))
        
    return y,m
    

def minimize(Eb=[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],
        Trang=[0,5],
        samp=200,
        guess=[[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]],
        m=[1.,0.6,0.5]):
    delta = (Trang[1]-Trang[0])/samp
    temp = np.linspace(Trang[0]+delta, Trang[1], samp)
    tp = np.linspace(Trang[0]+2*delta, Trang[1]-delta, samp-2)
    
    y=normalize(guess)
    print(y)

    mY,mF,E,C=[],[],[],[]
    xAe, xBe, xCe =[],[],[]
    xAo, xBo, xCo =[],[],[]
    xAt, xBt, xCt =[],[],[]
    for i in range(len(temp)):
        print('Calculating T='+str(temp[i]))
        T=temp[i]

        y,_ = search_y(y, Eb, T, m, None, MAX_ITER, True)
        
        x=Xe(y)
        xAe.append(2*x[0])
        xBe.append(2*x[1])
        xCe.append(2*x[2])

        x=Xo(y)
        xAo.append(2*x[0])
        xBo.append(2*x[1])
        xCo.append(2*x[2])
        
        x=Xt(y)
        xAt.append(x[0])
        xBt.append(x[1])
        xCt.append(x[2])

        mY.append(y)
        mF.append(F(Eb,y,T))

        if i>1:
            #calculate E
            dF=mF[i]-mF[i-2]
            dF/=(delta*2)
            E.append(mF[i-1]-temp[i-1]*dF)

            #calculate C
            ddF=mF[i]-2*mF[i-1]+mF[i-2]
            ddF/=(delta**2)
            C.append(-temp[i-1]*ddF)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle('2D square lattice, bond approx, ternary composition')
    
    gs=fig.add_gridspec(3,2)
    
    ax=fig.add_subplot(gs[0,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xAt,label='At', alpha=0.6)
    ax.plot(temp,xBt,label='Bt', alpha=0.6)
    ax.plot(temp,xCt,label='Ct', alpha=0.6)
    ax.legend()

    ax=fig.add_subplot(gs[1,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Free Energy')
    ax.plot(temp,mF)
    
    ax=fig.add_subplot(gs[2,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('E=F-T*dF/dT')
    ax.set_ylim(-5,5)
    ax.plot(tp, E)

    ax=fig.add_subplot(gs[2,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('C=-T*d2F/dT2')
    ax.set_ylim(-5,5)
    ax.plot(tp, C)
    
    ax=fig.add_subplot(gs[:2,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xAe,label='Ae', alpha=0.6)
    ax.plot(temp,xBe,label='Be', alpha=0.6)
    ax.plot(temp,xCe,label='Ce', alpha=0.6)

    ax.plot(temp,xAo,label='Ao', alpha=0.6)
    ax.plot(temp,xBo,label='Bo', alpha=0.6)
    ax.plot(temp,xCo,label='Co', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    #high temp slope should be ln(2)~0.693
    slope = mF[samp-1]-mF[math.floor(samp*0.75)]
    slope /= (temp[samp-1]-temp[math.floor(samp*0.75)])
    print('High temp slope: '+str(slope))
    #intercept should be ~0
    intercept=mF[samp-1]-slope*temp[samp-1]
    print('Intercept: ' + str(intercept))

    plt.tight_layout(pad=0, w_pad=0, h_pad=0, rect=(0,0,0.95,0.9))
    plt.show()

def getAllXValues(yList):

    ylen = len(yList)
    #Xes, Xos, Xts
    Xtypes = []
    Xtypes.append(list(map(Xe,yList)))
    Xtypes.append(list(map(Xo,yList)))
    Xtypes.append(list(map(Xt,yList)))

    #Ae, Be, Ce, ...
    Xseperated= []
    for Xtype in Xtypes:
        for i in range(3):
            Xseperated.append([Xtype[j][i] for j in range(ylen)])

    return tuple(Xseperated)

#phase diagram
def tsearch(Eb, m, Trang, guess, prevY, xTarget, debug):
    #recursive binary search
    Tavg = (Trang[1]+Trang[0])/2

    #base case
    if (Trang[1]-Trang[0])<T_PRECISION:
        return (Tavg, prevY)

    #Find whether to update top or bottom of range
    y=normalize(guess)
    if debug:
        print('    Tnew='+str(Tavg))

    y,m = search_y(y, Eb, Tavg, m, xTarget, MAX_ITER, debug)
    
    xdiff = abs(Xe(y)[0]-Xo(y)[0])
    if xdiff<=xA_TOL:
        return tsearch(Eb, m, [Trang[0], Tavg], guess, y, xTarget, debug)
    return tsearch(Eb, m, [Tavg, Trang[1]], guess, y, xTarget, debug)


def phase(Eb=[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],
        mBC=[1.0,1.0],
        Trang=[0,5],
        mrang=[0,2],
        mnum=200,
        guess=[[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]],
        debug=False):

    startTime = time.time()
    deltamA=mrang[1]-mrang[0]
    deltamA/=mnum
        
    mA_samp=np.linspace(mrang[0]+deltamA,mrang[1],mnum)
    
    Tc=[]
    ys=[]

    #Iterate through chemical potentials
    for mA in mA_samp:
        if debug:
            print('Calculating mA='+str(mA))
        
        m=[mA, mBC[0], mBC[1]]

        Tnew, ynew = tsearch(Eb, m, Trang, guess, guess, None, debug)
        ys.append(ynew)
        Tc.append(Tnew)

    #output to csv file
    with open('Tc_Comp_v_mA_mB'+str(math.floor(10*mBC[0]))+'_mC'+str(math.floor(10*mBC[1]))+'.csv', mode='w') as output:
        outputwriter=csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(mA_samp)):

            outputwriter.writerow([mA_samp[i],Tc[i],ys[i]])
        output.close()
     
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Transition Temp v. composition & chem potential with mB='+str(mBC[0])+' and mC='+str(mBC[1]))
    
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
    newplt.set_ylabel('Transition Temperature')
    newplt.set_xlabel('Composition (xA)')
    newplt.plot(At,Tc,label='At', alpha=0.6)

    print("Runtime: " + str(time.time()-startTime))
    plt.show()

def mix_mus(xTarg, xOld, mOld):
    xlen = len(xTarg)
    return [(xTarg[i]-xOld[i])/10+mOld[i] for i in range(xlen)]

def minimize_x(Eb=[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],
        Trang=[0,5],
        samp=200,
        yGuess=[[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]],
        xTarget=[0.4,0.3,0.3],
        mGuess=[1.,0.6,0.5],
        debug=False):

    delta = (Trang[1]-Trang[0])/samp
    temp = np.linspace(Trang[0]+2*delta, Trang[1], samp)
    tp = np.linspace(Trang[0]+3*delta, Trang[1]-delta, samp-2)
    
    y=normalize(yGuess)

    mY,mF,E,C=[],[],[],[]
    xAe, xBe, xCe =[],[],[]
    xAo, xBo, xCo =[],[],[]
    mA, mB, mC =[],[],[]


    for i in range(len(temp)):
        if debug:
            print('Calculating T='+str(temp[i]))
        T=temp[i]

        y,m = search_y(y, Eb, T, mGuess, xTarget, MAX_ITER, debug)
        
        x=Xe(y)
        xAe.append(2*x[0])
        xBe.append(2*x[1])
        xCe.append(2*x[2])

        x=Xo(y)
        xAo.append(2*x[0])
        xBo.append(2*x[1])
        xCo.append(2*x[2])
        
        mA.append(m[0])
        mB.append(m[1])
        mC.append(m[2])

        mY.append(y)
        mF.append(F(Eb,y,T))

        if i>1:
            #calculate E
            dF=mF[i]-mF[i-2]
            dF/=(delta*2)
            E.append(mF[i-1]-temp[i-1]*dF)

            #calculate C
            ddF=mF[i]-2*mF[i-1]+mF[i-2]
            ddF/=(delta**2)
            C.append(-temp[i-1]*ddF)

    #print(min(mA), max(mA), min(mB), max(mB), min(mC), max(mC))
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('2D square lattice, bond approx, ternary composition')
    
    gs=fig.add_gridspec(3,2)
    
    ax=fig.add_subplot(gs[0,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Chemical Potential')
    ax.plot(temp,mA,label='muA', alpha=0.6)
    ax.plot(temp,mB,label='muB', alpha=0.6)
    ax.plot(temp,mC,label='muC', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)


    ax=fig.add_subplot(gs[1,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Free Energy')
    ax.plot(temp,mF)
    
    ax=fig.add_subplot(gs[2,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('E=F-T*dF/dT')
    ax.set_ylim(-5,5)
    ax.plot(tp, E)

    ax=fig.add_subplot(gs[2,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('C=-T*d2F/dT2')
    ax.set_ylim(-5,5)
    ax.plot(tp, C)
    
    ax=fig.add_subplot(gs[:2,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xAe,label='Ae', alpha=0.6)
    ax.plot(temp,xBe,label='Be', alpha=0.6)
    ax.plot(temp,xCe,label='Ce', alpha=0.6)

    ax.plot(temp,xAo,label='Ao', alpha=0.6)
    ax.plot(temp,xBo,label='Bo', alpha=0.6)
    ax.plot(temp,xCo,label='Co', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    #high temp slope should be ln(2)~0.693
    slope = mF[samp-1]-mF[math.floor(samp*0.75)]
    slope /= (temp[samp-1]-temp[math.floor(samp*0.75)])
    print('High temp slope: '+str(slope))
    #intercept should be ~0
    intercept=mF[samp-1]-slope*temp[samp-1]
    print('Intercept: ' + str(intercept))

    plt.tight_layout(pad=0, w_pad=0, h_pad=0, rect=(0,0,0.95,0.9))
    plt.show()

def phase_x(Eb=[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],
        xBCrel=[0.5,0.5],
        Trang=[0,5],
        xArang=[0.3,0.8],
        xnum=50,
        guess=[[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]],
        mGuess = [1.,0.6,0.5],
        debug=False):

    startTime = time.time()
    deltaxA=xArang[1]-xArang[0]
    deltaxA/=xnum
        
    xA_samp=np.linspace(xArang[0]+deltaxA,xArang[1],xnum)
    
    Tc=[]
    ys=[]
    xs=[]

    xBC = xBCrel[0]+xBCrel[1]

    #Iterate through compositions 
    for xA in xA_samp:
        #Fix xB and xC for our given xA so they correclty sum to 1
        xBCnorm = 1-xA
        x=[xA, xBCrel[0]/xBC *xBCnorm, xBCrel[1]/xBC *xBCnorm]
        if debug:
            print('Calculating xA='+str(xA))

        Tnew, ynew = tsearch(Eb, mGuess, Trang, guess, guess, x, debug)
        ys.append(ynew)
        Tc.append(Tnew)
        xs.append(Xt(ynew)[0])

    #output to csv file
    with open('Tc_Comp_v_xA_xB'+str(math.floor(10*xBCrel[0]))+'_xC'+str(math.floor(10*xBCrel[1]))+'.csv', mode='w') as output:
        outputwriter=csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(xA_samp)):

            outputwriter.writerow([xA_samp[i],Tc[i],ys[i]])
        output.close()
     
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Transition Temp v. composition with relative xB='+str(xBCrel[0])+' and xC='+str(xBCrel[1]))
    
    gs = fig.add_gridspec(1,1)

    newplt = fig.add_subplot(gs[0,0])
    newplt.plot(xs, Tc)
    newplt.set_xlabel('xA')
    newplt.set_ylabel('Transition Temperature')

    # Ae, Be, Ce, Ao, Bo, Co, At, Bt, Ct = getAllXValues(ys)

    # newplt = fig.add_subplot(gs[1,:2])
    # newplt.set_xlabel('mA')
    # newplt.set_ylabel('Composition')
    # newplt.set_ylim(-0.05,1.05)

    # newplt.plot(mA_samp,Ae,label='Ae', alpha=0.6)
    # newplt.plot(mA_samp,Be,label='Be', alpha=0.6)
    # newplt.plot(mA_samp,Ce,label='Ce', alpha=0.6)

    # newplt.plot(mA_samp,Ao,label='Ao', alpha=0.6)
    # newplt.plot(mA_samp,Bo,label='Bo', alpha=0.6)
    # newplt.plot(mA_samp,Co,label='Co', alpha=0.6)
    # newplt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    # newplt = fig.add_subplot(gs[2,1])
    # newplt.set_xlabel('mA')
    # newplt.set_ylabel('Composition')
    # newplt.set_ylim(-0.05,1.05)
    # newplt.plot(mA_samp,At,label='At', alpha=0.6)
    # newplt.plot(mA_samp,Bt,label='Bt', alpha=0.6)
    # newplt.plot(mA_samp,Ct,label='Ct', alpha=0.6)
    # newplt.legend()

    print("Runtime: " + str(time.time()-startTime))
    plt.show()

#y = normalize([[1,1],[1,1]])
#Es = [[1,-1],[-1,1]]
#print(y)
#print(Xt(y))
print(Xt(normalize([[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]])))