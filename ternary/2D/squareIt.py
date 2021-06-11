import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy
import time

Z_PRECISION = 1e-18
MAX_ITER    = 2048

#s is for species (i,j,k,l all are for lattice point consideration)
C = [[[[[0  for i in range(3)] for j in range(3)] for l in range(3)] for k in range(3)] for s in range(3)]

#Construction of C matrix
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                for s in range(3):
                    C[i][j][k][l][s] += 1 if (i==s) else 0
                    C[i][j][k][l][s] += 1 if (j==s) else 0
                    C[i][j][k][l][s] += 1 if (k==s) else 0
                    C[i][j][k][l][s] += 1 if (l==s) else 0

def xlx(x):
    if x<0: return 1
    if x==0: return 0
    return x*math.log(x)

def dot(M, N):
    total=0
    rowsize = len(M[0])
    for i in range(rowsize):
        for j in range(rowsize):
            for k in range(rowsize):
                for l in range(rowsize):
                    total+= M[i][j][k][l]*N[i][j][k][l]
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
    return [[[[M[i][j][k][l] - N[i][j][k][l] for l in range(rowsize)] for k in range(rowsize)] for j in range(rowsize)] for i in range(len(M))]

def add(M,N):
    rowsize = len(M[0])
    return [[[[M[i][j][k][l] + N[i][j][k][l] for l in range(rowsize)] for k in range(rowsize)] for j in range(rowsize)] for i in range(len(M))]

def transpose(M):
    return [list(row) for row in zip(*M)]

#Note: Z is a four-dim matrix for square abdg (z_abdg)

def U(z, i1, i2 ,i3):
    '''Get u_i1i2i3 from z, where (i1,i2,i3) indicates vertices of triangle'''
    if (i1 == i2) or (i1 == i3) or (i2 == i3):
        raise IndexError
    
    dim = len(z)
    u = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    indices = [i,j,k,l]
                    u[indices[i1]][indices[i2]][indices[i3]] += z[i][j][k][l]
    return u

def Y_ij(us):
    '''Get y_ij from by summing up already-computed u_ijk,
    where ij is the indicatesthe bond on the square.
    Note that we require us to be filled as u_ij '''
    dim = len(us[0])
    
    y =  [[0 for _ in range(dim)] for _ in range(dim)]

    for l in range(2):
        for i in range(dim):
            for j in range(dim):
                y[0].append(0)
                for k in range(dim):
                        y[i][j] += us[l][i][j][k]
    return y

def Y(z, i1, i2):
    '''Get y_i1i2 from z, where (i1,i2) indicates the bond expected'''
    if i1 == i2:
        #Bad index
        raise IndexError
    
    dim = len(z)

    y = [[0 for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    indices = [i,j,k,l]
                    y[indices[i1]][indices[i2]] += z[i][j][k][l]
    
    return y

def X_i(ys):
    '''Gets component of x_i by summing up already-computed y_ij, 
       where i indicates point on square.
       Note that we require ys to be filled with y_ij, not y_ji'''
    #print("Ys: "+str(ys))
    dim = len(ys[0])
    x = [0 for _ in range(dim)]

    for k in range(3):
        for i in range(dim):
            for j in range(dim):
                    x[i] += ys[k][i][j]
    return x
        
def X(z,i):
    '''Get X_i from Z, where i indicates the point on the square'''

    vertices = [0,1,2,3]
    vertices.pop(i)

    ys = []
    for vertex in vertices:
        ys.append(Y(z,i,vertex))
    
    return X_i(ys)


def F(z,Es,T):
    '''E consists of only nnbs (J)'''
    dim = len(z)
    # ys matrix (done this way for x calculations)
    ys = [[None for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if (i<j):
                ys[i][j] = Y(z,i,j)
            elif (i>j):
                ys[i][j] = transpose(ys[j][i])

    # Nearest neighbor bonds
    y_nearest = [ys[0][1],ys[1][2],ys[2][1],ys[3][0]]
    y_all = [ys[0][1],ys[0][2],ys[0][3],ys[1][3],ys[2][1],ys[2][3]]
    # Compositionals (Technically faster because don't have to recompute Ys)
    xs = [X_i([ys[i][(i+j) % 4] for j in range(1,4)]) for i in range(4)]

    def H():
        # will have multiplicity of 4
        H = 0

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        Esum = 1/2*(Es[i][j] + Es[j][k] + Es[k][l] + Es[l][i])
                        H += Esum * z[i][j][k][l]

        print("H: "+str(H))
        return H

    def S():
        Sxlx, Syly, Szlz = 0,0,0

        for x in xs:
            for xi in x:
                Sxlx += xlx(xi)
        
        for y in y_nearest:
            for yi in y:
                for yij in yi:
                    Syly += xlx(yij)
        
        for zi in z:
            for zij in zi:
                for zijk in zij:
                    for zijkl in zijk:
                        Szlz += xlx(zijkl)
        
        print("S: "+str(-Sxlx/4+2*Syly/4-Szlz))
        return -Sxlx/4+2*Syly/4-Szlz

    if T==0: return H()
    return H()-T*S()

def normalize(ztil):
    total = 0
    dim = len(ztil[0])
    for zi in ztil:
        for zij in zi:
            for zijk in zij:
                for zijkl in zijk:  
                    total += zijkl
    return [[[[ztil[i][j][k][l]/(3*total) for l in range(dim)] for k in range(dim)] for j in range(dim)] for i in range(dim)]

def ztilde(zcur, Es, T, m):
    dim = len(zcur)
    zres = [[[[0 for l in range(dim)] for k in range(dim)] for j in range(dim)] for i in range(dim)]
    x_i, x_j, x_k, x_l = X(zcur,0), X(zcur,1), X(zcur,2), X(zcur,3)
    y_ij, y_jk, y_kl, y_li = Y(zcur,0,1), Y(zcur,1,2), Y(zcur,2,3), Y(zcur,3,0)

    total = 0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    
                    #This is so bad memory-wise
                    Esum = 1/2*(Es[i][j]+Es[j][k]+Es[k][l]+Es[l][i])
                    res =  math.exp(-Esum/T) * ((y_ij[i][j]*y_jk[j][k]*y_kl[k][l]*y_li[l][i])**(1/2))/(x_i[i]*x_j[j]*x_k[k]*x_l[l])**(1/4)
                    for s in range(dim):
                        res *= math.exp(m[s]*C[i][j][k][l][s]/T)
                        
                    zres[i][j][k][l] = res 
                    total += res 

    return [[[[zres[i][j][k][l]/(3*total) for l in range(dim)] for k in range(dim)] for j in range(dim)] for i in range(dim)]

def search_z(z, Eb, T, m, counter, debug=False):
    change = 1
    exited = False
    while change>Z_PRECISION:
            zold=z
            z=ztilde(zold, Eb, T, m)
            counter-=1
            if(counter<1):
                if debug:
                    print('  Max Iterations Reached')
                exited = True
                break
            change=norm(diff(z,zold))
    print("Change: " + str(change))
    if debug and (not exited):
        print('    Iterations taken:'+str(MAX_ITER-counter))
        
    return z
            
def minimize(Eb=[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],
        Trang=[0,5],
        samp=50,
        guess=None,
        m=[1.,0.6,0.5]):
    delta = (Trang[1]-Trang[0])/samp
    temp = np.linspace(Trang[0]+delta, Trang[1], samp)
    tp = np.linspace(Trang[0]+2*delta, Trang[1]-delta, samp-2)
    
    z=normalize(guess)

    mF,E,C=[],[],[]
    xA, xB, xC = ([[] for _ in range(4)] for _ in range(3))
    xAt, xBt, xCt =[],[],[]
    for i in range(len(temp)):
        print('Calculating T='+str(temp[i]))
        T=temp[i]

        z = search_z(z, Eb, T, m, MAX_ITER, True)
        
        #Get all site compositions
        xs = [X(z,i) for i in range(4)]

        #Extract out single site compositions
        for j in range(4):
            xA[j].append(xs[j][0])
            xB[j].append(xs[j][1])
            xC[j].append(xs[j][2])

        #Get total compositions
        x = [(xs[0][i]+xs[1][i]+xs[2][i]+xs[3][i])/4 for i in range(3)]
        xAt.append(x[0])
        xBt.append(x[1])
        xCt.append(x[2])

        mF.append(F(z,Eb,T))

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
    fig.suptitle('2D square lattice, square approx, ternary composition')
    
    gs=fig.add_gridspec(4,3)
    
    ax=fig.add_subplot(gs[0,2])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xAt,label='At', alpha=0.6)
    ax.plot(temp,xBt,label='Bt', alpha=0.6)
    ax.plot(temp,xCt,label='Ct', alpha=0.6)
    ax.legend()

    ax=fig.add_subplot(gs[1,2])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Free Energy')
    ax.plot(temp,mF)
    
    ax=fig.add_subplot(gs[3,2])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('E=F-T*dF/dT')
    ax.set_ylim(-5,5)
    ax.plot(tp, E)

    ax=fig.add_subplot(gs[2,2])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('C=-T*d2F/dT2')
    ax.set_ylim(-5,5)
    ax.plot(tp, C)
    
    ax=fig.add_subplot(gs[:2,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition on α')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xA[0],label='A_α', alpha=0.6)
    ax.plot(temp,xB[0],label='B_α', alpha=0.6)
    ax.plot(temp,xC[0],label='C_α', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    ax=fig.add_subplot(gs[:2,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition on β')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xA[1],label='A_β', alpha=0.6)
    ax.plot(temp,xB[1],label='B_β', alpha=0.6)
    ax.plot(temp,xC[1],label='C_β', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    ax=fig.add_subplot(gs[2:,1])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition on γ')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xA[2],label='A_γ', alpha=0.6)
    ax.plot(temp,xB[2],label='B_γ', alpha=0.6)
    ax.plot(temp,xC[2],label='C_γ', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)

    ax=fig.add_subplot(gs[2:,0])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Composition on δ')
    ax.set_ylim(-0.05,1.05)
    ax.plot(temp,xA[3],label='A_δ', alpha=0.6)
    ax.plot(temp,xB[3],label='B_δ', alpha=0.6)
    ax.plot(temp,xC[3],label='C_δ', alpha=0.6)
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
  

# Arbitrary Test Case/DEBUG

ybase = [[25,212.5,212.5],
               [12.5,12.5,0],
               [12.5,0,12.5]]
z = [[[[ ybase[i][j] * ybase[j][k] * ybase[k][l] * ybase[l][i] if (i == k and j == l) else 0 for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]

minimize(guess = z)
znormal = normalize(z)
print(z)
print("Normalized: "+str(Y(normalize(z),0,1)))

znew = search_z(znormal,[[0,-1,-1],
            [-1,0,0],
            [-1,0,0]],1, [1.,0.6,0.5],MAX_ITER,True)
            
print(F(znew, [[0,-1,-1],
             [-1,0,0],
             [-1,0,0]], 1))
# print("After one iteration: "+str(X(znew,0)))
