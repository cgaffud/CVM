import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy
import time

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

    for l in range(3):
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
    '''E consists of (h,J,K,L)'''
    dim = len(z)
    
    # this method is actually awful, we want to calculate unique U's once, and then abuse transpose somehow
    us = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    # ys matrix (done this way for x calculations)
    ys = [[None for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            #BAD!!!!
            for k in range(4):
                if (i!=j or j!=k or i!=k):
                    us[i][j][k] = U(z,i,j,k)
            if (i<j):
                ys[i][j] = Y(z,i,j) 
            elif (i>j):
                #This is correct but is it needed
                ys[i][j] = transpose(ys[j][i])

    # Nearest neighbor bonds
    y_nearest = [ys[0][1],ys[0][3],ys[2][1],ys[2][3]]
    y_all = [ys[0][1],ys[0][2],ys[0][3],ys[1][3],ys[2][1],ys[2][3]]
    # Compositionals (Technically faster because don't have to recompute Ys)
    xs = [X_i([ys[i][(i+j) % 4] for j in range(1,4)]) for i in range(4)]
    print(xs)

    def H():
        h,J,K,L = Es
        # will have multiplicity of 4
        Hx, Hy, Hu, Hz = 0,0,0,0

        for x in xs:
            for i in range(dim):
                Hx += x[i]*h[i]

        for y in y_all:
            for i in range(dim):
                for j in range(dim):
                    Hy += y[i][j]*J[i][j]
        
        for u in us:
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        Hu -= u[i][j][k]*K[i][j][k]

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        Hz += z[i][j][k][l]*L[i][j][k][l]

        return Hx/4-Hy/6-Hu/4-Hz

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

        return -Sxlx/4+2*Syly/4-Szlz

    if T==0: return H()
    return H()-T*S()

# Arbitrary Test Case/DEBUG
#z = [ [ [[2,1],[1,1]],[[1,1],[1,1]] ],[ [[1,1],[1,1]],[[1,1],[1,1]] ] ]
#print(X(z,0))
#F(z,None,0)
#h = [0.5,0.5]