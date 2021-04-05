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
    '''Gets component of X_i by summing up already-computed Y_ij, 
       where i indicates point on square.
       Note that we require ys to be filled with Y_ij, not Y_ji'''
    print("Ys: "+str(ys))
    dim = len(ys[0])
    x = []

    for i in range(dim):
        x.append(0)
        for j in range(dim):
            for k in range(3):
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
    
    # ys matrix (includes next-nearest neighbor bonds for H)
    ys = [[None for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if (i<j):
                ys[i][j] = Y(z,i,j) 
            elif (i>j):
                #This is correct but is it needed
                ys[i][j] = transpose(ys[j][i])

    # Nearest neighbor bonds
    y_nearest = [ys[0][1],ys[0][3],ys[2][1],ys[2][3]]
    # Compositionals (Technically faster because don't have to recompute Ys)
    xs = [X_i([ys[i][(i+j) % 4] for j in range(1,4)]) for i in range(4)]
    print(xs)

    #NEEDS TO BE WRITTEN
    def H():
        h,J,K,L = Es

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

    s = S()
    return s

    