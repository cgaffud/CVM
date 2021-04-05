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
                    y[indices[i1]][indices[i2]] = z[i][j][k][l]
    
    return y

def X_i(ys):
    '''Gets component of X_i by summing up already-computed Y_ij, 
       where i indicates point on square'''
    dim = len(ys[0])
    x = []
    for i in range(dim):
        x[i]=0
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


def F(z,Es):
    '''E consists of [h,J,K,L]'''
    dim = len(z)

    # Symmetric y matrix (so that xs can be simple) [includes nnnbs]
    y_all = []
    for i in range(dim):
        for j in range(dim):
            if (i<j):
                y_all[i][j] = Y(z,i,j) 
            elif (i>j):
                y_all[i][j] = y_all[j][i]
            else:
                y_all[i][j] = None

    # Nearest neighbor bonds
    ys = [y_all[0][1],y_all[0][4],y_all[2][1],y_all[2][3]]
    # Compositionals
    xs = [X_i([y_all[i][(j % 4)] for j in range(3)]) for i in range(dim)]

    #NEEDS TO BE WRITTEN
    def H():
        pass

    def S():

        Sxlx, Syly, Szlz = 0,0,0

        for x in xs:
            for xi in x:
                Sxlx += xlx(xi)
        
        for y in ys:
            for yi in y:
                for yij in yi:
                    Syly += xlx(yij)
        
        for zi in z:
            for zij in zi:
                for zijk in zij:
                    for zijkl in zijk:
                        Szlz += xlx(zijkl)

        return -Sxlx/4+2*Syly/4-Szlz

    
    