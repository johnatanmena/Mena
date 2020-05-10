import numpy as np
import matplotlib.pyplot as plt
from math import ceil #redondea los numeros por encima


def normalizar(X):
    m,n = np.shape(X)
    minimos = np.min(X,axis=0)
    maximos = np.max(X,axis=0)
    return (X-minimos)/(maximos-minimos)

def trainPerceptron(X,Y,alpha,MaxIter,Tol,plotTrainError=False):
    m,n = X.shape
    Xbar=np.hstack((np.ones((m,1)),X))
    W=np.random.rand(n+1,1)
    Iter=0
    trainError = []
    while True:
        Iter += 1
        nW = W - (alpha/m*np.dot((np.sign(np.dot(Xbar,W))-Y).T,Xbar).T)
        if Iter > MaxIter or np.linalg.norm(W-nW) < Tol :
            #print('Gradient descent stoped at ',Iter,' Iteractions')
            break
        else:
            W = nW
            Error= np.sum((Xbar@W-Y)**2)/m
            trainError.append(Error)
    if plotTrainError:        
        plt.plot(range(Iter-1),trainError)
        plt.show(block = False)
    return W

def validacion(X,Y):
    m,n = X.shape
    corte70 = round(m*0.7)
    sorteo = np.random.permutation(m)
    Xtrain = X[sorteo[0:corte70],:]
    Xtest = X[sorteo[corte70:],:]
    Ytrain = Y[sorteo[0:corte70],:]
    Ytest = Y[sorteo[corte70:],:]
    return Xtrain,Ytrain,Xtest,Ytest

def evaluatePerceptron(W,X):
    m,n = X.shape
    Xbar=np.hstack((np.ones((m,1)),X))
    Y = np.sign(Xbar@W)
    return Y

def confuMat(Y,Yes,orden=False):
    if orden == False:
        clases=np.unique(Y)
    else:
        clases=orden
    
    numClases=clases.size
    C=np.zeros((numClases,numClases))
    for i in range(numClases):
        for j in range(numClases):
            Estimados = Yes[Y == clases[i]]
            C[i,j]=np.sum(Estimados== clases[j])
    return C,clases
            
def k_fold_cross_validation(k,m):
    Fold_Index=np.zeros((m))
    Permutacion=np.random.permutation(m)
    start=0
    end=ceil(m/k) #reondea por encima
    for i in range(k):
        Fold_Index[Permutacion[start:end]]=i
        start=end
        end+=ceil(m/k)
        if end>m:
            end=m
    return Fold_Index

def EstadisticEvaluate(Vp,Vn,Fp,Fn):# Coge los valores y los opera para encontrar las medidas de desempeño
    sen=100*Vp/(Vp+Fn)
    esp=100*Vn/(Vn+Fp)
    exac=100*(Vp+Vn)/(Vp+Vn+Fp+Fn)
    return sen,esp,exac
    