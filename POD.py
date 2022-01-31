#!/usr/bin/python3
import os
import time
from concurrent.futures import ProcessPoolExecutor as Executor
import numpy as np
from numpy.fft import fft,fftfreq
import matplotlib.pyplot as plt
import sys
import shutil
import argparse
import math
from scipy.linalg import svd
from scipy.linalg import norm as slnorm
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

def SVD(D):
    print("SVD :: Creating covariance matrix ")
    M = np.dot(D.T,D) 
    print("SVD :: Performing eigendecomposition ")
    #EigVals,EigVecs = eigsh(M,k = M.shape[0]-1,which = "LM")
    
    EigVals,EigVecs = eigsh(M,k = min(M.shape[0]-1,500),which = "LM")    
    
    ind = np.argsort(-1*EigVals)
    EigVals = EigVals[ind] # Sorting from smallest to largest
    EigVecs = EigVecs[:,ind] # Sorting from smallest to largest
    print("SVD :: Calculating optimal rank for 0.999% of variance ")
    
    SingVals = np.sqrt(EigVals)
    
    totalVar = np.sum(SingVals)
    cumulativeVar = 0
    k = 0
    for i in range(0,len(SingVals)):
        cumulativeVar += SingVals[i]
        if(cumulativeVar > 0.999*totalVar):
            break
        else:
            k+=1
    
    optRank = k   
    print("SVD :: Optimal rank = ",optRank)    
    print("SVD :: Calculating U,S,Vh")  
    S = SingVals[0:optRank]
    V = EigVecs[:,0:optRank] 
    invS = 1./S
    U = np.dot(D,np.dot(V,np.diag(invS)))
    
    #DD =np.dot(U,np.dot(np.diag(S),V.T))
    #np.average(100*(np.abs(DD) - np.abs(D))/np.abs(D))
    
    return [U,S,V.T]
        

DATA_INPUT_METHOD = "readOpenFOAMRawFormatVector_ComponentZ"

SVD_EPS = 1e-4

def GramSchmidt(mat):
    for i in range(mat.shape[1]):
        for j in range(i):
            r = np.dot(mat[:, i], mat[:, j]);
            mat[:, i] -= r * mat[:, j];
        norm = np.linalg.norm(mat[:, i])
        if norm < SVD_EPS:
            for k in range(i, mat.shape[1]):
                mat[:, k] = 0.0
            return
        mat[:, i] *= 1.0 / norm

def redsvd(A):
    print("Calculating full spectrum of singular values ")    
    Sig = scipy.linalg.svd(A, full_matrices=True, compute_uv=False)
    ind = np.argsort(-1*Sig)
    Sig = Sig[ind]  
    print("Calculating optimal rank ")     
    totalVariance = np.sum(Sig)
    accumulatedVariance = 0
    optRank = 0 
    for i in range(Sig):
        accumulatedVariance+=Sig[i]
        optRank +=1
        if(accumulatedVariance > 0.99*totalVariance):
            break
    print("Optimal rank: ",optRank)
    k = optRank
    
    O = np.random.randn(A.shape[0]*k).reshape(A.shape[0], k)
    Y = A.T.dot(O)
    GramSchmidt(Y)
    B = A.dot(Y)

    P = np.random.randn(k*k).reshape(k, k)
    Z = np.dot(B, P)
    #print("Shape of the matrix Z: ",Z.shape)
    print("Performing Gram-Schmidt for the matrix shape: ",Z.shape)
    GramSchmidt(Z)
    C = np.dot(Z.T, B) # Z is actually Q in standard notation
    #print("Shape of the matrix C: ",C.shape)
    Uhat, S, Vhat = np.linalg.svd(C)
    #print("Shape of the matrix Uhat: ",Uhat.shape)
    U = np.dot(Z,Uhat)
    #print("Shape of the matrix U: ",U.shape)
    invSigma = np.diag(1./S)
    #print("Shape of the matrix invSigma: ",invSigma.shape)
    Vh = invSigma.dot(U.T.dot(A))
    #print("Shape of the matrix Vh: ",Vh.shape)
    return [U,S,Vh]


    


class DATA_INPUT_FUNCTIONS:
    # Most simple method, file contains only one column
    def readSingleColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        #print(path)
        return data
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatScalar(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-1]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentZ(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-1]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentY(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-2]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentX(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-3]
    # Usually openFoam raw output method
    def readFirstColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,0]
    # Usually openFoam raw output method
    def readSecondColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,1]
    # Usually openFoam raw output method
    def readThirdColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,2]
    # Usually openFoam raw output method
    def readAllThreeVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,-3:]
        #print(path)
        return data.flatten('F')
    # Usually openFoam raw output method
    def readXZVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,-3:]
        dataXZ = np.zeros((data.shape[0],2))
        dataXZ[:,0] = data[:,0]
        dataXZ[:,1] = data[:,2]

        #print(path)
        return dataXZ.flatten('F')

def dataInput(path):
    return getattr(DATA_INPUT_FUNCTIONS, DATA_INPUT_METHOD)(path)



def FFT(row):
    # Creating a Hanning window
    N = len(row)
    j = np.linspace(0,N-1,N)
    #w = 0.5 - 0.5*np.cos(2*np.pi*j/(N-1)) # Hamming window
    aw = 1.0 #- correction factor

    #yf = np.abs(fft(np.multiply(row,w)))
    yf = np.abs(fft(row))
    yf[1:N//2] *=2 # Scaling everythig between 0 and Nyquist

    return (aw/N) * yf[0:N//2]


def main():

    # Create input arguments:
    
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--sourceDirectory", required=True,help="String with a path to directory containing time directories")
    ap.add_argument("-f", "--sourceFileName", required=True,help="Exact name of the file that will be located in time directories")
    ap.add_argument("-i", "--inputMethod", required=True,help="Name of the method for data input")
    ap.add_argument("-r", "--resultsDirectory", required=True,help="Exact name of the results directory")

    ap.add_argument("-t0", "--timeStart", required=False,help="Time from which to start")
    ap.add_argument("-t1", "--timeFinish", required=False,help="Time with which to finish")
    ap.add_argument("-s", "--step", required=False,help="Sampling step")

    ap.add_argument("-nwc", "--noWriteCoord", required=False,help="Do not write coordinates")

    args = vars(ap.parse_args())

    #Parse input arguments
    

    directory        = str(args['sourceDirectory']) #r"D:\KarmanPimple\postProcessing\planeSample" #
    name             = str(args['sourceFileName']) #r"vorticity_plane1.raw" #
    resultsDirectory = str(args['resultsDirectory']) #r"C:\Users\gajev\Desktop\POD\Results"#

    if(os.path.exists(resultsDirectory)):
        try:
            os.rmdir(resultsDirectory)
            os.mkdir(resultsDirectory)
        except:
            print("Unable to remove {}, directory not empty".format(resultsDirectory))
            sys.exit()
    else:

        print("Creating directory: " + resultsDirectory)

        os.mkdir(resultsDirectory)


    global DATA_INPUT_METHOD
    DATA_INPUT_METHOD = str(args['inputMethod'])

    if(DATA_INPUT_METHOD not in dir(DATA_INPUT_FUNCTIONS)):
        print("ERROR: " + DATA_INPUT_METHOD + " is not among data input functions:")
        [print(d) for d in dir(DATA_INPUT_FUNCTIONS) if d.startswith('__') is False]
        print("Change name or modify DATA_INPUT_FUNCTIONS class at the start of the code")
        sys.exit()

    try:
        TSTART = float(args['timeStart'])
    except:
        TSTART = 0

    try:
        TEND = float(args['timeFinish'])
    except:
        TEND = 1e80

    try:
        STEP = int(args['step'])
    except:
        STEP = 1


    try:
        NWC = str(args['noWriteCoord'])
    except:
        NWC = 'y'

    #**********************************************************************************
    #**********************************************************************************
    #
    #    Read field
    #
    #**********************************************************************************
    #**********************************************************************************

    #timeFiles =  [float(t) for t in os.listdir(directory) if float(t) >= TSTART and float(t) <= TEND]
    timeFilesUnsorted =  set([t for t in os.listdir(directory) if float(t) >= TSTART and float(t) <= TEND])

    timeFilesStr = sorted(timeFilesUnsorted, key=lambda x: float(x))

    timeFilesStr = timeFilesStr[::STEP]

    timeFiles = [float(t) for t in timeFilesStr]

    if(TEND > timeFiles[-1]):
        TEND =  timeFiles[-1]

    N = len(timeFiles)

    TIME = timeFiles
    timePaths = [os.path.join(directory,str(t),name) for t in timeFilesStr]


    # At this point, prompt user
    dts = np.diff(TIME)
    dt = np.mean(np.diff(TIME))

    freq = fftfreq(N,dt)
    freq = freq[0:len(freq)//2]
    fs = 1.0/dt

    print("POD data:")
    print("------------------------------------------------------------------")

    print("   Start time                     = {} s".format(TIME[0]))
    print("   End time                       = {} s".format(TIME[-1]))
    print("   Number of samples              = {} ".format(N))
    print("   Min delta t                    = {} s".format(min(dts)))
    print("   Max delta t                    = {} s".format(max(dts)))
    print("   Avg delta t                    = {} s".format(dt))
    print("   Sampling frequency             = {} Hz".format(fs))
    print("   Frequency resolution           = {} Hz".format(fs/N))
    print("   Input method                   = {}   ".format(DATA_INPUT_METHOD))
    print("   Results directory              = {}   ".format(resultsDirectory))

    print("------------------------------------------------------------------")

    answer = input("If satisfied with frequency resolution, continue y/n?  ")

    if( answer not in ["y","Y","yes","Yes","z","Z"]):
        print("OK, exiting calculation")
        sys.exit()

    start = time.perf_counter()

    R = []

    print("Reading data files...")
    with Executor() as executor:
        R = list(tqdm(executor.map(dataInput, timePaths), total=len(timePaths)))


    finish = time.perf_counter()
    
    print("Finished in: " + str(finish - start) + "s" )
    
    global DATA_MATRIX
    DATA_MATRIX = np.vstack(R).T

    del R # Free memory

    DataMean = np.matrix(DATA_MATRIX.mean(axis=1)).T
        
    DATA_MATRIX -= DATA_MATRIX.mean(axis=1,keepdims = True) # np.subtract(DATA_MATRIX,np.matrix().T) # Mean padded
    
    
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Calculate SVD Modes
    #
    #**********************************************************************************
    #**********************************************************************************
    print("---------------------------------------------------")
    print("Performing SVD-decomposition of data matrix with shape:",DATA_MATRIX.shape)
    start = time.perf_counter()
    [U,Sig,Vh] = SVD(DATA_MATRIX)
    finish = time.perf_counter()
    print("Finished in: " + str(finish - start) + "s" )

    print("---------------------------------------------------")
    print("Saving results to ",resultsDirectory)

    np.savetxt(os.path.join(resultsDirectory,"SingularValues"),Sig)
    np.savetxt(os.path.join(resultsDirectory,"MeanField_SpatialDistribution"),DataMean)

    for i in range(0,20):
        print("		Saving Mode ",i+1)
        np.savetxt(os.path.join(resultsDirectory,"Mode_{}_SpatialDistribution".format(i+1)),U[:,i])
        np.savetxt(os.path.join(resultsDirectory,"Mode_{}_TimeDynamics".format(i+1)),Vh[:,i])

    print("Plotting data")

    plt.bar(np.linspace(1,len(Sig),len(Sig)),100*Sig/np.sum(Sig))
    plt.ylabel("Contained variance [%]")
    plt.xlabel("Mode No.")
    plt.xlim(0,20)
    plt.savefig(os.path.join(resultsDirectory,"ContainedVariance.png"))
 

    # Coordinates
    X = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readFirstColumn')(timePaths[0]))
    Y = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readSecondColumn')(timePaths[0]))
    Z = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readThirdColumn')(timePaths[0]))

    np.savetxt(os.path.join(resultsDirectory,"XYZ_Coordinates"), np.vstack((X,Y,Z)).T)
    print("		Saving XYZ coordinates")

    plt.show()
    print("All done!")

if __name__ == "__main__":
    main()
