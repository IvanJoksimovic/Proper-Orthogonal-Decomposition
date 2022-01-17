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

DATA_INPUT_METHOD = "foo"


class DATA_INPUT_FUNCTIONS:
    # Most simple method, file contains only one column
    def readSingleColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data
    # Usually openFoam raw output method 
    def readOpenFOAMRawFormatVector_ComponentZ(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-1]
    # Usually openFoam raw output method 
    def readOpenFOAMRawFormatVector_ComponentY(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-2]
    # Usually openFoam raw output method 
    def readOpenFOAMRawFormatVector_ComponentX(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-3]
    # Usually openFoam raw output method 
    def readFirstColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,0]
    # Usually openFoam raw output method 
    def readSecondColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,1]
    # Usually openFoam raw output method 
    def readThirdColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,2]
    # Usually openFoam raw output method 
    def readAllThreeVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None)
        data = data[:,-3:]
        print(path)
        return data.flatten('F')
    
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
    
    args = vars(ap.parse_args())
    
    # Parse input arguments
    
    
    directory        = str(args['sourceDirectory'])
    name             = str(args['sourceFileName'])
    
    resultsDirectory = str(args['resultsDirectory'])
    
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
        N_BLOCKS = int(args['NBLOCKS'])
    except:
        N_BLOCKS = 1

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

    timeFiles = [float(t) for t in timeFilesStr]
  
    if(TEND > timeFiles[-1]):
        TEND =  timeFiles[-1]
    
    N = len(timeFiles)
    
    TIME = timeFiles
    timePaths = [os.path.join(directory,str(t),name) for t in timeFilesStr]
    

    # At this point, prompt user 
    dts = np.diff(TIME)
    dt = np.mean(np.diff(TIME))
        
    freq = fftfreq(N_FFT,dt)
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
    print("   Frequency resolution           = {} Hz".format(fs/N_FFT)) 
    print("   Input method                   = {}   ".format(DATA_INPUT_METHOD)) 
    print("   Results directory              = {}   ".format(resultsDirectory))
 
    print("------------------------------------------------------------------")
    
    answer = input("If satisfied with frequency resolution, continue y/n?  ")
    
    if( answer not in ["y","Y","yes","Yes","z","Z"]):
        print("OK, exiting calculation")
        sys.exit()
    
    start = time.perf_counter()
    
    R = []
    with Executor() as executor:
        for r in executor.map(dataInput,timePaths):
            R.append(r)
                
    finish = time.perf_counter()
    print("===========================================================")
    print("Finished in: " + str(finish - start) + "s" )
        
    DATA_MATRIX = np.vstack(R).T   

    del R # Free memory
	
    DataMean = DATA_MATRIX.mean(axis=1)
     
    DATA_MATRIX -= DATA_MATRIX.mean(axis=1,keepdims = True) # np.subtract(DATA_MATRIX,np.matrix().T) # Mean padded 
			       
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Calculate SVD Modes
    #
    #**********************************************************************************
    #**********************************************************************************
    print("---------------------------------------------------")
    print("Performing eigendecomposition of covatiance matrix")
    [U,Sigma,Vh] = np.linalg.svd(DATA_MATRIX, full_matrices=False)

    Lambda = Sig*Sig

    print("---------------------------------------------------")
    print("Sorting eigenvalues")

    ind = np.argsort(-1*Lambda)
    Lambda = Lambda[ind]
    U = U[:,ind]
    V = Vh[ind,:]
    V = V.T
    

    print("---------------------------------------------------")
    print("Saving results to ",resultsDir)
	
    np.savetxt(os.path.join(resultsDirectory,"EigenValues"),Lambda) 
    np.savetxt(os.path.join(resultsDirectory,"MeanField_SpatialDistribution",DataMean)
	
    for i in range(0,20):
	print("		Saving Mode ",i+1)
	np.savetxt(os.path.join(resultsDirectory,"Mode_{}_SpatialDistribution".format(i+1),U[:,i])
 	np.savetxt(os.path.join(resultsDirectory,"Mode_{}_TimeDynamics".format(i+1),V[:,i])

    print("Plotting data")

    plt.plot(Lambda)
    plt.xlim(0,20)
    plt.savefig(os.path.join(resultsDirectory,"ContainedVariance.png"))
    answer = input("All done, write coordinates as well y/n?  ")
    
    if( answer not in ["y","Y","yes","Yes","z","Z"]):
        print("OK, All done!")
        sys.exit()    

    # Coordinates 
    X = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readFirstColumn')(timePaths[0]))
    Y = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readSecondColumn')(timePaths[0]))
    Z = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readThirdColumn')(timePaths[0]))

    np.savetxt(os.path.join(resultsDirectory,"XYZ_Coordinates"), np.vstack((X,Y,Z)).T)
    print("		Saving XYZ coordinates".format(iii))

    plt.show()





    
if __name__ == "__main__":
    main()    

       
