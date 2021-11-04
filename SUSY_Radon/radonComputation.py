'''
Created on 11.07.2014

@author: mkamp
'''

import gc

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

RADON_CALC_METHOD = 1 #2 #method 1 is to add a line 1,0,...,0 to S and set b = 0,...,0,1 in order to force a non-trivial solution. method 2 is to set S=[:-1] and b = S[-1], i.e., forcing the last column of S to have weight one and the other columns to add up to that last one, canceling each other out to zero.
EPS = 0.000001
MAX_REL_EPS = 0.0001

def getRadonPoint(S):
    alpha = []
    if RADON_CALC_METHOD == 1:
        A = np.vstack((np.transpose(S),np.ones(S.shape[0])))
        z = np.zeros(S.shape[0])
        z[0] = 1.0
        A = np.vstack((A,z))
        b = np.zeros(S.shape[0])
        b[-1] = 1.0
        alpha = np.linalg.lstsq(A, b)[0]
    else:        
        print( S)
        A = S[:-1]
        print( A)
        A = np.vstack((np.transpose(A),np.ones(A.shape[0])))
        print( A) 
        b = np.hstack((S[-1], np.ones(1)))
        print( b)
        alpha = np.linalg.solve(A, b)
    alpha_plus  = np.zeros(len(alpha))
    alpha_minus = np.zeros(len(alpha))
    for i in range(len(alpha)):
        if alpha[i] > 0:
            alpha_plus[i]  = alpha[i]
        if alpha[i] < 0:
            alpha_minus[i] = alpha[i]
    sumAlpha_plus  = 1.*np.sum(alpha_plus)
    sumAlpha_minus = -1.*np.sum(alpha_minus)
    if not floatApproxEqual(sumAlpha_plus, sumAlpha_minus):
        print( "Error: sum(a+) != sum(a-): " + str(sumAlpha_plus) + " != " + str(sumAlpha_minus) + " for |S| = " + str(S.shape) + " and R = " + str(getRadonNumber(S)))
    alpha /= sumAlpha_plus
    r = np.zeros(S.shape[1])
    r_minus = np.zeros(S.shape[1])
    for i in range(len(alpha)):
        if alpha[i] > 0:
            r += alpha[i] * S[i]
        if alpha[i] < 0:
            r_minus += alpha[i] * S[i]
    rtest_plus = r * 1./np.linalg.norm(r) #normiert
    rtest_minus = r_minus * 1./np.linalg.norm(r_minus) #normiert
    if np.linalg.norm(rtest_plus+rtest_minus) > EPS:
        print( "Something went wrong!!! r+ = " + str(r)+" but r- = "+str(-1*r_minus)+". They should be the same!")
    return r

def getRadonPointHierarchical(S,h):
    instCount = S.shape[0]
    if instCount == 1:
        return S[0]
    R = getRadonNumber(S)
    oldh = h
    #print("Radon number = ",R)
    if R**h != instCount:
        print( "Unexpected number of points (",instCount,") received for height "+str(h)+".")   
    while R**h > instCount:
        h -= 1
    if oldh !=h:
        print( "Had to adapt height to "+str(h)+".")
    S = S[:(instCount//R)*R] #ensures that |S| mod sampleSize == 0    
    if (instCount/R)*R == 0:
        print( R)
        print( h)
        print( instCount)
    #return getRadonPointRec(S,R)
    return getRadonPointIter(S,R)
    
def getRadonPointRec(S,R):
    S_new = []
    if S.shape[0] == 1:
        return S[0]
    if S.shape[0] < R:
        print( "Error: too few instances in S for radon point calculation! |S| = " + str(S.shape[0]) + " for radon number R = " + str(R) + " .")
        return S[0] 
    executor = ThreadPoolExecutor(max_workers = R)
    futures = []   
    for i in range(S.shape[0]/R):             
        futures.append(executor.submit(getRadonPoint(S[i*R:(i+1)*R])))            
    for f in as_completed(futures):
        S_new.append(f.result)
    gc.collect()
    return getRadonPointRec(np.array(S_new), R)

def getRadonPointIter(S,R):    
    while S.shape[0] >= R:
        S_new = []
        for i in range(S.shape[0]//R):
            S_new.append(getRadonPoint(S[i*R:(i+1)*R])) 
        S = np.array(S_new)
    if S.shape[0] > 1:
        print( "Error: too few instances in S for radon point calculation! |S| = " + str(S.shape[0]) + " for radon number R = " + str(R) + " .")
    r = S[0]
    return r
    
#def getRadonPointIter(S,R):    
#   while S.shape[0] >= R:
#        S_new = []
#        with ThreadPoolExecutor(max_workers = R) as executor:
#            futures = []
#            for i in range(S.shape[0]//R):
#                futures.append(executor.submit(getRadonPoint, S[i*R:(i+1)*R]))     
#            for f in as_completed(futures):
#                S_new.append(f.result()) 
#        S = np.array(S_new)        
#        gc.collect()
#    if S.shape[0] > 1:
#        print( "Error: too few instances in S for radon point calculation! |S| = " + str(S.shape[0]) + " for radon number R = " + str(R) + " .")
#    return S[0]

def getRadonNumber(S):
    return S.shape[1] + 2 #for Euclidean space R^d the radon number is R = d + 2

def floatApproxEqual(x, y):
    if x == y:
        return True
    relError = 0.0
    if (abs(y) > abs(x)):
        relError = abs((x - y) / y);
    else:
        relError = abs((x - y) / x);
    if relError <= MAX_REL_EPS:
        return True
    if abs(x - y) <= EPS:
        return True
    return False

if __name__=="__main__":
    #S = np.random.random_sample((4,2))
    S = np.array([[2.,2.],[3.,1.],[3.,3.],[4.,4.],[2.,1.],[3.,1.],[3.,1.],[4.,4.],[2.,8.],[5.,3.],[3.,3.],[4.,4.],[1.,5.],[3.,7.],[3.,3.],[2.,4.]])
    r = getRadonPointHierarchical(S,2)
    print( S)
    print( r)