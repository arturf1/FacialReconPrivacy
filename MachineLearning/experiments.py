import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random
from random import sample
import numpy as np
import time
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation
from sklearn import grid_search
import pywt

########################################################################
#                       DATA LOADING FUNCTIONS                       
########################################################################
def loadYaleData():
    X = np.matrix(scipy.io.loadmat('yalefacesFFT.mat')['DATA'])
    individuals = 15; 
    picsPerInd = 11;
    Y = np.zeros((individuals*picsPerInd))
    for i in range(0,individuals):
        Y[(i)*picsPerInd:(i+1)*picsPerInd] = i;
    return (X,Y)

def loadOlivettiData():
    X = np.matrix(scipy.io.loadmat('olivettifacesFFT.mat')['DATA'])
    individuals = 40; 
    picsPerInd = 10;
    Y = np.zeros((individuals*picsPerInd))
    for i in range(0,individuals):
        Y[(i)*picsPerInd:(i+1)*picsPerInd] = i;
    return (X,Y)

########################################################################
#                      TRANFORMATION FUNCTIONS                       
########################################################################

def removePhase(V):
    return V[0:V.shape[0]/2].reshape(V.shape[0]/2);

def recFilter(w,h,fw,fh):
    h = h + 1
    F = np.zeros((h,w));
    for i in range((w-1)/2+1 - fw/2, (w-1)/2+1 + fw/2 - 1):
        for j in range(h - fh - 1, h):
            F[j,i] = 1
    return np.reshape(F,(w*h),order='F')

def triFilter(w,h,fw,fh):
    h = h + 1
    F = np.zeros((h,w));
    for j in range(h - fh -1, h):
        span = (j - (h - fh)) * (fw/2)/fh;
        for i in range((w-1)/2+1 - span, (w-1)/2+1 + span - 1):
            F[j,i] = 1
    return np.reshape(F,(w*h),order='F')

def varFilter(train_X, numFeatures):
    F = np.zeros(train_X.shape[0])
    
    var = np.var(train_X, axis=1)
    varSorted = np.argsort(var)[::-1]

    F[varSorted[0:numFeatures]] = 1
    
    return F

def snrFilter(train_X, train_Y, numFeatures):
    F = np.zeros(train_X.shape[0])
    SNRsum = np.zeros(train_X.shape[0])
    numClasses = np.unique(train_Y).shape[0]
       
    for j in range(numClasses):
        pos = train_X[:,train_Y == j]
        neg = train_X[:,train_Y != j]
        mu_pos = np.mean(pos, axis = 1)
        mu_neg = np.mean(neg, axis = 1)
        sd_pos = np.std(pos, axis = 1)
        sd_neg = np.std(neg, axis = 1)
        SNRsum = SNRsum + np.abs((mu_pos - mu_neg)/(sd_pos + sd_neg))
        
    
    SNRavg = SNRsum/numClasses
    SNRavgSorted = np.argsort(SNRavg)[::-1]
    
    F[SNRavgSorted[0:numFeatures]] = 1
        
    return F

def fdrFilter(train_X, train_Y, numFeatures):
    F = np.zeros(train_X.shape[0])
    FDRsum = np.zeros(train_X.shape[0])
    numClasses = np.unique(train_Y).shape[0]
    
    for j in range(numClasses):
        pos = train_X[:,train_Y == j]
        neg = train_X[:,train_Y != j]
        mu_pos = np.mean(pos, axis = 1)
        mu_neg = np.mean(neg, axis = 1)
        sd_pos = np.std(pos, axis = 1)
        sd_neg = np.std(neg, axis = 1)
        FDRsum = FDRsum + np.square(mu_pos - mu_neg)/(np.square(sd_pos) + np.square(sd_neg))
        
    FDRavg = FDRsum/numClasses
    FDRavgSorted = np.argsort(FDRavg)[::-1]
    
    F[FDRavgSorted[0:numFeatures]] = 1
    
    return F

def sdFilter(train_X, train_Y, numFeatures):
    F = np.zeros(train_X.shape[0])
    SDsum = np.zeros(train_X.shape[0])
    numClasses = np.unique(train_Y).shape[0]
    
    for j in range(numClasses):
        pos = train_X[:,train_Y == j]
        neg = train_X[:,train_Y != j]
        mu_pos = np.mean(pos, axis = 1)
        mu_neg = np.mean(neg, axis = 1)
        var_pos = np.var(pos, axis = 1)
        var_neg = np.var(neg, axis = 1)
        SDsum = SDsum + 0.5 * (var_pos/var_neg + var_neg/var_pos) + 0.5 * (np.square(mu_pos - mu_neg)/(var_pos + var_neg)) - 1
        
    SDavg = SDsum/numClasses
    SDavgSorted = np.argsort(SDavg)[::-1]
    
    F[SDavgSorted[0:numFeatures]] = 1
   
    return F

def tFilter(train_X, train_Y, numFeatures):
    F = np.zeros(train_X.shape[0])
    Tsum = np.zeros(train_X.shape[0])
    numClasses = np.unique(train_Y).shape[0]
    
    for j in range(numClasses):
        pos = train_X[:,train_Y == j]
        neg = train_X[:,train_Y != j]
        N_pos = pos.shape[0]
        N_neg = neg.shape[0]
        mu_pos = np.mean(pos, axis = 1)
        mu_neg = np.mean(neg, axis = 1)
        var_pos = np.var(pos, axis = 1)
        var_neg = np.var(neg, axis = 1)
        Tsum = Tsum + np.abs(mu_pos - mu_neg)/np.sqrt(var_pos/N_pos + var_neg/N_neg)
        
    Tavg = Tsum/numClasses
    TavgSorted = np.argsort(Tavg)[::-1]
    
    F[TavgSorted[0:numFeatures]] = 1
    
    return F

########################################################################
#                      TRAINING FUNCTIONS                       
########################################################################

def trainTestSVM(train_X, train_y, test_X, test_y):

    clf = gridSearchSVM(train_X,train_y)

    prediction = clf.predict(test_X)
    print clf.best_params_
    # record performance
    accuracy = performance(prediction, test_y)
    print "Accuracy: " , accuracy
    
    return accuracy

def gridSearchSVM(X,y):
    #parameters = {'kernel':('linear','rbf'), 'C':[1, 2, 3, 5, 10, 13, 15,20]}
    parameters = {'kernel':('linear','rbf'), 'C':[1, 2]}
    svr  = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters, cv = 3, n_jobs = 4, verbose=1)
    clf.fit(X,y)
    return clf

def performance(prediction, target):
    acc = metrics.accuracy_score(target, prediction, normalize=True)
    return acc


########################################################################
#                      Run Classification Testing                       
########################################################################

numExperiments = 72

accuracy_phase = np.zeros(numExperiments)
accuracy_phase_2mom = np.zeros(numExperiments)
dim_phase = np.zeros(numExperiments)
accuracy_nophase = np.zeros(numExperiments)
accuracy_nophase_2mom = np.zeros(numExperiments)
dim_nophase = np.zeros(numExperiments) 

accuracy = np.array([])
numTests = 100

height = 63
width = 63

dims = [99, 208, 399, 598, 899, 1056, 32, 72, 162, 242, 392, 512]
 
# load the labels 
(X_o,Y_o) = loadOlivettiData()

for t in range(0, numTests):

    # leave 10% out for testing
    skf = cross_validation.StratifiedKFold(Y_o,n_folds=10,shuffle=True) 

    for cv_i,test_i in skf:
        exp = 0

        for w in range(10, 40, 5):
            print "\n"
            print "Yale Rect Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = recFilter(width, (height-1)/2, w, w)
            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1
        
        for w in range(10, 40, 5):

            print "\n"
            print "Yale Tri Experiment " + str(exp) + " trial " + str(t)
            
            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = triFilter(width, (height-1)/2, w, w)
            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1

        for d in dims:
            print "\n"
            print "Yale Var Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = varFilter(X_nophase[:,cv_i], d)

            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1

        for d in dims:
            print "\n"
            print "Yale SNR Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = snrFilter(X_nophase[:,cv_i], Y_o[cv_i], d)

            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1

        for d in dims:
            print "\n"
            print "Yale FDR Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = fdrFilter(X_nophase[:,cv_i], Y_o[cv_i], d)

            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1

        for d in dims:
            print "\n"
            print "Yale SD Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = sdFilter(X_nophase[:,cv_i], Y_o[cv_i], d)

            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1
             
        for d in dims:
            print "\n"
            print "Yale T Experiment " + str(exp) + " trial " + str(t)

            X = X_o

            X_nophase = np.zeros([X.shape[0]/2, X.shape[1]])
            for i in range(0,X.shape[1]):
                X_nophase[:,i] = removePhase(X[:,i])[:]
            
            # feature engineering and selection 
            F = tFilter(X_nophase[:,cv_i], Y_o[cv_i], d)

            X_nophase = X_nophase[F == 1]
            X = X[np.append(F, np.ones(X.shape[0]/2), axis=0) == 1]
            
            test_y = Y_o[test_i]
            train_y = Y_o[cv_i]

            # full dim test 
            train_X = np.transpose(X)[cv_i]           
            test_X = np.transpose(X)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_phase[exp] = accuracy_phase[exp] + accuracy
            accuracy_phase_2mom[exp] = accuracy_phase_2mom[exp] + accuracy*accuracy
            dim_phase[exp] = X.shape[0]
            
            # no phase test 
            train_X = np.transpose(X_nophase)[cv_i]  
            test_X = np.transpose(X_nophase)[test_i]
            accuracy = trainTestSVM(train_X, train_y, test_X, test_y)
            accuracy_nophase[exp] = accuracy_nophase[exp] + accuracy
            accuracy_nophase_2mom[exp] = accuracy_nophase_2mom[exp] + accuracy*accuracy
            dim_nophase[exp] = X_nophase.shape[0]
            
            exp = exp + 1

        break
    
    print "*******************************  Iterations " + str(t)

print accuracy_phase/numTests
print accuracy_nophase/numTests
print dim_phase
print dim_nophase
print accuracy_phase_2mom/numTests
print accuracy_nophase_2mom/numTests

np.savetxt("acc_phase", accuracy_phase/numTests)
np.savetxt("acc_2mom_phase", accuracy_phase_2mom/numTests)
np.savetxt("dim_phase", dim_phase)
np.savetxt("acc_nophase", accuracy_nophase/numTests)
np.savetxt("acc_2mom_nophase", accuracy_nophase_2mom/numTests)
np.savetxt("dim_nophase", dim_nophase)