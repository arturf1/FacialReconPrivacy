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

def normalize(train_X):
    mu = np.mean(train_X)
    sd = np.std(train_X)
        
    train_X = (train_X - mu)/sd
    
    return train_X, mu, sd

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
    clf = grid_search.GridSearchCV(svr, parameters, cv = 3, n_jobs = 12, verbose=1)
    clf.fit(X,y)
    return clf

def performance(prediction, target):
    acc = metrics.accuracy_score(target, prediction, normalize=True)
    return acc


########################################################################
#                      Run Classification Testing                       
########################################################################

accuracy = 0
numTests = 100

height = 63
width = 63

# load the labels 
(X_o,Y_o) = loadYaleData()

for t in range(0, numTests):

    # leave 10% out for testing
    skf = cross_validation.StratifiedKFold(Y_o,n_folds=10,shuffle=True) 

    for cv_i,test_i in skf:       

        X = X_o

        # normalize X
        X, mu, sd = normalize(X);
        
        test_y = Y_o[test_i]
        train_y = Y_o[cv_i]

        # test 
        train_X = np.transpose(X)[cv_i]           
        test_X = np.transpose(X)[test_i]
        accuracy = accuracy + trainTestSVM(train_X, train_y, test_X, test_y)

        break

    print "*******************************  Iterations " + str(t)

print accuracy/numTests

