import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import random
import colorsys
from radonComputation import *
import math
import pickle
import os

def getACC(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plotACCs(ax, accs, label, color=None, alpha=0.5):
    X = range(len(accs))
    m = len(accs[0])
    amount = 0.8/m 
    mean = [np.mean(accs[i]) for i in X]
    p = ax.plot(X,mean,label=label, zorder=m+1, color=color, linewidth=2)
    baseColor = p[0].get_color()
    for i in range(m):
        acc = [a[i] for a in accs]
        adjColor = lighten_color(baseColor, i*amount + 0.1)
        ax.plot(X,acc, c=adjColor, alpha=alpha)

def plotResults(trainACCs, testACCs, title, exp_path, bSave = True):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plotACCs(ax, trainACCs, 'train')
    plotACCs(ax, testACCs, 'test')
    plt.legend()
    plt.title(title)
    if bSave:
        plt.savefig(os.path.join(exp_path, "results.png"), dpi=300)
    else:
        plt.show()

def plotComparison(trainACCs1, testACCs1, trainACCs2, testACCs2, method1, method2, colors=None, alphas=[0.5,0.5], filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13.0,4.8))
    fig.suptitle('Comparison '+method1+' vs '+method2)
    rounds = len(trainACCs1)
    plotACCs(ax1, trainACCs1, method1+'_train', color = None if colors == None else colors[0], alpha=alphas[0])
    plotACCs(ax1, testACCs1,  method1+'_test',  color = None if colors == None else colors[1], alpha=alphas[0])
    plotACCs(ax2, trainACCs2, method2+'_train', color = None if colors == None else colors[2], alpha=alphas[1])
    plotACCs(ax2, testACCs2,  method2+'_test',  color = None if colors == None else colors[3], alpha=alphas[1])
    ax1.legend()
    ax2.legend()
    ax1.set_title(method1)
    ax2.set_title(method2)
    ax1.set_xlabel("rounds")
    ax2.set_xlabel("rounds")
    ax1.set_ylabel("accuracy")
    plt.subplots_adjust(wspace=0.1, hspace=0)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    
def splitIntoLocalData(X_train, y_train, m, n_local, rng):
    n = y_train.shape[0]
    if m*n_local > n:
        print("Error: not enough data (n=",n,") for",m,"sets of size ",n_local,". Reducing local size to",n//m,".")
        n_local = n // m
        
    idxs = np.arange(n)
    rng.shuffle(idxs)
    bucket_size = n_local
    i = 0
    Xs, Ys = [],[]
    while (i+1)*bucket_size <= n and i < m:
        idx = idxs[i*bucket_size:(i+1)*bucket_size]
        Xs.append(X_train[idx,:])
        Ys.append(y_train[idx])
        i += 1
    return Xs, Ys   

def averageSVCs(models):
    m = len(models)
    avgCoef = np.zeros(models[0].coef_.shape)
    avgIntercept = np.zeros(models[0].intercept_.shape)
    for model in models:
        avgCoef += model.coef_.copy()
        avgIntercept += model.intercept_.copy()
    avgCoef /= float(m)
    avgIntercept /= float(m)
    for model in models:
        model.coef_ = avgCoef.copy()
        model.intercept_ = avgIntercept.copy()
    return models
        
def simpleDaisyChain(models, perm):
    models = models[perm]
    return models

def runAveraging(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, classes = None):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if (r+1) % b == 0: #we don't want to average in the first round...
            localModels = averageSVCs(localModels)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
    #final model
    avgModel = averageSVCs(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs

def runSimpleDaisyChaining(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, classes = None, fix_permutation=False):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    perm = np.arange(m)
    rng.shuffle(perm)
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if r % b == 0:
            if not fix_permutation:
                rng.shuffle(perm)
            localModels = simpleDaisyChain(localModels, perm)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
    #final model
    avgModel = averageSVCs(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs

def runAverageAndDaisyChaining(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, bavg = 2, classes = None, fix_permutation=False):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    perm = np.arange(m)
    rng.shuffle(perm)
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if r % b == 0:
            if not fix_permutation:
                rng.shuffle(perm)
            localModels = simpleDaisyChain(localModels, perm)
        if (r+1) % bavg == 0:
            localModels = averageSVCs(localModels)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
    #final model
    avgModel = averageSVCs(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs
    
def radonPointLinearModelsMaxH(models):
    m = len(models)
    S = []
    coefDim = len(models[0].coef_[0].tolist())
    interceptDim = 1
    radonNumber = coefDim + interceptDim + 2
    maxHeight = math.floor(math.log(m)/math.log(radonNumber))
    for model in models:
        s = model.coef_[0].tolist()
        s.append(model.intercept_[0])
        S.append(s)
    S = np.array(S)
    r = getRadonPointHierarchical(S,maxHeight)
    for model in models:
        model.coef_ = np.array(r[:coefDim]).reshape(1,coefDim)
        model.intercept_ = np.array(r[coefDim:])
    return models
        
def simpleDaisyChain(models, perm):
    models = models[perm]
    return models

def runRadonPoint(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, classes = None, exp_path = ""):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if (r+1) % b == 0: #we don't want to average in the first round...
            localModels = radonPointLinearModelsMaxH(localModels)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
        if r % 10 == 0:
            print("round ",r)
            pickle.dump(trainACCs, open(os.path.join(exp_path, "trainACCs_tmp.pck"),'wb'))
            pickle.dump(testACCs, open(os.path.join(exp_path, "testACCs_tmp.pck"),'wb'))

    #final model
    avgModel = radonPointLinearModelsMaxH(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs

def runSimpleDaisyChaining(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, classes = None, fix_permutation=False):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    perm = np.arange(m)
    rng.shuffle(perm)
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if r % b == 0:
            if not fix_permutation:
                rng.shuffle(perm)
            localModels = simpleDaisyChain(localModels, perm)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
    #final model
    avgModel = averageSVCs(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs

def runRadonPointAndDaisyChaining(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=1, bavg = 2, classes = None, fix_permutation=False, exp_path = ""):
    trainACCs, testACCs = [], []
    m = len(localModels)
    if classes is None:
        classes = np.unique(y_test)
    trainACCs, testACCs = [],[]
    perm = np.arange(m)
    rng.shuffle(perm)
    for r in range(rounds):
        for i in range(m):
            localModels[i].partial_fit(local_Xtrains[i], local_ytrains[i], classes = classes)
        if r % b == 0:
            if not fix_permutation:
                rng.shuffle(perm)
            localModels = simpleDaisyChain(localModels, perm)
        if (r+1) % bavg == 0:
            localModels = radonPointLinearModelsMaxH(localModels)
        trainACCs.append([])
        testACCs.append([])
        for i in range(m):
            trainACCs[-1].append(getACC(localModels[i], local_Xtrains[i], local_ytrains[i]))
            testACCs[-1].append(getACC(localModels[i], X_test, y_test))
        if r % 10 == 0:
            print("round ",r)
            pickle.dump(trainACCs, open(os.path.join(exp_path, "trainACCs_tmp.pck"),'wb'))
            pickle.dump(testACCs, open(os.path.join(exp_path, "testACCs_tmp.pck"),'wb'))
    #final model
    avgModel = radonPointLinearModelsMaxH(localModels)[0]
    for i in range(m):
        trainACCs[-1].append(getACC(avgModel, local_Xtrains[i], local_ytrains[i]))
        testACCs[-1].append(getACC(avgModel, X_test, y_test))
    return avgModel, trainACCs, testACCs