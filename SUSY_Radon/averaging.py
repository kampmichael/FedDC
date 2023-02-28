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
from utils import *
import time
import os

randomState=42

data = np.genfromtxt("data/SUSY.csv", delimiter=',')

X = data[:,1:]
y = data[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

rounds = 500
b = 50
bavg = 50
m = 441
n_local = 2

random.seed(randomState)
rng = np.random.RandomState(randomState)

name = "Averaging"
#set up a folder for logging
exp_path = name + "_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
os.mkdir(exp_path)

#log basic experiment properties
f = open(exp_path+"/setup.txt",'w')
out  = "aggregator = average \n"
out += "m = "+str(m)+"\n n_local = "+str(n_local)+"\n"
out += "d = "+str(b)+"\n b = "+str(bavg)+"\n"
out += "rounds = "+str(rounds)+"\n"
out += "model = SGDClassifier(alpha=0.0001, random_state = rng, learning_rate='adaptive', eta0=0.01, early_stopping=False)\n"
out += "randomState = "+str(randomState)+"\n"
f.write(out)
f.close()


local_Xtrains, local_ytrains = splitIntoLocalData(X_train, y_train, m, n_local, rng)
print("Starting...")
localModels = np.array([SGDClassifier(alpha=0.0001, random_state = rng, learning_rate='adaptive', eta0=0.01, early_stopping=False) for _ in range(m)])
m_rad, trainACCs_rad, testACCs_rad = runAveraging(local_Xtrains, local_ytrains, localModels, X_test, y_test, rounds, rng, b=bavg, exp_path = exp_path)
print("averaging done.")

pickle.dump(m_rad, open(os.path.join(exp_path, "finalModel.pck"),'wb'))
pickle.dump(trainACCs_rad, open(os.path.join(exp_path, "trainACC.pck"),'wb'))
pickle.dump(testACCs_rad, open(os.path.join(exp_path, "testACC.pck"),'wb'))

plotResults(trainACCs_rad, testACCs_rad, 'Averaging', exp_path)

