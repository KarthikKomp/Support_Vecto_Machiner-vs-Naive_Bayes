#!/usr/bin/python

""" 
    Mini - Project
    SVM Vs Naive Bayes

    Use a SVM & NB to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1

    Within SVM implementation, both Linear & rbf kernels are observed in this algorithm.
    The code outputs, the accuracy scores of each algorithm & time taken for each of them
    to train & predict respectively. 
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import matplotlib.pyplot as plt


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

###features_train = features_train[:len(features_train)/10] 
###labels_train = labels_train[:len(labels_train)/10] 

#########################################################

import numpy as np

from sklearn import svm, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
###from sklearn.model_selection import train_test_split
###from sklearn.metrics import confusion_matrix

clfNB = GaussianNB()

t = time()
clfNB.fit(features_train,labels_train)
print "training time for GaussianNB:", round(time()-t, 3), "s"

tNB = time()
predNB = clfNB.predict(features_test)
### Confusion matrix for GaussianNB
print(metrics.classification_report(labels_test, predNB))
print(metrics.confusion_matrix(labels_test, predNB))

print "predicting time for GaussianNB:", round(time()-tNB, 3), "s"
print ("Accuracy score for Naive Bayes alg : ", clfNB.score(features_test,labels_test))

clf = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='rbf', C=10000) ##gamma = 1.51 for better accuracy

t0 = time()

clf.fit(features_train,labels_train)

print "training time for Linear:", round(time()-t0, 3), "s"

t1 = time()
clf2.fit(features_train,labels_train)
print "training time for rbf: ", round(time()-t1,3), "s"

t2 = time()
pred = clf.predict(features_test)
print "predicting time for linear: ", round(time()-t2,3), "s"

t3 = time()
pred2 = clf2.predict(features_test)
print "predicting time for rbf: ", round(time()-t3,3), "s"

print "For element 10: ", pred2[10], "For element 26:", pred2[26], "For element 50", pred2[50]

chris_count = 0

for i in range(len(pred2)):
    if pred2[i] == 1:
        chris_count += 1
print chris_count

print("Accuracy of Linear Kernel: ", clf.score(features_test,labels_test))

print("Accuracy of rbf Kernel: ", clf2.score(features_test,labels_test))

