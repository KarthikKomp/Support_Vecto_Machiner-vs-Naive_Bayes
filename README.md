# Naive_Bayes
Naive Bayes
March 14, 2017

 
""" 
    We use Naive bayes algorithms to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
and find out if the machine learning algorithm performs faster and accurate! 
"""
     
### preprocess is a function written inside
### dataset_preprocess.py

### features_train and features_test are the features for the training
### and testing data sets, respectively
### labels_train and labels_test are item labels
 
### features_train, features_test, labels_train, labels_test = preprocess()
 
### The dataset contains about 17000+ emails, so first take ###10% of the data to see which algorithm performs better.
features_train = features_train[:len(features_train)/10] 
labels_train = labels_train[:len(labels_train)/10] 
####################################################
 
Output for the Code is:
####################################################

no. of Chris training emails: 7936

no. of Sara training emails: 7884

training time for GaussianNB: 1.45 s


             precision    recall  f1-score   support
          0       1.00      0.95      0.97       893
          1       0.95      1.00      0.97       865
avg / total       0.97      0.97      0.97      1758

Confusion Matrix:

[[849  44]
 [  3 862]]

predicting time for GaussianNB: 0.259 s


Accuracy score for Naive Bayes alg : , 0.97326507394766781

training time for Linear: 165.879 s

training time for rbf:  113.941 s

predicting time for linear:  17.926 s

predicting time for rbf:  11.678 s

Number of emails sent by Chris: 877

Explanation:
############ 
 
So, in real time for the on-demand application, it is very likely that Naive Bayes could be preferred over many similar algorithms such as SVM & Decision Trees as NB takes least amount of time to learn & predict.
 
Please leave a feedback if you find this article interesting or if you think something else can be improved in this code!
 
