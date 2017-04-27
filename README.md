"""  We use both SVM & Naive bayes algorithms to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
and find out which machine learning algorithm performs faster and accurate! 
"""
    
preprocess is a function written inside
dataset_preprocess.py

features_train and features_test are the features for the training
and testing data sets, respectively
labels_train and labels_test are item labels
 
 
### The dataset contains about 17000+ emails, so first take ###10% of the data to see which algorithm performs better.

####################################################

 
We measure time taken to fit

Confusion matrix for GaussianNB

To count the number of emails sent by Chris
 
Alternatively we can eliminate the steps that shrink the
original data set to
10% to see how many emails were
sent by Chris totally. 
 
###Output for the Code is:
 
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

Accuracy of Linear Kernel: ', 0.98407281001137659

Accuracy of rbf Kernel: ', 0.99089874857792948
 
Explanation:
 
Accuracy: From the accuracy scores achieved, we could see that SVM classifiers performed better than Naive Bayes (At least for this dataset!). 
 
Time taken to fit & predict: For this data set, it can be observed that GaussianNB's classifier and SVM's classifier have huge difference in the amount of time taken. GaussianNB classifier trained and predicted the data way faster than what both SVM classifiers took.
 
So, in real time for the on-demand application, it is very likely that Naive Bayes could be preferred over SVM.
 
Please leave a feedback if you find this article interesting or if you think something else can be improved in this code!
 
print("Accuracy of rbf Kernel: ", clf2.score(features_test,labels_test))
