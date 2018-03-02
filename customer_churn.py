import pandas as pd
import numpy as np
import sys
from time import time
from sklearn import cross_validation
import matplotlib.pyplot as plt

data = pd.read_csv ("Customer Churn Data.csv")

data = data.drop (['Id', 'phone_number'], 1)

target = data['churn']
data = data.drop (['churn'], 1)

target = np.array (target)
data = np.array (data)

print "Numbers of tuples in the data : ", len (data)

def print_confusion_metrics (predict_labels, labels_test):
	print '-----------------------------------------------------------------------'
	print '\t\t\t\tActual True(+)\t| Actual False(-)|'
	print '-----------------------------------------------------------------------'
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in range (len (features_test)):
		if predict_labels[i] == 'True' and labels_test[i] == 'True':
			tp = tp + 1
		elif predict_labels[i] == 'False' and labels_test[i] == 'False':
			tn = tn + 1
		elif predict_labels[i] == 'True' and labels_test[i] == 'False':
			fp = fp + 1
		else:
			fn = fn + 1
		
	print 'Predicted True (+)\t|\t', tp, ' (TP)\t|', fp, ' (FP)\t|'
	print '-----------------------------------------------------------------------'
	print 'Predicted False (+)\t|\t', fn, ' (FN)\t|', tn, ' (TN)\t|'
	print '-----------------------------------------------------------------------'
	try:
		precision = float(tp) / float (tp + fp);
	except ZeroDivisionError:
		precision = 'NOT DEFINED'
	
	try:
		recall = float(tp) / float (tp + fn);
	except ZeroDivisionError:
		recall = 'NOT DEFINED'
	
	try:
		accuracy = float (tp + tn) / float (tp + fp + tn + fn)
	except ZeroDivisionError:
		accuracy = 'NOT DEFINED'
	
	print '\nPrecision = ', precision
	print 'Recall = ', recall
	print 'Accuracy = ', accuracy
	
	print labels_test[:10]
	print predict_labels[:10]
	
	# Plottinig the ROC curve for given classifier
	
	for i in range (len (labels_test)):
		labels_test[i] = int (labels_test[i] == 'True')
	
	for i in range (len (predict_labels)):
		predict_labels[i] = int (predict_labels[i] == 'True')
		
	predict_labels = predict_labels.astype (np.int)
		
	print labels_test[:10]
	print predict_labels[:10]
	
	from sklearn.metrics import roc_curve, auc
	
	fpr, tpr, threshold = roc_curve (labels_test, predict_labels)
	roc_auc = auc (fpr, tpr)
	
	plt.title ('Receiver Operating Characteristic')
	plt.plot (fpr, tpr, 'b', label = 'AUC = %0.2f' %roc_auc)
	plt.legend (loc = 'lower right')
	plt.plot ([0, 1], [0, 1], 'r--')
	plt.xlim ([0, 1])
	plt.ylim ([0, 1])
	plt.ylabel ('True Positive Rate')
	plt.xlabel ('False Positive Rate')
	plt.show ()
	
	return accuracy

### Preprocessing
maps = {}
v = 0
for i in range (len (data)):
	if data[i][0] in maps:
		data[i][0] = maps[data[i][0]]
	else:
		maps[data[i][0]] = v
		data[i][0] = v
		v = v + 1
		
### print 'maps = ', maps
		
maps = {}
v = 0
for i in range (len (data)):
	if data[i][3] in maps:
		data[i][3] = maps[data[i][3]]
	else:
		maps[data[i][3]] = v
		data[i][3] = v
		v = v + 1
		
	if data[i][4] in maps:
		data[i][4] = maps[data[i][4]]
	else:
		maps[data[i][4]] = v
		data[i][4] = v
		v = v + 1


for i in range (len (target)):
	if 'F' in target[i]:
		target[i] = 'False'
	else:
		target[i] = 'True'

		
### print 'maps = ', maps

### Train - Test split
print 'Train test splitting...'
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split (data, target, random_state = 42, test_size = 0.4)
print 'Length of train data : ', len (features_train)
print 'Length of test data : ', len (features_test)

accuracies = {}

# Applying naive-bayes classifier
print '+++++++++++++Applying naive-bayes classifier+++++++++'
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB ()

t0 = time()

clf.fit (features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time ()
predict_labels = clf.predict (features_test)
print "predicting time:", round(time () - t0, 3), "s"

accuracy = print_confusion_metrics (predict_labels, labels_test)
accuracies['naive-bayes'] = accuracy
print '-----------------------------------------------------'

# Applying Decision Tree classifier
print '+++++++++++++Applying decision tree classifier+++++++++'
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier (min_samples_split = 15)

t0 = time()

clf.fit (features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time ()
predict_labels = clf.predict (features_test)
print "predicting time:", round(time () - t0, 3), "s"

accuracy = print_confusion_metrics (predict_labels, labels_test)
accuracies['decision-tree'] = accuracy
print '-----------------------------------------------------'

# Applying Support Vector Machine classifier
print '+++++++++++++Applying svm classifier+++++++++'
from sklearn.svm import SVC
clf = SVC (kernel = 'rbf', C = 100000.0)

t0 = time()

clf.fit (features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time ()
predict_labels = clf.predict (features_test)
print "predicting time:", round(time () - t0, 3), "s"

accuracy = print_confusion_metrics (predict_labels, labels_test)
accuracies['svm'] = accuracy
print '-----------------------------------------------------'

# Conclusion
print ''
print 'Conclusion: '
maxAccuracy = 0.0
reqClassifier = "NONE"
print 'Accuracies of different classifiers is :'
for classifier in accuracies:
	print classifier, ' ==> ', accuracies[classifier]
	if accuracies[classifier] > maxAccuracy:
		maxAccuracy = accuracies[classifier]
		reqClassifier = classifier
		
print 'MAXIMUM ACCURACY ( ', maxAccuracy, ' ) is obtained by ', reqClassifier, ' classifier.'

