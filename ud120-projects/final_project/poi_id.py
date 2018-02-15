#!/usr/bin/python

import sys
import os
import pickle
from pprint import pprint
import numpy as np

os.chdir('/Users/brentan/Documents/DAND/Projects/ML/ud120-projects/final_project')

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
# show list of possible features
feature_names = list(list(data_dict.values())[0].keys())
pprint(feature_names)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',   #kept all features but email_address (text)
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 #'from_poi_to_this_person',
                 'exercised_stock_options',
                 'from_messages',
                 'other',
                 #'from_this_person_to_poi',
                 'long_term_incentive',
                 #'shared_receipt_with_poi',
                 'restricted_stock',
                 'director_fees']

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# split data into train / test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# check balance of POIs across data
print("Percent POI's in overall dataset: {}%".format(np.mean(labels)*100))
print("Percent POI's in training set: {}%".format(np.mean(y_train)*100))
print("Percent POI's in test set: {}%".format(round(np.mean(y_test)*100, 2)))

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:3
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # ML models
from sklearn.svm import SVC # model
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import GridSearchCV # for selecting the best model params
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix # for evaluating model performance
from sklearn.model_selection import train_test_split # for creating the test / train datasets

# make wee scoring function
def ScoreMe(model, y_test, preds):
    print("{}: accuracy report".format(model))
    print("The accuracy is {}".format(accuracy_score(y_test,preds)))
    print("The recall is {}".format(recall_score(y_test, preds)))
    print("The precision is {}".format(precision_score(y_test, preds)))
    print(confusion_matrix(y_test, preds))
    return


### models, raw, no grid search / tuning, or feature estimation
    
## logistic regression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
lr_preds = clf_lr.predict(X_test)
ScoreMe("logistic regression", y_test, lr_preds)

## support vectors
clf_svc = SVC()
clf_svc.fit(X_train, y_train)
svc_preds = clf_svc.predict(X_test)
ScoreMe("SVC", y_test, svc_preds)

## random forest
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
rf_preds = clf_rf.predict(X_test)
ScoreMe("RF", y_test, rf_preds)

## gradient boosting
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X_train, y_train)
gb_preds = clf_gb.predict(X_test)
ScoreMe("gradient boost",y_test, gb_preds)

## gaussian naive bayes
clf_gNB = GaussianNB()
clf_gNB.fit(X_train, y_train)
gNB_preds = clf_gNB.predict(X_test)
ScoreMe("GaussianNB", y_test, gNB_preds)

## K Neighbors Classifier
clf_knc = KNeighborsClassifier(n_neighbors = 2)
clf_knc.fit(X_train, y_train)
knc_preds = clf_knc.predict(X_test)
ScoreMe("k neighbors", y_test, knc_preds)

"""
I picked six supervised classification algorithms to test. My idea is to test 
the algorithms raw, without any dimensionality reduction, feature transformation 
or algorithm tuning. This will help me get a baseline understanding of how the
 models perform. After that, I will set up a grid of parameters for each 
 algorithm and use grid search to find each one's best estimator parameters.  

Based on that, I might try some dimensionality reduction, feature scaling, and
feature selection. 



"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

grid_params = {'learning_rate': [.01, .05, .1],
              'max_depth': [1,3,5,7],
              'max_features': [1,5,10]}

gb = GradientBoostingClassifier()

gb_GS = GridSearchCV(gb, grid_params)

gb_GS.fit(X_train, y_train)

gb_GS_preds = gb_GS.best_estimator_.predict(X_test)
print("score:", gb_GS.score(X_test, y_test))
confusion_matrix(y_test, gb_GS_preds)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)