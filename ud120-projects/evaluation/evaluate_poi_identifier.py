#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, 
                     sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)


### your code goes here 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
        train_test_split(features,labels,test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


print("score: {}".format(clf.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix, precision_score, recall_score

dt_preds = clf.predict(X_test)
cMatrix = confusion_matrix(y_test, dt_preds)
precision = precision_score(y_test, dt_preds)
recall = recall_score(y_test, dt_preds)






