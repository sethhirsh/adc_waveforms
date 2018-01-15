import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# Load data and split into training and test sets

file = np.load('adc.npz')
adc = file['adc']
pid = file['pid']
pmom = file['pmom']

adc_filt=adc[((pid==2212)|(pid==11))&(pmom > 53)]
pmom_filt = pmom[((pid==2212)|(pid==11))&(pmom > 53)]
pid_filt =pid[((pid==2212)|(pid==11))&(pmom > 53)]

pid_bin = (pid_filt != 2212)*1.
#0 is protons, 1 is electrons

from sklearn.preprocessing import StandardScaler
tf = StandardScaler(with_mean=False, with_std=False)
adc_filt = tf.fit_transform(adc_filt)

np.random.seed = 10001
indices = np.random.permutation(len(adc_filt))

train_cut = int(len(adc_filt)*0.8)

X_train = adc_filt[indices[:train_cut]]
y_train = pid_bin[indices[:train_cut]]

X_test = adc_filt[indices[train_cut:]]
y_test = pid_bin[indices[train_cut:]]

# Perform cross validation on single decision tree
clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X_train,y_train)
p = clf.predict_proba(X_test)

scores = []
confidence = []
for max_depth in range(1,15):
    print(max_depth)
    clf = DecisionTreeClassifier(max_depth=max_depth)
    
    cvs = cross_val_score(clf, X_train, 
                    y_train, cv=5,n_jobs=-1)
    scores.append(cvs.mean())
    confidence.append(cvs.std())

plt.figure()
plt.plot(np.arange(1,15),scores)
plt.xlabel("Max Depth",fontsize=14)
plt.ylabel("Mean Classification Accuracy",fontsize=14)
plt.show()


# Visualize tree
clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X_train,y_train)
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
