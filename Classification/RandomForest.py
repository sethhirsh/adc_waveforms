import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Load data and split in training, validation and test sets

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
valid_cut = int(len(adc_filt)*0.9)

X_train = adc_filt[indices[:train_cut]]
y_train = pid_bin[indices[:train_cut]]

X_valid = adc_filt[indices[train_cut:valid_cut]]
y_valid = pid_bin[indices[train_cut:valid_cut]]

X_test = adc_filt[indices[valid_cut:]]
y_test = pid_bin[indices[valid_cut:]]

r_adc_valid = tf.inverse_transform(X_valid)
energy=(np.max(r_adc_valid,axis=1)-np.mean(r_adc_valid[:,:4],axis=1))
fpr_pmp,tpr_pmp, _ = roc_curve(y_valid, energy)
roc_auc_pmp = auc(fpr_pmp, tpr_pmp)

clf = RandomForestClassifier(max_depth=7)
clf.fit(adc_filt[indices[:valid_cut]],pid_bin[indices[:valid_cut]])
p = clf.predict_proba(X_test)


plt.figure()
for max_depth in range(3,8):
	scores = []
	confidence = []
	for n_estimators in range(1,15):
		print(max_depth,n_estimators)
		clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,max_depth=max_depth)

		cvs = cross_val_score(clf, adc_filt[indices[:valid_cut]], 
		                pid_bin[indices[:valid_cut]], cv=5,n_jobs=-1)
		scores.append(cvs.mean())
		confidence.append(cvs.std())
	plt.scatter(np.arange(1,15),scores,label=str(max_depth))

plt.legend(title='Max Depth')
plt.xlabel("# of Estimators",fontsize=14)
plt.ylabel("Mean Classification Accuracy",fontsize=14)
plt.show()

