import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model, load_model
import seaborn as sns

np.random.seed(3)

#Load data
file = np.load('adc.npz')

#ADC output, particle id, particle momentum
#pid=11 is electrons, 2212 is protons
adc = file['adc']
pid = file['pid']
pmom = file['pmom']

#Preprocess to remove low momentum particles and make sure
#only have protons and electrons
adc_filt=adc[((pid==2212)|(pid==11))&(pmom > 53)]
pmom_filt = pmom[((pid==2212)|(pid==11))&(pmom > 53)]
pid_filt =pid[((pid==2212)|(pid==11))&(pmom > 53)]

#0 is protons, 1 is electrons
pid_bin = (pid_filt != 2212)*1.

#De-mean and scale data to unit variance
from sklearn.preprocessing import StandardScaler
tf = StandardScaler(with_mean=True,with_std=True)
adcTemp = np.copy(adc_filt)
adc_filt = tf.fit_transform(adc_filt)

#Separate into train, valid, and test sets
np.random.seed = 10001
indices = np.random.permutation(len(adc_filt))

train_cut = int(len(adc_filt)*0.8)

X_train = adc_filt[indices[:train_cut]]
y_train = pid_bin[indices[:train_cut]]

X_test = adc_filt[indices[train_cut:]]
y_test = pid_bin[indices[train_cut:]]


print('Training Set Size',len(X_train))
print('Test Set Size',len(y_test))


#Load previously trained model
D=load_model('CSE546_NN_300epochs_40nodes.h5')

from keras.models import load_model

D=load_model('CSE546_NN_300epochs_40nodes.h5')

#Evaluate on the test set
from sklearn.metrics import roc_curve, auc

#Make ROC curve for the neural network
fpr, tpr, thresholds = roc_curve(y_test, D.predict(X_test).flatten())
roc_auc = auc(fpr, tpr)

#Do peak-minus-pedestal and make ROC curve
r_adc_test = tf.inverse_transform(X_test)
energy=(np.max(r_adc_test,axis=1)-np.mean(r_adc_test[:,:4],axis=1))
fpr_pmp,tpr_pmp, _ = roc_curve(y_test, energy)
roc_auc_pmp = auc(fpr_pmp, tpr_pmp)

from sklearn.linear_model import LogisticRegression

#Do logistic regression and make ROC curve
logistic = LogisticRegression()
predictions_reg = logistic.fit(r_adc_test, y_test).predict_proba(r_adc_test)
fpr_reg, tpr_reg, _ = roc_curve(y_test,predictions_reg[:,0])
roc_auc_reg = auc(1-fpr_reg, 1-tpr_reg)

from sklearn.naive_bayes import GaussianNB
#Do Naive Bayes and make ROC curve
naive = GaussianNB()
predictions_gauss = naive.fit(r_adc_test, y_test).predict_proba(r_adc_test)
fpr_gauss, tpr_gauss, _ = roc_curve(y_test,predictions_gauss[:,0])
roc_auc_gauss = auc(1-fpr_gauss, 1-tpr_gauss)

from sklearn.tree import DecisionTreeClassifier
# Do decision tree and make ROC curve
tree = DecisionTreeClassifier(max_depth=7)
predictions_tree = tree.fit(r_adc_test, y_test).predict_proba(r_adc_test)
fpr_tree, tpr_tree, _ = roc_curve(y_test,predictions_tree[:,0])
roc_auc_tree = auc(1-fpr_tree, 1-tpr_tree)

from sklearn.ensemble import RandomForestClassifier
# Do random forest and make ROC curve
forest = RandomForestClassifier(n_estimators=3,n_jobs=-1,max_depth=7)
predictions_forest = forest.fit(r_adc_test, y_test).predict_proba(r_adc_test)
fpr_forest, tpr_forest, _ = roc_curve(y_test,predictions_forest[:,0])
roc_auc_forest = auc(1-fpr_forest, 1-tpr_forest)

from scipy.stats import multivariate_normal
protons_train = X_train[y_train == 0]
electrons_train = X_train[y_train == 1]
el = multivariate_normal(mean=np.mean(electrons_train,axis=0), cov=np.cov(electrons_train.T))
pr = multivariate_normal(mean=np.mean(protons_train,axis=0), cov=np.cov(protons_train.T))
el_prob = el.pdf(X_test)
pr_prob = pr.pdf(X_test)
bad_indices = np.where((el_prob + pr_prob) == 0.0)
prob = (pr_prob / (el_prob + pr_prob))
prob[bad_indices] = 1.0
prob  = 1 - prob
fpr_multi, tpr_multi, _  = roc_curve(y_test,1 - prob)
roc_auc_multi = auc(1-fpr_multi, 1-tpr_multi)

sns.set()

#Plot all the ROC curves together
plt.figure()
plt.plot(tpr, 1-fpr, label='NN ROC, AUC = %.3f' % roc_auc)
plt.plot(1-tpr_pmp, fpr_pmp, label='Peak - Ped. ROC, AUC = %.3f' % (1-roc_auc_pmp))
plt.plot(1-tpr_reg, fpr_reg, label='Logistic Reg., AUC = %.3f' % roc_auc_reg)
plt.plot(1-tpr_forest, fpr_forest, label='Random Forest, AUC = %.3f' % roc_auc_forest)
plt.legend()
plt.xlabel('Electron Acceptance Rate',fontsize=14)
plt.ylabel('Proton Rejection Rate',fontsize=14)
plt.savefig('Figures/ROC.png',dpi=600)
#plt.draw()

#Zoom in a bit
plt.figure()
plt.plot(tpr, 1-fpr, label='NN ROC, AUC = %.3f' % roc_auc)
plt.plot(1-tpr_pmp, fpr_pmp, label='Peak - Ped. ROC, AUC = %.3f' % (1-roc_auc_pmp))
plt.plot(1-tpr_reg, fpr_reg, label='Logistic Reg., AUC = %.3f' % roc_auc_reg)
#plt.plot(1-tpr_gauss, fpr_gauss, label='Naive Bayes, AUC = %.3f' % roc_auc_gauss)
#plt.plot(1-tpr_tree, fpr_tree, label='Decision Tree, AUC = %.3f' % roc_auc_tree)
plt.plot(1-tpr_forest, fpr_forest, label='Random Forest, AUC = %.3f' % roc_auc_forest)
plt.plot(1-tpr_multi, fpr_multi, label='Multivariate Gaussian, AUC = %.3f' % roc_auc_multi)
plt.legend()
plt.xlabel('Electron Acceptance Rate',fontsize=14)
plt.ylabel('Proton Rejection Rate',fontsize=14)
plt.axis([0.75,1.05,0.75,1.05])
#plt.savefig('CSE546_project_ROC_300epochs_40nodes_zoomed.pdf')
plt.savefig('Figures/ROC_zoomed.png',dpi=600)
#plt.draw()
plt.close()

sns.reset_orig()

#Plot neural net sigmoid outputs for true electrons/protons
plt.figure()
ax=plt.axes()
plt.hist(D.predict(X_test[y_test==1]),bins=50, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k',range=(0,1))
plt.hist(D.predict(X_test[y_test==0]),bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k',range=(0,1))
plt.yscale('log')
plt.xlabel('NN Prediction',fontsize=14)
plt.legend()
ax.set_ylim(1,3e4)
plt.savefig('Figures/NN.pdf',dpi=600)

#Plot logistic outputs for true electrons/protons
plt.figure()
ax=plt.axes()
plt.hist(predictions_reg[:,1][y_test==1],bins=50, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k',range=(0,1))
plt.hist(predictions_reg[:,1][y_test==0],bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k',range=(0,1))
plt.yscale('log')
plt.xlabel('Logistic Prediction',fontsize=14)
plt.legend()
ax.set_ylim(1,3e4)
plt.savefig('Figures/reg.pdf',dpi=600)

#Plot logistic outputs for true electrons/protons
plt.figure()
ax=plt.axes()
plt.hist(predictions_tree[:,1][y_test==1],bins=10, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k')
plt.hist(predictions_tree[:,1][y_test==0],bins=10, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k')
plt.yscale('log')
plt.xlabel('Decision Tree Prediction',fontsize=14)
plt.legend()
ax.set_ylim(1,3e4)
plt.savefig('Figures/tree.pdf',dpi=600)


#Plot Naive Bayes for true electrons/protons
plt.figure()
ax=plt.axes()
plt.hist(predictions_gauss[:,1][y_test==1],bins=50, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k')
plt.hist(predictions_gauss[:,1][y_test==0],bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k')
plt.yscale('log')
plt.xlabel('Naive Bayes Prediction',fontsize=14)
plt.legend()
ax.set_ylim(1,3e4)
plt.savefig('Figures/gauss.pdf',dpi=600)


#Plot peak-minus-pedestal for true electrons/protons
plt.figure()
energy=(np.max(r_adc_test,axis=1)-np.mean(r_adc_test[:,:4],axis=1))
plt.hist(energy[y_test==1],bins=30, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k')
plt.hist(energy[y_test==0],bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k')
plt.yscale('log')
plt.xlabel('Peak minus Pedestal',fontsize=14)
plt.legend()
plt.savefig('Figures/peakped.pdf',dpi=600)


#Plot random forest for true electrons/protons
plt.figure()
plt.hist(predictions_forest[:,1][y_test==1],bins=50, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k',range=(0,1))
plt.hist(predictions_forest[:,1][y_test==0],bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k',range=(0,1))
plt.yscale('log')
plt.xlabel('Random Forest Prediction',fontsize=14)
plt.legend()
plt.savefig('Figures/forest.pdf',dpi=600)


#Plot multivariate gaussian for true electrons/protons
plt.figure()
energy=(np.max(r_adc_test,axis=1)-np.mean(r_adc_test[:,:4],axis=1))
plt.hist(prob[y_test==1],bins=50, alpha=0.4, label='True Electrons', facecolor='r', edgecolor='k',range=(0,1))
plt.hist(prob[y_test==0],bins=50, alpha=0.4, label='True Protons', facecolor='b',edgecolor='k',range=(0,1))
plt.yscale('log')
plt.xlabel('Multivariate Gaussian',fontsize=14)
plt.legend()
#plt.savefig('CSE546_project_pmp.pdf')
plt.savefig('Figures/multi.pdf',dpi=600)




# Find correctly classified samples in decision tree that were incorrectly classified by peak
# minus pedestal

# Protons are zero
#ind = np.where(np.logical_and(np.logical_and(predictions_tree[:,1] > 0.8, energy > 1000),y_test == 1))
adc_test = adcTemp[indices[train_cut:]]
sns.set()

plt.figure()
time = np.arange(16)*20
waveform1 = adc_test[21763]
plt.plot(time,waveform1,'b')
plt.xlabel('Time [ns]',fontsize=14)
plt.ylabel('ADC [counts]',fontsize=14)
plt.savefig('Figures/Waveform1.png',dpi=600)


plt.figure()
waveform2 = adc_test[19493]
np.max(waveform2) - np.mean(waveform2)
plt.plot(time,waveform2,'b')
plt.xlabel('Time [ns]',fontsize=14)
plt.ylabel('ADC [counts]',fontsize=14)
plt.savefig('Figures/Waveform2.png',dpi=600)

plt.close()

