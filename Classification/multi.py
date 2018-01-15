import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load data and split into train and test sets
file = np.load('adc.npz')
adc = file['adc']
pid = file['pid']
pmom = file['pmom']

from sklearn.preprocessing import StandardScaler


adc_filt=adc[((pid==2212)|(pid==11))&(pmom > 53)]
tf = StandardScaler(with_mean=False, with_std=False)
adc_filt = tf.fit_transform(adc_filt)

pmom_filt = pmom[((pid==2212)|(pid==11))&(pmom > 53)]
pid_filt =pid[((pid==2212)|(pid==11))&(pmom > 53)]

np.random.seed = 10001
indices = np.random.permutation(len(adc_filt))

pid_bin = (pid_filt != 2212)*1.
train_cut = int(len(adc_filt)*0.8)

X_train = adc_filt[indices[:train_cut]]
y_train = pid_bin[indices[:train_cut]]

X_test = adc_filt[indices[train_cut:]]
y_test = pid_bin[indices[train_cut:]]



protons_train = X_train[y_train == 0]
electrons_train = X_train[y_train == 1]

# Construct multivariate normal distributions
el = multivariate_normal(mean=np.mean(electrons_train,axis=0), cov=np.cov(electrons_train.T))
pr = multivariate_normal(mean=np.mean(protons_train,axis=0), cov=np.cov(protons_train.T))


print("ELECTRON TIME")
for i in range(10):
	plt.plot(el.rvs())
	plt.show()

print("PROTON TIME")
for i in range(10):
	plt.plot(pr.rvs())
	plt.show()


# Generate data 
num_el_samples = 1000
el_sim = el.rvs(num_el_samples)

num_pr_samples = 1000
pr_sim = pr.rvs(num_pr_samples)