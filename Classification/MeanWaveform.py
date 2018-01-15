import numpy as np
import matplotlib.pyplot as plt

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
tf = StandardScaler()
adc_filt = tf.fit_transform(adc_filt)

#Separate into train, valid, and test sets
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

#Make array of all proton values at each time bin
p_adc=np.array(r_adc_valid[y_valid==0].T)
n,d=p_adc.shape
p_adc=p_adc.reshape((n*d,))

#Make corresponding time index array
idxs=np.zeros(d)
for i in range(1,16):
    idxs=np.concatenate((idxs,(np.zeros(d)+i)))

#Make 2d proton waveform histogram
from matplotlib.colors import LogNorm
plt.hist2d(idxs*20,p_adc,bins=[15,30], norm=LogNorm())  
plt.xlabel('Time [ns]')
plt.ylabel('ADC [Counts]')
plt.title('Distribution of Proton Waveforms')
#plt.savefig('CSE546_project_proton_wf_2dhist.pdf')
plt.show()

#Same thing for electrons
e_adc=np.array(r_adc_valid[y_valid==1].T)
n1,d1=e_adc.shape
e_adc=e_adc.reshape((n1*d1,))

idxs_e=np.zeros(d1)
for i in range(1,16):
    idxs_e=np.concatenate((idxs_e,(np.zeros(d1)+i)))

from matplotlib.colors import LogNorm
plt.hist2d(idxs_e*20,e_adc,bins=[15,30], norm=LogNorm())  
plt.xlabel('Time [ns]')
plt.ylabel('ADC [Counts]')
plt.title('Distribution of Electron Waveforms')
#plt.savefig('CSE546_project_electron_wf_2dhist.pdf')
plt.show()

#Find the bin-by-bin mean for both
import scipy.stats
means_result_p = scipy.stats.binned_statistic(idxs, p_adc, bins=15, range=(0,15), statistic='mean')
means_p = means_result_p.statistic
bin_edges_p = means_result_p.bin_edges
bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:])/2.

means_result_e = scipy.stats.binned_statistic(idxs_e, e_adc, bins=15, range=(0,15), statistic='mean')
means_e = means_result_e.statistic
bin_edges_e = means_result_e.bin_edges
bin_centers_e = (bin_edges_e[:-1] + bin_edges_e[1:])/2.

#Plot these means
plt.plot(bin_centers_p*20, means_p, label='Average Proton')
plt.plot(bin_centers_e*20, means_e, label='Average Electron')
plt.legend()
plt.ylabel('ADC [counts]')
plt.xlabel('Time [ns]')
#plt.savefig('CSE546_project_average_waveform.pdf')
plt.show()