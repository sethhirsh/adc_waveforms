import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.stats import multivariate_normal

D_p = load_model('GAN_disc_penalize_150.h5')
G_p = load_model('GAN_gen_penalize_150.h5')

D_e = load_model('GAN_disc_penalize_elec_150_mark2.h5')
G_e = load_model('GAN_gen_penalize_elec_150_mark2.h5')

from keras.utils import plot_model
plot_model(D_p, to_file='Discriminator_proton_net_pic.png', show_shapes=True, show_layer_names=False)

from keras.utils import plot_model
plot_model(G_p, to_file='Generator_proton_net_pic.png', show_shapes=True, show_layer_names=False)

D_p.summary()

file = np.load('adc.npz')
adc = file['adc']
pid = file['pid']
pmom = file['pmom']

from sklearn.preprocessing import StandardScaler

adc_filt=adc[((pid==2212)|(pid==11))&(pmom > 53)]
tf_mv = StandardScaler(with_mean=False, with_std=False)
adc_filt = tf_mv.fit_transform(adc_filt)

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

el = multivariate_normal(mean=np.mean(electrons_train,axis=0), cov=np.cov(electrons_train.T))
pr = multivariate_normal(mean=np.mean(protons_train,axis=0), cov=np.cov(protons_train.T))


# Generate data 
num_el_samples = len(X_test[y_test==1])
el_mv = el.rvs(num_el_samples)

num_pr_samples = len(X_test[y_test==0])
pr_mv = pr.rvs(num_pr_samples)

#Make electron data
real_e=X_test[y_test==1]

tf_e = StandardScaler()
real_e=tf_e.fit_transform(real_e)
scale_e=np.amax(real_e)
real_e=real_e/scale_e

noise_e = np.random.uniform(-1,1, size=(len(real_e),100))
pred_e = G_e.predict(noise_e)
fake_e = tf_e.inverse_transform(pred_e*scale_e)

score_fake_e = D_e.predict(pred_e).flatten()
score_real_e = D_e.predict(real_e).flatten()

#Make proton data
real_p=X_test[y_test==0]

tf_p = StandardScaler()
real_p=tf_p.fit_transform(real_p)
scale_p=np.amax(real_p)
real_p=real_p/scale_p

noise_p = np.random.uniform(-1,1, size=(len(real_p),100))
pred_p = G_p.predict(noise_e)
fake_p = tf_p.inverse_transform(pred_p*scale_p)

score_fake_p = D_p.predict(pred_p).flatten()
score_real_p = D_p.predict(real_p).flatten()

#Make go back to "real" waveforms for each
real_p_adc = tf_p.inverse_transform(real_p*scale_p)
real_e_adc = tf_e.inverse_transform(real_e*scale_e)

fake_p_adc = tf_p.inverse_transform(pred_p*scale_p)
fake_e_adc = tf_e.inverse_transform(pred_e*scale_e)

mv_p_adc = tf_mv.inverse_transform(pr_mv)
mv_e_adc = tf_mv.inverse_transform(el_mv)

#Make peak-minus-pedestal for each
real_energy_p=(np.max(real_p_adc,axis=1)-np.mean(real_p_adc[:,:4],axis=1))
real_energy_e=(np.max(real_e_adc,axis=1)-np.mean(real_e_adc[:,:4],axis=1))

fake_energy_p=(np.max(fake_p_adc,axis=1)-np.mean(fake_p_adc[:,:4],axis=1))
fake_energy_e=(np.max(fake_e_adc,axis=1)-np.mean(fake_e_adc[:,:4],axis=1))

mv_energy_p=(np.max(mv_p_adc,axis=1)-np.mean(mv_p_adc[:,:4],axis=1))
mv_energy_e=(np.max(mv_e_adc,axis=1)-np.mean(mv_e_adc[:,:4],axis=1))

#real_p_idx = np.where((real_energy_p>1000)& (real_energy_p<1100))[0][0]
#real_e_idx = np.where((real_energy_e>500)& (real_energy_e<600))[0][0]

#fake_p_idx = np.where((fake_energy_p>1000)& (fake_energy_p<1100))[0][0]
#fake_e_idx = np.where((fake_energy_e>500)& (fake_energy_e<600))[0][0]

#mv_p_idx = np.where((mv_energy_p>1000)& (mv_energy_p<1100))[0][0]
#mv_e_idx = np.where((mv_energy_e>500)& (mv_energy_e<600))[0][0]

#Protons
plt.plot(np.arange(16)*20, real_p_adc[20], label='Real Proton')
plt.plot(np.arange(16)*20, fake_p_adc[7], label='GAN Fake Proton')
plt.plot(np.arange(16)*20, pr_mv[6], label='Multivariate Fake Proton')
plt.legend()
plt.ylabel('ADC [Counts]')
plt.xlabel('Time [ns]')
#plt.savefig('CSE_546_project_fake_proton_wfs.png')
plt.show()

#Electrons
plt.plot(np.arange(16)*20, real_e_adc[30], label='Real Electron')
plt.plot(np.arange(16)*20, fake_e_adc[841], label ='GAN Fake Electron')
plt.plot(np.arange(16)*20, el_mv[11], label ='Multivariate Fake Electron')
plt.legend()
plt.ylabel('ADC [Counts]')
plt.xlabel('Time [ns]')
#plt.savefig('CSE_546_project_fake_electron_wfs.png')
plt.show()

#Peak-minus-pedestal plots for each
cut_fake_p = fake_energy_p[score_fake_p>0.5]
scale_fake_p = np.ones(len(cut_fake_p))*1.*len(real_energy_p)/len(cut_fake_p)

plt.hist(real_energy_p,range=[0,3000], bins=50, label='Real Protons',facecolor='None',edgecolor='r', histtype='step')
plt.hist(cut_fake_p,range=[0,3000], bins=50,weights=scale_fake_p, label='GAN Fake Protons',facecolor='None',edgecolor='b',histtype='step')
plt.hist(mv_energy_p,range=[0,3000], bins=50, label='MV Fake Protons',facecolor='None',edgecolor='g',histtype='step')
plt.yscale('log')
plt.xlabel('Peak-minus-pedestal')
plt.axis([0,3200,0.7,2e4])
plt.legend()
plt.savefig('CSE_546_project_fake_proton_pmp_0.5cut_scale.png')
plt.show()

cut_fake_e = fake_energy_e[score_fake_e>0.5]
scale_fake_e = np.ones(len(cut_fake_e))*1.*len(real_energy_e)/len(cut_fake_e)

plt.hist(real_energy_e,range=[0,3000], bins=50, label='Real Electrons',facecolor='None',edgecolor='r',histtype='step')
plt.hist(cut_fake_e,range=[0,3000], bins=50, weights=scale_fake_e, label='GAN Fake Electrons',facecolor='None',edgecolor='b',histtype='step')
plt.hist(mv_energy_e, range=[0,3000], bins=50, label='MV Fake Electrons',facecolor='None',edgecolor='g',histtype='step')
plt.yscale('log')
plt.xlabel('Peak-minus-pedestal')
plt.legend()
plt.savefig('CSE_546_project_fake_electron_pmp_0.5cut_mark2_scale.png')
plt.show()


from scipy import loadtxt, optimize

def fitfunc(p,x):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-(x-p[1])**2/(2*p[2]**2))+p[3]+p[4]*x 
def residual(p, x, y):
    return (fitfunc(p, x)-y)


widths_real_p = []
for y in real_p_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_real_p.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_real_p.append(2*pf[2])
    
widths_fake_p = []
for y in fake_p_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_fake_p.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_fake_p.append(2*pf[2])
    
widths_mv_p = []
for y in mv_p_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_mv_p.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_mv_p.append(2*pf[2])
    

widths_real_e = []
for y in real_e_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_real_e.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_real_e.append(2*pf[2])
    
widths_fake_e = []
for y in fake_e_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_fake_e.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_fake_e.append(2*pf[2])
    
widths_mv_e = []
for y in mv_e_adc:
    min_idx = np.argmax(y)-4
    max_idx = np.argmax(y)+4
    if((min_idx < 0)|(max_idx>16)):
        widths_mv_e.append(-10)
        continue
    y_1=y[min_idx:max_idx]
    x=np.arange(16)[min_idx:max_idx]*20.
    p0 = [10000., x[np.argmax(y_1)], 25., 50., 0.]

    pf, cov, info, mesg, success = optimize.leastsq(residual, p0, args=(x, y_1),
                                                full_output=1)
    widths_mv_e.append(2*pf[2])

widths_real_p = np.array(widths_real_p)
widths_fake_p= np.array(widths_fake_p)
widths_mv_p= np.array(widths_mv_p)
widths_real_e= np.array(widths_real_e)
widths_fake_e= np.array(widths_fake_e)
widths_mv_e= np.array(widths_mv_e)

cut_fake_p_w = widths_fake_p[score_fake_p>0.5]
scale_fake_p_w = np.ones(len(cut_fake_p_w))*1.*len(widths_real_p)/len(cut_fake_p_w)

plt.hist(widths_real_p, range=[10,200], bins=30, label='Real Protons',facecolor='None',edgecolor='r',histtype='step')
plt.hist(cut_fake_p_w, range=[10,200], bins=30, weights=scale_fake_p_w, label='GAN Protons',facecolor='None',edgecolor='b',histtype='step')
plt.hist(widths_mv_p, range=[10,200], bins=30, label='MV Protons',facecolor='None',edgecolor='g',histtype='step')
plt.xlabel('Peak Widths [ns]')
plt.legend()
plt.yscale('log')
plt.savefig('CSE546_project_gaussian_widths_protons_0.5cut.png')
plt.show()

cut_fake_e_w = widths_fake_e[score_fake_e>0.5]
scale_fake_e_w = np.ones(len(cut_fake_e_w))*1.*len(widths_real_e)/len(cut_fake_e_w)

plt.hist(widths_real_e, range=[10,200], bins=30, label='Real Electrons',facecolor='None',edgecolor='r',histtype='step')
plt.hist(cut_fake_e_w, range=[10,200], weights=scale_fake_e_w, bins=30, label='GAN Electrons',facecolor='None',edgecolor='b',histtype='step')
plt.hist(widths_mv_e, range=[10,200], bins=30, label='MV Electrons',facecolor='None',edgecolor='g',histtype='step')
plt.xlabel('Peak Widths [ns]')
plt.legend()
plt.yscale('log')
plt.savefig('CSE546_project_gaussian_widths_electrons_0.5cut.png')
plt.show()

a_vals = [0, 0.2, 0.4, 0.6, 0.8]

for a in a_vals:
    f_cut_energy_p = fake_energy_p[score_fake_p > a]
    scale = np.ones(len(f_cut_energy_p))*1.*len(real_energy_p)/len(f_cut_energy_p)
    plt.hist(real_energy_p,range=[0,5000], bins=50, label='Real Protons',facecolor='r',edgecolor='k',alpha=0.5)
    plt.hist(f_cut_energy_p,range=[0,5000], weights=scale, bins=50, label='GAN Fake Protons',facecolor='b',edgecolor='k',alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Peak-minus-pedestal')
    plt.title('Peak-minus-pedestal, D Prediction > %.1f' %a)
    plt.legend()
    plt.savefig('CSE546_project_pmp_successive_cuts_proton_Dgt%.1f.pdf' % a)
    plt.show()

a_vals = [0, 0.2, 0.4, 0.6, 0.8]

for a in a_vals:
    f_cut_energy_e = fake_energy_e[score_fake_e > a]
    scale = np.ones(len(f_cut_energy_e))*1.*len(real_energy_e)/len(f_cut_energy_e)
    plt.hist(real_energy_e,range=[0,3000], bins=50, label='Real Electrons',facecolor='r',edgecolor='k',alpha=0.5)
    plt.hist(f_cut_energy_e,range=[0,3000], weights=scale, bins=50, label='GAN Fake Electrons',facecolor='b',edgecolor='k',alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Peak-minus-pedestal')
    plt.title('Peak-minus-pedestal, D Prediction > %.1f' %a)
    plt.legend()
    plt.savefig('CSE546_project_pmp_successive_cuts_electron_Dgt%.1f.pdf' % a)
    plt.show()

