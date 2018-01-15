import numpy as np
import matplotlib.pyplot as plt

#Load data
file = np.load('adc.npz')

file.keys()

#pid=11 is electrons, 2212 is protons
adc = file['adc']
pid = file['pid']
pmom = file['pmom']

#Filter out anything not proton or electron and put lower
#momentum cut
adc_filt=adc[((pid==2212)|(pid==11))&(pmom > 53)]
pmom_filt = pmom[((pid==2212)|(pid==11))&(pmom > 53)]
pid_filt =pid[((pid==2212)|(pid==11))&(pmom > 53)]

#Restrict to making protons
#real_p_adc=adc_filt[pid_filt==2212]

#Restrict to making electrons
real_p_adc=adc_filt[pid_filt==11]

#Scale data
from sklearn.preprocessing import StandardScaler
tf = StandardScaler()
real_p_adc=tf.fit_transform(real_p_adc)
scale=np.amax(real_p_adc)
real_p_adc=real_p_adc/scale

#Get Keras set up
import keras.backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Conv1D,Reshape,Flatten, MaxPool1D,BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, load_model
from keras.initializers import RandomUniform

#Setup networks - see paper for more details
inputs = Input(shape=(100,))
Gx = Reshape((-1, 1))(inputs)
Gx = Conv1D(16, 4, activation='relu')(Gx)
#Gx = MaxPool1D()(Gx)
Gx= BatchNormalization()(Gx)
Gx = Flatten()(Gx)
Gx = Dense(100, activation="tanh")(Gx)
Gx = Dropout(0.3)(Gx)
Gx = Reshape((-1, 1))(Gx)
Gx = Conv1D(16, 4, activation='relu')(Gx)
Gx = MaxPool1D()(Gx)
Gx= BatchNormalization()(Gx)
Gx = Flatten()(Gx)
Gx = Dropout(0.3)(Gx)
Gx = Dense(100, activation="tanh")(Gx)
Gx = Dense(real_p_adc.shape[1], activation="tanh",kernel_initializer=RandomUniform(-0.005,0.005), bias_initializer='zeros')(Gx)
G = Model(inputs=[inputs], outputs=[Gx])

Din = Input(shape=(real_p_adc.shape[1],))
Dx = Reshape((-1, 1))(Din)
Dx = Conv1D(8, 4, activation='relu')(Dx)
Dx= BatchNormalization()(Dx)
Dx = Conv1D(16, 4, activation='relu')(Dx)
Dx= BatchNormalization()(Dx)
Dx = MaxPool1D()(Dx)
Dx = Dropout(0.25)(Dx)
Dx = Flatten()(Dx)
Dx = Dense(50, activation="tanh")(Dx)
Dx= Dense(1, activation="sigmoid")(Dx)
D = Model(inputs=[Din], outputs=[Dx])

Ain=G(inputs)
Ax=D(Ain)
A= Model(inputs=[inputs], outputs=[Ax])

#Load in old weights, if you want
#D.set_weights(load_model('GAN_disc_penalize_elec_150_mark2.h5').get_weights())
#G.set_weights(load_model('GAN_gen_penalize_elec_150_mark2.h5').get_weights())
#A.set_weights(load_model('GAN_GAN_penalize_elec_150_mark2.h5').get_weights())

from keras.optimizers import SGD
optD= SGD(lr=0.003)
optA= SGD(lr=0.001)
optG= SGD(lr=0.001)
G.compile(loss='binary_crossentropy', optimizer=optG, metrics=['accuracy'])
D.compile(loss='binary_crossentropy', optimizer=optD, metrics=['accuracy'])
A.compile(loss='binary_crossentropy', optimizer=optA, metrics=['accuracy'])

#Freeze/unfreeze weights - ideally just this first line, but
#have had issues in the past 
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

#Make mixed dataset and labels
noise_fakes = np.random.uniform(-1,1, size=(1000,100)) 
fake_p_adc = G.predict(noise_fakes)
idxs=np.random.permutation(3000)
real_p_adc_b = real_p_adc[idxs]
labels= np.zeros(len(fake_p_adc)+len(real_p_adc_b))
fake_labels=np.zeros(len(fake_p_adc))
real_labels=np.ones(len(real_p_adc_b))
labels[len(fake_p_adc):] = 1
mixed = np.concatenate((fake_p_adc, real_p_adc_b))

#from IPython import display

#Running plot function
def plot_losses(i, losses, fake_ex):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    ax1 = plt.subplot(211)   
    values = np.array(losses["D"])
    plt.plot(range(len(values)), values, label=r"$D$", color="blue")
    plt.legend(loc="upper right")
    
    ax2 = plt.subplot(212, sharex=ax1) 
    values = np.array(losses["A"])
    plt.plot(range(len(values)), values, label=r"$A$", color="green")
    plt.legend(loc="upper right")

    plt.show()

    
    plt.plot(np.arange(len(fake_ex))*20.,fake_ex)
    plt.show()

#Store losses
losses={'D':[], 'A':[]}
acc = {'D':[], 'A':[]}

#Main training loop
epochs=50
batch_size=40
for i in range(epochs):   
    if(i%5 == 0):
        fake_ex = G.predict(np.random.uniform(-1,1, size=(1,100)))
        fake_ex=tf.inverse_transform(fake_ex*scale)
        plot_losses(i,losses,fake_ex[0])
        print '%d of %d' % (i,epochs)

    #Make noise to feed into GAN, label with counterfeit 1's 
    noise_net = np.random.uniform(-1,1, size=(batch_size,100)) 
    noise_label = np.ones(len(noise_net))
    
    #Freeze weights of D
    set_trainability(G, True)
    set_trainability(D, False)

    #Compilation to ensure they actually freeze
    A.compile(loss='binary_crossentropy', optimizer=optA, metrics=['accuracy'])
    D.compile(loss='binary_crossentropy', optimizer=optD, metrics=['accuracy'])

    #Train GAN, update only weights of G 
    for j in range(5):
        A.train_on_batch(noise_net, noise_label)
    
    #Store gen loss and accuracy
    Amet= A.evaluate(noise_net,noise_label,verbose=0)
    
    losses['A'].append(Amet[0])
    acc['A'].append(Amet[1])
    
    #Unfreeze weights of D
    set_trainability(D, True)
    D.compile(loss='binary_crossentropy', optimizer=optD, metrics=['accuracy'])
    
    #loss2= D.evaluate(mixed,labels,verbose=0)
    #print loss2[0]
    
    #idxs = np.random.permutation(len(real_p_adc))[:batch_size]
    
    #Make distribution of distance from mean
    transf_full = tf.inverse_transform(real_p_adc*scale)
    mean_dist_full=np.mean(transf_full-np.mean(transf_full,axis=0),axis=1)
  
    #Define far and near indices
    far_idxs=np.where(abs(mean_dist_full)>10)[0]
    near_idxs=np.where(abs(mean_dist_full)<10)[0]
    
    #Set proportion of each
    b_far=int(batch_size*0.10)
    b_near=int(batch_size*0.90)
    
    #Take appropriate subsets 
    idxs_far=np.random.permutation(len(far_idxs))[:b_far]
    idxs_near=np.random.permutation(len(near_idxs))[:b_far]
    
    #Concatenate
    idxs=np.concatenate((far_idxs[idxs_far], near_idxs[idxs_near]))
    
    #Make mix of reals and fakes with appropriate labels
    noise_fakes = np.random.uniform(-1,1, size=(batch_size,100))
    fake_p_adc = G.predict(noise_fakes)
    real_p_adc_b = real_p_adc[idxs]
    labels= np.zeros(len(fake_p_adc)+len(real_p_adc_b))
    labels[len(fake_p_adc):] = 1
    mixed = np.concatenate((fake_p_adc, real_p_adc_b))
    
    #Train D
    D.train_on_batch(mixed,labels)
    
    #Grab random fakes
    fake1=fake_p_adc[np.random.randint(0,len(fake_p_adc))]
    fake2=fake_p_adc[np.random.randint(0,len(fake_p_adc))]
    
    #Penalize if too close
    #NOTE: this isn't the best penalty - should use a sum
    #squares
    if(np.mean(abs(fake1-fake2)) < 0.005):
        for i in range(3):
            D.train_on_batch(fake_p_adc,np.zeros(len(fake_p_adc)))
    
    #Store losses
    Dmet= D.evaluate(mixed,labels,verbose=0)
    losses['D'].append(Dmet[0])
    acc['D'].append(Dmet[1])

#Save models
#D.save('GAN_disc_penalize_elec_150_mark2.h5')
#G.save('GAN_gen_penalize_elec_150_mark2.h5')
#A.save('GAN_GAN_penalize_elec_150_mark2.h5')