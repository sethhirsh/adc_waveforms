import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model

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

#Hidden layer node values to cycle over
#Train each at 300 epochs
hlayer_nodes = [100, 80, 70, 60, 50, 40, 30, 10]

for nodes in hlayer_nodes:
    #Make network architecture
    inputs = Input(shape=(X_train.shape[1],))
    Dx = Dense(nodes, activation="tanh")(inputs)
    Dx = Dense(1, activation="sigmoid")(Dx)
    D = Model(inputs=[inputs], outputs=[Dx])

    #Compile
    D.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    
    #Train and save history
    history = D.fit(X_train,y_train, validation_data=(X_valid,y_valid), epochs=300)
    
    #Save history and model (here saving is commented out)
    hist=history.history
    #np.savez('CSE546_project_epochs_300_nodes_%d.npz' % nodes,val_acc=hist['val_acc'], train_acc=hist['acc'], val_loss=hist['val_loss'],train_loss= hist['loss'])
    #D.save('CSE546_NN_300epochs_%dnodes.h5' %nodes)

#Load each history
n10 = np.load('CSE546_project_epochs_300_nodes_10.npz')
n30 = np.load('CSE546_project_epochs_300_nodes_30.npz')
n40 = np.load('CSE546_project_epochs_300_nodes_40.npz')
n50 = np.load('CSE546_project_epochs_300_nodes_50.npz')
n60 = np.load('CSE546_project_epochs_300_nodes_60.npz')
n80 = np.load('CSE546_project_epochs_300_nodes_80.npz')
n100 = np.load('CSE546_project_epochs_300_nodes_100.npz')

#Make lists of 300 epoch value for each metric
node_list = [10, 30, 40, 50, 60, 80, 100]
val_acc = [n10['val_acc'][299], n30['val_acc'][299], n40['val_acc'][299], n50['val_acc'][299], n60['val_acc'][299], 
           n80['val_acc'][299], n100['val_acc'][299]]
train_acc = [n10['train_acc'][299], n30['train_acc'][299], n40['train_acc'][299], n50['train_acc'][299], n60['train_acc'][299], 
             n80['train_acc'][299], n100['train_acc'][299]]
val_loss = [n10['val_loss'][299], n30['val_loss'][299], n40['val_loss'][299], n50['val_loss'][299], n60['val_loss'][299], 
            n80['val_loss'][299], n100['val_loss'][299]]
train_loss = [n10['train_loss'][299], n30['train_loss'][299], n40['train_loss'][299], n50['train_loss'][299], n60['train_loss'][299], 
              n80['train_loss'][299], n100['train_loss'][299]]

#Plot accuracy vs number of hidden nodes
ax=plt.axes()
plt.plot(node_list, train_acc, label='Training Accuracy')
plt.plot(node_list, val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Nodes in Hidden Layer')
plt.legend()
ax.set_ylim(0.982, 0.984)
#plt.savefig('CSE546_project_accuracy_vs_nodes.pdf')
plt.show()

#Plot loss
plt.plot(node_list, train_loss, label='Training Loss')
plt.plot(node_list, val_loss, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Nodes in Hidden Layer')
plt.legend()
#plt.savefig('CSE546_project_loss_vs_nodes.pdf')
plt.show()