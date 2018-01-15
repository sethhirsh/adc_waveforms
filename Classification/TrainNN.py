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

#Setup neural net architecture 
inputs = Input(shape=(X_train.shape[1],))
Dx = Dense(100, activation="tanh")(inputs)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(inputs=[inputs], outputs=[Dx])

#Compile
D.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

#Train neural network and store training history
history = D.fit(X_train,y_train, validation_data=(X_valid,y_valid), epochs=300)

D.save('CSE546_NN_300epochs.h5')

#Plot train/valid accuracy for cross validation on epochs
val_acc=history.history['val_acc']
train_acc=history.history['acc']
plt.plot(np.arange(len(train_acc))[1:]+1,train_acc[1:], label='Train Accuracy')
plt.plot(np.arange(len(val_acc))[1:]+1,val_acc[1:], label='Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.savefig('CSE546_project_xvalid_acc_epochs_300.pdf')
plt.show()

#Plot train/valid loss for cross validation on epochs
val_loss = history.history['val_loss']
train_loss = history.history['loss']
plt.plot(np.arange(len(train_loss))[1:]+1,train_loss[1:], label='Train Loss')
plt.plot(np.arange(len(val_loss))[1:]+1,val_loss[1:], label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.savefig('CSE546_project_xvalid_epochs_150.pdf')
plt.show()
