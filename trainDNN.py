from __future__ import print_function
import numpy as np
np.random.seed(2671)  # for reproducibility

import copy

from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.utils.layer_utils import print_summary
import sys
#from sklearn.metrics import confusion_matrix,classification_report, roc_curve, auc

import matplotlib.pyplot as plt

#print(sys.argv)
#ifname="array_ZH_MS55_ctauS100.npy"
#if len(sys.argv) > 1:
#	ifname=sys.argv[1]



#X_signal = np.load("sig_array.npy")
X_bkg = np.load("jetarray_DY_a.npy")
X_sig = np.load("jetarray_ZH_MS40_ctauS100.npy")
#X_bkg = np.load("array_DY_a2.npy")
#X_sig = np.load("array_ZH_MS40_ctauS10.npy")
#X_sig = np.load("array_ZH_MS40_ctauS10000.npy")
#X_sig = np.load("array_ZH_MS15_ctauS100.npy")
#X_sig = np.load(ifname)
#X_bkg = np.load("array_DY_b2.npy")

nToUse = min(X_bkg.shape[0],X_sig.shape[0])

maskSig = np.ones(X_sig.shape[0], dtype=bool)
maskSig[[i for i in range(nToUse,X_sig.shape[0])]] = False
X_sig_skim = X_sig[maskSig,...]

maskBkg = np.ones(X_bkg.shape[0], dtype=bool)
maskBkg[[i for i in range(nToUse,X_bkg.shape[0])]] = False
X_bkg_skim = X_bkg[maskBkg,...]


y_bkg = np.empty(shape=(nToUse,),dtype=np.int32)
y_bkg.fill(0)
y_sig = np.empty(shape=(nToUse,),dtype=np.int32)
y_sig.fill(1)

X_interLeave = np.empty(shape=(X_sig_skim.shape[0]+X_bkg_skim.shape[0],X_sig.shape[1],))
X_interLeave[0::2] = X_sig_skim
X_interLeave[1::2] = X_bkg_skim

Y_interLeave = np.empty(shape=(y_bkg.shape[0]+y_sig.shape[0],))
Y_interLeave[0::2] = y_sig
Y_interLeave[1::2] = y_bkg

testFraction = 1.0/6.0
nTest = int(testFraction*nToUse)

X_test = X_interLeave[:nTest]
y_test = Y_interLeave[:nTest]

X_train = X_interLeave[nTest:]
y_train = Y_interLeave[nTest:]


nb_classes = 2
batch_size = 100
nb_epoch = 50
data_augmentation = False

#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

#print(Y_test[0])
#print(Y_test[1])
#print(X_test[0])
#print(X_test[1])

# input image dimensions


#X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#Y_train = y_train.reshape((-1,1))
#Y_test = y_test.reshape((-1,1))


print(X_train.shape)

model = Sequential()

#model.add(Dropout(0.25))
model.add(Dense(128,input_dim = X_sig.shape[1]))
#print(model.output_shape)
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
#print(model.output_shape)
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(Dense(64))
#print(model.output_shape)
#model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#print(model.output_shape)
model.add(Dense(2,init='uniform'))
print(model.output_shape)
model.add(Activation('softmax'))
print(model.output_shape)

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_split=1/12.0)
score = model.evaluate(X_test, Y_test, verbose=0)
Y_pred = model.predict(X_test)

#print(Y_pred.shape)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#print(Y_test)
nToPrint = 10
nPrinted = 0

tryCut = 0.99

num_outputs = Y_pred.shape[1]
confusion_matrix = np.zeros((num_outputs,num_outputs),dtype=np.int32)
for i,y in enumerate(Y_test):
	correct = np.argmax(y)
	first = 0
	if Y_pred[i][1] > tryCut: first = 1
	#first = np.argmax(Y_pred[i])
	confusion_matrix[correct, first] += 1
	#if nPrinted < nToPrint and correct == 1 and first != 0:
	#	print(i,Y_pred[i])
	#	nPrinted += 1
	
model.save("jet_dnn.keras")
#print(ifname)
print(confusion_matrix)
print("using cut: %0.3f" % (tryCut,))
print("background rejection: %0.4f" % (float(confusion_matrix[0][0])/float(confusion_matrix[0][0]+confusion_matrix[0][1]),))
print("signal efficiency: %0.4f" % (float(confusion_matrix[1][1])/float(confusion_matrix[1][1]+confusion_matrix[1][0]),))
