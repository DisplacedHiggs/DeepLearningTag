from __future__ import print_function
import numpy as np
np.random.seed(2671)  # for reproducibility

import copy

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.utils.layer_utils import print_summary
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt

def getResults(ifnamesig,ifnamebkg,ifnamemodel,mtype="CNN"):
	X_sig = np.load(ifnamesig)
	X_bkg = np.load(ifnamebkg)

	y_bkg = np.empty(shape=(X_bkg.shape[0],),dtype=np.int32)
	y_bkg.fill(0)
	y_sig = np.empty(shape=(X_sig.shape[0],),dtype=np.int32)
	y_sig.fill(1)

	nshape = list(X_sig.shape)
	nshape[0] = X_sig.shape[0]+X_bkg.shape[0]
	nshape = tuple(nshape)

	X_interLeave = np.empty(shape=nshape)
	X_interLeave[0:X_sig.shape[0]] = X_sig
	X_interLeave[X_sig.shape[0]:] = X_bkg

	Y_interLeave = np.empty(shape=(y_bkg.shape[0]+y_sig.shape[0],))
	Y_interLeave[0:X_sig.shape[0]] = y_sig
	Y_interLeave[X_sig.shape[0]:] = y_bkg

	X_test = X_interLeave[:]
	y_test = Y_interLeave[:]
	
	nb_classes = 2
	#img_rows, img_cols = X_sig.shape[1], X_sig.shape[2]

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	X_test = X_test.astype('float32')
	
	if mtype == "CNN" and K.image_dim_ordering() == 'th':
		X_test = np.swapaxes(X_test,1,3)
		X_test = np.swapaxes(X_test,2,3)
		#input_shape = (X_sig.shape[3], img_rows, img_cols)
	#else:
	#	input_shape = (img_rows, img_cols, X_sig.shape[3])
	
	
	model = load_model(ifnamemodel)

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
	
	print(ifnamesig)
	print(confusion_matrix)
	print("using cut: %0.3f" % (tryCut,))
	print("background rejection: %0.4f" % (float(confusion_matrix[0][0])/float(confusion_matrix[0][0]+confusion_matrix[0][1]),))
	print("signal efficiency: %0.4f" % (float(confusion_matrix[1][1])/float(confusion_matrix[1][1]+confusion_matrix[1][0]),))

	fpr, tpr, _ = roc_curve(Y_test[:,0], Y_pred[:,0])
	rauc = auc(fpr,tpr)

	return fpr,tpr,rauc



def main():
	ifname1="array_ZH_MS55_ctauS100.npy"
	ifname2="jetarray_ZH_MS55_ctauS100.npy"
	ofname="roc_ZH_MS55_ctauS100.png"
	if len(sys.argv) > 1:
		ifname1 = sys.argv[1]
		ifname2 = sys.argv[2]
		ofname=sys.argv[3]
	#cnnfpr, cnntpr, cnnauc = getCNNResults(ifname1,"array_DY_b2.npy")
	#dnnfpr, dnntpr, dnnauc = getDNNResults(ifname2,"jetarray_DY_b.npy")
	cnnfpr, cnntpr, cnnauc = getResults(ifname1,"array_DY_b2.npy","jet_cnn.keras","CNN")
	dnnfpr, dnntpr, dnnauc = getResults(ifname2,"jetarray_DY_b.npy","jet_dnn.keras","DNN")
	
	cnnbr = [1-fpr for fpr in cnnfpr]
	dnnbr = [1-fpr for fpr in dnnfpr]
	
	fig, ax = plt.subplots()
	ax.plot(cnnbr,cnntpr,label="CNN Results (area = %0.2f)" % cnnauc,color='b')
	ax.plot(dnnbr,dnntpr,label="DNN Results (area = %0.2f)" % dnnauc,color='g')
	ax.set_xlim([0.90,1.05])
	ax.set_ylim([0.0,1.05])
	plt.xlabel("Background Rejection")
	plt.ylabel("Signal Efficiency")
	plt.legend(loc = "lower left")
	fig.savefig(ofname)

	
	return


if __name__ == "__main__":
	main()
