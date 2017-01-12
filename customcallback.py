import keras.callbacks
import utils
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import scipy.misc
from keras import backend as K


class LossHistory(keras.callbacks.Callback):
	def __init__(self, imagefordebug, thresh_option):
		self.imagefordebug = imagefordebug
		self.thresh_option = thresh_option

	def setmodel(self, model):
                self.model = model

	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		#print 'on_batch_end: batch='+str(batch)
		#self.losses.append(logs.get('loss'))
		if (batch%20) == 0:
			utils.testfile(self.model, self.imagefordebug, waittime=100, confid_thresh=self.thresh_option, fordebug=True)

	def on_epoch_end(self, epoch, logs={}):
		self
		#if epoch>0 and (epoch%5)==0:
		#	self.model.save_weights('yolotiny_weight_'+str(epoch)+'.h5')

class ShowBox(keras.callbacks.Callback):
	def __init__(self, imgfilelist, onefeature,w,h,c):
		f = open(imgfilelist)
		for imgfile in f:
			frame = Image.open(imgfile.strip())
			#nim = scipy.misc.imresize(frame, (w, h, c))
			break

		self.frame = frame
		self.onefeature = onefeature
		self.w =w
		self.h =h
		self.c =c
		#print 'showbox:'+str(w)+', '+str(h)+', '+str(c)+', '+str(nim.shape)
		#print onefeature
		#exit()

	def on_batch_end(self, batch, logs={}):
		if (batch%100) == 0:
			#print 'batch='+str(batch)
			nim = scipy.misc.imresize(self.frame, (self.w, self.h, self.c))
                        utils.testfeature(self.model, nim, self.w, self.h, self.c, self.onefeature, waittime=100, confid_thresh=0.2)

class LrReducer(keras.callbacks.Callback):
	def __init__(self, patience=10, reduce_rate=0.2, reduce_nb=1, verbose=1):
		#super(Callback, self).__init__()
		self.patience = patience
		self.wait = 0
		self.best_loss = 1000000.
		self.reduce_rate = reduce_rate
		self.current_reduce_nb = 0
		self.reduce_nb = reduce_nb
		self.verbose = verbose
		self.stop_training = False

	def setmodel(self, model):
		self.model = model

	def istrainstop(self):
		return self.stop_training

	def on_epoch_end(self, epoch, logs={}):
		current_loss = logs.get('loss')
		if current_loss < self.best_loss:
			self.best_loss = current_loss
			self.wait = 0
			self.current_reduce_nb = 0
			#lr = K.get_value(self.model.optimizer.lr)
			#K.set_value(self.model.optimizer.lr, lr/self.reduce_rate)
			if self.verbose > 0:
				print('---current best loss : %.3f' % current_loss)
		else:
			self.current_reduce_nb += 1
			if self.current_reduce_nb <= self.reduce_nb:
				lr = K.get_value(self.model.optimizer.lr)
				print str(self.current_reduce_nb)+'try fail, lr='+str(lr)
				#K.set_value(self.model.optimizer.lr, lr*self.reduce_rate)
			else:
				self.current_reduce_nb =0
				self.wait += 1
				if self.wait >= self.patience:
					self.model.stop_training = True
					self.stop_training = True
					if self.verbose > 0:
						print("Epoch %d: early stopping" % (epoch))
				else:
					lr = K.get_value(self.model.optimizer.lr)
					K.set_value(self.model.optimizer.lr, lr*self.reduce_rate)
	



