from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
import numpy as np
import genregiontruth
import detregionloss
import os
import sys
import customcallback
from keras.optimizers import SGD

train_data = np.load(open('bottleneck_features_train.npy'))

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1225, activation='linear'))

if os.path.isfile('detbox.h5'):
	print 'Load pretrained model...'
	model=load_model('detbox.h5', custom_objects={'yololoss': detregionloss.yololoss})

showbox = customcallback.ShowBox('test1.txt', train_data[0], 448,448,3)

sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9)
model.compile(optimizer='rmsprop',
              loss=detregionloss.yololoss, 
              metrics=['accuracy'])

(orgX_train, train_labels) = genregiontruth.load_data('trainlist.txt', 448, 448, 3, truthonly=True)

numberofsamples = int(sys.argv[1])
print 'numberofsamples='+str(numberofsamples)

model.fit(train_data[0:numberofsamples], train_labels[0:numberofsamples],  callbacks=[showbox],
          nb_epoch=300, batch_size=2)
          #validation_data=(validation_data, validation_labels))

model.save('detbox.h5')
