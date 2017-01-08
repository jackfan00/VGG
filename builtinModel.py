from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BNOR
from keras.regularizers import l2

def add_regionDetect(model, truthtableszie):
        # add box region network
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name='dense_det1'))
        model.add(Dense(4096, activation='relu', name='dense_det2'))
        model.add(Dropout(0.5))
        model.add(Dense(truthtableszie, activation='linear', name='dense_truthtable')) #side=5
	return model

def vgg16_model(inputshape):
        model = Sequential()
# block1
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=inputshape, name='block1_conv1'))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block1_pool' ))
# block 2
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block2_pool' ))
# block 3
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block3_pool' ))
# block 4
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block4_pool' ))
# block 5
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block5_pool' ))

	#model.load_weights('~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        return model

def test1_model(inputshape):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=inputshape, name='conv1'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool1' ))
	model.add(BNOR(name='bnor1'))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool2' ))
	model.add(BNOR(name='bnor2'))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool3' ))
	model.add(BNOR(name='bnor3'))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='conv4'))
        #model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool4' ))
	model.add(BNOR(name='bnor4'))
        #model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv5'))
	#model.add(BNOR(name='bnor5'))
        #model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv6'))
	#model.add(BNOR(name='bnor6'))

        return model

def test_model(inputshape):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=inputshape, name='conv1'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool1' ))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool2' ))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool3' ))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='conv4'))

        return model


def yolotiny_model(inputshape):
	model = Sequential()
	model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), input_shape=inputshape, name='conv1'))
	model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool1' ))
	model.add(BNOR(name='bnor1'))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool2' ))
	model.add(BNOR(name='bnor2'))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv3'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool3' ))
	model.add(BNOR(name='bnor3'))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv4'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool4' ))
	model.add(BNOR(name='bnor4'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv5'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool5' ))
	model.add(BNOR(name='bnor5'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv6'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool6' ))
	model.add(BNOR(name='bnor6'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv7'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv8'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0), name='conv9'))

	return model

def yolosmall_model(inputshape):
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=inputshape, name='conv1'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool1' ))
        model.add(Convolution2D(192, 3, 3, activation='relu', border_mode='same', name='conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool2' ))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv3'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv4'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv5'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv6'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool6' ))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv7'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv8'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv9'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv10'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv11'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv12'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv13'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv14'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv15'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv16'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool16' ))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv17'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv18'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv19'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv20'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv21'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv22'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv23'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv24'))

        return model


def yolo_model(inputshape):
	model = Sequential()
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=inputshape, name='conv1'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool1' ))
        model.add(Convolution2D(192, 3, 3, activation='relu', border_mode='same', name='conv2'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool2' ))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv3'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv4'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv5'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv6'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool6' ))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv7'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv8'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv9'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv10'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv11'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv12'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv13'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv14'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv15'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv16'))
        model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2),name='maxpool16' ))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv17'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv18'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv19'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv20'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv21'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv22'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv23'))
        model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv24'))

	return model

