import numpy as np
from keras.applications.vgg16 import VGG16
import sys
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input

#VGG network
#input_1 (None, 448, 448, 3) -> (None, 448, 448, 3)
#block1_conv1 (None, 448, 448, 3) -> (None, 448, 448, 64)
#block1_conv2 (None, 448, 448, 64) -> (None, 448, 448, 64)
#block1_pool (None, 448, 448, 64) -> (None, 224, 224, 64)
#block2_conv1 (None, 224, 224, 64) -> (None, 224, 224, 128)
#block2_conv2 (None, 224, 224, 128) -> (None, 224, 224, 128)
#block2_pool (None, 224, 224, 128) -> (None, 112, 112, 128)
#block3_conv1 (None, 112, 112, 128) -> (None, 112, 112, 256)
#block3_conv2 (None, 112, 112, 256) -> (None, 112, 112, 256)
#block3_conv3 (None, 112, 112, 256) -> (None, 112, 112, 256)
#block3_pool (None, 112, 112, 256) -> (None, 56, 56, 256)
#block4_conv1 (None, 56, 56, 256) -> (None, 56, 56, 512)
#block4_conv2 (None, 56, 56, 512) -> (None, 56, 56, 512)
#block4_conv3 (None, 56, 56, 512) -> (None, 56, 56, 512)
#block4_pool (None, 56, 56, 512) -> (None, 28, 28, 512)
#block5_conv1 (None, 28, 28, 512) -> (None, 28, 28, 512)
#block5_conv2 (None, 28, 28, 512) -> (None, 28, 28, 512)
#block5_conv3 (None, 28, 28, 512) -> (None, 28, 28, 512)
#block5_pool (None, 28, 28, 512) -> (None, 14, 14, 512)

print 'Usage: python genfeature.py trainlist.txt'

input_tensor = Input(shape=(448, 448, 3))
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False )

x_train = []
f = open(sys.argv[1])
for img_path in f:
	img = image.load_img(img_path.strip(), target_size=(448, 448))
	x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)
	x_train.append(x)

bottleneck_features_train = model.predict(preprocess_input(np.asarray(x_train)))

print bottleneck_features_train.shape

np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
