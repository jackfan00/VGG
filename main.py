from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
import numpy as np
import genregiontruth
import detregionloss
import utils
import sys
import os
import customcallback
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import cfgconst
import builtinModel

def VGGregionModel(inputshape):
	input_tensor = Input(shape=inputshape) #(448, 448, 3))
	vgg_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False )

	# add region detection layers
	x = vgg_model.output
	x = Flatten()(x)
	#x = Dense(256, activation='relu')(x)
	#x = Dense(2048, activation='relu')(x)
	#x = Dropout(0.5)(x)
	x = Dense((cfgconst.side**2)*(cfgconst.classes+5)*cfgconst.bnum)(x)

	model = Model(input=vgg_model.input, output=x)
	#
	print 'returned model:'
	index = 0
	for l in model.layers:
		if index <= (18-8):
			l.trainable = False
                #print l.name+' '+str(l.input_shape)+' -> '+str(l.output_shape)+', trainable:'+str(l.trainable)
		index = index + 1

	return model

#
# pretrained
#model = VGGregionModel((448, 448, 3) )

#
model = builtinModel.add_regionDetect(builtinModel.yolotiny_model((448, 448, 3)), (cfgconst.side**2)*(cfgconst.classes+5)*cfgconst.bnum)
for l in model.layers:
	print l.name+' '+str(l.input_shape)+' -> '+str(l.output_shape)+', trainable:'+str(l.trainable)


if len(sys.argv)>4 and os.path.isfile(sys.argv[4]):
	print 'Load pretrained model:'+sys.argv[4]+'....'
	#model=load_model(sys.argv[4], custom_objects={'regionloss': detregionloss.regionloss})
	model.load_weights(sys.argv[4], by_name=True)
	print 'done!'


sgd = SGD(lr=0.1, decay=0, momentum=0.9)
model.compile(optimizer=sgd, loss=detregionloss.regionloss, metrics=[detregionloss.regionmetrics])
#model.compile(optimizer='rmsprop', loss=detregionloss.regionloss, metrics=[detregionloss.regionmetrics])
#
#
if len(sys.argv) < 2:
	print 'Usage:: python main.py train trainlist numberofsamples pretrined.h5 [1/0]'
	print 'Usage:: python main.py train_on_batch trainlist numberofsamples pretrined.h5'
	print 'Usage:: python main.py testfile testlist thresh pretrined.h5'
	print 'Usage:: python main.py testvideo videofile thresh pretrined.h5'
	exit()


nb_epoch =100
batch_size =64
DEBUG_IMG = False

history = customcallback.LossHistory(sys.argv[2])
history.setmodel(model)
adaptive_lr = customcallback.LrReducer(patience=10, reduce_rate=0.5, reduce_nb=3, verbose=1)
adaptive_lr.setmodel(model)

if sys.argv[1]=='train':
	if len(sys.argv)>3:
		numberofsamples = int(sys.argv[3])
	else:
		numberofsamples = 100000  

	train_img_paths = genregiontruth.load_img_paths(sys.argv[2])
	(train_data, train_labels) = genregiontruth.load_data(train_img_paths, 448, 448, 3, numberofsamples)
	print 'done!'
	#exit()

	numberofsamples = train_labels.shape[0]


	#
	if len(sys.argv)>5:
		if int(sys.argv[5]) ==1:
			DEBUG_IMG = True

	if DEBUG_IMG:
		model.fit(train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[adaptive_lr, history] )
	else:
		model.fit(train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[adaptive_lr] )
	#
	#for e in range(nb_epoch):
	#	ran_train_data = genregiontruth.randompixel(train_data)
	#	if DEBUG_IMG:
	#		model.fit(ran_train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=1, batch_size=batch_size, callbacks=[history, adaptive_lr])
	#	else:
	#		model.fit(ran_train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=1, batch_size=batch_size, callbacks=[adaptive_lr] )
	#	if adaptive_lr.istrainstop():
	#		break

	model.save_weights('vggregion_finetune_weight.h5')

# for prevent load all train data once from memory shortage
elif sys.argv[1]=='train_on_batch':
	if len(sys.argv)>3:
                numberofsamples = int(sys.argv[3])
        else:
                numberofsamples = 100000

	#adaptive_lr = customcallback.LrReducer(patience=10, reduce_rate=0.2, reduce_nb=3, verbose=1)
	#adaptive_lr.setmodel(model)

	batch_count =0
	seed = 0
	train_img_paths = genregiontruth.load_img_paths(sys.argv[2])
	val_img_paths = genregiontruth.load_img_paths('2007_test.txt')
	#
	for e in range(nb_epoch):
		print 'epoch='+str(e+1)+'/'+str(nb_epoch)
		seed = seed + 1
		batch_index =0
		randomize = True  # to make sure same random in 1 epoch, add seed parameter
		epoch_loss =0
		ave_train_result =[]
		for i in range(len(model.metrics_names)):
			ave_train_result.append(0)
		# 
		while (True):
			#
			if numberofsamples > (batch_size*batch_index):
				load_numberofsamples = batch_size
			elif numberofsamples == (batch_size*batch_index):
				break
			else:
				load_numberofsamples = numberofsamples - batch_size*(batch_index-1)
			#
			(train_data, train_labels) = genregiontruth.load_data(train_img_paths, 448, 448, 3, numberofsamples=load_numberofsamples, batch_index=batch_index, batch_size=batch_size, train_on_batch=True, randomize=randomize, seed=seed)
			train_result = model.train_on_batch(train_data, train_labels)
			epoch_loss += train_result[0]
			#
			sys.stdout.write("\r%04d " %(batch_index)+'   epochloss:'+"%0.4f" %(epoch_loss)+' ')
			for i in range(len(train_result)):
				sys.stdout.write('    '+model.metrics_names[i]+':'+"%0.4f" %(train_result[i])+' ')
				ave_train_result[i] += train_result[i]

			sys.stdout.flush()
			#
			batch_index = batch_index+1
			batch_count = batch_count+1
			if DEBUG_IMG:
				history.on_batch_end(batch_count)
			if len(train_data) < batch_size:  # end of train data
				break
		#
		sys.stdout.write("\r%04d " %(batch_index)+'   epochloss:'+"%0.4f" %(epoch_loss)+' ')
		for i in range(len(model.metrics_names)):
			sys.stdout.write('    '+model.metrics_names[i]+':'+"%0.4f" %(ave_train_result[i]/batch_index)+' ')
		sys.stdout.flush()

		adaptive_lr.on_epoch_end(epoch=e, logs={'loss':epoch_loss})
		#
		# calcu valid loss
		#
		batch_index =0
		epoch_testloss =0
		ave_test_result =[]
                for i in range(len(model.metrics_names)):
                        ave_test_result.append(0)
                #
		if ((e+1) % 20)==0 :
			valtest = True
		else:
			valtest = False
                while (valtest):
                        #
                        if numberofsamples > (batch_size*batch_index):
                                load_numberofsamples = batch_size
			elif numberofsamples == (batch_size*batch_index):
				break
                        else:
                                load_numberofsamples = numberofsamples - batch_size*(batch_index-1)
                        #
                        (test_data, test_labels) = genregiontruth.load_data(val_img_paths, 448, 448, 3, numberofsamples=load_numberofsamples, batch_index=batch_index, batch_size=batch_size, train_on_batch=True, randomize=False, seed=seed)
                        test_result = model.test_on_batch(test_data, test_labels)
                        epoch_testloss += test_result[0]
                        #
                        sys.stdout.write("\r%04d " %(batch_index)+'epochvalloss:'+"%0.4f" %(epoch_testloss)+' ')
                        for i in range(len(test_result)):
                                sys.stdout.write('val_'+model.metrics_names[i]+':'+"%0.4f" %(test_result[i])+' ')
				ave_test_result[i] += test_result[i]
                        sys.stdout.flush()
                        #
                        batch_index = batch_index+1
                        #batch_count = batch_count+1
                        #if DEBUG_IMG:
                        #        history.on_batch_end(batch_count)
                        if len(test_data) < batch_size:  # end of train data
                                break
		#
		if (valtest):
			sys.stdout.write("\r%04d " %(batch_index)+'epochvalloss:'+"%0.4f" %(epoch_testloss)+' ')
			for i in range(len(model.metrics_names)):
				sys.stdout.write('val_'+model.metrics_names[i]+':'+"%0.4f" %(ave_test_result[i]/batch_index)+' ')
				sys.stdout.flush()

			print '-'

		if adaptive_lr.istrainstop():
			break

	model.save_weights('vggregion_finetune_weight.h5')


elif sys.argv[1]=='testfile':
	utils.testfile(model, imglist_path=sys.argv[2], confid_thresh=float(sys.argv[3]), fordebug=True)
elif sys.argv[1]=='testvideo':
	utils.testvideo(model, videofile=sys.argv[2], confid_thresh=float(sys.argv[3]))
