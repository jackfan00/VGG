from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
import numpy as np
import random
import genregiontruth_bnum
import detregionloss_bnum
import utils
import sys
import os
import customcallback
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import cfgconst
import builtinModel
import statusSever_socket
import SocketServer

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

if len(sys.argv) < 2:
	print 'Command error --'
        print 'Usage:: python main_bnum.py train [pretrined.h5]'
        print 'Usage:: python main_bnum.py train_on_batch [pretrined.h5]'
        print 'Usage:: python main_bnum.py testonefile pretrined.h5 xxx.jpg '
        print 'Usage:: python main_bnum.py testfile pretrined.h5 '
        print 'Usage:: python main_bnum.py testvideo pretrined.h5 '
        exit()

# here change to network
model = builtinModel.add_regionDetect(builtinModel.yolotiny_model((448, 448, 3)), (cfgconst.side**2)*(cfgconst.classes+5)*cfgconst.bnum)
for l in model.layers:
	print l.name+' '+str(l.input_shape)+' -> '+str(l.output_shape)+', trainable:'+str(l.trainable)


if len(sys.argv)>2 and os.path.isfile(sys.argv[2]):
	print 'Load pretrained model:'+sys.argv[2]+'....'
	#model=load_model(sys.argv[4], custom_objects={'regionloss': detregionloss_bnum.regionloss})
	model.load_weights(sys.argv[2], by_name=True)
	print '----load weight done!'


sgd = SGD(lr=cfgconst.lr, decay=0, momentum=0.9)
model.compile(optimizer=sgd, loss=detregionloss_bnum.regionloss, metrics=[detregionloss_bnum.regionmetrics])
#model.compile(optimizer='rmsprop', loss=detregionloss_bnum.regionloss, metrics=[detregionloss_bnum.regionmetrics])
#
#

thresh_option = cfgconst.confid_thresh

#for i in range(len(sys.argv)):
#        if sys.argv[i] == '-thresh':
#                thresh_option = float(sys.argv[i+1])
#                break
#
nb_epoch =cfgconst.nb_epoch
batch_size =cfgconst.batch_size
DEBUG_IMG = cfgconst.debugimg

history = customcallback.LossHistory(imagefordebug=cfgconst.imagefordebugtrain, thresh_option=thresh_option)
history.setmodel(model)
adaptive_lr = customcallback.LrReducer(patience=cfgconst.patience, reduce_rate=cfgconst.lr_reduce_rate, reduce_nb=cfgconst.lr_reduce_nb, verbose=1)
adaptive_lr.setmodel(model)


if sys.argv[1]=='train':
	#if len(sys.argv)>3:
	#	numberofsamples = int(sys.argv[3])
	#else:
	#	numberofsamples = 100000  
	

	train_img_paths = genregiontruth_bnum.load_img_paths(cfgconst.trainset) #sys.argv[2])
	(train_data, train_labels) = genregiontruth_bnum.load_data(train_img_paths, 448, 448, 3, cfgconst.numberof_train_samples, randomize=(cfgconst.randomize==1))
	print '----load data done!'
	#exit()

	numberofsamples = train_labels.shape[0]


	#
	#if len(sys.argv)>5:
	#	if int(sys.argv[5]) ==1:
	#		DEBUG_IMG = True

	if DEBUG_IMG==1:
		model.fit(train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[history] )
	else:
		model.fit(train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[adaptive_lr] )
	#
	#for e in range(nb_epoch):
	#	ran_train_data = genregiontruth_bnum.randompixel(train_data)
	#	if DEBUG_IMG:
	#		model.fit(ran_train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=1, batch_size=batch_size, callbacks=[history, adaptive_lr])
	#	else:
	#		model.fit(ran_train_data[0:numberofsamples], train_labels[0:numberofsamples],nb_epoch=1, batch_size=batch_size, callbacks=[adaptive_lr] )
	#	if adaptive_lr.istrainstop():
	#		break

	#model.save_weights('vggregion_finetune_weight.h5')

# for prevent load all train data once from memory shortage
elif sys.argv[1]=='train_on_batch':
	#if len(sys.argv)>3:
        #        numberofsamples = int(sys.argv[3])
        #else:
        #        numberofsamples = 100000
	numberofsamples = cfgconst.numberof_train_samples

	#adaptive_lr = customcallback.LrReducer(patience=10, reduce_rate=0.2, reduce_nb=3, verbose=1)
	#adaptive_lr.setmodel(model)

	batch_count =0
	seed = 0
	train_img_paths = genregiontruth_bnum.load_img_paths(cfgconst.trainset) #sys.argv[2])
	val_img_paths = genregiontruth_bnum.load_img_paths(cfgconst.valset) #'2007_test.txt')
	#
	for e in range(nb_epoch):
		print 'epoch='+str(e+1)+'/'+str(nb_epoch)
		seed = seed + 1
		batch_index =0
		randomize = (cfgconst.randomize==1)  # to make sure same random in 1 epoch, add seed parameter
		#
		if randomize and numberofsamples > len(train_img_paths):
			random.seed(seed)
			random.shuffle(train_img_paths)
		#

		epoch_loss =0
		ave_train_result =[]
		for i in range(len(model.metrics_names)):
			ave_train_result.append(0)
		# 
		while (True):
			#
			if numberofsamples > (batch_size*(batch_index+1)):
				load_numberofsamples = batch_size
			elif numberofsamples == (batch_size*batch_index):
				break
			else:
				load_numberofsamples = numberofsamples - batch_size*(batch_index)
			#
			(train_data, train_labels) = genregiontruth_bnum.load_data(train_img_paths, 448, 448, 3, numberofsamples=load_numberofsamples, batch_index=batch_index, batch_size=batch_size, train_on_batch=True, randomize=randomize )
			train_result = model.train_on_batch(train_data, train_labels)
			epoch_loss += train_result[0]
			#
			sys.stdout.write("\r%04d " %(batch_index)+'   epochloss:'+"%0.4f" %(epoch_loss)+' ')
			for i in range(len(train_result)):
				sys.stdout.write('    '+model.metrics_names[i]+':'+"%0.4f" %(train_result[i])+' ')
				ave_train_result[i] += train_result[i]
			sys.stdout.write("\n")

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
		sys.stdout.write("\n")
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
                        if numberofsamples > (batch_size*(batch_index+1)):
                                load_numberofsamples = batch_size
			elif numberofsamples == (batch_size*batch_index):
				break
                        else:
                                load_numberofsamples = numberofsamples - batch_size*(batch_index)
                        #
                        (test_data, test_labels) = genregiontruth_bnum.load_data(val_img_paths, 448, 448, 3, numberofsamples=load_numberofsamples, batch_index=batch_index, batch_size=batch_size, train_on_batch=True )
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

		if adaptive_lr.istrainstop() and DEBUG_IMG==0:
			break

	model.save_weights('vggregion_finetune_weight.h5')


elif sys.argv[1]=='testonefile':
	if len(sys.argv) <4:
		print 'testfile command is not correct:: python main.py testonefile pretrained.h5 xxx.jpg '
		exit()
	utils.testonefile(model, img_path=sys.argv[3], confid_thresh=thresh_option, fordebug=False)

elif sys.argv[1]=='testfile':
	if len(sys.argv) <3:
		print 'testfile command is not correct:: python main.py testfile pretrained.h5 '
		exit()
	utils.testfile(model, imglist_path=cfgconst.testfile, confid_thresh=thresh_option, fordebug=True)
elif sys.argv[1]=='testvideo':
	if len(sys.argv) <3:
		print 'testvideo command is not correct:: python main.py testvideo pretrained.h5 '
		exit()
	utils.testvideo(model, videofile=cfgconst.videofile, confid_thresh=thresh_option)
elif sys.argv[1]=='testsocketvideo':
	if len(sys.argv) <3:
                print 'testvideo command is not correct:: python main.py testsocketvideo pretrained.h5 '
                exit()

	MyTCPHandler = statusSever_socket.MyTCPHandler
	MyTCPHandler.testmodel = model
	MyTCPHandler.confid_thresh = thresh_option
	HOST, PORT = "localhost", 9999
	server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
	server.serve_forever()
else:
	print 'unsupported command option:'+sys.argv[1]
