#from PIL import Image, ImageDraw
import numpy as np
import sys
import cfgconst
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import random 

class regionbox():
	def __init__(self):
		self

# img value is 0~255
def randompixel(img, contrast=1.0, brightness=1.0):
	cr_delta = contrast*(random.random()-0.5)/10.
	cg_delta = contrast*(random.random()-0.5)/10.
	cb_delta = contrast*(random.random()-0.5)/10.
	br_delta = brightness*255*(random.random()-0.5)/10.
	bg_delta = brightness*255*(random.random()-0.5)/10.
	bb_delta = brightness*255*(random.random()-0.5)/10.
	result = img * (1+np.asarray([cr_delta,cg_delta,cb_delta])) + np.asarray([br_delta,bg_delta,bb_delta])
	result = np.maximum(0., result)
	result = np.minimum(255., result)
	return result

def readlabel(fn):
	#print 'readlabel '+ fn
	boxlist = []
	f = open(fn)
	for l in f:
		try:
			ss= l.strip().split(' ')
			#print ss
			box = regionbox()
			box.id = int(ss[0])
			box.x = float(ss[1]) 
			box.y = float(ss[2]) 
			box.w = float(ss[3]) 
			box.h = float(ss[4]) 
		except:
			box.id = -1
		boxlist.append(box)
	return boxlist
		
def load_data(train_images, h, w, c,numberofsamples, truthonly=False, batch_index=0, batch_size=1, train_on_batch=False, randomize=False, seed=0):
	if not train_on_batch:
		print 'Loading train data:'+train_images+'...'
	f = open(train_images)
	paths = []
	for l in f:
		paths.append(l)

	# randomize file list
	if randomize:
		random.seed(seed) 
		random.shuffle(paths)


	bckptsPercell = 5
	side = cfgconst.side 
	bnumPercell = cfgconst.bnum
	classes = cfgconst.classes

	X_train = []
	Y_train = []
	count = 1
	batch_start = batch_index*batch_size
	fn_count =0
	batch_count =0
	for fn in paths:
		if train_on_batch:
			if fn_count < batch_start:
				fn_count = fn_count+1
				continue
			elif batch_count >= batch_size:
				break
			else:
				batch_count = batch_count+1

		#print 'load_data fn:'+fn
		if not truthonly:
			img = image.load_img( fn.strip(),  target_size=(w, h))
			xx = image.img_to_array(img)
			xx = randompixel(xx)
			#xx = preprocess_input(xx)
			#(orgw,orgh) = img.size
			#nim = img.resize( (w, h), Image.BILINEAR )
			#data = np.asarray( nim )
			#if data.shape != (w, h, c):
			#	continue
			X_train.append(xx) #data)

		# replace to label path
		fn=fn.replace("/images/","/labels/")
		fn=fn.replace("/JPEGImages/","/labels/")  #VOC
		fn=fn.replace(".JPEG",".txt")
		fn=fn.replace(".jpg",".txt")              #VOC
		#fn=fn.replace(".JPG",".txt")
		#print fn

		#
		# may have multi bounding box for 1 image
		boxlist = readlabel(fn.strip())
		
		truth = np.zeros(side**2*(bckptsPercell+classes)*bnumPercell)
		for box in boxlist:
			if box.id == -1:
				print 'read bbox fail'
				continue


			#
			# let truth size == pred size, different from yolo.c 
			# trurh data arrangement is (confid,x,y,w,h)(..)(classes)
			#
			col = int(box.x * side)
			row = int(box.y * side)
			x = box.x * side - col
			y = box.y * side - row

			# support bnum box for 1 cell
			#
			index = (col+row*side)
			for i in range(bnumPercell):
				truth[index+i*(side**2)] = 1
				truth[1*(side**2)*bnum+index+i*(side**2)] = x
				truth[2*(side**2)*bnum+index+i*(side**2)] = y
				truth[3*(side**2)*bnum+index+i*(side**2)] = box.w
				truth[4*(side**2)*bnum+index+i*(side**2)] = box.h
				truth[(5+box.id)*(side**2)*bnum+index+i*(side**2)] =1

		#exit()
		#
		Y_train.append(truth)

		#print 'draw rect bounding box'
		#draw = ImageDraw.Draw(img)
		#draw.rectangle([(box.x-box.w/2)*orgw,(box.y-box.h/2)*orgh,(box.x+box.w/2)*orgw,(box.y+box.h/2)*orgh])
		#del draw
		#img.save('ttt.png')
		#exit()
		#for k in range(7):
		#	print 'L'+str(k)
		#	for row_cell in range(7):
		#		for col_cell in range(7):
		#			sys.stdout.write( str(truth[k*49+col_cell+row_cell*(7)])+', ' )
		#		print '-'

		#print truth[720:740]
		#exit()
		# this is for debug
		if count > (numberofsamples-1):
			break
		else:
			count = count + 1

	#print len(X_train)
	XX_train = np.asarray(X_train)
	YY_train = np.asarray(Y_train)
	if not train_on_batch:
		print 'XX_train:'+str(XX_train.shape)
		print 'YY_train:'+str(YY_train.shape)
	#np.savetxt("XX.csv", XX_train, delimiter=",")
	#np.savetxt("YY.csv", YY_train, delimiter=",")
	#exit()

	return preprocess_input(XX_train), YY_train
		


