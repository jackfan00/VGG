from PIL import Image, ImageDraw
import scipy.misc

import numpy as np
import sys
import cfgconst
#from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import random 

class regionbox():
	def __init__(self):
		self

# convert gray to RGB
def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


# img value is 0~255
def random_distort_image(img, contrast=1.0, brightness=1.0):
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


def crop_image(img_path, outw, outh):
	jitter = 0.1
	#img = Image.open(img_path.strip())
	ckimg = scipy.misc.imread(img_path.strip())
	try:
		(orgh,orgw,c) = ckimg.shape
		img = Image.open(img_path.strip())
	except:
		rgbimg = to_rgb2(ckimg)
		img = Image.fromarray(rgbimg) # update img obj
		(orgh,orgw,c) = rgbimg.shape
		#print 'img shape err='+img_path.strip()+',shape='+str(ckimg.shape)
		#return -1,-1,-1,-1,-1,-1,-1,-1
	if c !=3:
		print 'img shape err='+img_path.strip()+',c='+str(c)
		return -1,-1,-1,-1,-1,-1,-1,-1
	#
	dw = int(orgw * jitter)
	dh = int(orgh * jitter)
	pleft = int(np.random.uniform(-dw,dw))
	pright = int(np.random.uniform(-dw,dw))
	ptop = int(np.random.uniform(-dh,dh))
	pbot = int(np.random.uniform(-dh,dh))
	swidth = orgw - pleft - pright
	sheight = orgh - ptop - pbot
	sx = float(swidth) / orgw
	sy = float(sheight) / orgh
	dx = float(pleft) / swidth
	dy = float(ptop) / sheight
	flip = int(random.random()*2)
	asratio = int(random.random()*3)
	#print 'dw='+str(dw)+',dh='+str(dh)+',pleft='+str(pleft)+',pright='+str(pright)+',ptop='+str(ptop)+',pbot='+str(pbot)+',swidth='+str(swidth)+',sheight='+str(sheight)
	# crop, 0 for ouside image
	cropped = img.crop((pleft, ptop, orgw-pright, orgh-pbot))
	#scipy.misc.imsave('debug_cropped.jpg', cropped)
	# resize
	ssy =1.0
	ssx =1.0
	if asratio>=1:  # maintain aspect ratio
		r0 = float(outw)/swidth
		r1 = float(outh)/sheight
		if asratio==1:
			r = r1
			ssx = (r*swidth) / outw

		else:
			r = r0
			ssy = (r*sheight) / outh
		as_resized = cropped.resize( (int(r*swidth), int(r*sheight)), Image.BILINEAR )
		resized = as_resized.crop( (0,0,outw,outh))
		#
	else:
		resized = cropped.resize( (outw, outh), Image.BILINEAR )
	#scipy.misc.imsave('debug_resized.jpg', resized)
	# flip
	if flip ==1:
		resized = resized.transpose( Image.FLIP_LEFT_RIGHT )
		#scipy.misc.imsave('debug_flip.jpg', resized)
	# disort
	if int(random.random()*3)>1:
		disorted = random_distort_image(np.asarray(resized))
	else:
		disorted = np.asarray(resized, dtype=np.float32)
	#scipy.misc.imsave('debug_disorted.jpg', disorted)

	return disorted, sx, sy, dx, dy, flip, ssx, ssy


def readlabel(fn, sx, sy, dx, dy, flip, ssx, ssy):
	#print 'readlabel '+ fn
	boxlist = []
	f = open(fn)
	for l in f:
		try:
			ss= l.strip().split(' ')
			#print ss
			box = regionbox()
			box.id = int(ss[0])
			box.orgx = float(ss[1]) 
			box.orgy = float(ss[2]) 
			box.orgw = float(ss[3]) 
			box.orgh = float(ss[4]) 
			#
			# ignore small block
			if box.orgw < 0.1 or box.orgh < 0.1:
				continue
			#
			box.x = (box.orgx / sx - dx) * ssx
			if flip ==1:
				box.x = 1.0 - box.x
			box.y = (box.orgy / sy - dy) * ssy
			box.w = (box.orgw / sx) * ssx 
			box.h = (box.orgh / sy) * ssy 
			#print 'sx='+str(sx)+',sy='+str(sy)+',dx='+str(dx)+',dy='+str(dy)+',flip='+str(flip)
                        #print 'box.orgx='+str(box.orgx)+',box.orgy='+str(box.orgy)+',box.orgw='+str(box.orgw)+',box.orgh='+str(box.orgh)
                        #print 'box.x='+str(box.x)+',box.y='+str(box.y)+',box.w='+str(box.w)+',box.h='+str(box.h)
			# consider out of image
			left = max(0.001,box.x - box.w/2.0)
			right = min(0.999,box.x + box.w/2.0)
			top = max(0.001,box.y - box.h/2.0)
			bot = min(0.999,box.y + box.h/2.0)

			box.x = (left+right)/2.0 
                        box.y = (top+bot)/2.0 
                        box.w = right-left 
                        box.h = bot - top 

			# constraint
			box.x = min(0.999, max(0.001, box.x))
			box.y = min(0.999, max(0.001, box.y))
			box.w = min(0.999, max(0.001, box.w))
			box.h = min(0.999, max(0.001, box.h))
			#print 'sx='+str(sx)+',sy='+str(sy)+',dx='+str(dx)+',dy='+str(dy)+',flip='+str(flip)
			#print 'box.orgx='+str(box.orgx)+',box.orgy='+str(box.orgy)+',box.orgw='+str(box.orgw)+',box.orgh='+str(box.orgh)
			#print 'box.x='+str(box.x)+',box.y='+str(box.y)+',box.w='+str(box.w)+',box.h='+str(box.h)
		except:
			box.id = -1
		boxlist.append(box)
	return boxlist
		
def load_img_paths(train_images):
	f = open(train_images)
	paths = []
	for l in f:
		paths.append(l)

	return paths

def load_data(paths, h, w, c,numberofsamples, truthonly=False, batch_index=0, batch_size=1, train_on_batch=False ):
        #if not train_on_batch:
        #        print 'Loading train data:'+train_images+'...'

	# randomize file list
	#if randomize:
	#	random.seed(seed) 
	#	random.shuffle(paths)


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
		sx =1.0
		sy =1.0
		dx =0.0
		dy =0.0
		flip =0
		ssx = 1.0
		ssy = 1.0
		if not truthonly:
			xx,sx,sy,dx,dy,flip,ssx,ssy = crop_image(fn.strip(), w, h)
			if flip ==-1:  # invalid img
				batch_count = batch_count -1
				continue
			#img = image.load_img( fn.strip(),  target_size=(w, h))
			#xx = image.img_to_array(img)
			#xx = randompixel(xx)
			#xx = preprocess_input(xx)
			#(orgh,orgw) = img.size
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
		boxlist = readlabel(fn.strip(), sx,sy,dx,dy,flip,ssx,ssy)
		
		truth = np.zeros(side**2*(bckptsPercell+classes)*bnumPercell)
		for box in boxlist:
			if box.id == -1:
				print 'read bbox fail'
				continue


			#
			# let truth size == pred size, different from yolo.c 
			# trurh data arrangement is (confid,x,y,w,h)(..)(classes)
			#
			#truth = np.zeros(side**2*(bckptsPercell*bnumPercell+classes))
			col = int(box.x * side)
			row = int(box.y * side)
			x = box.x * side - col
			y = box.y * side - row

			# only 1 box for 1 cell
			#for i in range(bnumPercell):
			index = (col+row*side)
			truth[index] = 1
			truth[side**2+index] = x
			truth[2*(side**2)+index] = y
			truth[3*(side**2)+index] = box.w
			truth[4*(side**2)+index] = box.h
			#print 'index='+str(index)+' '+str(box.x)+' '+str(box.y)+' '+str(box.w)+' '+str(box.h)
			truth[(5+box.id)*(side**2)+index] =1

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
		


