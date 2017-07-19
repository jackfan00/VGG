import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2
import scipy.misc
import cfgconst
import math
import genregiontruth_bnum
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def iou(boxlist, x0_list, y0_list, x1_list, y1_list, w, h):
	ioulist = []
	for box in boxlist:
		maxv =0
		for x0,y0,x1,y1 in zip(x0_list, y0_list, x1_list, y1_list):
			box_x0 = (box.x - box.w/2)*w
			box_x1 = (box.x + box.w/2)*w
			box_y0 = (box.y - box.h/2)*h
			box_y1 = (box.y + box.h/2)*h
			ow = min(x1,box_x1) - max(x0,box_x0)
			oh = min(y1,box_y1) - max(y0,box_y0)
			intersec = max(0,ow) * max(0,oh)
 
			union = box.w*box.h*(w*h) + (x1-x0)*(y1-y0) - intersec
			v = intersec / union 
			if v > maxv:
				maxv =v
		ioulist.append(maxv)
	return ioulist

def limit(x):
	if x > 100:
		y= 100
	elif x <-100:
		y=-100
	else:
		y=x
	return y

def sigmoid(x):
	#print 'x='+str(x)
	if x<-100:
		y = -100
	else:
		y = x
	return 1 / (1 + math.exp(-y))

def predict(X_test, testmodel, confid_thresh,w,h,c):
	#print 'predict, confid_thresh='+str(confid_thresh)
	
	pred = testmodel.predict(X_test)
	#(s,w,h,c) = testmodel.layers[0].input_shape
	
	# find confidence value > 0.5
	confid_index_list =[]
	confid_value_list =[]
	x_value_list = []
	y_value_list =[]
	w_value_list =[]
	h_value_list =[]
	class_id_list =[]
	classprob_list =[]
	x0_list = []
	x1_list = []
	y0_list = []
	y1_list = []
	#
	bnum = cfgconst.bnum
        side = cfgconst.side
	classes = cfgconst.classes
	xtext_index =0
	foundindex = False
	max_confid =0
	#
	for p in pred:
		#foundindex = False
		for k in range(bnum): #5+classes):
			#print 'L'+str(k)
			for i in range(side):
				for j in range(side):
					#if k==0:
					max_confid = max(max_confid,p[k*(side**2)+i*side+j])

					#sys.stdout.write( str(sigmoid(p[k*(side**2)+i*side+j]))+', ' )
					#if k==0 and sigmoid(p[k*(side**2)+i*side+j])>confid_thresh:
					if sigmoid(p[k*(side**2)+i*side+j])>confid_thresh:
						confid_index_list.append(k*(side**2)+i*side+j)
						foundindex = True
				#print '-'
		#print 'max_confid='+str(max_confid)
		#
		for confid_index in confid_index_list:
			confid_value = max(0,sigmoid(limit(p[0*(side**2)*bnum+confid_index])))
			confid_value_list.append(confid_value)

			x_value = max(0,sigmoid(limit(p[1*(side**2)*bnum+confid_index])))
			y_value = max(0,sigmoid(limit(p[2*(side**2)*bnum+confid_index])))
			w_value = max(0,sigmoid(limit(p[3*(side**2)*bnum+confid_index])))
			h_value = max(0,sigmoid(limit(p[4*(side**2)*bnum+confid_index])))
			#print 'x_value='+str(x_value)+',y_value='+str(y_value)+',w_value='+str(w_value)+',h_value='+str(h_value)
			maxclassprob = 0
			maxclassprob_i =-1
			for i in range(classes):
				if p[(5+i)*(side**2)*bnum+confid_index] > maxclassprob and foundindex:
					maxclassprob = p[(5+i)*(side**2)*bnum+confid_index]
					maxclassprob_i = i

			classprob_list.append( sigmoid(maxclassprob))
			class_id_list.append( maxclassprob_i)

			#print 'max_confid='+str(max_confid)+',c='+str(confid_value)+',x='+str(x_value)+',y='+str(y_value)+',w='+str(w_value)+',h='+str(h_value)+',cid='+str(maxclassprob_i)+',prob='+str(maxclassprob)
		#
			# in case confid_index at 2nd,3rd,4th... bbox
			row = (confid_index % (side**2)) / side
			col = (confid_index % (side**2)) % side
			x = (w / side) * (col + x_value)
			y = (h / side) * (row + y_value)

			#print 'max_confid='+str(max_confid)+',c='+str(confid_value)+',x='+str(x_value)+',y='+str(y_value)+',w='+str(w_value)+',h='+str(h_value)+',cid='+str(maxclassprob_i)+',prob='+str(maxclassprob)+',row='+str(row)+',col='+str(col)
			#print 'confid_index='+str(confid_index)+',x='+str(x)+',y='+str(y)+',row='+str(row)+',col='+str(col)

		#draw = ImageDraw.Draw(nim)
		#draw.rectangle([x-(w_value/2)*w,y-(h_value/2)*h,x+(w_value/2)*w,y+(h_value/2)*h])
		#del draw
		#nim.save('predbox.png')
		
		#sourceimage = X_test[xtext_index].copy()

			x0_list.append( max(0, int(x-(w_value/2)*w)) )
			y0_list.append( max(0, int(y-(h_value/2)*h)) )
			x1_list.append( min(w, int(x+(w_value/2)*w)) )
			y1_list.append( min(h, int(y+(h_value/2)*h)) )
		
		break
		#xtext_index = xtext_index + 1

	#print pred
	sourceimage = X_test[0].copy()
	return sourceimage, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list


def testfeature(testmodel, img,w,h,c, onefeature, confid_thresh=0.2, waittime=1000):
# predict
	tttimg, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list = predict(np.asarray([onefeature]), testmodel, confid_thresh,w,h,c)
# 
	for x0,y0,x1,y1,classprob,class_id,confid_value in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list):
		#
		if x0 >w or x1>w or y0>h or y1>h :
			continue
		#print 'box='+str(x0)+','+str(y0)+','+str(x1)+','+str(y1)
		# draw bounding box
		cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
		# draw classimg
		classimg = cv2.imread(cfgconst.label_names[class_id])
		if y0-classimg.shape[0] <= 0:
			yst =0
			yend =classimg.shape[0]
		elif y0 >= img.shape[0]:
			yst = img.shape[0]-classimg.shape[0]-1
			yend = img.shape[0]-1
		else:
			yst = y0 - classimg.shape[0]
			yend = y0
		#
		if x0+classimg.shape[1] >= img.shape[1]:
			xst = img.shape[1]-classimg.shape[1]-1
			xend = img.shape[1]-1
		elif x0 <=0:
			xst = 0
			xend = classimg.shape[1]
		else:
			xst = x0
			xend = x0+classimg.shape[1]
		#

		img[yst:yend, xst:xend] = classimg
		# draw text
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(classprob), (x0,y0+classimg.shape[0]-1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
		cv2.putText(img, str(confid_value), (x0,y1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
		#
		cv2.imshow('prediction',img)
		if cv2.waitKey(waittime):
			if 0xFF == ord('q'):
				break
			else:
				continue

def testonefile(testmodel, img_path, confid_thresh=0.3, fordebug=False ):
       	(s,w,h,c) = testmodel.layers[0].input_shape
	fimg, sx, sy, dx, dy, flip,ssx,ssy = genregiontruth_bnum.crop_image(img_path.strip(), w, h, randomize=False)
	xx = fimg.copy()
	img = fimg.astype(float)
	if fordebug:  # read label
		fn=img_path.replace("/JPEGImages/","/labels/")
		fn=fn.replace(".jpg",".txt")              #VOC
		boxlist = genregiontruth_bnum.readlabel(fn.strip(), sx, sy, dx, dy, flip, ssx, ssy)
		for box in boxlist:
			draw.rectangle([(box.x-box.w/2)*w,(box.y-box.h/2)*h,(box.x+box.w/2)*w,(box.y+box.h/2)*h])
	#
	ttimg, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list = predict(preprocess_input(np.asarray([xx])), testmodel, confid_thresh,w,h,c)

	iimg = Image.fromarray(img.astype(np.uint8))
	draw = ImageDraw.Draw(iimg, 'RGBA')

	sortedindexlist = np.argsort(confid_value_list)
	colors=[]
	for i in range(3):
		for j in range(7):
			if i==0:
				rcolor = (j+1)*32
				gcolor = 0
				bcolor = 0
			elif i==1:
				rcolor = 0
				gcolor = (j+1)*32
				bcolor = 0
			else:
				rcolor = 0
				gcolor = 0
				bcolor = (j+1)*32
			colors.append( (rcolor, gcolor, bcolor, 127) )
	#print colors
		
	for i in range(len(confid_value_list)):
		index = sortedindexlist[len(confid_value_list)-i-1]
		for k in range(5): # thick line
			draw.rectangle([x0_list[index]+k,y0_list[index]+k,x1_list[index]-k,y1_list[index]-k], outline=colors[class_id_list[index]])

		labelim = Image.open(cfgconst.label_names[class_id_list[index]])
		draw.bitmap((x0_list[index],y0_list[index]),labelim)

		x = (x0_list[index]+x1_list[index])/2.
		y = (y0_list[index]+y1_list[index])/2.
		x0 = int(x/w*cfgconst.side)*w/cfgconst.side
		y0 = int(y/h*cfgconst.side)*h/cfgconst.side
		x1 = x0 + float(w)/cfgconst.side
		y1 = y0 + float(h)/cfgconst.side
		draw.rectangle([x0,y0,x1,y1], fill=colors[class_id_list[index]] )
		print cfgconst.label_names[class_id_list[index]].split('/')[1].split('.')[0]+': '+str(confid_value_list[index])
	del draw
	iimg.save('predicted.png')
	


def testfile(testmodel, imglist_path, confid_thresh=0.2, waittime=50000, fordebug=False ):
        #print 'testfile: '+imglist_path
        # custom objective function
        #print (s,w,h,c)
        #exit()
	randomize= (cfgconst.randomize==1)
	if os.path.isfile(imglist_path):
        	#testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
        	(s,w,h,c) = testmodel.layers[0].input_shape
		f = open(imglist_path)
		for img_path in f:
		#
			#if fordebug:  # read label
			#	fn=img_path.replace("/JPEGImages/","/labels/")
			#	fn=fn.replace(".jpg",".txt")              #VOC
			#	boxlist = genregiontruth_bnum.readlabel(fn.strip())

        		#X_test = []
        		if os.path.isfile(img_path.strip()):
                		#frame = Image.open(img_path.strip())
                		#(orgw,orgh) = img.size
				#nim = scipy.misc.imresize(frame, (w, h, c))
				#if nim.shape != (w, h, c):
				#	continue
				#img = nim

				#timg = image.load_img( img_path.strip(),  target_size=(w, h))
				#xx = image.img_to_array(timg)

				#
				fimg, sx, sy, dx, dy, flip,ssx,ssy = genregiontruth_bnum.crop_image(img_path.strip(), w, h, randomize=randomize)
				if flip == -1:  # not rgb color image
					continue
				xx = fimg.copy()
				img = fimg.astype(float)
				if fordebug:  # read label
					fn=img_path.replace("/JPEGImages/","/labels/")
					fn=fn.replace(".jpg",".txt")              #VOC
					boxlist = genregiontruth_bnum.readlabel(fn.strip(), sx, sy, dx, dy, flip, ssx, ssy)

				#print 'img='+str(img.shape)+str(img[0][0])

                		#nim = img.resize( (w, h), Image.BILINEAR )
				ttimg, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list = predict(preprocess_input(np.asarray([xx])), testmodel, confid_thresh,w,h,c)
                		#X_test.append(np.asarray(nim))
        			#predict(np.asarray(X_test), testmodel, confid_thresh)
				# found confid box
				# iou
				font = cv2.FONT_HERSHEY_SIMPLEX
				if fordebug:
					ioulist = iou(boxlist, x0_list, y0_list, x1_list, y1_list, w, h)
					ii = 0
					for iouvalue in ioulist:
						#print 'iouvalue='+str(iouvalue)+',ii='+str(ii)
						cv2.putText(img, "%0.4f" %(iouvalue), (10+100*ii,20), font, 0.4,(128,255,128),1,cv2.LINE_AA)
						ii = ii+ 1

					for box in boxlist:
						x0 = int((box.x - box.w/2)*w)
						x1 = int((box.x + box.w/2)*w)
						y0 = int((box.y - box.h/2)*h)
						y1 = int((box.y + box.h/2)*h)
						cv2.rectangle(img, (x0, y0), (x1, y1), (128,255,128), 2)
						cv2.putText(img, str(box.id), (x0,y0), font, 1,(128,255,128),1,cv2.LINE_AA)
					
				#
                		for x0,y0,x1,y1,classprob,class_id,confid_value in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list):
                		#
					# draw bounding box
					cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
					# draw classimg
					classimg = cv2.imread(cfgconst.label_names[class_id])
					if y0-classimg.shape[0] <= 0:
						yst =0
						yend =classimg.shape[0]
					elif y0 >= img.shape[0]:
						yst = img.shape[0]-classimg.shape[0]-1
						yend = img.shape[0]-1
					else:
						yst = y0 - classimg.shape[0]
						yend = y0
					
					if x0+classimg.shape[1] >= img.shape[1]:
						xst = img.shape[1]-classimg.shape[1]-1
						xend = img.shape[1]-1
					elif x0 <=0:
						xst = 0
						xend = classimg.shape[1]
					else:
						xst = x0
						xend = x0+classimg.shape[1]

					#
					#print 'box='+str(x0)+','+str(y0)+','+str(x1)+','+str(y1)+', '+str(img.shape)+', '+str(classimg.shape)+', '+str(yst)+', '+str(yend)+', '+str(xst)+', '+str(xend)
					#
					img[yst:yend, xst:xend] = classimg
					# draw text
					#font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, str(classprob), (x0,y0+classimg.shape[0]-1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
					cv2.putText(img, str(confid_value), (x0,y1), font, 0.5,(128,255,255),1,cv2.LINE_AA)
					#
				cv2.imshow('prediction',cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
				if cv2.waitKey(waittime):
					if 0xFF == ord('q'):
						break
					else:
						continue


			else:
				print img_path+' predict fail'

			# only show first image
			if fordebug:
				break

		#cv2.destroyAllWindows()
	else:
		print imglist_path+' does not exist'
	


def testvideo(testmodel, videofile, confid_thresh=0.2):
	print 'testdemo '+videofile
	#testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape

	cap = cv2.VideoCapture(videofile)

	while (cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			break
		#print frame
		nim = scipy.misc.imresize(frame, (w, h, c))
		img = nim
		xx = image.img_to_array(cv2.cvtColor(nim, cv2.COLOR_RGB2BGR))

		ttimg, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list = predict(preprocess_input(np.asarray([xx])), testmodel, confid_thresh,w,h,c)
		# found confid box
                for x0,y0,x1,y1,classprob,class_id,confid_value in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list):
		#
			# draw bounding box
			cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
			# draw classimg
                        classimg = cv2.imread(cfgconst.label_names[class_id])
                        if y0-classimg.shape[0] <= 0:
                        	yst =0
                        	yend =classimg.shape[0]
                        elif y0 >= img.shape[0]:
                        	yst = img.shape[0]-classimg.shape[0]-1
                        	yend = img.shape[0]-1
                        else:
                        	yst = y0 - classimg.shape[0]
                        	yend = y0

                        if x0+classimg.shape[1] >= img.shape[1]:
                        	xst = img.shape[1]-classimg.shape[1]-1
                        	xend = img.shape[1]-1
                        elif x0 <=0:
                        	xst = 0
                        	xend = classimg.shape[1]
                        else:
                        	xst = x0
                        	xend = x0+classimg.shape[1]

                        #

			img[yst:yend, xst:xend] = classimg
			# draw text
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img, str(classprob), (x0,y0+classimg.shape[0]-1), font, 0.5,(255,255,255),2,cv2.LINE_AA)
			cv2.putText(img, str(confid_value), (x0,y1), font, 0.5,(128,255,255),1,cv2.LINE_AA)
			#
		cv2.imshow('frame',img)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

