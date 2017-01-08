from keras import backend as K
import tensorflow as tf
import numpy as np
import cfgconst


#
bnum = cfgconst.bnum
side = cfgconst.side
gridcells = cfgconst.side**2
lamda_confid_obj = cfgconst.object_scale
lamda_confid_noobj = cfgconst.noobject_scale
lamda_xy = cfgconst.coord_scale
lamda_wh = cfgconst.coord_scale
reguralar_wh = 0
lamda_class = cfgconst.class_scale
classes = cfgconst.classes

DEBUG_loss = True

# shape is (gridcells,)
def yoloconfidloss(y_true, y_pred, t):
	pobj = K.sigmoid(y_pred)
	lo = K.square(y_true-pobj)
	value_if_true = lamda_confid_obj*(lo)
	value_if_false = lamda_confid_noobj*(lo)
	loss1 = tf.select(t, value_if_true, value_if_false)
	loss = K.mean(loss1) #,axis=0)
	#
	ave_anyobj = K.mean(pobj)
	obj = tf.select(t, pobj, K.zeros_like(y_pred))
	objcount = tf.select(t, K.ones_like(y_pred), K.zeros_like(y_pred))
	ave_obj = K.mean(K.sum(obj, axis=1) / K.sum(objcount, axis=1))
	return loss, ave_anyobj, ave_obj

# shape is (gridcells*2,)
def yoloxyloss(y_true, y_pred, t):
        lo = K.square(y_true-K.sigmoid(y_pred))
        value_if_true = lamda_xy*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.select(t, value_if_true, value_if_false)
	return K.mean(loss1)

# different with YOLO
# shape is (gridcells*2,)
def yolowhloss(y_true, y_pred, t):
        lo = K.square(y_true-K.sigmoid(y_pred))
	# let w,h not too small or large
        #lo = K.square(y_true-y_pred)+reguralar_wh*K.square(0.5-y_pred)
        value_if_true = lamda_wh*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.select(t, value_if_true, value_if_false)
	return K.mean(loss1)

# shape is (gridcells*classes,)
def yoloclassloss(y_true, y_pred, t):
        lo = K.square(y_true-y_pred)
        value_if_true = lamda_class*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.select(t, value_if_true, value_if_false)
	# only extract predicted class value at obj location
	cat = K.sum(tf.select(t, y_pred, K.zeros_like(y_pred)), axis=1)
	# check valid class value
	objsum = K.sum(y_true, axis=1)
	# if objsum > 0.5 , means it contain some valid obj(may be 1,2.. objs)
	isobj = K.greater(objsum, 0.5)
	# only extract class value at obj location
	valid_cat = tf.select(isobj, cat, K.zeros_like(cat))
	# prevent div 0
	ave_cat = tf.select(K.greater(K.sum(objsum),0.5), K.sum(valid_cat) / K.sum(objsum) , -1)
	return K.mean(loss1), ave_cat

def overlap(x1, w1, x2, w2):
        l1 = (x1) - w1/2
        l2 = (x2) - w2/2
        left = tf.select(K.greater(l1,l2), l1, l2)
        r1 = (x1) + w1/2
        r2 = (x2) + w2/2
        right = tf.select(K.greater(r1,r2), r2, r1)
        result = right - left
	return result

def iou(x_true,y_true,w_true,h_true,x_pred,y_pred,w_pred,h_pred,t):
	xoffset = K.cast_to_floatx(np.tile(np.tile(np.arange(side),side),bnum))
	yoffset = K.cast_to_floatx(np.tile(np.repeat(np.arange(side),side),bnum))
	x = tf.select(t, K.sigmoid(x_pred), K.zeros_like(x_pred)) 
	y = tf.select(t, K.sigmoid(y_pred), K.zeros_like(y_pred))
	w = tf.select(t, K.sigmoid(w_pred), K.zeros_like(w_pred))
	h = tf.select(t, K.sigmoid(h_pred), K.zeros_like(h_pred))

	ow = overlap(x+xoffset, w*side, x_true+xoffset, w_true*side)
	oh = overlap(y+yoffset, h*side, y_true+yoffset, h_true*side)
	ow = tf.select(K.greater(ow,0), ow, K.zeros_like(ow))
	oh = tf.select(K.greater(oh,0), oh, K.zeros_like(oh))
	intersection = ow*oh
	union = w*h*(side**2) + w_true*h_true*(side**2) - intersection + K.epsilon()  # prevent div 0

	iouall = intersection / union
	iouall = K.reshape(iouall, (-1, bnum, gridcells))
	# bestiou deminsion become gridcells, shape=(-1, gridcells)
	# 
	bestiou = K.max(iouall, axis=1)
	# maxiou shape=(-1,bnum,gridcells)
	maxiou_inbox = K.repeat(bestiou, bnum) #K.repeat func is like np.tile
	#
	bestiou_flag = (iouall == maxiou_inbox)
	bestiou_flag = K.reshape(bestiou_flag, (-1, bnum*gridcells))

	#iou = K.sum(intersection / union, axis=1)
	obj_count = K.sum(tf.select(t, K.ones_like(x_true), K.zeros_like(x_true)), axis=1)
	ave_iou = K.sum(bestiou) / (K.sum(obj_count) / bnum)
	return ave_iou, bestiou_flag 

# shape is (gridcells*(5+classes), )
def yololoss(y_true, y_pred):
        truth_confid_tf = tf.slice(y_true, [0,0], [-1,gridcells*bnum])
        truth_x_tf = tf.slice(y_true, [0,gridcells*bnum*1], [-1,gridcells*bnum])
        truth_y_tf = tf.slice(y_true, [0,gridcells*bnum*2], [-1,gridcells*bnum])
        truth_w_tf = tf.slice(y_true, [0,gridcells*bnum*3], [-1,gridcells*bnum])
        truth_h_tf = tf.slice(y_true, [0,gridcells*bnum*4], [-1,gridcells*bnum])

	truth_classes_tf = []
	for i in range(classes):
        	ctf = tf.slice(y_true, [0,gridcells*bnum*(5+i)], [-1,gridcells*bnum])
		truth_classes_tf.append(ctf)


        pred_confid_tf = tf.slice(y_pred, [0,0], [-1,gridcells*bnum])
        pred_x_tf = tf.slice(y_pred, [0,gridcells*bnum*1], [-1,gridcells*bnum])
        pred_y_tf = tf.slice(y_pred, [0,gridcells*bnum*2], [-1,gridcells*bnum])
        pred_w_tf = tf.slice(y_pred, [0,gridcells*bnum*3], [-1,gridcells*bnum])
        pred_h_tf = tf.slice(y_pred, [0,gridcells*bnum*4], [-1,gridcells*bnum])

        #
        # below transformation is for softmax calculate
        # slice classes parta, shape is (samples, classes for one sample)
        classall = tf.slice(y_pred, [0,gridcells*bnum*5], [-1,gridcells*bnum*classes])
        # shape (samples, class for one sample) --> shape (samples, classes rows, gridcells cols)
        # every row contain 1 class with all cells
        classall_celltype = K.reshape(classall, (-1, classes, gridcells*bnum))
        # transpose shape to (samples, gridcells rows, classes cols)
        # this is for softmax operation shape
        # every row contain all classes with 1 cell
        classall_softmaxtype = tf.transpose(classall_celltype, perm=(0,2,1))  # backend transpose function didnt support this kind of transpose
        # doing softmax operation, shape is (samples, gridcells rows, classes cols)
        class_softmax_softmaxtype = K.softmax(classall_softmaxtype)
        # transpose back to shape (samples, classes rows, gridcells cols)
        classall_softmax_celltype = tf.transpose(class_softmax_softmaxtype, perm=(0,2,1))  # backend transpose function didnt support this kind of transpose
        # change back to original matrix type,  but with softmax value
        pred_classall_softmax_tf = K.reshape(classall_softmax_celltype, (-1, classes*gridcells*bnum))

	#return classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_tf



	pred_classes_tf = []
	for i in range(classes):
        	#ctf = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
        	ctf = tf.slice(pred_classall_softmax_tf, [0,gridcells*bnum*(0+i)], [-1,gridcells*bnum])
		pred_classes_tf.append(ctf)


	t = K.greater(truth_confid_tf, 0.5) 
	ave_iou, bestiou_flag = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)

	# constraint bestiou
	t = tf.logical_and(t, bestiou_flag)

	confidloss, ave_anyobj, ave_obj = yoloconfidloss(truth_confid_tf, pred_confid_tf, t)
	xloss = yoloxyloss(truth_x_tf, pred_x_tf, t)
	yloss = yoloxyloss(truth_y_tf, pred_y_tf, t)
	wloss = yolowhloss(truth_w_tf, pred_w_tf, t)
	hloss = yolowhloss(truth_h_tf, pred_h_tf, t)


	classesloss =0
	ave_cat =0
	closslist = []
	catlist = []
	for i in range(classes):
		closs, cat = yoloclassloss(truth_classes_tf[i], pred_classes_tf[i], t)
		closslist.append(closs)
		catlist.append(cat)
		classesloss += closs
		ave_cat = tf.select(K.greater(cat ,0), (ave_cat+cat)/2 , ave_cat) 

	#return classesloss, ave_cat

	loss = confidloss+xloss+yloss+wloss+hloss+classesloss
	#loss = wloss+hloss
	#
	return loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, intersection, union,ow,oh,x,y,w,h
	#return loss, ave_cat, ave_obj, ave_anyobj, ave_iou


def limit(x):
	y = tf.select(K.greater(x,100000), 1000000.*K.ones_like(x), x)
	z = tf.select(K.lesser(y,-100000), -1000000.*K.ones_like(x), y)
	return z

def regionloss(y_true, y_pred):
	limited_pred = limit(y_pred)
	loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	#return confidloss+xloss+yloss+wloss+hloss
	return loss

def regionmetrics(y_true, y_pred):
	limited_pred = limit(y_pred)
        loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	pw = K.sum(w)
	ph = K.sum(h)
	return {
		#'loss' : loss,
		#'confidloss' : confidloss,
		#'xloss' : xloss,
		#'yloss' : yloss,
		#'wloss' : wloss,
		#'hloss' : hloss,
		'classesloss' : classesloss,
		'ave_cat' : ave_cat,
		'ave_obj' : ave_obj,
		'ave_anyobj' : ave_anyobj,
		'ave_iou' : ave_iou
		#'predw' : pw,
		#'predh' : ph,
		#'ow' : K.sum(ow),
		#'oh' : K.sum(oh),
		#'insec' : K.sum(intersection),
		#'union' : K.sum(union)
	}


def check(detection_layer,model):
        expected = gridcells*(5+classes)
        real = model.layers[len(model.layers)-1].output_shape[1]
        if expected != real:
                print 'cfg detection layer setting mismatch::change cfg setting'
                print 'output layer should be '+str(expected)+'neurons'
                print 'actual output layer is '+str(real)+'neurons'
                exit()

#
#
if DEBUG_loss:

	side = 5
	bnum = 2
        obj_row = 2
        obj_col = 2
	obj_class = 6

        x_true =K.placeholder(ndim=2)
        x_pred =K.placeholder(ndim=2)
	classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_t = yololoss(x_true, x_pred)
	#classesloss, ave_cat = yololoss(x_true, x_pred)
	classcheck_f = K.function([x_true, x_pred], [classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_t])
	#classcheck_f = K.function([x_true, x_pred], [classesloss, ave_cat])
	tx = np.zeros((side**2)*(classes+5)*bnum)
	for i in range(bnum):
	        tx[side*obj_row+obj_col+(side**2)*i] = 1
	        tx[1*bnum*(side**2)+side*obj_row+obj_col+(side**2)*i] = 0.1
	        tx[2*bnum*(side**2)+side*obj_row+obj_col+(side**2)*i] = 0.2
	        tx[3*bnum*(side**2)+side*obj_row+obj_col+(side**2)*i] = 0.3
	        tx[4*bnum*(side**2)+side*obj_row+obj_col+(side**2)*i] = 0.4
		tx[bnum*(side**2)*(5+obj_class)+side*obj_row+obj_col+(side**2)*i] = 1

	px = np.arange((side**2)*(classes+5)*bnum)

	a0,a1,a2,a3,a4,a5 = classcheck_f([np.asarray([tx]),np.asarray([px])])
	#a0,a1 = classcheck_f([np.asarray([tx]),np.asarray([px])])
	print a0

        #t =K.placeholder(ndim=2, dtype=tf.bool)
        #truth_x_tf =K.placeholder(ndim=2)
        #truth_y_tf =K.placeholder(ndim=2)
        #truth_w_tf =K.placeholder(ndim=2)
        #truth_h_tf =K.placeholder(ndim=2)
        #pred_x_tf =K.placeholder(ndim=2)
        #pred_y_tf =K.placeholder(ndim=2)
        #pred_w_tf =K.placeholder(ndim=2)
        #pred_h_tf =K.placeholder(ndim=2)

        #ave_iou, intersection, union,ow,oh,x,y,w,h = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)
	#iouf = K.function([truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t], [ave_iou, intersection, union,ow,oh,x,y,w,h])
	# 0.507 0.551051051051 0.39 0.51951951952
	#np_t = np.zeros((side**2)*2).reshape(2,side**2)
	#obj_t = np_t >1
	#obj_t[0][obj_row*side+obj_col] = True
	#obj_t[1][obj_row*side+obj_col] = True
	#tx = np.zeros((side**2)*2).reshape(2,side**2)
	#ty = np.zeros((side**2)*2).reshape(2,side**2)
	#tw = np.zeros((side**2)*2).reshape(2,side**2)
	#th = np.zeros((side**2)*2).reshape(2,side**2)
	#tx[0][obj_row*side+obj_col] = 0.507*side - int(0.507*side)
	#ty[0][obj_row*side+obj_col] = 0.551051051051*side - int(0.551051051051*side)
	#tw[0][obj_row*side+obj_col] = 0.39
	#th[0][obj_row*side+obj_col] = 0.51951951952
	#px = np.random.random((side**2)*2).reshape(2,side**2)
	#py = np.random.random((side**2)*2).reshape(2,side**2)
	#pw = np.random.random((side**2)*2).reshape(2,side**2)
	#ph = np.random.random((side**2)*2).reshape(2,side**2)
	#px[0][obj_row*side+obj_col] = 0.5
	#py[0][obj_row*side+obj_col] = 0.5
	#pw[0][obj_row*side+obj_col] = 0.39/0.66
	#ph[0][obj_row*side+obj_col] = 0.51951951952/0.66

	#tx[1][obj_row*side+obj_col] = tx[0][obj_row*side+obj_col]
	#ty[1][obj_row*side+obj_col] = ty[0][obj_row*side+obj_col]
	#tw[1][obj_row*side+obj_col] = tw[0][obj_row*side+obj_col]
	#th[1][obj_row*side+obj_col] = th[0][obj_row*side+obj_col]
        #px[1][obj_row*side+obj_col] = px[0][obj_row*side+obj_col]
        #py[1][obj_row*side+obj_col] = py[0][obj_row*side+obj_col]
        #pw[1][obj_row*side+obj_col] = pw[0][obj_row*side+obj_col]
        #ph[1][obj_row*side+obj_col] = ph[0][obj_row*side+obj_col]


	#[a0,a1,a2,b0,b1,c0,c1,c2,c3]= iouf([tx,ty,tw,th,px,py,pw,ph,obj_t])
	#print a0


	#x =K.placeholder(ndim=2)
	#y =K.placeholder(ndim=2)
	#loss,confidloss,xloss,yloss,wloss,hloss,classesloss = yololoss(y,x)

	#f = K.function([y,x], [loss,confidloss,xloss,yloss,wloss,hloss,classesloss])

	#xtrain = np.ones(343*10).reshape(10,343)
	#ytrain = np.zeros(343*10).reshape(10,343)
	#ytrain[0][0]=1
	#ytrain[0][49]=0.1
	#ytrain[0][49*2]=0.2
	#ytrain[0][49*3]=0.3
	#ytrain[0][49*4]=0.4
	#ytrain[0][49*5]=1


	#print f([ytrain,xtrain])

