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

DEBUG_loss = False

# shape is (gridcells,)
def yoloconfidloss(y_true, y_pred, t):
	real_y_true = tf.select(t, y_true, K.zeros_like(y_true))
	pobj = K.sigmoid(y_pred)
	lo = K.square(real_y_true-pobj)
	value_if_true = lamda_confid_obj*(lo)
	value_if_false = lamda_confid_noobj*(lo)
	loss1 = tf.select(t, value_if_true, value_if_false)

	loss = K.mean(loss1) 
	#
	noobj = tf.select(t, K.zeros_like(y_pred), pobj)
	noobjcount = tf.select(t, K.zeros_like(y_pred), K.ones_like(y_pred))
	ave_anyobj = K.sum(noobj) / K.sum(noobjcount)
	#ave_anyobj = K.mean(pobj)
	obj = tf.select(t, pobj, K.zeros_like(y_pred))
	objcount = tf.select(t, K.ones_like(y_pred), K.zeros_like(y_pred))
	#ave_obj = K.mean( K.sum(obj, axis=1) / (K.sum(objcount, axis=1)+0.000001) ) # prevent div 0
	ave_obj =  K.sum(obj) / (K.sum(objcount)+0.000001)  # prevent div 0
	return loss, ave_anyobj, ave_obj

# shape is (gridcells*2,)
def yoloxyloss(y_true, y_pred, t):
	real_y_true = tf.select(t, y_true, K.zeros_like(y_true))
        lo = K.square(real_y_true-K.sigmoid(y_pred))
        value_if_true = lamda_xy*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.select(t, value_if_true, value_if_false)
	#return K.mean(value_if_true)
	objsum = K.sum(y_true)
	return K.sum(loss1)/(objsum+0.0000001)

# different with YOLO
# shape is (gridcells*2,)
def yolowhloss(y_true, y_pred, t):
	real_y_true = tf.select(t, y_true, K.zeros_like(y_true))
        lo = K.square(K.sqrt(real_y_true)-K.sqrt(K.sigmoid(y_pred)))   
	# let w,h not too small or large
        #lo = K.square(y_true-y_pred)+reguralar_wh*K.square(0.5-y_pred)
        value_if_true = lamda_wh*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.select(t, value_if_true , value_if_false)
	#return K.mean(loss1/(y_true+0.000000001))
	#return K.mean(value_if_true)
	objsum = K.sum(y_true)
	return K.sum(loss1)/(objsum+0.0000001)

# shape is (gridcells*classes,)
def yoloclassloss(y_true, y_pred, t):
	#real_y_true = tf.select(t, y_true, K.zeros_like(y_true))
        lo = K.square(y_true-y_pred)
        value_if_true = lamda_class*(lo)
        value_if_false = K.zeros_like(y_true)
	tlist =[]
	for i in range(classes):
		tlist.append(t)
	tt = K.concatenate(tlist,1)
        loss1 = tf.select(tt, value_if_true, value_if_false)


	## only extract predicted class value at obj location
	#nouse_cat = K.sum(tf.select(t, y_pred, K.zeros_like(y_pred)), axis=1)
	## check valid class value
	#nouse_objsum = K.sum(y_true, axis=1)
	## if objsum > 0.5 , means it contain some valid obj(may be 1,2.. objs)
	#nouse_isobj = K.greater(objsum, 0.5)
	## only extract class value at obj location
	#nouse_valid_cat = tf.select(isobj, cat, K.zeros_like(cat))
	## prevent div 0
	#nouse_ave_cat = tf.select(K.greater(K.sum(objsum),0.5), K.sum(valid_cat) / K.sum(objsum) , -1)

	t_y_true = K.greater(y_true, 0.5)
	cat = K.sum(tf.select(t_y_true, y_pred, K.zeros_like(y_pred)))
	objsum = K.sum(y_true)
	return K.sum(loss1)/(objsum+0.0000001), cat/(objsum+0.0000001), loss1, lo

def overlap(x1, w1, x2, w2):
        l1 = (x1) - w1/2
        l2 = (x2) - w2/2
        left = tf.select(K.greater(l1,l2), l1, l2)
        r1 = (x1) + w1/2
        r2 = (x2) + w2/2
        right = tf.select(K.greater(r1,r2), r2, r1)
        result = right - left
	return result

def iou(x_true,y_true,w_true,h_true,x_pred,y_pred,w_pred,h_pred,t,pred_confid_tf):
	xoffset = K.cast_to_floatx(np.tile(np.tile(np.arange(side),side),bnum))
        yoffset = K.cast_to_floatx(np.tile(np.repeat(np.arange(side),side),bnum))

	#xoffset = K.cast_to_floatx((np.tile(np.arange(side),side)))
	#yoffset = K.cast_to_floatx((np.repeat(np.arange(side),side)))
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

	#
	# find best iou among bboxs
	# iouall shape=(-1, bnum*gridcells)
        iouall = intersection / union
	obj_count = K.sum(tf.select(t, K.ones_like(x_true), K.zeros_like(x_true)))
	ave_iou = K.sum(iouall) / (obj_count+0.0000001)
	recall_t = K.greater(iouall, 0.5)
	#recall_count = K.sum(tf.select(recall_t, K.ones_like(iouall), K.zeros_like(iouall)))

	fid_t = K.greater(pred_confid_tf, cfgconst.confid_thresh)
	recall_count_all = K.sum(tf.select(fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

	#  
	obj_fid_t = tf.logical_and(fid_t, t)
	obj_fid_t = tf.logical_and(fid_t, recall_t)
	effevtive_iou_count = K.sum(tf.select(obj_fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

	recall = effevtive_iou_count / (obj_count+0.00000001)
	precision = effevtive_iou_count / (recall_count_all+0.0000001)
	
	#bestiou_flag = t   # nouse


        #nouse_iouall = K.reshape(iouall, (-1, bnum, gridcells))
	##
        ## bestiou deminsion become gridcells, shape=(-1, gridcells)
        #nouse_bestiou = K.max(iouall, axis=1)
	##
        ## maxiou_inbox shape=(-1,bnum,gridcells)
        #nouse_maxiou_inbox = K.repeat(bestiou, bnum) #K.repeat func is like np.tile
        ##
        #nouse_bestiou_flag = K.equal( K.reshape(iouall, (-1, bnum*gridcells)) , K.reshape(maxiou_inbox, (-1, bnum*gridcells)) )
	##
	##return iouall, maxiou_inbox, bestiou_flag 
	## only obj and bestiou coexist become True
        ##bestiou_flag = K.reshape(bestiou_flag, (-1, bnum*gridcells))
	#nouse_bestiou_flag = tf.logical_and(t, bestiou_flag)
	##
	#nouse_recall_t = K.greater(bestiou, 0.5)
	#nouse_recall_count = K.sum(tf.select(recall_t, K.ones_like(bestiou), K.zeros_like(bestiou)))

	##
	##recall_iou = intersection / union
	##recall_t = K.greater(recall_iou, 0.5)
	##recall_count = K.sum(tf.select(recall_t, K.ones_like(recall_iou), K.zeros_like(recall_iou)))
	##
	##iou = K.sum(intersection / union, axis=1)
	#nouse_obj_count = K.sum(tf.select(t, K.ones_like(x_true), K.zeros_like(x_true)))
	##
	##ave_iou = K.sum(iou) / K.sum(obj_count)
	#nouse_ave_iou = K.sum(bestiou) / ((obj_count)) # / bnum)
	##recall = recall_count / (obj_count)
	#nouse_recall = recall_count / ((obj_count)) # / bnum)



	return ave_iou, recall, precision, obj_count, intersection, union,ow,oh,x,y,w,h
	#return obj_count, ave_iou, bestiou 

# shape is (gridcells*(5+classes), )
def yololoss(y_true, y_pred):
        truth_confid_tf = tf.slice(y_true, [0,0], [-1,gridcells*bnum])
        truth_x_tf = tf.slice(y_true, [0,gridcells*bnum], [-1,gridcells*bnum])
        truth_y_tf = tf.slice(y_true, [0,gridcells*bnum*2], [-1,gridcells*bnum])
        truth_w_tf = tf.slice(y_true, [0,gridcells*bnum*3], [-1,gridcells*bnum])
        truth_h_tf = tf.slice(y_true, [0,gridcells*bnum*4], [-1,gridcells*bnum])
        truth_classes_tf_flattern = tf.slice(y_true, [0,gridcells*bnum*5], [-1,gridcells*bnum*classes])

	truth_classes_tf = K.reshape(truth_classes_tf_flattern, [-1,classes,gridcells*bnum])

	#truth_classes_tf = []
	#for i in range(classes):
        #	ctf = tf.slice(y_true, [0,gridcells*bnum*(5+i)], [-1,gridcells*bnum])
	#	truth_classes_tf.append(ctf)


        pred_confid_tf = tf.slice(y_pred, [0,0], [-1,gridcells*bnum])
        pred_x_tf = tf.slice(y_pred, [0,gridcells*bnum], [-1,gridcells*bnum])
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



	pred_classes_tf = K.reshape(pred_classall_softmax_tf, [-1,classes,gridcells*bnum])

	#pred_classes_tf = []
	#for i in range(classes):
        #	#ctf = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
        #	ctf = tf.slice(pred_classall_softmax_tf, [0,gridcells*bnum*(0+i)], [-1,gridcells*bnum])
	#	pred_classes_tf.append(ctf)

	t = K.greater(truth_confid_tf, 0.5) 
	ave_iou, recall, precision, obj_count, intersection, union,ow,oh,x,y,w,h = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t, pred_confid_tf)

	# constraint bestiou
        #t = tf.logical_and(t, bestiou_flag)

	#bestiou_truth_confid_tf = tf.select(t, truth_confid_tf, K.zeros_like(truth_confid_tf))

	confidloss, ave_anyobj, ave_obj = yoloconfidloss(truth_confid_tf, pred_confid_tf, t)
	xloss = yoloxyloss(truth_x_tf, pred_x_tf, t)
	yloss = yoloxyloss(truth_y_tf, pred_y_tf, t)
	wloss = yolowhloss(truth_w_tf, pred_w_tf, t)
	hloss = yolowhloss(truth_h_tf, pred_h_tf, t)


	classesloss, ave_cat, loss1, lo = yoloclassloss(truth_classes_tf_flattern, pred_classall_softmax_tf, t)

	#classesloss =0
	#ave_cat =0.
	#count =0.
	#closslist = []
	#catlist = []
	#for i in range(classes):
	#	closs, cat, objsum = yoloclassloss(truth_classes_tf[i], pred_classes_tf[i], t)
	#	#closslist.append(closs)
	#	#catlist.append(cat)
	#	classesloss += closs
	#	ave_cat += K.sum(cat) 
	#	count += objsum
	#ave_cat = ave_cat / count

	#return classesloss, ave_cat

	loss = confidloss+xloss+yloss+wloss+hloss+classesloss
	#loss = wloss+hloss
	#
	if DEBUG_loss:
		return pred_classall_softmax_tf, truth_classes_tf_flattern, classesloss, ave_cat, loss1, lo
	else:
		return loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall, precision, obj_count, intersection, union,ow,oh,x,y,w,h


def limit(x):
	y = tf.select(K.greater(x,100000), 1000000.*K.ones_like(x), x)
	z = tf.select(K.lesser(y,-100000), -1000000.*K.ones_like(x), y)
	return z

def regionloss(y_true, y_pred):
	limited_pred = limit(y_pred)
	loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall, precision, obj_count, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	#return confidloss+xloss+yloss+wloss+hloss
	return loss

def regionmetrics(y_true, y_pred):
	limited_pred = limit(y_pred)
        loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall, precision, obj_count, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	pw = K.sum(w)
	ph = K.sum(h)
	return {
		#'loss' : loss,
		#'confidloss' : confidloss,
		#'xloss' : xloss,
		#'yloss' : yloss,
		#'wloss' : wloss,
		#'hloss' : hloss,
		#'classesloss' : classesloss,
		'ave_cat' : ave_cat,
		'ave_obj' : ave_obj,
		'ave_anyobj' : ave_anyobj,
		'ave_iou' : ave_iou,
		'recall' : recall,
		'obj_count' : obj_count,
		'precision' : precision
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

	debugoutfile = open('debugout', 'w')
	#side = 5
	#bnum = 2
        obj_row = 2
        obj_col = 2
	obj_class = 1
	batch_size = 2

        x_true_tf =K.placeholder(ndim=2)
        x_pred_tf =K.placeholder(ndim=2)
	pred_classall_softmax_tf, truth_classes_tf_flattern, classesloss, ave_cat, loss1, lo = yololoss(x_true_tf, x_pred_tf)

	classcheck_f = K.function([x_true_tf, x_pred_tf], [pred_classall_softmax_tf, truth_classes_tf_flattern, classesloss, ave_cat, loss1, lo])

	tx = np.zeros((side**2)*bnum*(classes+5)*batch_size).reshape(batch_size,(side**2)*bnum*(classes+5))
        tx[0][side*obj_row+obj_col] = 1
	tx[0][(side**2)*(5+obj_class)*bnum+side*obj_row+obj_col] = 1
	tx[1][side*obj_row+obj_col+1] = 1
	tx[1][(side**2)*(5+obj_class)*bnum+side*obj_row+obj_col+1] = 1


	px = (10*np.random.random((side**2)*bnum*(classes+5)*batch_size)-5).reshape(batch_size,(side**2)*bnum*(classes+5))

	[v_pred_classall_softmax_tf, v_truth_classes_tf_flattern, v_classesloss, v_ave_cat, v_loss1, v_lo] = classcheck_f([tx, px])
	#a0,a1 = classcheck_f([np.asarray([tx]),np.asarray([px])])
	#print a0
	debugoutfile.write(str([px, v_pred_classall_softmax_tf, v_truth_classes_tf_flattern, v_classesloss, v_ave_cat, v_loss1, v_lo])+str([tx]))
	debugoutfile.close()
	exit()


        t =K.placeholder(ndim=2, dtype=tf.bool)
        truth_x_tf =K.placeholder(ndim=2)
        truth_y_tf =K.placeholder(ndim=2)
        truth_w_tf =K.placeholder(ndim=2)
        truth_h_tf =K.placeholder(ndim=2)
        pred_x_tf =K.placeholder(ndim=2)
        pred_y_tf =K.placeholder(ndim=2)
        pred_w_tf =K.placeholder(ndim=2)
        pred_h_tf =K.placeholder(ndim=2)

        ave_iou,recall,precision,objcount, intersection, union,ow,oh,x,y,w,h = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)
	#iouf = K.function([truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t], [ave_iou,recall,precision, intersection, union,ow,oh,x,y,w,h])

	#iouall, maxiou_inbox, bestiou_flag = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)
	iouf = K.function([truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t], [ave_iou,recall,precision,objcount,intersection, union,ow,oh,x,y,w,h])

	# 0.507 0.551051051051 0.39 0.51951951952
	np_t = np.zeros((side**2)*bnum*2).reshape(2,(side**2)*bnum)
	obj_t = np_t >1
	obj_t[0][obj_row*side+obj_col] = True
	obj_t[0][obj_row*side+obj_col+side**2] = True
	obj_t[1][1+obj_row*side+obj_col] = True  # second bbox
	obj_t[1][1+obj_row*side+obj_col+(side**2)] = True  # second bbox
	tx = np.zeros((side**2)*bnum*2).reshape(2,side**2*bnum)
	ty = np.zeros((side**2)*bnum*2).reshape(2,side**2*bnum)
	tw = np.zeros((side**2)*bnum*2).reshape(2,side**2*bnum)
	th = np.zeros((side**2)*bnum*2).reshape(2,side**2*bnum)
	tx[0][obj_row*side+obj_col] = 0.507*side - int(0.507*side)
	ty[0][obj_row*side+obj_col] = 0.551051051051*side - int(0.551051051051*side)
	tw[0][obj_row*side+obj_col] = 0.39
	th[0][obj_row*side+obj_col] = 0.51951951952
        tx[0][obj_row*side+obj_col+side**2] = 0.507*side - int(0.507*side)
        ty[0][obj_row*side+obj_col+side**2] = 0.551051051051*side - int(0.551051051051*side)
        tw[0][obj_row*side+obj_col+side**2] = 0.39
        th[0][obj_row*side+obj_col+side**2] = 0.51951951952

        tx[1][1+obj_row*side+obj_col] = 0.507*side - int(0.507*side)
        ty[1][1+obj_row*side+obj_col] = 0.551051051051*side - int(0.551051051051*side)
        tw[1][1+obj_row*side+obj_col] = 0.39
        th[1][1+obj_row*side+obj_col] = 0.51951951952
        tx[1][1+obj_row*side+obj_col+side**2] = 0.507*side - int(0.507*side)
        ty[1][1+obj_row*side+obj_col+side**2] = 0.551051051051*side - int(0.551051051051*side)
        tw[1][1+obj_row*side+obj_col+side**2] = 0.39
        th[1][1+obj_row*side+obj_col+side**2] = 0.51951951952

	px = np.random.random((side**2)*bnum*2).reshape(2,side**2*bnum)
	py = np.random.random((side**2)*bnum*2).reshape(2,side**2*bnum)
	pw = np.random.random((side**2)*bnum*2).reshape(2,side**2*bnum)
	ph = np.random.random((side**2)*bnum*2).reshape(2,side**2*bnum)
	px[0][obj_row*side+obj_col] = 0.5
	py[0][obj_row*side+obj_col] = 0.5
	pw[0][obj_row*side+obj_col] = 0.39/0.9
	ph[0][obj_row*side+obj_col] = 0.51951951952/0.9


        px[1][1+obj_row*side+obj_col+(side**2)] = px[0][obj_row*side+obj_col]
        py[1][1+obj_row*side+obj_col+(side**2)] = py[0][obj_row*side+obj_col]
        pw[1][1+obj_row*side+obj_col+(side**2)] = pw[0][obj_row*side+obj_col]
        ph[1][1+obj_row*side+obj_col+(side**2)] = ph[0][obj_row*side+obj_col]


	[v_ave_iou,v_recall,v_precision,v_objcount,v_intersection, v_union,v_ow,v_oh,v_x,v_y,v_w,v_h] = iouf([tx,ty,tw,th,px,py,pw,ph,obj_t])
	#[a0,a1,a2,a3,a4,b0,b1,c0,c1,c2,c3]= iouf([tx,ty,tw,th,px,py,pw,ph,obj_t])
	#[a0,a1,a2]= iouf([tx,ty,tw,th,px,py,pw,ph,obj_t])
	debugoutfile.write(str([v_ave_iou,v_recall,v_precision,v_objcount,v_intersection, v_union,v_ow,v_oh,v_x,v_y,v_w,v_h])+str([tx,ty,tw,th])) 
	debugoutfile.close()


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

