f = open('detregion.cfg')
for l in f:
	ss = l.split('=')
	if ss[0].strip() == 'classes':
		classes = int(ss[1].strip())
		print 'classes='+str(classes)
	elif ss[0].strip() == 'side':
		side = int(ss[1].strip())
		print 'side='+str(side)
	elif ss[0].strip() == 'bnum':
		bnum = int(ss[1].strip())
		print 'bnum='+str(bnum)
	elif ss[0].strip() == 'object_scale':
		object_scale = float(ss[1].strip())
		print 'object_scale='+str(object_scale)
	elif ss[0].strip() == 'noobject_scale':
		noobject_scale = float(ss[1].strip())
		print 'noobject_scale='+str(noobject_scale)
	elif ss[0].strip() == 'class_scale':
		class_scale = float(ss[1].strip())
		print 'class_scale='+str(class_scale)
	elif ss[0].strip() == 'coord_scale':
		coord_scale = float(ss[1].strip())
		print 'coord_scale='+str(coord_scale)
	elif ss[0].strip() == 'trainset':
		trainset = ss[1].strip()
		print 'trainset='+trainset
        elif ss[0].strip() == 'valset':
                valset = ss[1].strip()
                print 'valset='+valset
        elif ss[0].strip() == 'numberof_train_samples':
                numberof_train_samples = int(ss[1].strip())
                print 'numberof_train_samples='+str(numberof_train_samples)
        elif ss[0].strip() == 'testfile':
                testfile = ss[1].strip()
                print 'testfile='+testfile
        elif ss[0].strip() == 'videofile':
                videofile = ss[1].strip()
                print 'videofile='+videofile
        elif ss[0].strip() == 'debugimg':
                debugimg = int(ss[1].strip())
                print 'debugimg='+str(debugimg)
        elif ss[0].strip() == 'imagefordebugtrain':
                imagefordebugtrain = ss[1].strip()
                print 'imagefordebugtrain='+imagefordebugtrain
        elif ss[0].strip() == 'lr':
                lr = float(ss[1].strip())
                print 'lr='+str(lr)
        elif ss[0].strip() == 'patience':
                patience = int(ss[1].strip())
                print 'patience='+str(patience)
        elif ss[0].strip() == 'lr_reduce_rate':
                lr_reduce_rate = float(ss[1].strip())
                print 'lr_reduce_rate='+str(lr_reduce_rate)
        elif ss[0].strip() == 'lr_reduce_nb':
                lr_reduce_nb = int(ss[1].strip())
                print 'lr_reduce_nb='+str(lr_reduce_nb)
        elif ss[0].strip() == 'nb_epoch':
                nb_epoch = int(ss[1].strip())
                print 'nb_epoch='+str(nb_epoch)
        elif ss[0].strip() == 'batch_size':
                batch_size = int(ss[1].strip())
                print 'batch_size='+str(batch_size)
	elif ss[0].strip() == 'randomize':
		randomize = int(ss[1].strip())
		print 'randomize='+str(randomize)
        elif ss[0].strip() == 'labelnames':
                labelnames_file = ss[1].strip()
                print 'labelnames='+labelnames_file



f = open(labelnames_file)
label_names =[]
for ln in f:
	label_names.append('uilabel_images/'+ln.strip()+'.png')

