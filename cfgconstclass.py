

class CFG:
	def __init__(self, cfgpath):
		self.cfgpath = cfgpath

		f = open(cfgpath) #'detregion.cfg')
		for l in f:
			ss = l.split('=')
			if ss[0].strip() == 'classes':
				self.classes = int(ss[1].strip())
				print 'classes='+str(self.classes)
			elif ss[0].strip() == 'side':
				self.side = int(ss[1].strip())
				print 'side='+str(self.side)
			elif ss[0].strip() == 'bnum':
				self.bnum = int(ss[1].strip())
				print 'bnum='+str(self.bnum)
			elif ss[0].strip() == 'object_scale':
				self.object_scale = float(ss[1].strip())
				print 'object_scale='+str(self.object_scale)
			elif ss[0].strip() == 'noobject_scale':
				self.noobject_scale = float(ss[1].strip())
				print 'noobject_scale='+str(self.noobject_scale)
			elif ss[0].strip() == 'class_scale':
				self.class_scale = float(ss[1].strip())
				print 'class_scale='+str(self.class_scale)
			elif ss[0].strip() == 'coord_scale':
				self.coord_scale = float(ss[1].strip())
				print 'coord_scale='+str(self.coord_scale)
			elif ss[0].strip() == 'trainset':
				self.trainset = ss[1].strip()
				print 'trainset='+self.trainset
			elif ss[0].strip() == 'valset':
				self.valset = ss[1].strip()
				print 'valset='+self.valset
			elif ss[0].strip() == 'numberof_train_samples':
				self.numberof_train_samples = int(ss[1].strip())
				print 'numberof_train_samples='+str(self.numberof_train_samples)
			elif ss[0].strip() == 'testfile':
				self.testfile = ss[1].strip()
				print 'testfile='+self.testfile
			elif ss[0].strip() == 'videofile':
				self.videofile = ss[1].strip()
				print 'videofile='+self.videofile
			elif ss[0].strip() == 'debugimg':
				self.debugimg = int(ss[1].strip())
				print 'debugimg='+str(self.debugimg)
			elif ss[0].strip() == 'imagefordebugtrain':
				self.imagefordebugtrain = ss[1].strip()
				print 'imagefordebugtrain='+self.imagefordebugtrain
			elif ss[0].strip() == 'lr':
				self.lr = float(ss[1].strip())
				print 'lr='+str(self.lr)
			elif ss[0].strip() == 'patience':
				self.patience = int(ss[1].strip())
				print 'patience='+str(self.patience)
			elif ss[0].strip() == 'lr_reduce_rate':
				self.lr_reduce_rate = float(ss[1].strip())
				print 'lr_reduce_rate='+str(self.lr_reduce_rate)
			elif ss[0].strip() == 'lr_reduce_nb':
				self.lr_reduce_nb = int(ss[1].strip())
				print 'lr_reduce_nb='+str(self.lr_reduce_nb)
			elif ss[0].strip() == 'nb_epoch':
				self.nb_epoch = int(ss[1].strip())
				print 'nb_epoch='+str(self.nb_epoch)
			elif ss[0].strip() == 'batch_size':
				self.batch_size = int(ss[1].strip())
				print 'batch_size='+str(self.batch_size)
			elif ss[0].strip() == 'labelnames':
				self.labelnames_file = ss[1].strip()
				print 'labelnames='+self.labelnames_file


		f = open(self.labelnames_file)
		self.label_names =[]
		for ln in f:
			self.label_names.append('uilabel_images/'+ln.strip()+'.png')

