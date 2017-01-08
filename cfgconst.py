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

f = open('labelnames.txt')
label_names =[]
for ln in f:
	label_names.append('uilabel_images/'+ln.strip()+'.png')

