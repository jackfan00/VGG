import sys
import os
sys.path.append(os.path.abspath("/home/jack/DataSets/COCO/coco/PythonAPI"))
from pycocotools.coco import COCO
import skimage.io as io
import random

random.seed(0)
maxfiles = 1000

imgfolder = '/home/jack/DataSets/COCO/JPEGImages/'
labelfolder = '/home/jack/DataSets/COCO/labels/'

train_annFile = '/home/jack/DataSets/COCO/annotations/instances_train2014.json'
#val_annFile = '/home/jack/DataSets/COCO/annotations/instances_val2014.json'
kps_annFile = '/home/jack/DataSets/COCO/annotations/person_keypoints_train2014.json'

coco_kps=COCO(kps_annFile)
coco= COCO(train_annFile)
#coco_val = COCO(val_annFile)


# COCO label number
coco_label_num = [1, 16, 17, 21, 18, 19, 20, 5, 2, 9, 6, 3, 4, 7, 44, 62, 67, 64, 63, 72]
# VOC label number
voc_label_num = [15, 3, 8, 10, 12, 13, 17, 1, 2, 4, 6, 7, 14, 19, 5, 9, 11, 16, 18, 20]
# get all images containing given categories, select one at random
catNms=['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train','bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv' ]

coco_catids =[]
allimgIds =[]
for catnm in catNms:
	catIds = coco.getCatIds(catNms=[catnm])
	coco_catids.extend(catIds)
	imgIds = coco.getImgIds(catIds=catIds)
	random.shuffle(imgIds)
	allimgIds.extend(imgIds[0:maxfiles-1])

# create labels
for imgid in allimgIds:
	f = open(labelfolder+str(imgid)+'.txt','w')
	img = coco.loadImgs(imgid)
	width = img[0]['width']
	height = img[0]['height']
	annIds = coco.getAnnIds(imgid)
	anns = coco.loadAnns(annIds)
	for an in anns:
		x = an['bbox'][0]
		y = an['bbox'][1]
		w = an['bbox'][2]
		h = an['bbox'][3]
		# filter too small object
		if w/width < 0.05 or h/height < 0.05:
			continue
		coco_cat = an['category_id']
		# check the cat is in VOC catelogs
		try:
			coco_label_num.index(coco_cat)
		except:
			continue
		#
		if coco_cat ==1:  #if person, check appearity
			annid = an['id']
			kps_anns = coco_kps.loadAnns(annid)
			num_keypoints = kps_anns[0]['num_keypoints']
			if num_keypoints <7:
				continue

		# convert to yolo voc format
		x = (x+w/2)/width
		y = (y+h/2)/height
		w = w/width
		h = h/height
		id = voc_label_num[coco_label_num.index(coco_cat)]-1  #0-index
		f.write(str(id)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
	f.close()


# use url to load image
for imgid in allimgIds:
	if os.path.isfile(imgfolder+str(imgid)+'.jpg'):
		continue
	I = io.imread('http://mscoco.org/images/%d'%(imgid))
	io.imsave(imgfolder+str(imgid)+'.jpg',I)

# create trainlist file
labs = os.listdir(labelfolder)
f = open(labelfolder+'coco_trainlist.txt' , 'w')
for l in labs:
	s = l.replace('.txt','.jpg')
	f.write(imgfolder+s+'\n')
f.close()
