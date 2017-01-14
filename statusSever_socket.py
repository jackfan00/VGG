import SocketServer
import cv2
import numpy as np
import scipy.misc
import utils
import cfgconst
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def drawbbox(self, img, xx, testmodel, confid_thresh, w, h, c):
	ttimg, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list = utils.predict(preprocess_input(np.asarray([xx])), testmodel, confid_thresh,w,h,c)
	for x0,y0,x1,y1,classprob,class_id,confid_value in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list, confid_value_list):
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
	cv2.waitKey(1)


    def shownosourceimage(self, w, h, c):
	defimg = cv2.imread('t.png')
	cv2.imshow('frame',defimg)
	cv2.waitKey(1000)
	

    def handle(self):
	#self.request.settimeout(5)
	#self.request.setblocking(1)
        # self.request is the TCP socket connected to the client
	metadata = self.request.recv(1024)
	print 'raw metadata='+metadata
	print 'metadata='+str(metadata)
	sw, sh, sc = str(metadata).replace('(','').replace(')','').split(',')
	w = int(sw.strip())
	h = int(sh.strip())
	c = int(sc.strip())
	if w>0 and h>0 and c==3:
		self.request.sendall('gooooo')
	else:
		self.request.sendall('gg')

	# receive img
	imglength = w*h*c
	count =0
	imgdata =[]
	timeout_count =0
	while True:
		if (imglength-count) >= 16384:
			bufsize = 16384
		else:
			bufsize = imglength-count
		try:
			#print str(count)+':'+str(bufsize)
			rd = self.request.recv(bufsize)
			if len(rd)==0:
				timeout_count = timeout_count +1
			if timeout_count > 100:
				#self.shownosourceimage(w,h,c)
				self.request.close()
				print 'timeout_count timeout'
				break
		except:
			#self.shownosourceimage(w,h,c)
			self.request.close()
			print 'recv timeout'
			break
		#
		imgdata.extend(rd)
		count = count + len(rd)
		if count == (w*h*c):
			count =0
			try:
				self.request.sendall('go')
			except:
				self.request.close()
				print 'sendall timeout'
				break
			#
			#print len(imgdata)
			#print imgdata
			image_array = np.asarray(bytearray(imgdata)).reshape(w,h,c)
			nim = scipy.misc.imresize(image_array, (448, 448, 3))
			img = nim
			xx = image.img_to_array(cv2.cvtColor(nim, cv2.COLOR_RGB2BGR))

			#scipy.misc.imsave('outfile.jpg', image_array)
			#cv2.imshow('frame',image_array)
			#cv2.waitKey(1) 
			self.drawbbox(img, xx, MyTCPHandler.testmodel, MyTCPHandler.confid_thresh, 448, 448, 3)
			imgdata =[]
		
    #def initpara(self, testmodel, confid_thresh):
    #    self.testmodel = testmodel
    #    self.confid_thresh = confid_thresh


"""
	self.imgdata =[]
	while True:
		rawdata = self.request.recv(1024)
        print "{} connected.".format(self.client_address[0])
        #print self.data
        # just send back the same data, but upper-cased
        #self.request.sendall(self.data.upper())
"""

"""
if __name__ == "__main__":

	HOST, PORT = "localhost", 9999

	# Create the server, binding to localhost on port 9999
	server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

	# Activate the server; this will keep running until you
	# interrupt the program with Ctrl-C
	server.serve_forever()

"""
