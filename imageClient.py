import cv2
import os
import sys
import socket
import numpy as np
from keras.preprocessing import image
import time


if len(sys.argv)<2:
	print 'syntax error::need videofile'
	exit()

videofile = sys.argv[1]
if not os.path.isfile(videofile):
        print videofile+' open error'
        exit()
#
#
HOST, PORT = "localhost", 9999
# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect to server and send data
sock.connect((HOST, PORT))

def sendimg(sock, data):
	my_bytes = bytearray()
	for i in range(len(data)):
		my_bytes.append(data[i])
	sock.sendall(my_bytes)
	received = sock.recv(1024)
	#print 'sendmsg received:'+received


def imagefromlist(testlist):
	sock.sendall('(448, 448, 3)')
	received = sock.recv(1024)
	#print received
	#
	f = open(testlist)
	for img_path in f:
		timg = image.load_img(img_path.strip(),  target_size=(448, 448))
		xx = image.img_to_array(timg)
		try:
			(orgh,orgw,c) = xx.shape
			#print xx.shape
			data = xx.reshape(448*448*3).astype(np.uint8)
			sendimg(sock, data)
			time.sleep(3)
		except:
			continue
	sock.close()
		
		

def imagefromvideo(videofile):
	cap = cv2.VideoCapture(videofile)
	ret, frame = cap.read()
	if ret:
		(h, w, c) = frame.shape
		print frame.shape
		sock.sendall('('+str(h)+','+str(w)+','+str(c)+')')
		received = sock.recv(1024)
	else:
		print videofile+' open fail'
		exit()

		
	while (cap.isOpened()):
		ret, frame = cap.read()
		#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		if ret:
			data = frame.reshape(w*h*c)
			sendimg(sock, data)
		else:
			print videofile+' open fail'
			break
		#cv2.waitKey(1000)
		#break

	cap.release()
	sock.close()

if videofile.split('.')[-1] == 'txt':
	imagefromlist(videofile)
else:
	imagefromvideo(videofile)

