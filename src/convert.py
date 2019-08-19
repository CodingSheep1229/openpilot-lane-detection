import cv2
from numpy import *
import numpy as np  
screenLevels = 255.0

def readYuvFile(filename,width,height):
	fp=open(filename,'rb')
	uv_width=width//2
	uv_height=height//2

	Y=zeros((height,width),uint8,'C')
	U=zeros((uv_height,uv_width),uint8,'C')
	V=zeros((uv_height,uv_width),uint8,'C')

	for m in range(height):
		for n in range(width):
			Y[m,n]=ord(fp.read(1))
	for m in range(uv_height):
		for n in range(uv_width):
			V[m,n]=ord(fp.read(1))
	U[m,n]=ord(fp.read(1))
	fp.close()
	return (Y,U,V)

width=512
height=256

#convert png to yuv raw binary files
for i in range(50):
	(Y,U,V)=readYuvFile('data/yuvs/'+str(i)+'.yuv',width,height)
	Y = list(Y.reshape(-1))
	U = list(U.reshape(-1))
	V = list(V.reshape(-1))
	total = np.array(Y + U + V).astype(np.float32).reshape(1,6,128,256)
	total.tofile('data/raws/'+str(i)+'.raw')
