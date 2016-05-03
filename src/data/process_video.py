#background subtraction
import cv2
from numpy import zeros
import numpy as np
from numpy import uint8
from os.path import basename
from os import sep
import sys
#Find frames with blast
#find judge the mean brightness



def findBlast(videoFile):
	"""
	This function finds the last frame whos mean brightness is more than 
	two standard devitations away from the mean.
	"""
	cap=cv2.VideoCapture(videoFile)
	numFrames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

	count=0
	meanBightness=zeros(numFrames)
	while(cap.isOpened()): 
		ret,frame=cap.read()
		if ret==True:
			meanBightness[count]=frame.mean()
			count=count+1
		else:
			break

	mean=meanBightness.mean()
	std=meanBightness.std()
	blastInd=np.max(np.where(meanBightness>(mean+2*std)))

	#write out tmp video and cropp
	cv2.destroyAllWindows()
	cap.release()
	return(blastInd)

def isolatePendulum(videoFile,tmpVideoDir,template,blastInd=0):

	#set input
	cap=cv2.VideoCapture(videoFile)

	#set output
	name=basename(videoFile)
	videoOut=tmpVideoDir+"/processed_"+name.split(".")[0]+".avi"

	#set compression
	fourcc = cv2.cv.CV_FOURCC('I','Y','U','V')

	count=0
	while(cap.isOpened()): 
		ret,frame=cap.read()
		if ret==True:

			#find thrust stand
			if count==blastInd:

				method=eval('cv2.TM_CCOEFF_NORMED')
				template=cv2.imread(template)
				h,w=template.shape[0:2]
				res = cv2.matchTemplate(frame,template,method)
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
				top_left = max_loc
				bottom_right = (top_left[0] + w, top_left[1] + h)
				frame_width=bottom_right[0]-top_left[0]
				frame_height=(bottom_right[1]-top_left[1])/2

				video = cv2.VideoWriter(videoOut,fourcc,30,(frame_width,frame_height),True)
				tmp=zeros(frame[(top_left[1]):(bottom_right[1]-frame_height),top_left[0]:bottom_right[0],:].shape,dtype=uint8)

			if count>=blastInd:

				#crop out thrust stand
				frame=frame[(top_left[1]):(bottom_right[1]-frame_height),top_left[0]:bottom_right[0],:]
				
				#filter images
				blur = cv2.GaussianBlur(frame,(5,5),0)
				ret1,frame = cv2.threshold(cv2.cvtColor(blur,cv2.cv.CV_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

				#write out
				tmp[:,:,0]=frame
				tmp[:,:,1]=frame
				tmp[:,:,2]=frame
				video.write(tmp)

			count=count+1
		else:
			break

	cv2.destroyAllWindows()
	video.release()
	cap.release()
	print("Processed video written to:", videoOut)

if __name__ == "__main__":

	videoFile=sys.argv[1]
	tmpVideoDir=sys.argv[2]
	template=sys.argv[3]

	blastInd=findBlast(videoFile)
	isolatePendulum(videoFile,tmpVideoDir,template,blastInd=blastInd)



