import cv2
import cv2.cv as cv
import sys
import time
import numpy as np

def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
	if len(rects) == 0:
		return []
	rects[:,2:] += rects[:,:2]
	return rects

def draw_rects(img, rects, color):
	for x1, y1, x2, y2 in rects:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
		
def draw_single_rect(img,x1,y1,x2,y2,color):
	cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)


if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) != 2:                                         ## Check for error in usage syntax
		print "Usage : python faces.py <image_file>"

	else:
		img = cv2.imread(sys.argv[1],cv2.CV_LOAD_IMAGE_COLOR)  ## Read image file
		img = cv2.resize(img,(0,0), fx=0.05,fy=0.05)
		
		rows,cols,z = img.shape
		
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
		img = cv2.warpAffine(img,M,(cols,rows))
		cv2.imwrite("test.jpg",img);

		if (img == None):                                      ## Check for invalid input
			print "Could not open or find the image"
		else:
			cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
			cascade_eye = cv2.CascadeClassifier("haarcascade_eye.xml")
			cascade_mouth = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
			gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
			#gray_face = cv2.equalizeHist(gray)
	
			kernel = np.ones((5,5),np.float32)/25
			#gray_face = cv2.filter2D(gray_face,-1,kernel)
			gray = cv2.filter2D(gray,-1,kernel)
			cv2.imshow('Display face ROI', gray)

			rects = detect(gray, cascade_face)
			
			print len(rects)
			
			## Extract face coordinates			
			x1 = rects[0][1]
			y1 = rects[0][0]
			x2 = rects[0][3]
			y2 = rects[0][2]
			
			## Extract face ROI
			faceROI = gray[x1:x2, y1:y2]

			## Detect eyes on the face
			gray_eyes = cv2.equalizeHist(faceROI)
			rects_eyes = detect(gray_eyes, cascade_eye)
			numrow_eyes = len(rects_eyes)

			# Keep only the two rectangles at the top of the face
			#rects_eyes_tmp = sorted(rects_eyes_tmp,key=lambda x: x[1])
			#rects_eyes = (rects_eyes_tmp[0],rects_eyes_tmp[1])
			#numrow_eyes = len(rects_eyes)

			## Coordinates base change
			for i in range(0,numrow_eyes):		
				rects_eyes[i][1] = rects_eyes[i][1] + x1
				rects_eyes[i][0] = rects_eyes[i][0] + y1
				rects_eyes[i][3] = rects_eyes[i][3] + x1
				rects_eyes[i][2] = rects_eyes[i][2] + y1

			## Detect mouth on the face
			gray_mouth = cv2.equalizeHist(faceROI)
			rects_mouth = detect(gray_mouth, cascade_mouth)
			numrow_mouth = len(rects_mouth)

			#Keep only the rectangle at the bottom of the face
			#rects_mouth = sorted(rects_mouth_tmp, key=lambda x:x[3])
			#rects_mouth = (rects_mouth_tmp[numrow_mouth-1])
			#numrow_mouth = len(rects_mouth)
			#print numrow_mouth

			## Coordinates base change	
			#rects_mouth[numrow_mouth-1][1] = rects_mouth[numrow_mouth-1][1] + x1
			#rects_mouth[numrow_mouth-1][0] = rects_mouth[numrow_mouth-1][0] + y1
			#rects_mouth[numrow_mouth-1][3] = rects_mouth[numrow_mouth-1][3] + x1
			#rects_mouth[numrow_mouth-1][2] = rects_mouth[numrow_mouth-1][2] + y1
			
			for i in range(0,numrow_mouth):
				rects_mouth[i][1] = rects_mouth[i][1] + x1
				rects_mouth[i][0] = rects_mouth[i][0] + y1
				rects_mouth[i][3] = rects_mouth[i][3] + x1
				rects_mouth[i][2] = rects_mouth[i][2] + y1

			rects_mouth = sorted(rects_mouth, key=lambda x:x[3])
			rects_mouth = (rects_mouth[numrow_mouth-1])
			print rects_mouth
			#rects_mouth = rects_mouth[[numrow_mouth-1]]		
	
			## Show face ROI
			cv2.imshow('Display face ROI', faceROI)

			vis = img.copy()
			draw_rects(vis, rects, (0, 255, 0))
			draw_rects(vis, rects_eyes, (255, 0, 0))
			draw_single_rect(vis, rects_mouth[0],rects_mouth[1],rects_mouth[2],rects_mouth[3], (0,0,255))
			#draw_rects(vis, rects_mouth, (0,0,255))
        		cv2.namedWindow('Display image')          ## create window for display
        		cv2.imshow('Display image', vis)          ## Show image in the window

        		print "size of image: ", img.shape        ## print size of image
			print("--- %s seconds ---" % (time.time() - start_time))

        		cv2.waitKey(0)                            ## Wait for keystroke
        		cv2.destroyAllWindows()                   ## Destroy all windows

