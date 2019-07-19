import cv2
import numpy as np
from keras.models import load_model


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


model = load_model('final_model.h5')


def detectFace(img,gray):
	face = face_cascade.detectMultiScale(gray,1.25,6)
	for x_ax,y_ax,w,h in face:
		#cv2.rectangle(img,(x_ax,y_ax),(x_ax+w,y_ax+h),(255,0,0),2)
		roi_gray = gray[y_ax:y_ax+h,x_ax:x_ax+h] #get the coordinate of the face
		roi_color = img[y_ax:y_ax+h,x_ax:x_ax+h]
		original_shape = roi_gray.shape #store the shape of the face, which will later be used.
		print(roi_gray) #test
		

		
		preImg = preProcessImage(roi_gray) #preprocess the image for feeding the image to the classifier
		color_img_small = preProcessImage(roi_color)
		preds = model.predict(preImg.reshape(1,96,96,1))
		pred = preds.reshape(30,)
		x = pred[::2] #x coordinates of the landmarks
		y = pred[1::2] #y coordinates of the landmarks
		color_img_small_copy = color_img_small.copy() 
		for i in range(15):
			cv2.circle(color_img_small_copy,(x[i],y[i]),1,(255,255,0),1) #drawing a circle over all the landmarks detected
		
		sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED) # loading the filter
		sunglass_width = int((x[7]-x[9])*1.1) #setting the width of the filter
		sunglass_height = int((y[10]-y[8])/1.1) #setting the height of the filter
		sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC) 
		transparent_region = sunglass_resized[:,:,:3] != 0 #removing the dark portion at the sides of the sunglasses
		print(transparent_region)
		color_img_small[int(y[9]):int(y[9])+sunglass_height, int(x[9]):int(x[9])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region] #overlaying the filter over the image
		#img[y:y+h, x:x+w] = cv2.resize(color_img_small,original_shape, interpolation = cv2.INTER_CUBIC)
		img[y_ax:y_ax+h,x_ax:x_ax+w] = cv2.resize(color_img_small,original_shape,interpolation = cv2.INTER_AREA) #fitting the image with filter in the main frame
		cv2.imshow('gray',color_img_small_copy) 
		return roi_gray

def preProcessImage(image):
	image = cv2.resize(image,(96,96),interpolation=cv2.INTER_AREA) #making the image fit for feeding into the classifier
	return image

cap = cv2.VideoCapture(0) #start the webcam
#img = []
while True:
	ret,img = cap.read() #capture frames from the webcam

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	roi_gray = detectFace(img,gray)


	
	cv2.imshow("image",img)
	#cv2.imshow('cropped',preprocessedImage)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()