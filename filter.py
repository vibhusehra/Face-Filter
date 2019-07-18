import cv2
import numpy as np
from keras.models import load_model


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


model = load_model('final_model.h5')


def detectFace(img,gray):
	face = face_cascade.detectMultiScale(gray,1.25,6)
	for x_ax,y_ax,w,h in face:
		#cv2.rectangle(img,(x_ax,y_ax),(x_ax+w,y_ax+h),(255,0,0),2)
		roi_gray = gray[y_ax:y_ax+h,x_ax:x_ax+h]
		roi_color = img[y_ax:y_ax+h,x_ax:x_ax+h]
		original_shape = roi_gray.shape
		print(roi_gray)
		

		
		preImg = preProcessImage(roi_gray)
		color_img_small = preProcessImage(roi_color)
		preds = model.predict(preImg.reshape(1,96,96,1))
		pred = preds.reshape(30,)
		x = pred[::2]
		y = pred[1::2]
		color_img_small_copy = color_img_small.copy()
		for i in range(15):
			cv2.circle(color_img_small_copy,(x[i],y[i]),1,(255,255,0),1)
		
		sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
		sunglass_width = int((x[7]-x[9])*1.1)
		sunglass_height = int((y[10]-y[8])/1.1)
		sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
		transparent_region = sunglass_resized[:,:,:3] != 0
		color_img_small[int(y[9]):int(y[9])+sunglass_height, int(x[9]):int(x[9])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
		#img[y:y+h, x:x+w] = cv2.resize(color_img_small,original_shape, interpolation = cv2.INTER_CUBIC)
		#if color_img_small is not None:
		img[y_ax:y_ax+h,x_ax:x_ax+w] = cv2.resize(color_img_small,original_shape,interpolation = cv2.INTER_AREA)
		cv2.imshow('gray',color_img_small_copy)
		return roi_gray

def preProcessImage(image):
	image = cv2.resize(image,(96,96),interpolation=cv2.INTER_AREA)
	return image

cap = cv2.VideoCapture(0)
#img = []
while True:
	ret,img = cap.read()

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	roi_gray = detectFace(img,gray)


	
	cv2.imshow("image",img)
	#cv2.imshow('cropped',preprocessedImage)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()