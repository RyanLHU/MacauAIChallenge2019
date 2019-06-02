import os

from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.filters import roberts, sobel, prewitt
from skimage import color, io, feature
from skimage.filters import (threshold_otsu, threshold_niblack)
from skimage.feature import local_binary_pattern

__author__ = "Shaoning Zeng"
__license__ = "GNU GPL 3.0 or later"

# Dataset
#dataset_path = './crop_resize/roi'
dataset_path = 'traffic_sign_macau'
save_path = './feature/'
images_name_list = []

def preprocessing(dir,images_name_list):
	newDir = dir
	if os.path.isfile(dir):
		images_name_list.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir,s)
			preprocessing(newDir, images_name_list)
	return images_name_list

# 
def extract_feature(X, featureName):
	# operate on smaller image
	#small_size = (32, 32)
	#X = [cv2.resize(x, small_size) for x in X]

	# normalize all intensities to be between 0 and 1

	# X = np.array(X).astype(np.float32) / 255
	# subtract mean
	# X = [x - np.mean(x) for x in X]

	# Iteratively processing each image
	X_fea = []
	num = 1
	show_once = 1
	for x in X: 
		print(featureName)
		# 1. Point-based Harris features
		if featureName == 'Harris':
			x  = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) # Change BGR format to RGB format
			img_rgb  = x
			img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) # Convert RGB color image to gray scale image
			#img_gray = np.float32(img_gray)
			dst = cv2.cornerHarris(img_gray,2,3,0.04) # Harris corner
			dst = cv2.dilate(dst, (),dst,(-1,-1,),3) # result is dilated for marking the corners, not important
			# TODO: One should tune the parameters to obtain a better result for a specific image
			# Threshold for an optimal value, it may vary depending on the image
			img_rgb[dst>0.01*dst.max()]=[255,0,0] # Parameters
			x1 = img_rgb

		# 2. Edge Roberts features
		elif featureName == 'Roberts':
			#img_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) # Change BGR format to RGB format
			#img_rgb  = x
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			#gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
			x1 = roberts(gray)
		# 3. Edge Sobel features
		elif featureName == 'Sobel':
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			x1 = sobel(gray)
		# 4. Edge Prewitt features
		elif featureName == 'Prewitt':
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			x1 = prewitt(gray)
		# 5. Edge Canny features
		elif featureName == 'Canny':
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			x1=feature.canny(gray)

		# 6. Hough Line features
		elif featureName == 'Line':
			gray1 = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray1,10,150,apertureSize = 3)
			lines = cv2.HoughLines(edges,1,np.pi/180,150)
			try:
				for line in lines:
					for rho,theta in line:
						a = np.cos(theta)
						b = np.sin(theta)
						x0 = a*rho
						y0 = b*rho
						x1 = int(x0 + 3000*(-b))
						y1 = int(y0 + 3000*(a))
						x2 = int(x0 - 3000*(-b))
						y2 = int(y0 - 3000*(a))
						cv2.line(gray1,(x1,y1),(x2,y2),(0,0,255),2)
				x1 = cv2.cvtColor(gray1,cv2.COLOR_BGR2RGB)
			except TypeError as te:
				print('No line found.')
		# 7. Hough Circle features
		elif featureName == 'Circle':
		    img_circle_gray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
		    circles = cv2.HoughCircles(img_circle_gray,cv2.HOUGH_GRADIENT,1,20,
		                                param1=50,param2=30,minRadius=150,maxRadius=500)
		    circles = np.uint16(np.around(circles))
		    for i in circles[0,:]:
		        # draw the outer circle
		        cv2.circle(x,(i[0],i[1]),i[2],(0,255,0),2)
		        # draw the center of the circle
		        cv2.circle(x,(i[0],i[1]),2,(255,0,0),3) 
		    #Show
		    x1 = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

		# 8. Binary Otsu features
		elif featureName == 'Otsu':
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
			thresh_otsu = threshold_otsu(gray)
			x1 = gray < thresh_otsu
		# 9. Binary Niblack features
		elif featureName == 'Niblack':
			#gray = color.rgb2gray(x)
			gray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
			thresh_niblack = threshold_niblack(gray, window_size =31, k =0.05)
			x1 = gray < thresh_niblack

		# 10. Color Histgram features
		elif featureName == 'Histgram':
			histr3 =[];
			img_source = x
			img_source_RGB = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) 
			#img_source_RGB = x 
			r,g,b = cv2.split(img_source_RGB) # or you can use the following commands
			# r = img5[:,:,0]
			# g = img5[:,:,1]
			# b = img5[:,:,2]
			color = ('b','g','r')  ### color RGB image has three channel: Red, Green and Blue. 
			for i, col in enumerate(color):
				histr = cv2.calcHist([img_source],[i],None,[256],[0,256])
				histr3.append(histr)
			x1 = histr3

		# 11. Texture features
		elif featureName == 'Texture':
			img_source = x
			#img_source_RGB = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB) # this is our original image
			img_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
			# settings for LBP 
			radius = 3 
			n_points = 8 * radius
			lbp_default = local_binary_pattern(img_gray, n_points, radius,  method='default')
			x1 = local_binary_pattern(img_gray, n_points, radius,  method='var')

		# 12. ORB features
		elif featureName == 'ORB':
			img_gray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
			img_source_RGB = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
			#img_source_RGB = x
			# Initiate ORB detector
			orb = cv2.ORB_create()
			# find the keypoints with ORB
			kp = orb.detect(img_source_RGB,None)
			# compute the descriptors with ORB
			kp, des = orb.compute(img_source_RGB, kp)
			# draw only keypoints location,not size and orientation
			x1 = cv2.drawKeypoints(x, kp, None, color=(0,255,0), flags=0)

		try:
			print(x1)
			X_fea.append(x1)
			# if os.isdir():
			# 	os.mkdir( path, 0755 )
			# Save image
			print("./feature/"+featureName+"/"+str(num)+".jpeg")
			cv2.imwrite("./feature/"+featureName+"/"+str(num)+".jpeg", x1*255)
			# fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(10, 10))
			# ax.imshow(x1, cmap=plt.cm.gray)
			# ax.set_title(featureName)
			# plt.savefig("./feature/"+featureName+"/"+str(num)+".jpeg")
			num = num+1
		except NameError:
			print("Detect nothing")

	X = X_fea 
	X = [x.flatten() for x in X]
	X = np.array(X)
	return X

# Extract and save features

images_name_list = preprocessing(dataset_path,images_name_list)
num = len(images_name_list)
X = []
for k in range(num):
	img = cv2.imread(images_name_list[k])
	X.append(img)

# X = extract_feature(imageAll, 'Harris')
# np.save(save_path+'1.Point_harris.npy', X)
#print('Shape of the Harris feature:', X.shape)
# 1. Point-based Harris features
# X = extract_feature(X, 'Harris')
# np.save("1.Point_Harris.npy",X)
# 2. Edge Roberts features
# X = extract_feature(X, 'Roberts')
# np.save("2.Edge_Roberts.npy",X) 
# # # 3. Edge Sobel features
# X = extract_feature(X, 'Sobel')
# np.save("3.Edge_Sobel.npy",X) 
# # 4. Edge Prewitt features
# X = extract_feature(X, 'Prewitt')
# np.save("4.Edge_Prewitt.npy",X) 
# 5. Edge Canny features
# X = extract_feature(X, 'Canny')
# np.save("5.Edge_Canny.npy",X) 
# 6. Hough Line features
# X = extract_feature(X, 'Line')
# np.save("6.Hough_Line.npy",X) 
# 7. Hough Circle features
# X = extract_feature(X, 'Circle')
# np.save("7.Hough_Circle.npy",X) 
# 8. Binary Otsu features
# X = extract_feature(X, 'Otsu')
# np.save("8.Binary_Otsu.npy",X) 
# 9. Binary Niblack features
# X = extract_feature(X, 'Niblack')
# np.save("9.Binary_Niblack.npy",X) 
# 10.Color Histgram features
# X = extract_feature(X, 'Histgram')
# np.save("10.Color_Histgram.npy",X) 
# # 11.Texture features
# X = extract_feature(X, 'Texture')
# np.save("11.Texture.npy",X) 
# # 12.ORB features
X = extract_feature(X, 'ORB')
np.save("12.ORB.npy",X) 

