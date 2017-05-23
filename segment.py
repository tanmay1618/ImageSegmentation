# -*- coding: utf-8 -*-
"""
Created on Mon May 22 23:01:56 2017

@author: Tanmay
"""

import numpy as np
import cv2

img = cv2.imread('IMG_20170509_154032.jpg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(bw,200,205,cv2.THRESH_BINARY_INV)
res = cv2.bitwise_and(img,img,mask=threshold)
Z = res.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)

lower_blue = np.array([63,88,81])
upper_blue = np.array([65,90,83])


res = center[label.flatten()]
res2 = res.reshape((img.shape))
mask = cv2.inRange(res2,lower_blue, upper_blue)
cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
cv2.imshow('dst_rt',res2)

cv2.waitKey(0)
cv2.destroyAllWindows()