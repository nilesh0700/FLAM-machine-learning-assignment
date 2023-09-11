# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:12:28 2023

@author: acer
"""

import cv2
import numpy as np
import imutils

model = cv2.dnn.readNetFromCaffe('D:\FLAM\MobileNetSSD_deploy.prototxt',
                                 'D:\FLAM\MobileNetSSD_deploy.caffemodel')

CONF_THR = 0.5

frame = cv2.imread("D:\FLAM\drive-download-20230908T062050Z-001\IMG_0846.JPG")
frame = imutils.resize(frame, width=800, height=1000)

h, w = frame.shape[0:2]

blob = cv2.dnn.blobFromImage(frame, 1/127.5, (80, 120),
                             (127.5, 127.5, 127.5), False)

model.setInput(blob)
output = model.forward()

group_bbox = None 

for i in range(output.shape[2]):
    conf = output[0, 0, i, 2]

    if conf > CONF_THR:
        x0, y0, x1, y1 = (output[0, 0, i, 3:7] * [w, h, w, h]).astype(int)

        if group_bbox is None:
            group_bbox = (x0, y0, x1, y1)
        else:
            group_bbox = (min(group_bbox[0], x0),
                          min(group_bbox[1], y0),
                          max(group_bbox[2], x1),
                          max(group_bbox[3], y1))

if group_bbox is not None:
    x0, y0, x1, y1 = group_bbox
    
    # Move the bounding box upwards by a certain number of pixels (e.g., 20 pixels)
    move_upwards = 20
    
    y0 -= move_upwards
    y1 -= move_upwards
    
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 5)

#frame = imutils.resize(frame, width=600, height=500) 

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
