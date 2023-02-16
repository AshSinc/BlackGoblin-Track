###
#python track_multi.py -s 2 -e 7 -n 2
###

import numpy as np
import tensorflow as tf
import re
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import metrics
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import preprocess_input
from keras import optimizers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tabulate import tabulate
from PIL import Image
import argparse

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


import math
import cv2
import sys
import getopt
import os.path
 
argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'i:s:e:n:')

VID_PATH = "./resources/People Walking Free Stock Footage.mp4"
START_TIME = 0
END_TIME = math.inf
NUM_TO_TRACK = 1

MODEL_FILE = "arch7_epochs20_optsgd"

for opt in opts :
    if opt[0] == '-i' : VID_PATH = opt[1]
    if opt[0] == '-s' : START_TIME = int(opt[1])*1000
    if opt[0] == '-e' : END_TIME = int(opt[1])*1000
    if opt[0] == '-n' : NUM_TO_TRACK = int(opt[1])

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create() 
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create() 
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Get the video file and read it
video = cv2.VideoCapture(VID_PATH)
if not video.isOpened() :
    print("Video file not found")
    exit()

video.set(cv2.CAP_PROP_POS_MSEC, START_TIME)

ret, frame = video.read()

scale_percent = 1
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)*scale_percent)
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale_percent)
dim = (frame_width, frame_height)

# Resize the video for a more convenient view
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

OUT_PATH = ("s{}e{}_{}-0.avi".format(int(START_TIME/1000), int(END_TIME/1000), tracker_type))
#increment output filename number if already exists
c = 0
while os.path.exists(OUT_PATH) :
    split = OUT_PATH.split(".")[0].split("-")
    OUT_PATH = split[0] + "-" + str(c) + ".avi"
    c+=1

# Initialize video writer to save the results
output = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), 60.0, dim, True)
if not ret:
    print('cannot read the video')

bboxes = []
trackers = []

for x in range(0,NUM_TO_TRACK):
    bboxes.append(cv2.selectROI(frame, False))
    trackers.append(cv2.TrackerCSRT_create())
    trackers[x].init(frame, bboxes[x])

# Start tracking
print("starting")
count=0
while (video.get(cv2.CAP_PROP_POS_MSEC) <= END_TIME):

    success, frame = video.read()
                
    if not success:
        print('something went wrong')
        break

    for tracker in trackers:
        success, bbox = tracker.update(frame)

        if success:
            #draw bounding box around foot
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            #draw box around expected terrain, could try either side as well
            footheight = p1[1] - p2[1]
            p3 = (p1[0], p2[1] - footheight)
            crop_img = frame[p2[1]:p3[1], p3[0]:p2[0]].copy()
            cv2.rectangle(frame, p3, p2, (0,255,0), 2, 1)
            cv2.imshow("crop", crop_img)
            cv2.imwrite("crop.png", crop_img)

            #physical_devices = tf.config.experimental.list_physical_devices('GPU')
            #if len(physical_devices) > 0:
            #    print("Not enough GPU hardware devices available")
            #config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
            
            input_arr = tf.keras.preprocessing.image.img_to_array(crop_img)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            model = keras.models.load_model(MODEL_FILE)
            probabilities = model.predict(input_arr)
            print(probabilities)

            exit()

        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
    cv2.imshow("Tracking", frame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
        
video.release()
output.release()
cv2.destroyAllWindows()