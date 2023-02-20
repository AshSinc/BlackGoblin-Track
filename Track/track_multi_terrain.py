###
#python Track/track_multi_terrain.py -s 2 -e 7 -n 1
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

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import math
import cv2
import sys
import getopt
import os.path
 
argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'i:s:e:n:v:')

MODEL_FILE = "outputs/models/arch7_epochs40_optsgd"

VID_PATH = "./resources/People Walking Free Stock Footage.mp4"

START_TIME = 0
END_TIME = int(sys.maxsize)
NUM_TO_TRACK = 1

SHOULD_SHOW_CLASS = True

GROUND_CLASSES = ["hard_stone", "loose_stone", "soft", "veg"]

for opt in opts :
    if opt[0] == '-i' : VID_PATH = opt[1]
    if opt[0] == '-s' : START_TIME = int(opt[1])*1000
    if opt[0] == '-e' : END_TIME = int(opt[1])*1000
    if opt[0] == '-n' : NUM_TO_TRACK = int(opt[1])
    if opt[0] == '-v' : VID_PATH = opt[1]

OUTPUT_PATH = "outputs/track/"

# Get the video file and read it
video = cv2.VideoCapture(VID_PATH)
if not video.isOpened() :
    print("Video file not found")
    exit()

if END_TIME == int(sys.maxsize) :
    # count the number of frames
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # calculate duration of the video
    seconds = round(frames / fps)
    #video_time = datetime.timedelta(seconds=seconds)
    #print(f"duration in seconds: {seconds}")
    #print(f"video time: {video_time}")
    END_TIME = seconds*1000
    print(seconds)

video.set(cv2.CAP_PROP_POS_MSEC, START_TIME)

ret, frame = video.read()

scale_percent = 1
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)*scale_percent)
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale_percent)
dim = (frame_width, frame_height)

# Resize the video for a more convenient view
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

OUT_PATH = (OUTPUT_PATH+"s{}e{}_{}-0.avi".format(int(START_TIME/1000), int(END_TIME/1000), "CRST"))
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
ypos_tracked = []

for x in range(0,NUM_TO_TRACK):
    bboxes.append(cv2.selectROI(frame, False))
    trackers.append(cv2.TrackerCSRT_create())
    trackers[x].init(frame, bboxes[x])
    ypos_tracked.append([])

f = open(OUTPUT_PATH+"output-timesteps.txt", "w")
model = keras.models.load_model(MODEL_FILE)

# Start tracking
print("starting")

while (video.get(cv2.CAP_PROP_POS_MSEC) < END_TIME):
    success, original_frame = video.read()
    frame = original_frame.copy()

    if not success:
        print('something went wrong')
        break

    for index, tracker in enumerate(trackers):
        
        success, bbox = tracker.update(frame)

        if success:
            
            #bounding box around foot
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            
            #box around expected terrain, could try either side as well
            footheight = p1[1] - p2[1]
            footwidth = p2[0] - p1[0]
            p3 = (p1[0], p2[1] - footheight)

            #crop out image
            crop_img = original_frame[p2[1]:p3[1], p3[0]:p2[0]]
            if(crop_img.shape[0] < 10 or crop_img.shape[1] < 10) : #if cropped image is only 10 pixels then just take the whole foot box, not great but no time
                p3 = p1
                crop_img = original_frame[p1[1]:p2[1], p1[0]:p2[0]]
            cv2.imshow("crop", crop_img)

            #get expected center of foot
            center = [int(p1[0]+(footwidth/2)),int(p1[1]-footheight/2)]
            ypos_tracked[index].append(frame_height-center[1])
            #cv2.circle(frame, center, 10, (0, 0, 255), 2, 1)

            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(frame, p3, p2, (0,255,0), 2, 1)

            #preprocess cropped section of image (underfoot)
            resize = cv2.resize(crop_img, (240, 240), interpolation= cv2.INTER_LINEAR)
            input_arr = tf.keras.preprocessing.image.img_to_array(resize)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            input_arr = input_arr.astype('float32') / 255.
            
            #predict
            probabilities = model.predict(input_arr)
            predicted_class = np.argmax(probabilities, axis=-1)

            if(SHOULD_SHOW_CLASS):
                cv2.putText(frame, GROUND_CLASSES[int(predicted_class)], p3, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 1)

            print(predicted_class)

            f.write("obj: {}, y: {}, p: {}, t: {}\n".format(index, center[1], predicted_class, video.get(cv2.CAP_PROP_POS_MSEC)))

        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
    cv2.imshow("Tracking", frame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break


for index, list in enumerate(ypos_tracked):
    fig, ax = plt.subplots(1)
    ax.plot(list)
    ax.set_ylim(ymin=0)
    #plt.show()
    plt.savefig(OUTPUT_PATH+'ypos_tracked'+str(index)+'.png', bbox_inches='tight')

f.close()   
video.release()
output.release()
cv2.destroyAllWindows()