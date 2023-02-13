import math
import cv2
import sys
import getopt
import os.path
import numpy as np
 
argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'i:s:e:n:')

VID_PATH = "./resources/People Walking Free Stock Footage.mp4"
START_TIME = 0
END_TIME = math.inf
NUM_TO_TRACK = 1

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

first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Creates a mask image with zeros
mask = np.zeros_like(frame)
# Saturation for mask image is set to 255
mask[..., 1] = 255

# Start tracking
print("starting")
while (video.get(cv2.CAP_PROP_POS_MSEC) <= END_TIME):

    success, frame = video.read()
                
    if not success:
        print('something went wrong')
        break

    # Each frame is converted into grayscale
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # optical flow using Farneback method
    optical_flow = cv2.calcOpticalFlowFarneback(first_gray, current_gray,
                    None,
                    0.2, 4, 12, 2, 3, 1.1, 0)
    
    # Magnitude and Angle of 2D vector is calculated
    mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    
    # Determines mask value using normalized magnitude
    mask[..., 0] = ang * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Converts HSV (Hue Saturation Value) to RGB (or BGR)
    rgb_frame = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # Displays new output frame
    #cv2.imshow("optical flow", rgb_frame)
    
    first_gray = current_gray

    for tracker in trackers:
        success, bbox = tracker.update(rgb_frame)

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(rgb_frame, p1, p2, (255,0,0), 2, 1)
        else:
            cv2.putText(rgb_frame, "Tracking failure detected", (100,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
    cv2.imshow("Tracking", rgb_frame)
    output.write(rgb_frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
        
video.release()
output.release()
cv2.destroyAllWindows()