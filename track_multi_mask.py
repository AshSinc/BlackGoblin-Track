import math
import cv2
import sys
import getopt
import os.path
 
argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'i:s:e:n:')

VID_PATH = "./resources/People Walking Free Stock Footage2.mp4"
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

# Start tracking
print("starting")
while (video.get(cv2.CAP_PROP_POS_MSEC) <= END_TIME):

    success, frame = video.read()
                
    maskedFrame = frame.copy()

    if not success:
        print('something went wrong')
        break

    timer = cv2.getTickCount()

    for tracker in trackers:
        success, bbox = tracker.update(maskedFrame)

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            maskedFrame = cv2.rectangle(frame.copy(), p1, p2, (0,0,0), -1)
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
    cv2.imshow("Tracking", frame)
    cv2.imshow("Tracking Mask", maskedFrame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
        
video.release()
output.release()
cv2.destroyAllWindows()