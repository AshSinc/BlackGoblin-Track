import os
import numpy as np
import json
from PIL import Image, ImageDraw
import shutil

datadir = "/run/media/ash/2TB/Datasets/andriluka14cvpr"
jsonpath = "dataset.json"
outputdir = "/run/media/ash/2TB/Datasets/andriluka14cvpr/testoutput"

MAX_IMAGES = 100
FOOT_WIDTH = 200*0.20 # 200 pixels tall, 10% of height, so 20 pixels of foot for a 200 pixel tall person, + 5% to be safe because of variance in poses
FOOT_HEIGHT = 200*0.20
DROP_HEIGHT = 10

usableimages = []

f = open(jsonpath)
dataset = json.load(f)
f.close()

def getUsableImagesList() :
    #for each image instance
    for i in dataset['annolist']:
        image_name = i['image']['name']

        #for each person in image
        person_number = -1
        for p in i['annorect'] :
            person_number += 1
            scale = p['scale']

            foundLeftAnkle = False
            foundRightAnkle = False

            left_ankle = {}
            right_ankle = {}
            left_foot_yolo = {}
            right_foot_yolo = {}

            #if there are point annotations
            if 'annopoints' in p :
                annopoints = p['annopoints']

                #for each point
                for ap in annopoints['point'] :
                    if(ap['is_visible'] == 1 or ap['is_visible'] == "1") :

                        #if the point is left or right ankle
                        if(ap['id'] == 0) :
                            foundLeftAnkle = True
                            left_ankle['x'] = ap['x']
                            left_ankle['y'] = ap['y']
                        if(ap['id'] == 5) :
                            foundRightAnkle = True
                            right_ankle['x'] = ap['x']
                            right_ankle['y'] = ap['y']
        
            if(foundLeftAnkle and foundRightAnkle) :
                im = Image.open(datadir + '/mpii_human_pose_v1/images/' + image_name)
                width = im.size[0]
                height = im.size[1]

                #calculate some yolo values as a percentage of image width and height
                left_foot_yolo['center_x'] = left_ankle['x']/width
                left_foot_yolo['center_y'] = (left_ankle['y']+DROP_HEIGHT*scale)/height

                left_foot_yolo['width'] = scale*FOOT_WIDTH/width
                left_foot_yolo['height'] = scale*FOOT_HEIGHT/height

                right_foot_yolo['center_x'] = right_ankle['x']/width
                right_foot_yolo['center_y'] = (right_ankle['y']+DROP_HEIGHT*scale)/height

                right_foot_yolo['width'] =  scale*FOOT_WIDTH/width
                right_foot_yolo['height'] = scale*FOOT_HEIGHT/height

                found = {
                            'image' : image_name,
                            'person_number' : person_number,
                            'scale' : scale,
                            'left_ankle' : left_ankle,
                            'right_ankle' : right_ankle,
                            'left_foot_yolo' : left_foot_yolo,
                            'right_foot_yolo' : right_foot_yolo
                        }
                usableimages.append(found)

        if(len(usableimages) > MAX_IMAGES) :
            break

def generateYOLODataSetFromUsableImages() :
    for instance in usableimages :
        destination = outputdir + '/images/train/' + instance['image']
        if not os.path.exists(destination):
            shutil.copyfile(datadir + '/mpii_human_pose_v1/images/' + instance['image'], destination)
            #need to write yolo.txt files
            #file = open('test.txt', 'w')

def drawExampleImage(obj):

    im = None

    if os.path.exists(outputdir + '/images/examples/' + obj['image']) :
        im = Image.open(outputdir + '/images/examples/' + obj['image'])
    else :
        im = Image.open(datadir + '/mpii_human_pose_v1/images/' + obj['image'])

    image_width = im.size[0]
    image_height = im.size[1]

    draw = ImageDraw.Draw(im)

    left_center = {}
    left_center['x'] = obj['left_foot_yolo']['center_x'] * image_width
    left_center['y'] = obj['left_foot_yolo']['center_y'] * image_height

    right_center = {}
    right_center['x'] = obj['right_foot_yolo']['center_x'] * image_width
    right_center['y'] = obj['right_foot_yolo']['center_y'] * image_height

    half_foot_width = FOOT_WIDTH*obj['scale']/2
    half_foot_height = FOOT_HEIGHT*obj['scale']/2

    #left foot
    upperleft_x = left_center['x']-half_foot_width
    upperleft_y = left_center['y']-half_foot_height

    lowerright_x = left_center['x']+half_foot_width
    lowerright_y = left_center['y']+half_foot_height

    draw.rectangle(
        (
            upperleft_x, upperleft_y, lowerright_x, lowerright_y
        ),
        outline=(0, 0, 0))

    #right foot
    upperleft_x = right_center['x']-half_foot_width
    upperleft_y = right_center['y']-half_foot_height

    lowerright_x = right_center['x']+half_foot_width
    lowerright_y = right_center['y']+half_foot_height

    draw.rectangle(
        (
            upperleft_x, upperleft_y, lowerright_x, lowerright_y
        ),
        outline=(0, 0, 0))
    
    #im.show()
    im.save(outputdir + '/images/examples/' + obj['image'])

def drawExamples():
    for instance in usableimages:
        drawExampleImage(instance)

def prepDirectories():
    if os.path.exists(outputdir + '/images/examples') :
        shutil.rmtree(outputdir + '/images/examples')

    os.makedirs(outputdir + '/images/train/', exist_ok=True)
    os.makedirs(outputdir + '/images/examples/', exist_ok=True)
    os.makedirs(outputdir + '/labels/train/', exist_ok=True)

def outputUsedImagedToJSON():
    with open('dataset_output.json', 'w') as f:
        json.dump(usableimages, f, ensure_ascii=False, indent=4)

prepDirectories()
getUsableImagesList()
generateYOLODataSetFromUsableImages()
drawExamples()
outputUsedImagedToJSON()