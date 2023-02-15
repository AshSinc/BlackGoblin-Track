import os
import shutil

### DATA_DIR = gtos directory
### TARGET_SET = test or train . txt file in GTOS/labels
### OUTPUT_TYPE = either test or train, could match TARGET_SET above as a general rule (outputs to test or train which is required by trainModel.py)
### OUTPUT_DIR = where to copy resulting images

DATA_DIR = "/home/ash/Downloads/gtos/gtos"
TARGET_SET = DATA_DIR + "/labels/test1.txt" 
OUTPUT_TYPE = "test"
OUTPUT_DIR = "Terrain/gtos_keras/" + OUTPUT_TYPE

def generateKerasFileStructure() :
    file = open(TARGET_SET, 'r')
    lines = file.readlines()
    for line in lines :
        split = line.split(" ")[0].split("/")

        for filename in os.scandir(DATA_DIR + "/color_imgs/" + split[1]):
            if filename.is_file():
                if not os.path.exists(OUTPUT_DIR + "/" + split[0]):
                    os.makedirs(OUTPUT_DIR + "/" + split[0], exist_ok=True)
                
                destination = OUTPUT_DIR + "/" + split[0] + "/" + filename.name           
                if not os.path.exists(destination):
                    print(destination)
                    shutil.copyfile(filename.path, destination)

def prepDirectories():
    if os.path.exists(OUTPUT_DIR) :
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

prepDirectories()
generateKerasFileStructure()