import os
import shutil

DATA_DIR = "resources/gtos/"
TRAIN_SETS = ["train1.txt", "train2.txt"]
TEST_SETS = ["test1.txt", "test2.txt"]

APPROVED_LIST = [
"asphalt",
"asphalt_stone",
"brick",
"cement",
"dry_grass",
"dry_leaf",
"grass",
"leaf",
"moss",
"mud",
"pebble",
"sand",
"soil",
"stone_asphalt",
"stone_brick",
"stone_cement",
"stone_mud",
"turf",
"wood_chips",
]

GROUPS = {
"asphalt" : "hard_stone",
"asphalt_stone" : "loose_stone",
"brick" : "hard_stone",
"cement" : "hard_stone",
"dry_grass" : "veg",
"dry_leaf" : "veg",
"grass" : "veg",
"leaf" : "veg",
"moss" : "veg",
"mud" : "veg",
"pebble" : "loose_stone",
"sand" : "soft",
"soil" : "soft",
"stone_asphalt" : "loose_stone",
"stone_brick" : "hard_stone",
"stone_cement" : "hard_stone",
"stone_mud" : "loose_stone",
"turf" : "veg",
"wood_chips" : "soft",
}

def generateKerasFileStructure(output_dir, target_set) :
    file = open(target_set, 'r')
    lines = file.readlines()
    for line in lines :
        split = line.split(" ")[0].split("/")
        if(APPROVED_LIST.__contains__(split[0])) :
            category = GROUPS[split[0]]
            for filename in os.scandir(DATA_DIR + "/color_imgs/" + split[1]):
                if filename.is_file():
                    if not os.path.exists(output_dir + "/" + category):
                        os.makedirs(output_dir + "/" + category, exist_ok=True)
                    
                    destination = output_dir + "/" + category + "/" + filename.name           
                    if not os.path.exists(destination):
                        print(destination)
                        shutil.copyfile(filename.path, destination)

def prepDirectories(output_dir):
    if os.path.exists(output_dir) :
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

for trainPath in TRAIN_SETS :
    prepDirectories( "Terrain/gtos_keras/train")
    target_set = DATA_DIR + "labels/" + trainPath
    generateKerasFileStructure("Terrain/gtos_keras/train", target_set)

for testPath in TEST_SETS :
    prepDirectories( "Terrain/gtos_keras/test")
    target_set = DATA_DIR + "labels/" + testPath
    generateKerasFileStructure("Terrain/gtos_keras/test", target_set)