import os
import random
import shutil

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def makeFreshDir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def createFileList(myDir, formats=['.tif', '.png', '.tiff', '.jpg', 'jpeg']):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            for format in formats:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
    return fileList

def pictureConfusionMatrix(cm, labels, figureName="confusion_matrix.png"):
    df_cm = pd.DataFrame(cm, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("confusion_matrix.png")
