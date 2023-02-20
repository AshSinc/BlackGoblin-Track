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
import argparse

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import common as c

import tensorflow_addons as tfa

if __name__ == "__main__":
    img_w = 240
    img_h = 240
    #model_file = "arch1_epochs20_optsgd_best"
    model_file = "outputs/models/arch7_epochs40_optsgd"
    OUTPUT_DIR = "outputs/predict/"

    print("\n\n\n")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("Not enough GPU hardware devices available")
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #tf.keras.backend.set_session(tf.Session(config=config))

    datadir = "Terrain/gtos_keras/test"

    parser = argparse.ArgumentParser(description="Runs a dataset through a model, spits out the list of files with their prediction and label \
                                                    Example: \n \
                                                    python modelPredict.py -d Images_Blutac/train -m arch1_epochs20_optsgd_best ")
    parser.add_argument("-d", "--data_dir", help="The directory of the dataset with label directories immediately under it (defaults to Images_Blutac/train)")
    parser.add_argument("-m", "--model_name", help="The name of the directory that holds the model (defaults to arch1_epochs20_optsgd_best)")
    parser.add_argument("-w", "--image_dim", help="The dimensions of the (square) images in the new dataset", type=int)
    args = parser.parse_args()
    if args.data_dir:
        datadir = args.data_dir
    if args.model_name:
        model_file = args.model_name
    if args.image_dim:
        img_w = args.image_dim
        img_h = args.image_dim


    fileList = c.createFileList(datadir)
    samples = len(fileList)
    batch_size = samples
    batch_size = 16


    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 3)

    datagen= ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=tfa.image.gaussian_filter2d,
        #preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=90,
        #brightness_range=[1.0, 1.15],
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_it = datagen.flow_from_directory(datadir,
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=False)

    num_classes = len(list(test_it.class_indices.values()))


    model = keras.models.load_model(model_file)

    probabilities = model.predict_generator(generator=test_it)
    #print(probabilities)
    y_pred = np.argmax(probabilities, axis=-1)
    #print(y_pred)
    y_true = test_it.classes
    y_files = test_it.filenames
    #print(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=list(test_it.class_indices.values()))
    c.pictureConfusionMatrix(cm, list(test_it.class_indices.keys()),figureName=OUTPUT_DIR+"confusion_matrix.png")
    f1 = f1_score(y_true, y_pred, average='micro')
    f1_all = f1_score(y_true, y_pred, average=None)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred, normalize=True)
    
    with open(OUTPUT_DIR+'predictions.txt', 'w') as f:
        
        f.writelines("The stats for model in {}:".format(model_file))
        #f.writelines("Here is the confusion matrix with labels {}".format(list(test_it.class_indices.keys())))
        #f.writelines(cm)
        f.writelines("Here is the classification report")
        f.writelines(classification_report(y_true, y_pred, labels=list(test_it.class_indices.values()), target_names=list(test_it.class_indices.keys())))
        f.writelines("End of classification report")
        f.writelines("f1 micro = {} and all {} ".format(f1, f1_all))
        f.writelines("accuracy = {}".format(acc))
        f.writelines("mcc = {}".format(mcc))
        
        f.writelines("The overall results:")
        resultsList = list(zip(list(y_files), list(y_pred), list(y_true)))
        f.writelines(tabulate(resultsList, headers=["file", "prediction", "actual"]))
        #f.writelines("\n", file=f)
        
    clear_session()