import os
import numpy as np
import tensorflow as tf
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

import common as c

def modelArchitecture(input_shape, num_classes, architectureNumber):
    print(num_classes)
    if architectureNumber == 0:
        modelName = "first_architecture"
        model = Sequential()
        model.add(Conv2D(32, (4, 4), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 1:
        modelName = "three_3x3_convs_and_fc"
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 2:
        modelName = "AlexNet_modified"
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))


        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Model dropout to prevent overfitting
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 3:
        modelName = "VGG-16_architecture"
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 4:
        # previous stride of 1 too big, changed to stride 4
        modelName = "MNIST_99.25Simple_Stride2"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    if architectureNumber == 5:
        # previous stride of 1 too big, changed to stride 4
        modelName = "MNIST_99.25Simple_Stride4"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(4, 4), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    if architectureNumber == 6:
        # previous stride of 1 too big, changed to stride 4
        modelName = "MNIST_99.25Simple_Stride1"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    if architectureNumber == 7:
        modelName = "Super simple LeNet"
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (5, 5)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 8:
        modelName = "Simpler than LeNet"
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    
    print(modelName)
    print(model.summary())
    print("Number of parameters")
    print(model.count_params())
    return model, modelName

def decideDataGeneration(dataGenType=0):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        #vertical_flip = True,
        #zca_whitening = True,
        rotation_range = 45
    )

    if dataGenType == 1:
        datagen= ImageDataGenerator(
            preprocessing_function=preprocess_input,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            rotation_range=90,
            #brightness_range=[1.0, 1.15],
            zoom_range=0.2,
            horizontal_flip=True,
            #vertical_flip=True
        )
    if dataGenType == 2: # This one for the test set?
        datagen= ImageDataGenerator(
            rescale=1. / 255,
        )
    return datagen



"""
No command line options for this one yet because this is the file that does the heavy lifting.
You need a dataset with test/train split and, under that, labels.
Labels can be binary (e.g. positive or negative presence of anomaly) or you can have multiple labels.

You also need to choose:
- number of epochs (this affects training time and should be selected so that training and validation loss is minimised)
- model architecture (model 1 is recommended)
- an optimiser (sgd is recommended, but adam2 can also work with slightly longer training times)
- image dimensions (default to 224x224)
- batch size (small dataset, so defaults to 16)

This is set up to train and test a number of models as specificed in the array listOfTests.
The models take a while to train, so it's worth leaving them running.
A model will only be saved if it beats a preceding model.
Therefore the best model is the last one to be saved.
All results will be reported via the command line.


"""


if __name__ == "__main__":
    img_w = 240
    img_h = 240
    doTheSaving = True

    batch_size = 16
    print("\n\n\n")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("Not enough GPU hardware devices available")
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #tf.keras.backend.set_session(tf.Session(config=config))

    datadir = "Terrain/gtos_keras"
    traindir = '{}/train'.format(datadir)
    filelist = []
    for root, dirs, files in os.walk(traindir):
        for file in files:
            filelist.append(file)

    trainSamples = len(filelist)

    testdir = '{}/test'.format(datadir)
    
    filelist = []
    for root, dirs, files in os.walk(testdir):
        for file in files:
            filelist.append(file)

    testSamples = len(filelist)



    if K.image_data_format() == 'channels_first':
        print("channels first")
        input_shape = (3, img_w, img_h)
    else:
        print("channels last")
        input_shape = (img_w, img_h, 3)

    datagen = decideDataGeneration(0)
    datagen_test = decideDataGeneration(2)

    train_it = datagen.flow_from_directory('{}/train'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=True)

    test_it = datagen_test.flow_from_directory('{}/test'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=False)

    num_classes = len(list(train_it.class_indices.values()))

    bestf1 = 0
    bestNetwork = "unknown"
    bestMCC = 0
    bestAccuracy = 0
    bestf1model = 0
    bestMCCmodel = 0
    bestAccuracyModel = 0



    # Note there is a list of tests, but you can also define only 1 test in the list if you want!
    #listOfTests = [[architectureNumber, epochs, optimiser],]
    listOfTests = [ [7, 20, "sgd" ],
                  #  [1, 20, "adam2" ],
    #                [1, 20, "sgd" ]
    ]
    #listOfTests = [ [7, 30, "adam2" ],
    #                [8, 50, "adam2" ],
    #                [1, 30, "sgd" ],
    #                [7, 30, "sgd" ],
    #                [8, 50, "sgd" ],
    #                [1, 40, "sgd" ]
    #]

    architectureNumber = 1
    optimiser = "sgd"
    epochs = 30

    resultsList = []
    for test in listOfTests:

        architectureNumber = test[0]
        epochs = test[1]
        optimiser = test[2]
        optimiser_name = test[2]

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%{}".format(optimiser_name))
        if "sgd1" in optimiser:
            # Trying something with SGD.
            print("switching to optimer SGD1")
            mySgd = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
            optimiser = mySgd
        if "adam1" in optimiser:
            # Trying something with adam.
            print("switching to optimer Adam1")
            optimiser = optimizers.Adam(learning_rate=0.0001)
        elif "adam2" in optimiser:
            # Trying something with adam.
            print("switching to optimer Adam2")
            optimiser = optimizers.Adam(learning_rate=0.00001)
        elif "adam3" in optimiser:
            # Trying something with adam.
            print("switching to optimer Adam3")
            optimiser = optimizers.Adam(learning_rate=0.000001)

        model, modelName = modelArchitecture(input_shape, num_classes, architectureNumber)


        print("Compiling the model: {}".format(modelName))
        model.compile(loss='mse',
                      optimizer=optimiser,
                      metrics=[metrics.categorical_accuracy])


        stepsPerEpoch = trainSamples // batch_size
        if stepsPerEpoch < 20:
            stepsPerEpoch = 20
        print("There will be {} steps per epoch".format(stepsPerEpoch))



        valSteps = testSamples // batch_size

        # can miss out the early termination
        early = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')

        print("Fitting the model: {}".format(modelName))
        model.fit_generator(
            train_it,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            validation_data=test_it,
            validation_steps=valSteps,
            callbacks=[early]
            )

        probabilities = model.predict_generator(generator=test_it)
        #print(probabilities)
        y_pred = np.argmax(probabilities, axis=-1)
        #print(y_pred)
        y_true = test_it.classes
        #print(y_true)

        cm = confusion_matrix(y_true, y_pred, labels=list(test_it.class_indices.values()))
        c.pictureConfusionMatrix(cm, list(test_it.class_indices.keys()))
        print("The stats for {} after {} epochs with {} opt:".format(modelName, epochs, optimiser_name))
        f1 = f1_score(y_true, y_pred, average='micro')
        f1_all = f1_score(y_true, y_pred, average=None)
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred, normalize=True)
        print("Here is the confusion matrix with labels {}".format(list(test_it.class_indices.keys())))
        print(cm)
        print("Here is the classification report")
        print(classification_report(y_true, y_pred, labels=list(test_it.class_indices.values()), target_names=list(test_it.class_indices.keys())))
        print("End of classification report")
        print("f1 micro = {} and all {} ".format(f1, f1_all))
        print("accuracy = {}".format(acc))
        print("mcc = {}".format(mcc))
        myResults = (architectureNumber, epochs, optimiser_name, f1, acc, mcc)
        resultsList.append(myResults)
        # print out the results as we go...
        print(tabulate(resultsList, headers=["archNo", "epochs", "opt", "f1", "acc", "mcc"]))
        print(cm[0][0])

        theseResults = [(datadir, modelName, epochs, optimiser_name, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1)]
        print(theseResults)
        #tryit = tabulate(theseResults, headers=["datadir", "augmentation", "Model", "epochs", "opt", "cm00", "cm01", "cm10", "cm11", "f1"])
        with open('output.txt', 'a') as f:
            print(tabulate(theseResults, headers=["datadir", "augmentation", "Model", "epochs", "opt", "cm00", "cm01", "cm10", "cm11", "f1"]), file=f)
            print("\n", file=f)
        with open('output_alt.txt', 'a') as f:
            print(theseResults)
            print("\n", file=f)

        if doTheSaving:
            saveModel = False

            if f1 > bestf1:
                bestf1 = f1
                bestf1model = model
                saveModel = True

            if mcc > bestMCC:
                bestMCC = mcc
                bestMCCmodel = model
                saveModel = True

            if acc > bestAccuracy:
                bestAccuracy = acc
                bestAccuracyModel = model
                saveModel = True

            # save model to file
            if saveModel:
                modelBaseFilename = "arch{}_epochs{}_opt{}".format(architectureNumber, epochs, optimiser_name)
                print("Saving to {}".format(modelBaseFilename))
                model.save(modelBaseFilename)
                #model_json = model.to_json()
                #with open("{}.json".format(modelBaseFilename), "w") as json_file:
                #    json_file.write(model_json)
                #model.save_weights("{}.h5".format(modelBaseFilename))
        clear_session()

    print("The overall results:")
    print(resultsList)
