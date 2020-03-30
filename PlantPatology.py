from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

import threading
import csv

import pickle
import glob
import efficientnet.tfkeras as efn
from PIL import Image
import imagehash

# import keras
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D
# from keras.layers import AveragePooling2D, Input, Flatten
# from keras.regularizers import l2
# from keras.models import Model


import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

IMAGE_SIZE = 180
NUMBER_OF_THREADS = 8
results_1 = [0 for x in range(NUMBER_OF_THREADS)]
results_2 = [0 for x in range(NUMBER_OF_THREADS)]
X_train_golden = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
y_train_golden = np.full((1, 4), 0)
golden_count = 0

IMAGE_GENERATOR_FACTOR_DICT = {0: 0, 1: 0, 2: 0, 3: 0}
IMAGE_BIN_SIZE = 10000

SUB_MEAN = False

TRAIN_IMAGES = "./data\\images_clean\\Train*"


def cleanFolders():
    path_1 = './golden/*'
    path_2 = './pickles/*'
    path_3 = './augmented/*'
    pathList = [path_1, path_2, path_3]
    for path in pathList:
        files = glob.glob(path)
        for f in files:
            os.remove(f)


def get_model(nb_classes=4):
    base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg',
                                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


def showHistory():
    if not os.path.isfile('./history/history.pkl'):
        return

    history = pickle.load(open('./history/history.pkl', "rb"))
    pd.DataFrame(history).plot(figsize=(16, 16))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    exit(1)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=4):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def findDuplicatedTrainImages():
    imageNameToLables = {}

    dataMappingFile = './data/train.csv'
    with open(dataMappingFile, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            line = line.split(',')
            imageName = line[0]
            line = line[1:]
            line = ','.join(line).replace('\n', '')
            imageNameToLables[imageName] = line
            pass
    hash_to_image_name = {}

    TRAIN_IMAGES_TO_LOAD = glob.glob(TRAIN_IMAGES)
    for im in TRAIN_IMAGES_TO_LOAD:
        hash = imagehash.average_hash(Image.open(im))
        imageName = im.split('\\')[-1].replace('.jpg', '')

        imageKabels = imageNameToLables[imageName]
        imageEnt = [(imageName, imageKabels)]

        if hash not in hash_to_image_name.keys():
            hash_to_image_name[hash] = imageEnt
        else:
            currentList = hash_to_image_name[hash]
            currentList.append(imageEnt)
            hash_to_image_name[hash] = currentList

    for k in hash_to_image_name.keys():
        if len(hash_to_image_name[k]) > 1:
            print("{}:{}".format(k, hash_to_image_name[k]))
    print("Done")


def saveImagesByCategory():
    global TRAIN_IMAGES
    dataMappingFile = './data/train.csv'
    TRAIN_IMAGES_TO_LOAD = glob.glob(TRAIN_IMAGES)

    imageCatToFolderPath = {0: './data\\healthy',
                            1: './data\\multiple_diseases',
                            2: './data\\rust',
                            3: './data\\scab'
                            }

    for k in imageCatToFolderPath:
        if not os.path.exists(imageCatToFolderPath[k]):
            os.makedirs(imageCatToFolderPath[k])

    imageNameToLables = {}
    with open(dataMappingFile, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            line = line.split(',')
            imageName = line[0]
            line = line[1:]
            line = ','.join(line).replace('\n', '')
            imageNameToLables[imageName] = line
            pass

    for im in TRAIN_IMAGES_TO_LOAD:
        image = cv2.imread(im)
        imageName = im.split('\\')[-1].replace('.jpg', '')
        print(imageName)

        label = imageNameToLables[imageName].split(',')
        label = [int(x) for x in label]
        labelIndex = label.index(1)
        print("Label {}".format(labelIndex))

        filePath = imageCatToFolderPath[labelIndex] + '\\' + imageName + '.jpg'

        cv2.imwrite(filePath, image)


def createDataSet():
    global TRAIN_IMAGES
    cleanFolders()

    TRAIN_IMAGES_TO_LOAD = glob.glob(TRAIN_IMAGES)

    imageNameToLables = {}

    dataMappingFile = './data/train.csv'
    with open(dataMappingFile, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            line = line.split(',')
            imageName = line[0]
            line = line[1:]
            line = ','.join(line).replace('\n', '')
            imageNameToLables[imageName] = line

    X_train = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
    y_train = np.full((1, 4), 0)
    totalImages = len(TRAIN_IMAGES_TO_LOAD)

    histogram = [0 for x in range(4)]

    for im in TRAIN_IMAGES_TO_LOAD:
        print('X_train  {}/{}'.format(X_train.shape[0], totalImages))

        image = cv2.imread(im)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # cv2.imshow('232',image)
        # cv2.waitKey()

        image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
        X_train = np.concatenate((X_train, image), axis=0)

        imageName = im.split('\\')[-1].replace('.jpg', '')
        label = imageNameToLables[imageName].split(',')
        label = [int(x) for x in label]

        histogram[label.index(1)] = histogram[label.index(1)] + 1

        label = np.reshape(label, (1, 4))
        y_train = np.concatenate((y_train, label), axis=0)

    X_train = np.delete(X_train, 0, 0)
    y_train = np.delete(y_train, 0, 0)

    pickle.dump(X_train, open("./pickles/X_train.pkl", "wb"))
    pickle.dump(y_train, open("./pickles/y_train.pkl", "wb"))


def imagesToSaveThreaded(imagesToSave, label, threadId, arrayId):
    global results_1, results_2, IMAGE_GENERATOR_FACTOR_DICT
    print("Thread {} working on array {}".format(threadId, arrayId))

    datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')

    X_train = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
    y_train = np.full((1, 4), 0)

    NUMBER_OF_Generator = IMAGE_GENERATOR_FACTOR_DICT[list(label[0]).index(1)]

    i = 0
    for batch in datagen.flow(imagesToSave, batch_size=1):
        if i >= NUMBER_OF_Generator:
            break
        X_train = np.concatenate((X_train, batch), axis=0)
        y_train = np.concatenate((y_train, label), axis=0)

        i += 1

    X_train = np.delete(X_train, 0, 0)
    y_train = np.delete(y_train, 0, 0)

    if X_train.shape[0] != y_train.shape[0]:
        print("ERROR in imagesToSaveThreaded")
        exit(0)

    if arrayId == 0:
        results_1[threadId] = X_train, y_train
    else:
        results_2[threadId] = X_train, y_train


def copyDataThread(arrayId):
    print("Copy...")
    global X_train_golden, y_train_golden, results_1, results_2, golden_count
    if arrayId == 0:
        for t in results_1:
            if t == 0:
                continue
            X_train = t[0]
            y_train = t[1]

            if X_train.shape[0] != y_train.shape[0]:
                print("ERROR in copy")
                exit(0)

            X_train_golden = np.concatenate((X_train_golden, X_train), axis=0)
            y_train_golden = np.concatenate((y_train_golden, y_train), axis=0)
        results_1 = [0 for x in range(NUMBER_OF_THREADS)]
    else:
        for t in results_2:
            if t == 0:
                continue
            X_train = t[0]
            y_train = t[1]

            if X_train.shape[0] != y_train.shape[0]:
                print("ERROR in copy")
                exit(0)

            X_train_golden = np.concatenate((X_train_golden, X_train), axis=0)
            y_train_golden = np.concatenate((y_train_golden, y_train), axis=0)
        results_2 = [0 for x in range(NUMBER_OF_THREADS)]

    if X_train_golden.shape[0] > 1999:
        X_train_golden = np.delete(X_train_golden, 0, 0)
        y_train_golden = np.delete(y_train_golden, 0, 0)

        pickle.dump(X_train_golden, open("./augmented/X_train_{}.pkl".format(golden_count), "wb"))
        pickle.dump(y_train_golden, open("./augmented/y_train_{}.pkl".format(golden_count), "wb"))

        golden_count = golden_count + 1

        X_train_golden = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
        y_train_golden = np.full((1, 4), 0)


def createAugmentedDataSet():
    global X_train_golden, y_train_golden, IMAGE_GENERATOR_FACTOR_DICT
    X_train = None
    y_train = None

    data = glob.glob("./pickles/X_train*")
    data.sort()

    labels = glob.glob("./pickles/y_train*")
    labels.sort()

    for i in range(len(data)):
        X = data[i]
        y = labels[i]

        if X.replace('X_train', '') != y.replace('y_train', ''):
            print("ERROR in createAugmentedDataSet ")
            exit(0)

        if X_train is None:
            X_train = pickle.load(open(X, "rb"))
            y_train = pickle.load(open(y, "rb"))
            continue

        X_train_tmp = pickle.load(open(X, "rb"))
        y_train_tmp = pickle.load(open(y, "rb"))

        X_train = np.concatenate((X_train, X_train_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)

        print('Data {}/{}'.format(i, len(data)))

    if X_train.shape[0] != y_train.shape[0]:
        print("ERROR in createAugmentedDataSet")
        exit(0)

    histogram = [0 for x in range(4)]

    for i in y_train:
        histogram[list(i).index(1)] = histogram[list(i).index(1)] + 1

    print("Histogram {}".format(histogram))

    for x in range(len(histogram)):
        IMAGE_GENERATOR_FACTOR_DICT[x] = IMAGE_BIN_SIZE // histogram[x]
    print("FACTORS {}".format(IMAGE_GENERATOR_FACTOR_DICT))

    pickle.dump(X_train, open("./augmented/X_train.pkl", "wb"))
    pickle.dump(y_train, open("./augmented/y_train.pkl", "wb"))

    arry_id = 0
    threads = list()

    copyThread = threading.Thread(target=copyDataThread, args=(arry_id,))
    copyThread.start()

    for i in range(X_train.shape[0]):
        print("Image {}/{}".format(i, X_train.shape[0]))
        print(X_train_golden.shape)

        image = X_train[i]

        image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        label = y_train[i]
        label = np.reshape(label, (1, 4))

        t = threading.Thread(target=imagesToSaveThreaded, args=(image, label, len(threads), arry_id,))
        threads.append(t)
        t.start()

        if len(threads) == NUMBER_OF_THREADS:
            for t in threads:
                t.join()

            if copyThread.is_alive():
                copyThread.join()

            copyThread = threading.Thread(target=copyDataThread, args=(arry_id,))
            copyThread.start()

            arry_id = 0 if arry_id == 1 else 1
            threads.clear()

    if copyThread.is_alive():
        copyThread.join()

    copyThread = threading.Thread(target=copyDataThread, args=(arry_id,))
    copyThread.start()
    copyThread.join()

    X_train_golden = np.delete(X_train_golden, 0, 0)
    y_train_golden = np.delete(y_train_golden, 0, 0)

    pickle.dump(X_train_golden, open("./augmented/X_train_{}.pkl".format(golden_count), "wb"))
    pickle.dump(y_train_golden, open("./augmented/y_train_{}.pkl".format(golden_count), "wb"))

    print("Done")


def getGoldenDataSet():
    if os.path.isfile('./golden/X_train.pkl'):
        X_train = pickle.load(open('./golden/X_train.pkl', "rb"))
        y_train = pickle.load(open('./golden/y_train.pkl', "rb"))
        return X_train, y_train

    X_train = None
    y_train = None

    data = glob.glob("./augmented/X_train*")
    data.sort()

    labels = glob.glob("./augmented/y_train*")
    labels.sort()

    for i in range(len(data)):
        X = data[i]
        y = labels[i]

        if X.replace('X_train', '') != y.replace('y_train', ''):
            print("ERROR loadDataSet")
            exit(0)

        print("Loading {}".format(X))
        print("Loading {}".format(y))

        if X_train is None:
            X_train = pickle.load(open(X, "rb"))
            y_train = pickle.load(open(y, "rb"))
            print('Data {}/{}'.format(i, len(data)))
            print("Size Total {}".format(X_train.shape))
            continue

        X_train_tmp = pickle.load(open(X, "rb"))
        y_train_tmp = pickle.load(open(y, "rb"))

        print("X_train_tmp Size {}".format(X_train_tmp.shape))
        print("y_train_tmp Size {}".format(y_train_tmp.shape))

        if X_train_tmp.shape[0] != y_train_tmp.shape[0]:
            print("ERROR in loadDataSet")
            exit(0)

        X_train = np.concatenate((X_train, X_train_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)

        if X_train.shape[0] != y_train.shape[0]:
            print("ERROR in loadDataSet 2 ")
            exit(0)

        print('Data {}/{}'.format(i, len(data)))
        print("Size Total {}".format(X_train.shape))

    print("Size {}".format(X_train.shape))

    histogram = [0 for x in range(4)]

    for i in y_train:
        histogram[list(i).index(1)] = histogram[list(i).index(1)] + 1

    print("Histogram {}".format(histogram))

    #    pickle.dump(X_train, open("./golden/X_train.pkl", "wb"), protocol=4)
    #    pickle.dump(y_train, open("./golden/y_train.pkl", "wb"), protocol=4)

    return X_train, y_train


LR_START = 0.00001
LR_MAX = 0.0001
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 100
LR_SUSTAIN_EPOCHS = 10
LR_EXP_DECAY = .94


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


# rng = [i for i in range(200)]
# y = [lrfn(x) for x in rng]
# plt.plot(rng, y)
# plt.show()

# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def trainModelResNet():
    batch_size = 10
    epochs = 200
    num_classes = 4
    n = 2

    depth = n * 9 + 2

    model_type = 'ResNet%dv%d' % (depth, 2)
    print(model_type)

    X_train, y_train = getGoldenDataSet()
    X_train = X_train.astype('float32') / 255

    if SUB_MEAN:
        GLOBAL_xMean = np.mean(X_train, axis=0)
        pickle.dump(GLOBAL_xMean, open("./GLOBAL_xMean.pkl", "wb"))
        X_train -= GLOBAL_xMean

    for x in range(4):
        X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    input_shape = X_train.shape[1:]

    print('x_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
    model = get_model()

    # opt = keras.optimizers.SGD(lr=0.01, momentum=0.8, nesterov=False)
    opti = tf.keras.optimizers.Adam(learning_rate=lrfn(0))
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])

    fileName = 'Model_epoch{epoch:02d}_loss{val_loss:.5f}_acc{val_accuracy:.5f}.h5'

    modelCheckPoint = tf.keras.callbacks.ModelCheckpoint(
        './model/' + fileName,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    # earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, verbose=1, cooldown=0,
                                                     min_lr=0.5e-6)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    # callbacks_list = [modelCheckPoint, reduce_lr, lr_scheduler, earlyStopping]
    callbacks_list = [modelCheckPoint, reduce_lr, lr_scheduler]

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                        shuffle=True, callbacks=callbacks_list)

    scores = model.evaluate(X_test, y_test, verbose=1)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    pickle.dump(history.history, open('./history/history.pkl', 'wb'))


def provideResults():
    requestImages = []
    requestImagesTups = []

    with open('./data/test.csv', 'r') as in_file:
        in_file.readline()
        for line in in_file:
            line = line.replace('\n', '').strip()
            requestImages.append(line)

    for req in requestImages:
        reqs = int(req.split("_")[1])
        requestImagesTups.append((reqs, req))

    requestImagesTups.sort(key=lambda tup: tup[0], reverse=False)
    requestImages = [t[1] for t in requestImagesTups]

    TEST_IMAGES_TO_LOAD = glob.glob("C:\\Users\\Avi\\Desktop\\PyProj\\PlantPathology\\data\\images\\Test*.jpg")
    TEST_IMAGES_TO_LOAD.sort()

    modelsToLoad = glob.glob("./model/Model*")

    lossToModelName = []
    for modelName in modelsToLoad:
        index = modelName.find("loss")
        end = modelName.find("_", index)
        loss = modelName[index:end].replace('loss', '')
        loss = float(loss)
        lossToModelName.append((loss, modelName))

    lossToModelName.sort(key=lambda tup: tup[0], reverse=True)
    print("Selected Model {}".format(lossToModelName[-1][1]))
    print("Loading Model ...")
    model = tf.keras.models.load_model(lossToModelName[-1][1])

    imageNameToResult = {}

    for img in TEST_IMAGES_TO_LOAD:
        imName = img.split('\\')[-1].split('.')[0]
        if imName not in requestImages:
            continue

        image = cv2.imread(img)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        image = image.astype('float32') / 255

        if SUB_MEAN:
            X_train_Mean = pickle.load(open("./GLOBAL_xMean.pkl", "rb"))
            image -= X_train_Mean

        image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        res = model.predict(image)
        imageNameToResult[imName] = res

        print(res)

    if len(imageNameToResult) != len(requestImages):
        print("ERROR imageNameToResult not in the same size as requestImages")
        exit(1)

    with open('Final_results.csv', mode='w', newline='') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])

        for reqImg in requestImages:
            res = imageNameToResult[reqImg]
            writer.writerow([reqImg, res[0][0], res[0][1], res[0][2], res[0][3]])


# findDuplicatedTrainImages()
# saveImagesByCategory()

# createDataSet()
# createAugmentedDataSet()
# trainModelResNet()
provideResults()
