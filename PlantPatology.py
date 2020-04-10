import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

import threading
import csv

import pickle
import glob
import efficientnet.tfkeras as efn
from PIL import Image
import imagehash

import tensorflow as tf
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model

IMAGE_SIZE = 512
NUMBER_OF_THREADS = 8
results_1 = [0 for x in range(NUMBER_OF_THREADS)]
results_2 = [0 for x in range(NUMBER_OF_THREADS)]
X_train_golden = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
y_train_golden = np.full((1, 4), 0)
golden_count = 0

IMAGE_GENERATOR_FACTOR_DICT = {0: 0, 1: 0, 2: 0, 3: 0}
IMAGE_BIN_SIZE = 4200

TRAIN_IMAGES = "./data\\images_clean\\*"


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
    base_model = efn.EfficientNetB5(weights='imagenet', include_top=False, pooling='avg',
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


def findDuplicatedTrainImages():
    global TRAIN_IMAGES
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


def calcDistanceMatrix():
    ImagesToLoad = glob.glob('./data/scab/*')

    for x in range(len(ImagesToLoad)):
        y = x + 1
        distance_vector = []
        xImageHash = imagehash.average_hash(Image.open(ImagesToLoad[x]))
        print("- {}".format(ImagesToLoad[x]))
        while y != len(ImagesToLoad):
            yImageHash = imagehash.average_hash(Image.open(ImagesToLoad[y]))
            print("-- {}".format(ImagesToLoad[y]))
            print("--- {}".format(abs(xImageHash - yImageHash)))
            distance_vector.append(abs(xImageHash - yImageHash))
        print('{} --> {}'.format(ImagesToLoad[x], distance_vector))


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

    pack = 0
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

        if X_train.shape[0] >= 200:
            X_train = np.delete(X_train, 0, 0)
            y_train = np.delete(y_train, 0, 0)
            pickle.dump(X_train, open("./pickles/X_train_{}.pkl".format(pack), "wb"), protocol=4)
            pickle.dump(y_train, open("./pickles/y_train_{}.pkl".format(pack), "wb"), protocol=4)
            X_train = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
            y_train = np.full((1, 4), 0)
            pack += 1

    if X_train.shape[0] >= 1:
        X_train = np.delete(X_train, 0, 0)
        y_train = np.delete(y_train, 0, 0)
        pickle.dump(X_train, open("./pickles/X_train_{}.pkl".format(pack), "wb"), protocol=4)
        pickle.dump(y_train, open("./pickles/y_train_{}.pkl".format(pack), "wb"), protocol=4)


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

    if X_train_golden.shape[0] > 1000:
        X_train_golden = np.delete(X_train_golden, 0, 0)
        y_train_golden = np.delete(y_train_golden, 0, 0)

        X_train_golden = X_train_golden.astype('uint8')

        pickle.dump(X_train_golden, open("./augmented/X_train_{}.pkl".format(golden_count), "wb"), protocol=4)
        pickle.dump(y_train_golden, open("./augmented/y_train_{}.pkl".format(golden_count), "wb"), protocol=4)

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
            print('Data {}/{}'.format(i + 1, len(data)))
            continue

        X_train_tmp = pickle.load(open(X, "rb"))
        y_train_tmp = pickle.load(open(y, "rb"))

        X_train = np.concatenate((X_train, X_train_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)

        print('Data {}/{}'.format(i + 1, len(data)))

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

    X_train = X_train.astype('uint8')

    pickle.dump(X_train, open("./augmented/X_train.pkl", "wb"), protocol=4)
    pickle.dump(y_train, open("./augmented/y_train.pkl", "wb"), protocol=4)

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

    pickle.dump(X_train_golden, open("./augmented/X_train_{}.pkl".format(golden_count), "wb"), protocol=4)
    pickle.dump(y_train_golden, open("./augmented/y_train_{}.pkl".format(golden_count), "wb"), protocol=4)

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
            X_train = X_train.astype('uint8')

            y_train = pickle.load(open(y, "rb"))
            print('Data {}/{}'.format(i, len(data)))
            print("Size Total {}".format(X_train.shape))
            continue

        X_train_tmp = pickle.load(open(X, "rb"))
        X_train_tmp = X_train_tmp.astype('uint8')

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

        break

    print("Size {}".format(X_train.shape))

    histogram = [0 for x in range(4)]

    for i in y_train:
        histogram[list(i).index(1)] = histogram[list(i).index(1)] + 1

    print("Histogram {}".format(histogram))

    print("Shuffle data ...")
    for x in range(4):
        X_train, y_train = shuffle(X_train, y_train)

    try:
        pickle.dump(X_train, open("./golden/X_train.pkl", "wb"), protocol=4)
        pickle.dump(y_train, open("./golden/y_train.pkl", "wb"), protocol=4)
    except:
        print("-E- Failed to save augmented")

    return X_train, y_train


class MyeScheduler(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.index = 0

        self.reduceFactor = np.sqrt(0.1)

        self.LR_START = 0.00001
        self.LR_MAX = 0.001
        self.LR_MIN = 0.00001
        self.LR_RAMPUP_EPOCHS = 15
        self.LR_SUSTAIN_EPOCHS = 3
        self.LR_EXP_DECAY = .8
        self.bestLoss = None
        self.epoch = None

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('val_loss')

        self.losses.append(loss)

        if self.bestLoss is None or loss < self.bestLoss:
            self.bestLoss = loss

        if self.index > 40 or self.epoch > 40:
            minLoss = np.minimum(self.losses)
            if minLoss > self.bestLoss:
                print("-I- Stoping traning ...")
                self.model.stop_training = True

    def lrfn(self, epoch, lr):
        print("In my class with {}".format(lr))

        self.epoch = epoch

        if len(self.losses) > 25:
            self.losses = self.losses[-25:]

        elif len(self.losses) < 3:
            return self.calcLearningRateForVal(self.index)

        if self.losses[-1] < self.losses[-2]:
            print("-I- LR is OK")
            return lr

        if self.losses[-1] == self.losses[-2] == self.losses[-3]:
            print("-I- Reducing Learning rate on Plateau")
            return lr * self.reduceFactor

        self.index = self.index + 1
        print("-I- On new index: {}".format(self.index))
        n_lr = self.calcLearningRateForVal(self.index)

        return n_lr

    def calcLearningRateForVal(self, i):
        if i < self.LR_RAMPUP_EPOCHS:
            lr = (self.LR_MAX - self.LR_START) / self.LR_RAMPUP_EPOCHS * i + self.LR_START
        elif i < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
            lr = self.LR_MAX
        else:
            lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                        i - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS) + self.LR_MIN
        return lr


def trainModelResNet():
    print("--" * 34)
    batch_size = 3
    epochs = 1000

    X_train, y_train = getGoldenDataSet()

    print("Data set size {}".format(X_train.shape))

    print("Normalize data ...")
    X_train = X_train.astype('float32') / 255
    print("Done")

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    print("Training Data info:")
    print('X_train shape: {} y_train shape: {}'.format(X_train.shape, y_train.shape))

    print('X_test shape: {} y_test shape:{}'.format(X_test.shape, y_test.shape))

    print("Loading model ...")
    model = get_model()

    sweetSpotFinder = MyeScheduler()

    optimizer = tf.keras.optimizers.Adam(learning_rate=sweetSpotFinder.lrfn(0, 0))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(sweetSpotFinder.lrfn, verbose=True)

    callbacks_list = [modelCheckPoint, lr_scheduler, sweetSpotFinder]

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

    TEST_IMAGES_TO_LOAD = glob.glob("./data\\images\\Test*.jpg")
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
    print("Done")

    imageNameToResult = {}

    for img in TEST_IMAGES_TO_LOAD:
        imName = img.split('\\')[-1].split('.')[0]
        if imName not in requestImages:
            continue

        image = cv2.imread(img)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        image = image.astype('float32') / 255

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


findDuplicatedTrainImages()
saveImagesByCategory()

createDataSet()
createAugmentedDataSet()
trainModelResNet()
provideResults()
