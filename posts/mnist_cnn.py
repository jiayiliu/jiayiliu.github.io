#!/usr/bin/python3
'''Trains a simple convnet on the MNIST dataset.
Modified from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

Also see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
'''

import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--numpy_seed", default=0, type=int)
parser.add_argument("-t", "--tf_seed", default=0, type=int)
parser.add_argument("-r", "--random_seed", default=0, type=int)
parser.add_argument("-e", "--epoch", default=0, type=int)
parser.add_argument("-f", "--filename", default="test.csv", type=str)
parser.add_argument("-s", "--tf_session", action="store_true")
parser.add_argument("-a", "--hash", default="0", type=str)
parser.add_argument("-w", "--workers", default=1, type=int)
args = parser.parse_args()

if args.numpy_seed:
    import numpy as np
    np.random.seed(args.numpy_seed)
if args.tf_seed:
    import tensorflow as tf
    tf.set_random_seed(args.tf_seed)
if args.random_seed:
    import random
    random.seed(args.random_seed)
if args.hash != 0:
    os.environ["PYTHONHASHSEED"] = args.hash

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

if args.tf_session:
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    
batch_size = 128
num_classes = 10
epochs = args.epoch

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = LossHistory()

if args.workers > 0:
    datagen = ImageDataGenerator()
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=0, workers=args.workers, callbacks=[history])
else:
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
              epochs=epochs, verbose=0, callbacks=[history])
    
score = model.evaluate(x_test, y_test, verbose=0)

print("%f,%f"%(score[0], score[1]))

with open(args.filename, 'w') as f:
    for i in history.losses:
        f.write("%f\n"%i)
    
