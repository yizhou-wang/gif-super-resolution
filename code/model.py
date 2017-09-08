from __future__ import print_function

import keras
import gen_img_sets
from keras.models import Sequential, Input
from keras.layers import Dense, Activation, TimeDistributed, SeparableConv2D
from keras.layers import SimpleRNN, Conv2D, LSTM, Embedding, MaxPool2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import initializers
from keras.optimizers import RMSprop, adam

import os
from PIL import Image
import keras.callbacks
import numpy as np


class PeriodicImageGenerator(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 25 == 0:
            testVal = x_train[np.random.randint(len(x_train)-1)]
            image = Image.fromarray(testVal.astype('uint8'), 'RGB')
            image.save('../result/image'+str(self.epochs)+'.jpg')

            testVal=testVal.reshape(1, height, width, channels)

            val=model.predict(testVal,1,verbose=1)
            val=val.reshape(32,32,3)
            print("val: %s" % (val.astype('uint8')))
            image = Image.fromarray(val.astype('uint8'), 'RGB')
            image.save('../result/image'+str(self.epochs)+'_predicted.jpg')

        model.save('../result/kerasModel_anshul_noAttn.h5')
            # Do stuff like printing metrics


if __name__=='__main__':

    if not os.path.exists('../result'):
        os.mkdir('../result')

    batch_size = 8
    epochs = 100
    hidden_units = 100

    learning_rate = 1e-6
    clip_norm = 1.0

    height = 32
    width = 64
    channels = 3

    # the data, shuffled and split between train and test sets
    print('Generating Dataset ...')

    dataset, gif_id = gen_img_sets.gen_GT_HR_sets("../data/")
    print('Dataset Generated!')
    print('dataset.shape =', dataset.shape)

    x_train = dataset[:-10,:-1,:,:,:]; y_train = dataset[:-10,-1,:,:,:]
    x_test = dataset[-10:,:-1,:,:,:]; y_test = dataset[-10:,-1,:,:,:]
    #x_train = x_train.reshape(x_train.shape[0],height,width,channels)

    #x_gt has all even column, x_hr has all odd, or vice-versa
    x_gt = x_train[:,0,:,:,:]
    x_hr = x_train[:,1,:,:,:]
    x_train = np.insert(x_hr, np.arange(32), x_gt, axis=2)

    x_gt = x_test[:,0,:,:,:]
    x_hr = x_test[:,1,:,:,:]
    x_test = np.insert(x_hr, np.arange(32), x_gt, axis=2)

    print('x_train.shape =', x_train.shape)
    print('x_test.shape =', x_test.shape)
    print('y_train.shape =', y_train.shape)
    print('y_test.shape =', y_test.shape)

    testVal = x_train[0].reshape(1, height, width, channels)
    # print(x_train[0].shape)
    input_shape = x_train.shape[1:]
    print('input.shape =', input_shape)

    print('Evaluating...')

    PIG = PeriodicImageGenerator()

    model = Sequential()

    row, col, pixel = x_train.shape[1:]
    row_hidden = 512
    col_hidden = 512
    # 4D input.
    x = Input(shape=(row, col, pixel))

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)

    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu',
                     input_shape=(height,width,channels)))
    model.add(Dropout(0.15))
    model.add(MaxPool2D(pool_size=(1,2)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(SeparableConv2D(128, (1,1)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Dropout(0.30))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros'))
    model.add(Conv2D(3, (1, 1), padding="same", activation="relu"))

    #model.add(Activation('relu'))
    #model.add(Embedding(256, output_dim=256))
    # model.add(LSTM( 128,
    #                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
    #                 recurrent_initializer=initializers.Identity(gain=1.0),
    #                 activation='relu'))

    adam = adam(lr=learning_rate)
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.02,
              callbacks=[PIG])

    # Test model
    for i in xrange(x_test.shape[0]):

        testVal = x_test[i]
        image = Image.fromarray(testVal.astype('uint8'), 'RGB')
        image.save('../result/image'+str(i)+'_Test.jpg')

        testVal=testVal.reshape(1, height, width, channels)

        val=model.predict(testVal,1,verbose=1)
        val=val.reshape(32,32,3)
        image = Image.fromarray(val.astype('uint8'), 'RGB')
        image.save('../result/image'+str(i)+'_Test_predicted.jpg')

    # Save out model
    model.save('../result/kerasModel_anshul_noAttn.h5')




