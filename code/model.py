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
from keras import backend as K

import os
from PIL import Image
import keras.callbacks
import numpy as np

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

GIF_ID = []

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 20.0 * K.log(255.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


class PeriodicImageGenerator(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):

        if not os.path.exists('../result/test_while_train'):
            os.mkdir('../result/test_while_train')
        if not os.path.exists('../result/test_while_train/in_imgs'):
            os.mkdir('../result/test_while_train/in_imgs')
        if not os.path.exists('../result/test_while_train/out_imgs'):
            os.mkdir('../result/test_while_train/out_imgs')
        if not os.path.exists('../result/model'):
            os.mkdir('../result/model')

        self.epochs += 1

        if self.epochs % 10 == 0:
            rand_id = np.random.randint(len(x_train)-1)
            gif_id = GIF_ID[rand_id]
            testVal = x_train[rand_id]
            image = Image.fromarray(testVal.astype('uint8'), 'RGB')
            image.save('../result/test_while_train/in_imgs/' + str(gif_id) + '_' + str(self.epochs)+'.png')

            testVal=testVal.reshape(1, height, width, channels)

            val=model.predict(testVal,1,verbose=1)
            val=val.reshape(32,32,3)
            # print("val: %s" % (val.astype('uint8')))
            image = Image.fromarray(val.astype('uint8'), 'RGB')
            image.save('../result/test_while_train/out_imgs/' + str(gif_id) + '_' + str(self.epochs)+'.png')

        model.save('../result/model/while_train.h5')
            # Do stuff like printing metrics


if __name__ == '__main__':

    if not os.path.exists('../result'):
        os.mkdir('../result')
    if not os.path.exists('../result/model'):
        os.mkdir('../result/model')

    batch_size = 8
    epochs = 100
    hidden_units = 100

    learning_rate = 1e-3
    clip_norm = 1.0

    height = 32
    width = 64
    channels = 3

    # the data, shuffled and split between train and test sets
    print('Generating Dataset ...')

    global GIF_ID
    dataset, GIF_ID, frame_num = gen_img_sets.gen_GT_HR_sets("../data/")
    print('Dataset Generated!')
    print('dataset.shape =', dataset.shape)
    test_size = frame_num[-1]

    x_train = dataset[:-1*test_size, :-1, :, :, :] 
    y_train = dataset[:-1*test_size, -1, :, :, :]
    # x_train = dataset[:, :-1, :, :, :]
    # y_train = dataset[:, -1, :, :, :]
    x_test = dataset[-1*test_size:, :-1, :, :, :]
    y_test = dataset[-1*test_size:, -1, :, :, :]
    # x_train = x_train.reshape(x_train.shape[0],height,width,channels)

    # x_gt has all even column, x_hr has all odd, or vice-versa
    x_gt = x_train[:, 0, :, :, :]
    x_hr = x_train[:, 1, :, :, :]
    x_train = np.insert(x_hr, np.arange(32), x_gt, axis=2)

    x_gt = x_test[:, 0, :, :, :]
    x_hr = x_test[:, 1, :, :, :]
    x_test = np.insert(x_hr, np.arange(32), x_gt, axis=2)

    print('x_train.shape =', x_train.shape)
    print('x_test.shape =', x_test.shape)
    print('y_train.shape =', y_train.shape)
    print('y_test.shape =', y_test.shape)

    testVal = x_train[0].reshape(1, height, width, channels)
    # print(x_train[0].shape)
    input_shape = x_train.shape[1:]
    print('input.shape =', input_shape)

    print('Training Model ...')

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
                  metrics=[PSNRLoss])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.02,
              callbacks=[PIG])

    # Test model
    if not os.path.exists('../result/final_test'):
        os.mkdir('../result/final_test')
    if not os.path.exists('../result/final_test/in_imgs'):
        os.mkdir('../result/final_test/in_imgs')
    if not os.path.exists('../result/final_test/out_imgs'):
        os.mkdir('../result/final_test/out_imgs')

    for i in xrange(x_test.shape[0]):

        testVal = x_test[i]
        image = Image.fromarray(testVal.astype('uint8'), 'RGB')
        image.save('../result/final_test/in_imgs/test_' + str(i) + '.png')

        testVal=testVal.reshape(1, height, width, channels)

        val=model.predict(testVal,1,verbose=1)
        val=val.reshape(32,32,3)
        image = Image.fromarray(val.astype('uint8'), 'RGB')
        image.save('../result/final_test/out_imgs/test_' + str(i) + '.png')

    # Save out model
    model.save('../result/model/final.h5')




