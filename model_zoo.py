"""
Models that are used in the paper "An Investigation on Deep Learning-Based Activity Recognition Using IMUs and Stretch Sensors"
(c) Nguyen Thi Hoai Thu
"""
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import regularizers

import random
random.seed(11)
np.random.seed(11)


def model_CNN_1D(_inputshape, _n_classes, _n_hiddens=128):
    """ Require 3D data: [n_samples, n_timesteps, n_channels]"""
    _model = Sequential()
    _model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=_inputshape, data_format="channels_last"))
    _model.add(BatchNormalization())
    _model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last"))
    _model.add(MaxPooling1D(pool_size=2, data_format="channels_last"))
    _model.add(Conv1D(filters=64, kernel_size=3, activation='relu', data_format="channels_last"))
    _model.add(Conv1D(filters=64, kernel_size=3, activation='relu', data_format="channels_last"))
    _model.add(MaxPooling1D(pool_size=2, data_format="channels_last"))
    _model.add(Flatten())

    _model.add(Dense(_n_hiddens, activation='relu'))
    _model.add(Dense(_n_classes, activation='softmax'))

    # _model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])

    return _model


def model_LSTM(_inputshape, _n_classes, _n_hiddens=10, _dropout=0.2):
    """ Require 3D data: [n_samples, n_timesteps (or sequence size), n_features]"""
    _model = Sequential()
    _model.add(BatchNormalization(input_shape=_inputshape))       # Tried but the result is really bad
    _model.add(LSTM(_n_hiddens))
    # _model.add(Dropout(_dropout))
    _model.add(Dense(_n_hiddens, activation='relu', kernel_regularizer=regularizers.l2()))
    _model.add(Dense(_n_classes, activation='softmax'))

    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


def model_stacked_LSTM(_inputshape, _n_classes, _n_hiddens=128, _dropout=0.2):
    """ Require 3D data: [n_samples, n_timesteps (or sequence size), n_features]"""
    _model = Sequential()
    _model.add(LSTM(_n_hiddens, input_shape=_inputshape, return_sequences=True))
    _model.add(LSTM(_n_hiddens))
    # _model.add(Dropout(0.3))
    _model.add(Dense(_n_hiddens, activation='relu', kernel_regularizer=regularizers.l1()))
    _model.add(Dense(_n_classes, activation='softmax'))

    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(_model, show_shapes=True, to_file='model_LSTM.png')
    return _model


def model_CNN1D_LSTM_v2(_inputshape, _n_classes=10, _n_hiddens=128):
    """ Require 3D data: [n_samples, sequence_size, n_channels]"""
    input_layer = Input(shape=_inputshape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(input_layer)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(x)
    x = MaxPooling1D(pool_size=2, data_format="channels_last")(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', data_format="channels_last")(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', data_format="channels_last")(x)
    x = MaxPooling1D(pool_size=2, data_format="channels_last")(x)
    # x = Reshape(x.shape[1], -1)(x)    NO NEED reshape in the case of 1D CNN
    x = LSTM(units=_n_hiddens)(x)
    x = Dense(_n_hiddens, activation='relu')(x)
    output_layer = Dense(_n_classes, activation='softmax')(x)

    _model = Model(inputs=input_layer, outputs=output_layer)

    return _model


if __name__ == "__main__":
    model = model_CNN1D_LSTM_v2(_inputshape=(128, 9), _n_classes=10, _n_hiddens=128)
    model.summary()





