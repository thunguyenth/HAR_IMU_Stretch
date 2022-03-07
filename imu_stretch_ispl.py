""" ****************************************** IMUs + STRETCH SENSORS ******************************************
 - This is a draft of Deep learning based HAR models using IMUs and Stretch sensors
 - Testing the model with 2 datasets: 1) our lab dataset, 2) w-HAR dataset
 """

import numpy as np
from dataclasses import dataclass
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.callbacks as C
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import datetime

import model_zoo as zoo
import utils

import random
seed_value = 1234
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def load_sensor_fusion_data(path='data/ispl_data'):
    """
    Load the train-test sets that were already split and saved in the "data" folder (HARD-CODE)

    :return:
    X_IMUs_train, X_IMUs_test: shape=[n_wins, 50, 9, 3] = [n_wins, win_size, n_signals, n_sensors]
    X_Stretch_train, X_Stretch_test: shape=[n_wins, 50, 1, 2] = [n_wins, win_size, n_signals, n_sensors]
    y_train, y_test: shape=[n_wins, n_classes] (already be one-hot encoded)
    """
    from scipy.io import loadmat
    print(f'Loading data from {path} \n Loading ...')
    train_data = loadmat(f'{path}_train.mat')
    X_IMUs_train, X_Stretch_train, y_train = train_data['X_IMUs_train'], train_data['X_Stretch_train'], train_data['y_train']
    print("Train data\'s shape: ", X_IMUs_train.shape, X_Stretch_train.shape, y_train.shape)

    test_data = loadmat(f'{path}_test.mat')
    X_IMUs_test, X_Stretch_test, y_test = test_data['X_IMUs_test'], test_data['X_Stretch_test'], test_data['y_test']
    print("Test data\'s shape: ", X_IMUs_test.shape, X_Stretch_test.shape, y_test.shape)

    return X_IMUs_train, X_IMUs_test, X_Stretch_train, X_Stretch_test, y_train, y_test


class PrintLR(C.Callback):
    """ To print out the current learning rate value (when using learning rate scheduling method) """
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning Rate = ", K.eval(lr_with_decay))


def evaluate_model(model, _X_train, _y_train, _X_val, _y_val, _X_test, _y_test):
    config = Config()
    rlrop = C.ReduceLROnPlateau(monitor='loss', factor=1/config.RLROP, patience=20)
    # early = C.EarlyStopping
    print_lr = PrintLR()
    opt = tf.keras.optimizers.Adam(learning_rate=config.LR)

    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    _history = model.fit(_X_train, _y_train, validation_data=(_X_val, _y_val),
                         callbacks=[rlrop, print_lr],
                         epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, verbose=config.VERBOSE)
    # evaluate model:
    _, _accuracy = model.evaluate(_X_test, _y_test, batch_size=config.BATCH_SIZE, verbose=config.VERBOSE)
    _y_predict = model.predict(_X_test, batch_size=config.BATCH_SIZE)

    return _accuracy, _history, _y_predict


@dataclass
class Config:
    BATCH_SIZE = 256
    N_EPOCHS = 250
    LR = 0.001
    VERBOSE = 2
    RLROP = 10  # reduction rate of reducing learning rate on plateau


if __name__ == '__main__':

    DATA_PATH = 'data/ispl_data/data_2021120101'
    WIN_SIZE = 50
    N_HIDDENS = 128
    CLASS_NAMES = utils.labels_basic_iSPL_2020_06_25
    COMBINE_OPTIONS = ['single_wrist', 'single_waist', 'single_ankle', 'multi_imus', 'stretch', 'imu_stretch']
    MODEL_NAMES = ['LSTM', 'CNN_1D', 'CNN1D_LSTM_v2']

    combine_option = 'stretch'
    model_name = "LSTM"

    """---------- LOAD DATASET ----------"""
    X_IMUs_train, X_IMUs_test, X_Stretch_train, X_Stretch_test, y_train, y_test = load_sensor_fusion_data(path=DATA_PATH)

    """ Make change and test data based on the SENSOR COMBINATION """
    # Single IMU sensor
    for i in range(3):
        if combine_option == COMBINE_OPTIONS[i]:    # wrist IMU
            X_train, X_test = X_IMUs_train[:, :, :, i], X_IMUs_test[:, :, :, i]
            print('Enter the combine checking step')
            break

    # Multi-IMUs
    if combine_option == COMBINE_OPTIONS[3]:
        X_train, X_test = X_IMUs_train, X_IMUs_test

    # Stretch sensors
    if combine_option == COMBINE_OPTIONS[4]:
        X_train, X_test = X_Stretch_train, X_Stretch_test

    # IMUs + Stretch sensors
    if combine_option == COMBINE_OPTIONS[5]:
        X_IMUs_train = X_IMUs_train.reshape(X_IMUs_train.shape[0], X_IMUs_train.shape[1], -1)
        X_IMUs_test = X_IMUs_test.reshape(X_IMUs_test.shape[0], X_IMUs_test.shape[1], -1)
        X_Stretch_train = X_Stretch_train.reshape(X_Stretch_train.shape[0], X_Stretch_train.shape[1], -1)
        X_Stretch_test = X_Stretch_test.reshape(X_Stretch_test.shape[0], X_Stretch_test.shape[1], -1)

        X_train = np.concatenate((X_IMUs_train, X_Stretch_train), axis=2)
        X_test = np.concatenate((X_IMUs_test, X_Stretch_test), axis=2)

    # 3D data
    X_train, X_test = X_train.reshape(X_train.shape[0], WIN_SIZE, -1), X_test.reshape(X_test.shape[0], WIN_SIZE, -1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Test: Using only Acc data
    # X_train, X_test = X_train[:, :, 0:3], X_test[:, :, 0:3]
    # End test

    """ Run SEVERAL experiments and then get the average results """
    n_exps = 10
    Accuracies = []
    Y_pred_total = np.zeros((n_exps, y_test.shape[0]))
    F1scores = []
    for exp in range(n_exps):
        print(f"**************************************** Experiment {exp} ****************************************")

        if model_name == 'LSTM':
            my_model = zoo.model_LSTM(_inputshape=X_train.shape[1:], _n_classes=y_train.shape[1], _n_hiddens=N_HIDDENS, _dropout=0.2)
        if model_name == 'Stacked_LSTM':
            my_model = zoo.model_stacked_LSTM(_inputshape=X_train.shape[1:], _n_classes=y_train.shape[1], _n_hiddens=N_HIDDENS, _dropout=0.2)
        if model_name == 'CNN_1D':
            my_model = zoo.model_CNN_1D(_inputshape=X_train.shape[1:], _n_classes=y_train.shape[1], _n_hiddens=N_HIDDENS)
        if model_name == 'CNN1D_LSTM_v2':
            my_model = zoo.model_CNN1D_LSTM_v2(_inputshape=X_train.shape[1:], n_classes=y_train.shape[1], _n_hiddens=N_HIDDENS)
        my_model.summary()

        accuracy, history, y_predict = evaluate_model(my_model, X_train, y_train, X_test, y_test, X_test, y_test)

        accuracy = accuracy * 100.0
        print(f'Accuracy of experiment {exp} = ', accuracy)
        y_test2 = np.argmax(y_test, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        f1score = f1_score(y_test2, y_predict, average='weighted') * 100.0
        print('F1-score = ', f1score)

        # Append and print average results
        Accuracies.append(accuracy)
        print('All the accuracies = ', Accuracies)
        utils.summarize_results(Accuracies)

        # Save predicted results
        dt = datetime.now()
        accuracy = np.round(accuracy, decimals=2)
        f1score = np.round(f1score, decimals=2)

        Y_pred_total[exp, :] = y_predict
        # np.savetxt(f'results_20211129/{model_name}-{combine_option}-{dt.date()}_{dt.hour}_{dt.minute}-acc_{accuracy}-f1_{f1score}.txt', y_predict, delimiter=',')

        # Plot
        if exp == 0 or exp == n_exps - 1:   # Only plot the last experiment
            utils.plot_process(history)
            utils.plot_CM(y_test2, y_predict, CLASS_NAMES, f'{model_name}--{combine_option}')
            plt.show()

    # Save the results into mat file
    from scipy.io import savemat
    mat_file = {'Y_pred_total': Y_pred_total, 'Accuracies': Accuracies}
    savemat(f"results_20211201/{model_name}-{combine_option}-{dt.date()}_{dt.hour}_{dt.minute}.mat", mat_file)

    # # Using only once to save the y_test to mat file
    # y_test_mat_file = {'y_test': y_test2}
    # savemat('data/data_2021120101_y_test.mat', y_test_mat_file)






