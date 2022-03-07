"""
This file contains util functions for plotting train process and output results (accuracy, confusion matrix...)
(c) Nguyen Thi Hoai Thu
"""

import numpy as np
import pandas as pd
from pandas import read_csv, read_excel, DataFrame
from numpy import dstack
from tensorflow.keras.utils import to_categorical
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


"""
#************************************ RESULTS VISUALIZATION functions ************************************#
"""
# plot train and test process
def plot_process(_history, _title='Training Curves'):
    # accuracy
    fig1 = plt.figure(1)
    plt.plot(_history.history['accuracy'], label='Training Acc')
    plt.plot(_history.history['val_accuracy'], label='Test Acc')
    plt.title(_title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # loss
    fig2 = plt.figure(2)
    plt.plot(_history.history['loss'], label='Training Loss')
    plt.plot(_history.history['val_loss'], label='Test Loss')
    plt.title(_title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # learning rate
    try:
        fig3, axs = plt.subplots(2, 1)
        fig3.suptitle(_title)
        axs[0].plot(_history.history['loss'], label='Training Loss')
        axs[0].plot(_history.history['val_loss'], label='Test Loss')
        axs[1].plot(_history.history['lr'], label='Learning Rate')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    except KeyError as e:
        print("ERROR: There is no learning rate in the training history")


def plot_CM(_y_true, _y_predict, _class_names, _title="Confusion Matrix", precision=3):
    """ Plot confusion matrix
    :param _y_true:  The groundtruth labels with shape = [n_samples, ]
    :param _y_predict: The predicted labels with shape = [n_samples, ]
    :param _class_names: A list that contains names of all the classes in string
    :param _title: Title of the figure
    """

    cm = confusion_matrix(_y_true, _y_predict)

    """ ********* TYPE 1 with a color bar with only % ********* """
    import itertools
    figure1 = plt.figure()
    # Normalize the confusion matrix.
    cm1 = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=precision)

    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Normalized Confusion Matrix - {_title}')
    plt.colorbar()
    tick_marks = np.arange(len(_class_names))
    plt.xticks(tick_marks, _class_names, rotation=45)
    plt.yticks(tick_marks, _class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        color = "white" if cm1[i, j] > threshold else "black"
        plt.text(j, i, cm1[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    """ ********* TYPE 2 with a color bar with only real number ********* """
    figure2 = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(_title)
    plt.colorbar()
    tick_marks = np.arange(len(_class_names))
    plt.xticks(tick_marks, _class_names, rotation=90)
    plt.yticks(tick_marks, _class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def summarize_results(_scores):
    """ Calculate the average accuracy and standard deviation of all experiments' accuracy"""
    print(_scores)
    m, s = np.mean(_scores), np.std(_scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# *********************************** CLASS NAMES (for plotting confusion matrix) *********************************** #
# labels of the iSPL IMU-Stretch dataset
labels_basic_iSPL_2020_06_25 = [
    "WALKING",
    "STANDING",
    "SITTING",
    "LYING",
    "RUNNING",
    "JUMPING",
    "SIT-UP",
    "PUSH-UP",
    "DANCING"
]

# labels of the w-HAR (IMU-Stretch) dataset
labels_wHAR = [
    'JUMP',
    'LIE-DOWN',
    'SIT',
    'DOWNSTAIRS',
    'UPSTAIRS',
    'STAND',
    'TRANSITION',
    'WALK'
]
