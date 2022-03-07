""" ---------------------- Plot CONFUSION MATRIX in Python ---------------------- """

import numpy as np
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import utils



CM_TITLE = 'w-HAR dataset'
CLASS_NAMES = utils.labels_wHAR
PATH_TEST = 'data/w-har_data/data_wHAR_2021120101_y_test.mat'
# PATH_PRED = 'results/w-har_data/results_wHAR_20211201/CNN1D_LSTM_v2-imu-2021-12-01_12_19.mat'          # IMU
# PATH_PRED = 'results/w-har_data/results_wHAR_20211201/CNN1D_LSTM_v2-stretch-2021-12-01_14_2.mat'       # Stretch
PATH_PRED = 'results/w-har_data/results_wHAR_20211201/CNN1D_LSTM_v2-imu_stretch-2021-12-01_14_11.mat'  # IMU-Stretch

# CM_TITLE = 'Stretch sensor - iSPL dataset'
# CLASS_NAMES = utils.labels_basic_iSPL_2020_06_25
# PATH_TEST = 'data/ispl_data/data_2021120101_y_test.mat'
# PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-single_wrist-2021-12-01_8_40.mat'     # IMU-Wrist
# # PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-single_waist-2021-12-01_10_18.mat'    # IMU-Waist
# # PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-single_ankle-2021-12-01_11_53.mat'    # IMU-Ankle
# # PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-multi_imus-2021-12-01_1_35.mat'       # 3 IMUs
# # PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-stretch-2021-12-01_2_24.mat'          # Stretch
# # PATH_PRED = 'results/ispl_data/results_20211201/CNN1D_LSTM_v2-imu_stretch-2021-12-01_8_33.mat'      # IMU-Stretch


# # Load predicted y
Y_pred_total = loadmat(PATH_PRED)['Y_pred_total']
n_exps = Y_pred_total.shape[0]

# # Load the ground-truth y
y_test = loadmat(PATH_TEST)['y_test']

# As the results contain multiple running experiments (e.g., 10), so need
# to create the Y_test_total to have same amount of labels as Y_pred_total
Y_test_total = y_test
for i in range(n_exps - 1):
    Y_test_total = np.concatenate((Y_test_total, y_test), axis=0)

# Reshape the two Y_test_total and Y_pred_total to 1D arrays
Y_test_total = np.reshape(Y_test_total, (-1, 1))
Y_pred_total = np.reshape(Y_pred_total, (-1, 1))

utils.plot_CM(Y_test_total, Y_pred_total, CLASS_NAMES, CM_TITLE, precision=2)
plt.show()
print('End')
np.concatenate