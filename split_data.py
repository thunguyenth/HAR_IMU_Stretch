
def load_and_split_data02(path_list=[], win_size=50, n_signals=9, n_classes=9, test_size=0.3):
    """
    ++++++++++ 2021-11-29 ++++++++++
    - This function is to LOAD the ORIGINAL DATA from original mat files, then randomly split the data into train-test
    sets with the size of the test set = test_size.
    - The reason why creating this function is: if we keep randomly splitting the data each time we run the model, it will
    be not fair when comparing the models' performance. Thus we need to fixedly SPLIT DATA into train-test sets ONLY
    ONCE!!! Then save them to mat files in THIS FOLDER. And load it every time we run the model.
    - This method is used only for this dataset (kind of hard coding).
    :param path_list: a list that contain the paths of different mat files
    :param win_size: size of the window (unit=data samples)
    :param n_signals: number of signals in each IMU sensor
    :param n_classes: number of activities in the dataset
    :param test_size: size of the test data compared to the whole dataset
    :return:
    """
    import numpy as np
    from scipy.io import loadmat, savemat
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from datetime import datetime   # just to mark the date that the data was split

    if len(path_list) == 0:
        print('Error: There is no path of data file. Check path_list')
        return 0

    for i in range(len(path_list)):
        mat_file = loadmat(path_list[i])
        temp_IMUs = mat_file['win_IMUs']
        temp_Stretch = mat_file['win_stretch']
        temp_labels = mat_file['labels']

        # At the first mat file, need to create at the X_IMUs, X_stretch and y
        if i == 0:
            X_IMUs = temp_IMUs
            X_Stretch = temp_Stretch
            y = temp_labels
        else:
            X_IMUs = np.concatenate((X_IMUs, temp_IMUs), axis=0)
            X_Stretch = np.concatenate((X_Stretch, temp_Stretch), axis=0)
            y = np.concatenate((y, temp_labels), axis=0)

    # Reshape from (n_wins, win_size*n_signals) to (n_wins, win_size, n_signals)
    X_IMUs = np.reshape(X_IMUs, (X_IMUs.shape[0], win_size, n_signals, -1), order='F')
    X_Stretch = np.reshape(X_Stretch, (X_Stretch.shape[0], win_size, -1), order='F')

    # Categorize labels y
    y = to_categorical(y, num_classes=n_classes)

    # Split into train-test sets
    X_IMUs_train, X_IMUs_test, X_Stretch_train, X_Stretch_test, y_train, y_test = train_test_split(X_IMUs, X_Stretch, y,
                                                                                                   test_size=test_size)

    # Change to dictionaries
    train_data = {'X_IMUs_train': X_IMUs_train, 'X_Stretch_train': X_Stretch_train, 'y_train': y_train}
    test_data = {'X_IMUs_test': X_IMUs_test, 'X_Stretch_test': X_Stretch_test, 'y_test': y_test}

    # Save to mat files
    time = datetime.now()
    time_string = f'{time.year}{time.month}{time.day}{time.hour}'
    savemat(f"data/w-har_data/data_{time_string}_train.mat", train_data)
    savemat(f"data/w-har_data/data_{time_string}_test.mat", test_data)


if __name__ == '__main__':
    """ w-HAR dataset (only the data that has stretch sensor's frequency = 25Hz is used)"""
    path_list = ['data/w-har_data/step3_windata_synced.mat']
    load_and_split_data02(path_list=path_list, win_size=50, n_signals=6, n_classes=8, test_size=0.3)  # Run this function ONLY when to re-split train-test sets

    print('End')