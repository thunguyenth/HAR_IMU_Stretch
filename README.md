# HAR_IMU_Stretch
GitHub repo for the iSPL IMU-Stretch dataset and implementation code of the paper "An Investigation on Deep Learning-Based Human Activity Recognition Using IMUs and Stretch Sensors" accepted by the **ICAIIC2022** conference.

## 1. iSPL IMU-Stretch Dataset
The iSPL IMU-Stretch dataset was collected from one subject wearing 3 IMUs (at the right wrist, waist and ankle) and 2 fabric stretch sensors (at the knees). 9 different activities are collected: *walking, standing, sitting, lying, running, jumping, sit-up, push-up and dancing*.

- From each IMU sensor, 3D acceleration, 3D angular velocity and 3D linear acceleration are collected.
- From each stretch sensor, the stretch degree (actually is the capacity of the stretch capacitor) is collected (the data is 1D).
- The data is already pre-processed (interpolation & windowing) and randomly split into train and test sets with a proportion of **70/30** and saved as mat files. Each mat file contains:
	- Window data of 3 IMUs: **X_IMUs_train** (**X_IMUs_test**): 4-D array: [number of windows; window size; ]
	- Window data of 2 stretch sensors: **X_Stretch_train** (**X_Stretch_test**)
	- Ground truth label: **y_train** (**y_test**)
 
## 2. w-HAR dataset
- The w-HAR dataset was introduced in the "Bhat, G.; Tran, N.; Shill, H.; Ogras, U.Y. w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices. Sensors 2020, 20, 5356." and is available at the GitHub repo: https://github.com/gmbhat/human-activity-recognition

- The preprocessing process is implemented using **MATLAB**, the codes are in the folder **data/w-har_data** 
	- **Step 0**: Import the csv data and save them into mat files for smaller data size and a faster data loading process
	- **Step 1**: Save the data of each (user-scenario-trial) as a cell in the cell array
		- *m_step1_data_combining_motion.m*
		- *m_step1_data_combining_stretch.m*
		- *m_step1b_check_frequency.m*
	- **Step 2**: Resampling the data to a new sampling rate
		- *m_step2_resampling_v2.m*
	- **Step 3**: Windowing (split the data into a fixed length window with an overlap between adjacent windows)
		- *m_step3_windowing.m*
- The train/test split process is implemented using **python**, the codes are in the file ***split_data.py***

## 3. HAR Models Implementation
- The deep learning models are in the file ***model_zoo.py***
- The training and evaluation tasks are in the files:
	- ***imu_stretch_ispl.py***
	- ***imu_stretch_wHAR.py***


***Citation***: If you would like to use the iSPL IMU-Stretch Dataset or the materials in this repo, please cite our work:
N. T. H. Thu and D. S. Han, "An Investigation on Deep Learning-Based Activity Recognition Using IMUs and Stretch Sensors," 2022 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), 2022, pp. 377-382, doi: 10.1109/ICAIIC54071.2022.9722621.

If you have any question, please feel free to contact me (thunguyen@knu.ac.kr) or if you find out any error, please let me know by creating a new issue in this repo.

Thank you for your interest in this work!

** Stand on the shoulders of giants **