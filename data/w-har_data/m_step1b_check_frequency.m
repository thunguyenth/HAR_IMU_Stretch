% There are some experiments whose stretch sensor data has sampling frequency of 100Hz instead of 25Hz.
% So this freq_rate is the ratio between motion sampling frequency and stretch sampling frequency.
% freq_rate = 2.5 means the frequency of stretch sensor in that experiment is 100Hz.
clear; clc; close all
load('step1_combined_motiondata_ver2.mat')
load('step1_combined_stretchdata_ver2.mat')
freq_rate = [];
for i1 = 1:size(combined_motiondata, 1)
freq_rate(i1) = size(combined_motiondata{i1, 1}, 1)/size(combined_stretchdata{i1, 1}, 1);
end

freq_rate = freq_rate';
save('step1_sampling_freq_rate.mat', 'freq_rate')
figure; stem(freq_rate); xlabel("experiment"); ylabel("sampling frequency ratio (motion/stretch)")