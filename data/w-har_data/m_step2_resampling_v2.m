%% --------- Resampling the data to a new sampling rate ---------
% - Because there are some experiments of stretch sensor have sampling
% frequency = 100Hz instead of 25Hz
% - Can also be used for resampling motion data
% Note: there is a experiment that has wrong freq_rate [exp_idx = 62]
% => Delete the data of this experiment

%% Difference with version 1
% - Version 1 uses the "resample" function from matlab, this ones can work
% with non-integer downsampling rate, but the resampled data look kind of
% weird
% - Version 2: Because the downsampling rate of stretch and motion data are
% 4 and 10, so we can downsample by taking the average values
%   + Details of the customized downsampling function are at the END of
%   this file


clc; clear; close all
%% Load data
load step1_combined_stretchdata_ver2.mat
load step1_combined_motiondata_ver2.mat
load step1_sampling_freq_rate.mat

%% CONSTANTS
MOTION_FREQ_OLD = 250;
MOTION_FREQ_NEW = 25;
STRETCH_FREQ_OLD = 100;
STRETCH_FREQ_NEW = 25

%% Remove the 62th experiment from all the variables
% Because the 62th experiment has abnormal sampling frquency ratio (=1.375)
remove_idx = find(freq_rate < 2.5); % Find the idx of that experiment
combined_motiondata(remove_idx,:) = [];
combined_stretchdata(remove_idx, :) = [];
freq_rate(remove_idx, :) = [];
exp_idx_cell(remove_idx, :) = [];

%% Resample the stretch and motion data
resampled_motiondata = {};
resampled_stretchdata = {};
for i1 = 1: size(combined_stretchdata, 1)      
    i1
    % STRETCH DATA:
    temp_stretchdata = combined_stretchdata{i1, 1};
    % Find the experiments that have frequency = 100Hz and resampled_stretchdata
    if freq_rate(i1, 1) < 3        
        resampled_stretchdata{i1, 1} = mf_average_downsample(temp_stretchdata, STRETCH_FREQ_OLD/STRETCH_FREQ_NEW);
    else
        resampled_stretchdata{i1, 1} = temp_stretchdata;
    end
    % round the label to integer
    resampled_stretchdata{i1, 1}(:, end) = round(resampled_stretchdata{i1, 1}(:, end));
    
    % MOTION DATA
    temp_motiondata = combined_motiondata{i1, 1};   % Note {} will return an array but () will return cell
    resampled_motiondata{i1, 1} = mf_average_downsample(temp_motiondata, MOTION_FREQ_OLD/MOTION_FREQ_NEW);
    % round the label to integer
    resampled_motiondata{i1, 1}(:, end) = round(resampled_motiondata{i1, 1}(:, end));
end

save('step2_resampled_data_25Hz_v2.mat', 'resampled_motiondata', 'resampled_stretchdata', 'exp_idx_cell')        
        
        
%% Moving average downsampling FUNCTION
function [downsampled_data] = mf_average_downsample(data_array,rate)
%MF_AVERAGE_DOWNSAMPLE Summary of this function goes here
%   Detailed explanation goes here
%   - data_array: can be a 2-D array: is the original data that need to be
%   downsampled. Shape = [n_samples, n_features]
%   - rate: (must be integer) is the rate that how many samples of the
%   original data will be averaged into 1 sample in the downsampled data

old_idx = 1;
new_idx = 1;
downsampled_data = [];
while old_idx <= size(data_array, 1) - rate
    downsampled_data(new_idx, :) = mean(data_array(old_idx:old_idx + rate -1, :));
    new_idx = new_idx + 1;
    old_idx = old_idx + rate;
end
        
end        
        

        