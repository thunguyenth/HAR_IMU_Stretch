%% --------- Resampling the data to a new sampling rate ---------
% - Because there are some experiments of stretch sensor have sampling
% frequency = 100Hz instead of 25Hz
% - Can also be used for resampling motion data
% Note: there is a experiment that has wrong freq_rate [exp_idx = 62]
% => Delete the data of this experiment

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
        resampled_stretchdata{i1, 1} = resample(temp_stretchdata, STRETCH_FREQ_NEW, STRETCH_FREQ_OLD);
    else
        resampled_stretchdata{i1, 1} = temp_stretchdata;
    end
    % round the label to integer
    resampled_stretchdata{i1, 1}(:, end) = round(resampled_stretchdata{i1, 1}(:, end));
    
    % MOTION DATA
    temp_motiondata = combined_motiondata{i1, 1};   % Note {} will return an array but () will return cell
    resampled_motiondata{i1, 1} = resample(temp_motiondata, MOTION_FREQ_NEW, MOTION_FREQ_OLD);
    % round the label to integer
    resampled_motiondata{i1, 1}(:, end) = round(resampled_motiondata{i1, 1}(:, end));
end

% save('step2_resampled_data_25Hz_v1.mat', 'resampled_motiondata', 'resampled_stretchdata', 'exp_idx_cell')
        
        
        
        
        