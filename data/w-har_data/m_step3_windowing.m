%% ---------------------- WINDOWING ----------------------
%% Mismatch in number of data samples between motion and stretch data
% - There are big difference in number of data samples in the first 67
% experiments (68 if not deleting the one has freq_rate=1.37).
% - These experiments are the one that has frequency of stretch data = 100Hz
%% => For the windata_total, only the remain experiments (68th to 163rd) are used

clc; clear; close all;
%% CONSTANTS
WIN_SIZE = 50;
STRIDE = 25;
%% Load data
load('step2_resampled_data_25Hz_v2.mat')
n_exps = size(resampled_motiondata, 1);

for i = 1:n_exps
size_diff(i) = size(resampled_motiondata{i, 1}, 1) - size(resampled_stretchdata{i, 1}, 1);
end

%% Windowing
windata_stretch = {};
labels_stretch = {};
windata_motion = {};
labels_motion = {};
for i1 = 1:n_exps
    % STRETCH DATA
    temp_stretchdata = resampled_stretchdata{i1, 1};
    [windata_stretch{i1,1}, labels_stretch{i1, 1}] = mf_window(temp_stretchdata, WIN_SIZE, STRIDE); 
    
    % MOTION DATA
    temp_motiondata = resampled_motiondata{i1, 1};
    [windata_motion{i1,1}, labels_motion{i1, 1}] = mf_window(temp_motiondata, WIN_SIZE, STRIDE); 
end


%% windata_total (HARD-CODE): only use data of experiments 68th-163rd
% Save the motion and stretch data into 2 2D arrays (combine all the
% experiments)
min_idx = [];
labels = [];
win_IMUs = [];      % shape = [total_n_wins, win_size*n_signals]
win_stretch = [];   % shape = [total_n_wins, win_size*n_signals]
for i1 = 68: n_exps
    min_idx = min(size(windata_motion{i1, 1}, 1), size(windata_stretch{i1, 1}, 1));
    win_IMUs = [win_IMUs; windata_motion{i1, 1}(1:min_idx,:)];
    win_stretch = [win_stretch; windata_stretch{i1, 1}(1:min_idx, :)];
    labels = [labels; labels_motion{i1, 1}(1:min_idx, 1)];
        
end
    

%% Save data
save('step3_windata.mat', 'windata_motion', 'windata_stretch', 'labels_motion', 'labels_stretch')
save('step3_windata_synced.mat', 'win_IMUs', 'win_stretch', 'labels')



%% Windowing function        
function [windata, labels] = mf_window(data_array, win_size, stride)
%MF_WINDOW Split the array data into fixed-length window 
% INPUT:
%   - data_array: can be a 2-D array: is the original data that need to be
%   downsampled. Shape = [n_samples, n_signals+1]. The last column is the
%   label column
%   - win_size: (must be integer) is the size of the window
%   - stride: (must be integer) is the step size. Overlap = win_size - stride
% OUTPUT: 
%   - windata is reshaped from [n_wins, win_size, n_signals] to
% [n_wins, win_size*n_signals]
%   - labesl: shape = [n_wins, 1]

    n_signals = size(data_array, 2) - 1;  % (-1) because the last column is label
    sample_idx = 1;
    win_idx = 1;
    windata = [];
    labels = [];
    while sample_idx <= size(data_array, 1) - win_size  + 1
        temp_data = data_array(sample_idx:sample_idx + win_size - 1, :);
        windata(win_idx, :) = reshape(temp_data(:, 1:n_signals), 1, win_size*n_signals);
        labels(win_idx, 1) = mode(temp_data(:, end));   % the last column is label
        win_idx = win_idx + 1;
        sample_idx = sample_idx + stride;
    end
        
end          
      