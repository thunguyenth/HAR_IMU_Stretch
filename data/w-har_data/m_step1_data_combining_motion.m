clc;
clear; close all;
%% Step 0: Import the csv data and save them into mat files
% (for smaller data size and faster data loading)
% - Import the csv data to the MATLAB workspace as "table"
% - Save the data as .mat files with names "step0_motiondata.mat" and
% "step0_stretchdata.mat"

%% STEP 1: Save the data of each (user-scenario-trial) as a cell in the cell array
load('step0_motiondata.mat')      % Uncomment for motion data
% load('step0_stretchdata.mat')     % Uncomment for stretch data

% *** NOTE: Change the following 5 values to adapt with the data (motion or stretch)
raw_data = motiondata;     % CHANGE to stretchdata for stretch data
user_col_idx = 2;
scen_col_idx = 3;
tria_col_idx = 4;
data_col_idx_start = 5;
data_col_idx_end = 10;
labe_col_idx = 11;

CLASS_NAMES = ["jump", "liedown", "sit", "stairsdown", "stairsup", "stand", "transition", "walk", "undefined"];
% index =       0          1        2          3            4         5         6           7           8    

%% Step 1a: Create the new compact experiment indices
% List of unique user idx: Is not a compact list from 1 to 105 (total = 22 users)
unique_user_idx = unique(table2array(raw_data(:, user_col_idx)));

% List of unique scenario idx: Is a compact list from 1 to 10
unique_scenario_idx = unique(table2array(raw_data(:, scen_col_idx)));   

% List of unique trial idx: Is a compact list from 1 to 15
unique_trial_idx = unique(table2array(raw_data(:, tria_col_idx)));     

% New index column: [the last 2 digits: trial idx]; [next 2 digits: scenario idx]; [remain digits: user idx]
% xxx/yy/zz (zz: trial idx, yy: scenario idx, xxx: user idx)
idx_col = table2array(raw_data(:, 2)).*(100*100) +...
    table2array(raw_data(:, 3)).*100 + ... 
    table2array(raw_data(:, 4));

%% Change the label from categorical into integer (based on the CLASS_NAMES)
labels = string(table2array(raw_data(:, labe_col_idx)));
label_col = [];
for i=1:size(labels, 1)
    label_col(i,1) = find(CLASS_NAMES==labels(i, 1));
end
label_col = label_col - 1; % Zero offset

%%
data = table2array(raw_data(:, data_col_idx_start:data_col_idx_end));
% Add the idx_col and label_col as the FIRST and LAST columns of the data
data = [data label_col];    

clear raw_data labels    % to free the memory

%% Start making combined_data (a cell column vector)
combined_data = {};
exp_idx_cell = {}; % so later on can check user idx
cur_idx = 0;
temp_data = [];     % to temporarily store the data of an experiment
for i1 = 1:size(idx_col, 1)    
    new_idx = idx_col(i1);
    
    % If the new experiment starts or the current experiment is discrete
    % because of "undefined" label
    if cur_idx ~= new_idx || label_col(i1) == 8
        % fprintf('*********** New experiment ***********')
        cur_idx
        % Check whether the previous temp_data is NULL or not
        if ~isempty(temp_data)            
            % If temp_data is NOT NULL, save it to the combined_data
            combined_data = [combined_data; temp_data];
            exp_idx_cell = [exp_idx_cell; cur_idx];
            % Reset the temp_data to save the new experiment
            temp_data = [];
        end
                
    end
    
    if label_col(i1) == 8
        continue
    else
        cur_idx = new_idx;
    end
    
    % Insert the data sample into the temp_data
    temp_data = [temp_data; data(i1, :)];
end

%% Save to mat file
%% Uncomment for motion data
combined_motiondata = combined_data;
save('step1_combined_motiondata_ver2.mat', 'combined_motiondata', 'exp_idx_cell')

%% Uncomment for stretch data
% combined_stretchdata = combined_data;
% save('step1_combined_stretchdata_ver2.mat', 'combined_stretchdata', 'exp_idx_cell')



