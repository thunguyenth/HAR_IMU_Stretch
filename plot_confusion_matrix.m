%%%%%%%%%% CONFUSION MAXTRIX %%%%%%%%%%%
% This file draws confusion matrix from the predY and testY files extracted
% from the python models

%% 1: Load the predy.txt and testy.txt files
% Load test data
load('data\w-har_data\data_wHAR_2021120101_y_test.mat')
% Load predicted results
load('results\w-har_data\results_wHAR_20211201\CNN_1D-imu_stretch-2021-12-01_14_15.mat')

% As the results contain multiple running experiments (e.g., 10), so need
% to create the Y_test_total to have same amount of labels as Y_pred_total
Y_test_total = []
n_exps = size(Y_pred_total, 1)
for i1 = 1:n_exps
    Y_test_total = [Y_test_total; y_test];
end

% Reshape Y_test_total and Y_test_total to vector form
y_test = reshape(Y_test_total,[], 1);
y_pred = reshape(Y_pred_total, [], 1);

%% 2: Categorize the data; ex: B = categorical(A,[1 2 3],{'red' 'green' 'blue'})
% % iSPL data
% cat_labels = {'WALKING' 'STANDING' 'SITTING' 'LYING' 'RUNNING' 'JUMPING' 'SIT-UP' 'PUSH-UP' 'DANCING'};
% num_labels = [0 1 2 3 4 5 6 7 8];

% wHAR data
cat_labels = {'JUMPING' 'LIE-DOWN' 'SITTING' 'DOWNSTAIRS' 'UPSTAIRS' 'STANDING' 'TRANSITION' 'WALKING'};
num_labels = [0 1 2 3 4 5 6 7];
y_pred = categorical(y_pred, num_labels, cat_labels);
y_test = categorical(y_test, num_labels, cat_labels);

%% 3: plot confusion matrix
figure;
plotconfusion(y_test, y_pred)
% plotconfusion(y_test, y_pred, 'Confusion matrix - HiHAR-8')
figure;
cm = confusionchart(y_test, y_pred,...
'ColumnSummary','column-normalized',...
'RowSummary','row-normalized',...
'FontSize', 14)
% 'FontSize', 14,...
% 'Title', 'HiHAR-8 - Transition grouping');