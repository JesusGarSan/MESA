%% load data
close all; clear; clc
load external/examples/EEG.mat

disp("Data size:")
disp(size(data))

%% Read labels
row_label_names = strtrim(string(label_names(1,:)));
col_label_names = strtrim(string(label_names(2,:)));

% (S*O, T*F) % Collapsed channels
% Rows
subjects    = [obs_l{1,:}];
operations  = [obs_l{2,:}];
% Cols
channels    = [];
freqs       = [var_l{1,:}];
times       = [var_l{2,:}];

%% Filter frequencies
data_filtered = data;

subjects_filtered = subjects;
operations_filtered = operations;
channels_filtered = channels;
freqs_filtered = freqs;
times_filtered = times;
if false
    idx = (freqs <= 15) && (freqs > 0.5); %Hz
    data_filtered = data_filtered(:,idx);
    % Cols
    if sum("subjects" == col_label_names)
        subjects_filtered    = subjects(idx); end
    if sum("operations" == col_label_names)
        operations_filtered    = operations(idx); end
    if sum("channels" == col_label_names)
        channels_filtered    = channels(idx); end
    if sum("freq" == col_label_names)
        freqs_filtered    = freqs(idx); end
    if sum("time" == col_label_names)
        times_filtered    = times(idx); end
        
end
disp("Filtered data size:")
disp(size(data_filtered))

%% Decorate labels
subjects_label = "subject " + subjects_filtered;
operations_label = string(categorical(operations_filtered, [0, 1], {'Baseline', 'Arithmetic'}));
channels_label = channels_filtered;
freqs_label = freqs_filtered;
times_label = times_filtered;

disp("Labels ready")

%% ASCA 
%% parglm
disp("Running parglm...")
F = [subjects; operations]';
[T, parglmo] = parglm(data_filtered, F, 'Preprocessing', 1);

for i=1:size(row_label_names, 2)
    T{i,1} = {char(row_label_names(i))};
end

PercSS = T{2,3};
% Open file for appending ('a')
csv_file = "EEG_params.csv";
fid = fopen(csv_file, 'a');
if fid == -1
    error('Cannot open file');
end

% Write the value followed by a newline to finish the CSV row
fprintf(fid, '%f\n', PercSS);
fclose(fid);

exit;