%% load data
close all; clear; clc
load external/examples/EEG/EEG.mat

disp("Data size:")
disp(size(data))

%% Read labels
row_label_names = strtrim(string(label_names(1,:)));
col_label_names = strtrim(string(label_names(2,:)));

% % (O*T, S*C*F)
% % Rows
% operations  = [obs_l{1,:}];
% times       = [obs_l{2,:}];

% % Cols
% subjects = [var_l{1,:}];
% channels = [var_l{2,:}];
% freqs    = [var_l{3,:}];

% % (S*O*T, C*F)
% % Rows
% subjects    = [obs_l{1,:}];
% operations  = [obs_l{2,:}];
% times       = [obs_l{3,:}];
% % Cols
% channels = [var_l{1,:}];
% freqs    = [var_l{2,:}];

% % (S*O*C, T*F)
% % Rows
% subjects    = [obs_l{1,:}];
% operations  = [obs_l{2,:}];
% channels    = [obs_l{3,:}];
% % Cols
% freqs    = [var_l{1,:}];
% times = [var_l{2,:}];

% % (S*O, C*T*F)
% % Rows
% subjects    = [obs_l{1,:}];
% operations  = [obs_l{2,:}];
% % Cols
% channels    = [var_l(1,:)];
% freqs       = [var_l{2,:}];
% times       = [var_l{3,:}];

% (S*O, T*F) % Collapsed channels
% Rows
subjects    = [obs_l{1,:}];
operations  = [obs_l{2,:}];
% Cols
channels    = [];
freqs       = [var_l{1,:}];
times       = [var_l{2,:}];


% % (S*O, F) % Collapsed channels and time
% % Rows
% subjects    = [obs_l{1,:}];
% operations  = [obs_l{2,:}];
% % Cols
% channels    = [];
% freqs       = [var_l{1,:}];
% times       = [];
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

%% Preprocess
X = preprocess2D(data_filtered, 'Preprocessing', 1);
%% Var PCA plot
% plot_pca = false;
% if plot_pca
%     varPca(X, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", true);
% end
% %% Model
% if plot_pca
%     model.lvs=1:2;
%     model = pcaEig(X,'PCs',model.lvs);
%     model.var = trace(X'*X);
% end
% %% Scores plot
% if plot_pca
%     disp("Plotting scores...")
%     if sum("subjects" == row_label_names)
%         scores(model, "ObsLabel", subjects_label, "ObsClass", subjects_label, "Color", "parula"); title("Subjects");legend('off');
%     end
%     if sum("operations" == row_label_names)
%         scores(model, "ObsLabel", operations_label, "ObsClass", operations_label, "Color", "parula"); title("Operations");
%     end
%     if sum("channels" == row_label_names)
%         scores(model, "ObsLabel", channels_label, "ObsClass", channels_label, 'Color', 'parula'); title("Channels");legend('off')
%     end
%     if sum("freq" == row_label_names)
%         scores(model, "ObsLabel", freqs_label, "ObsClass", freqs_label, 'Color', 'parula'); title("Frequencies");
%     end
%     if sum("time" == row_label_names)
%         scores(model, "ObsLabel", times_label, "ObsClass", times_label, "Color", "parula"); title("Time");
%     end
% end

% %% Loadings plots
% if plot_pca
%     disp("Plotting loadings...")
%     if sum("subjects" == col_label_names)
%         loadings(model, "VarsLabel", subjects_label, "VarsClass", subjects_label, "Color", "parula"); title("Subjects");legend('off');
%     end
%     if sum("operations" == col_label_names)
%         loadings(model, "VarsLabel", operations_label, "VarsClass", operations_label, "Color", "parula"); title("Operations");
%     end
%     if sum("channels" == col_label_names)
%         loadings(model, "VarsLabel", channels_label, "VarsClass", channels_label, 'Color', 'parula'); title("Channels");legend('off')
%     end
%     if sum("freq" == col_label_names)
%         loadings(model, "VarsLabel", freqs_label, "VarsClass", freqs_label, 'Color', 'parula'); title("Frequencies");
%     end
%     if sum("time" == col_label_names)
%         loadings(model, "VarsLabel", times_label, "VarsClass", times_label, "Color", "parula"); title("Time");
%     end
% end
%%

% %% oMEDA: Baseline vs. Arithmetic
% % Baseline: -1 | Arithmetic: +1
% dummy = ones(size(operations_label));
% idx = (operations == 0); 
% dummy(idx) = -1;

% omeda_vec = omeda(X, dummy, model.loads);

% %%
% % close all;
% if true
%     subject_ids = 5:10;
%     channel_id = 4;

%     for subject_id = subject_ids
%         idx1 = channels_filtered == channel_id;
%         idx2 = subjects_filtered == subject_id;
%         idx = idx1 & idx2;
%         omeda_vec_filtered = omeda_vec(idx);

%         plotVec(omeda_vec_filtered, 'ObsClass', freqs_label(idx)); title("oMEDA Baseline vs. Arithmetic | Subject " + ...
%         string(subject_id)+", Channel " + string(channel_id))
%         xlabel("Frequencies (Hz)")
%     end
% end
% %%
% if true
%     [~, idx] = sort(freqs_filtered);
%     plotVec(omeda_vec(idx), 'ObsClass', freqs_label(idx)); title("oMEDA Baseline vs. Arithmetic: Frequencies")
%     [~, idx] = sort(channels_filtered);
%     plotVec(omeda_vec(idx), 'ObsClass', channels_label(idx)); title("oMEDA Baseline vs. Arithmetic: Channels")
%     [~, idx] = sort(subjects_filtered);
%     plotVec(omeda_vec(idx), 'ObsClass', subjects_label(idx)); title("oMEDA Baseline vs. Arithmetic: Subjects")
% end


%% ASCA 
%% parglm
disp("Running parglm...")
F = [subjects; operations]';
[T, parglmo] = parglm(data_filtered, F, 'Preprocessing', 1);

for i=1:size(row_label_names, 2)
    T{i,1} = {char(row_label_names(i))};
end
disp(T)
%% build model
disp("Creating ASCA model...")
ascao = asca(parglmo);
%% ASCA Visualization - Scores
disp("Plotting ASCA scores...")
for factor_id = 1:ascao.nFactors
    factor_name = string(row_label_names(factor_id));
    factor_model = ascao.factors{factor_id};

    lvs = min([max(factor_model.lvs), 2]);
    factor_model.lvs = 1:lvs;

    if factor_name == "subjects"
        scores(factor_model, "ObsLabel", subjects_label, "ObsClass", subjects_label, "Color", "okabeIto", "BlurIndex", 0.1); title("Factor " + factor_name + " Scores - Color: Subjects");legend('off');
    end
    if factor_name == "operations"
        scores(factor_model, "ObsLabel", subjects_label, "ObsClass", operations_label, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Operations");
    end
    if factor_name == "channels"
        scores(factor_model, "ObsLabel", channels_label, "ObsClass", channels_label, 'Color', 'parula'); title("Factor " + factor_name + " Scores - Color: Channels");legend('off')
    end
    if factor_name == "freq"
        scores(factor_model, "ObsLabel", freqs_label, "ObsClass", freqs_label, 'Color', 'parula'); title("Factor " + factor_name + " Scores - Color: Frequencies");
    end
    if factor_name == "time"
        scores(factor_model, "ObsLabel", times_label, "ObsClass", times_label, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Time");
    end
end
%% ASCA Visualization - Loadings
disp("Plotting ASCA loadings...")
for factor_id = 1:ascao.nFactors
    factor_name = string(row_label_names(factor_id));
    factor_model = ascao.factors{factor_id};
    
    lvs = min([max(factor_model.lvs), 2]);
    factor_model.lvs = 1:lvs;
    aux = factor_model;


    if sum("subjects" == col_label_names)
        [~, idx] = sort(subjects_label);
        labels = subjects_label(idx)';
        classes = subjects_label(idx);
        title_label = "Subjects";
        aux.loads = aux.loads(idx, :);
        loadings(aux, "VarsLabel", labels, "VarsClass", classes, "Color", "parula"); title("Factor " + factor_name + " Loadings - Color: " + title_label);legend('off');

    end
    if sum("operations" == col_label_names)
        [~, idx] = sort(operations_label);
        labels = subjects_label(idx);
        classes = operations_label(idx);
        title_label = "Operations";
        aux.loads = aux.loads(idx, :);
        loadings(aux, "VarsLabel", labels, "VarsClass", classes, "Color", "parula"); title("Factor " + factor_name + " Loadings - Color: " + title_label);legend('off');

    end
    if sum("channels" == col_label_names)
        [~, idx] = sort(channels_label);
        labels = channels_label(idx);
        classes = channels_label(idx);
        title_label = "Channels";
        aux.loads = aux.loads(idx, :);
        loadings(aux, "VarsLabel", labels, "VarsClass", classes, "Color", "parula"); title("Factor " + factor_name + " Loadings - Color: " + title_label);legend('off');

    end
    if sum("freq" == col_label_names)
        [~, idx] = sort(freqs_label);
        labels = freqs_label(idx);
        classes = freqs_label(idx);
        title_label = "Frequencies";
        aux.loads = aux.loads(idx, :);
        loadings(aux, "VarsLabel", labels, "VarsClass", classes, "Color", "parula"); title("Factor " + factor_name + " Loadings - Color: " + title_label);legend('off');

    end
    if sum("time" == col_label_names)
        [~, idx] = sort(times_label);
        labels = times_label(idx);
        classes = times_label(idx);
        title_label = "Time";
        aux.loads = aux.loads(idx, :);
        loadings(aux, "VarsLabel", labels, "VarsClass", classes, "Color", "parula"); title("Factor " + factor_name + " Loadings - Sort: " + title_label);legend('off');
    end

end