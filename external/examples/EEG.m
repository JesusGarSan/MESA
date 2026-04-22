%% load data
close all; clear; clc
load external/examples/EEG.mat

disp("Data size:")
disp(size(data))

%%
row_label_names = string(label_names{1});
col_label_names = string(label_names{2});


subjects    = [obs_l{1,:}];
operations  = [obs_l{2,:}];
times       = [obs_l{3,:}];

subjects = "subject " + subjects;
operations = string(categorical(operations, [0, 1], {'Baseline', 'Arithmetic'}));

channels = [var_l{1,:}];
freqs    = [var_l{2,:}];

channels = "channel " + channels;

disp("Labels ready")

%% Var PCA plot
X = preprocess2D(data, 'Preprocessing', 1);
% varPca(X, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", true);
%% Model
model.lvs=1:2;
model = pcaEig(X,'PCs',model.lvs);
model.var = trace(X'*X);
%% Scores plot
disp("Plotting scores...")
scores(model, "ObsLabel", times, "ObsClass", times, "Color", "parula"); title("Time");
scores(model, "ObsLabel", subjects, "ObsClass", subjects, "Color", "parula"); title("Subjects");
scores(model, "ObsLabel", operations, "ObsClass", operations, "Color", "parula"); title("Operations");

%% Loadings plots
disp("Plotting loadings...")
loadings(model, "VarsLabel", freqs, "VarsClass", freqs); title("Frequencies");
loadings(model, "VarsLabel", channels, "VarsClass", channels); title("Channels");

%%