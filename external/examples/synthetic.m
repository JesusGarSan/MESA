%% load data
close all; clear; clc
load external/examples/trumpet.mat

%%
row_label_names = string(label_names{1});
col_label_names = string(label_names{2});

times = cell2mat(obs_l);
sensors  = [var_l{1,:}];
channels = string(var_l(2,:));
freqs    = [var_l{3,:}];

disp("Labels ready")

%% Var PCA plot
X = preprocess2D(data, 'Preprocessing', 1);
varPca(X, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", true);
%% Model
model.lvs=1:2;
model = pcaEig(X,'PCs',model.lvs);
model.var = trace(X'*X);
%% Scores plot
scores(model, "ObsLabel", times, "ObsClass", times, "Color", "parula");

%% Loadings plots
loadings(model, "VarsLabel", freqs, "VarsClass", freqs);
loadings(model, "VarsLabel", sensors, "VarsClass", sensors);
loadings(model, "VarsLabel", channels, "VarsClass", channels);


%%