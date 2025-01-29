%%
clc 
close all

%% Load the data files together
files = dir('simulation_data/*.mat');

data = [];
var_class = [];
var_l = [];

for i = 1:length(files)
    file_data = load(fullfile(files(i).folder, files(i).name));
    
    matrix = file_data.matrix;
    
    data = [data, matrix];
    var_l = [var_l; string(file_data.column_names)];
    var_class = [var_class, repmat(["sensor "+ string(i)], 1, length(file_data.column_names))];
end


%% Preprocessing
prep = 0; % 0 = No Preprocessing, 1 = Mean Centering ; 2 = autoscaling

prep_methods = ["No preprocessing", "Mean Centering", "Autoscaling"];
disp("Preprocesing method: " + prep_methods(prep+1))
clear prep_methods
[Xcs,model.av,model.sc] = preprocess2D(data, 'Preprocessing',prep);

%% Choosing the number of PCs
% VarX + ckf
disp("Displaying var+ckf plot")
pcs = 0:20;
x_var = varPca(Xcs, 'Pcs', pcs, 'Preprocessing', 0, 'PlotCkf', true); 

%% Create PCA model
pcs = 1:2;
disp("Creating PCA model with "+ string(pcs(end))+ "PCs")

model.lvs = pcs;
model.var = trace(Xcs'*Xcs);
model=pcaEig(Xcs,'Pcs',model.lvs);

T = model.scores;
d = diag(T'*T);
var_PC1 = 100*d(1)/model.var;

%% Plot scores
disp("Displaying scores plot")
scores(model, 'ObsLabel', 1:size(data, 1), 'ObsClass', 1:size(data, 1));
title('Time')

%% Plot loadings
disp("Displaying loadings plot")
loadings(model, 'VarsLabel', var_l, 'VarsClass', var_class, 'Color', 'okabeIto', ...
    'BlurIndex', 2);
legend()

%%
disp("MEDA analysis completed.")