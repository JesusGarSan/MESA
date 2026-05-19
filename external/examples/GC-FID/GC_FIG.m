%% load data
close all; clear; clc
load external/examples/GC-FID/GC-FIG.mat

disp("Data size:")
disp(size(data))

%% Read labels
row_label_names = strtrim(string(label_names(:,1)));
col_label_names = strtrim(string(label_names{:,2}));

% (S, T*F)
% Rows
subjects    = [obs_l{1,:}];
% Cols
freqs       = [var_l{1,:}];
times       = [var_l{2,:}];

%% Decorate labels
subjects_label = "subject " + subjects;
freqs_label = freqs;
times_label = times;

disp("Labels ready")

%% ASCA 
%% parglm
disp("Running parglm...")
% F = [subjects];
F = F_data;
[T, parglmo] = parglm(data, F, 'Preprocessing', 1);

factor_names = ["Time", "Treatment", "Sex", "Order"];
T{1,1} = {'Time'};        % How long after they were treated
T{2,1} = {'Treatment'};   % Wounded (injured + nothing), primed (injured + PBS (salt)), inoculated (injured + bacteria)
T{3,1} = {'Sex'};         % Sex of the beetle
T{4,1} = {'Order'};       % Order in which the samples were run
for i = 1:size(factor_names,2)
    T{i,1} = {factor_names(i)};
end
disp(T)

%% build model
disp("Creating ASCA model...")
ascao = asca(parglmo);

%% ASCA Visualization - Scores
disp("Plotting ASCA scores...")
for factor_id = 1:ascao.nFactors
    % factor_name = string(row_label_names(factor_id));
    factor_name = factor_names{factor_id};
    factor_model = ascao.factors{factor_id};

    lvs = min([max(factor_model.lvs), 2]);
    factor_model.lvs = 1:lvs;
    class = F(:,factor_id);

    scores(factor_model, "ObsLabel", class, "ObsClass", class, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: " + factor_name);
end

%% ASCA Visualization - Loadings
disp("Plotting ASCA loadings...")
for factor_id = 1:ascao.nFactors
    % factor_name = string(row_label_names(factor_id));
    factor_name = factor_names{factor_id};
    factor_model = ascao.factors{factor_id};

    lvs = min([max(factor_model.lvs), 2]);
    factor_model.lvs = 1:lvs;
    class = F(:,factor_id);

    % loadings(factor_model, "VarsLabel", times, "VarsClass", times, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Time");
    loadings(factor_model, "VarsLabel", freqs, "VarsClass", freqs, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Frequencies");
end