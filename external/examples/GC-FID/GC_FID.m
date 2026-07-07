%% load data
disp('Running GC_FID.m ...')
close all; clear; clc
load external/examples/GC-FID/GC-FID.mat

% disp("Data size:")
% disp(size(data))
plot_bool = true;

%% Read labels
row_label_names = strtrim(string(label_names(:,1)));
col_label_names = strtrim(string(label_names{:,2}));

% (S, T*F)
% Rows
subjects    = [obs_l{1,:}];
% Cols
times = [];
freqs       = [var_l{1,:}];
times       = [var_l{2,:}];

%% Decorate labels
subjects_label = "subject " + subjects;
freqs_label = freqs;
times_label = times;

% disp("Labels ready")

%% Preprocess
% X = preprocess2D(data','Preprocessing', 2)';
X = preprocess2D(data, 'Preprocessing', 0);
% X = preprocess2D(data, 'Preprocessing', 1);
% X = preprocess2D(data, 'Preprocessing', 2);


%% ASCA 
%% parglm
disp("Running parglm...")
% F = [subjects];
F = F_data;
% [T, parglmo] = parglm(X, F, 'Preprocessing', 1, 'Model', {[1,2]});
[T, parglmo] = parglm(X, F, 'Preprocessing', 2, 'Model', 'linear');

factor_names = ["Time", "Treatment", "Sex", "Order"];
% T{2,1} = {'Time'};        % How long after they were treated
% T{3,1} = {'Treatment'};   % Wounded (injured + nothing), primed (injured + PBS (salt)), inoculated (injured + bacteria)
% T{4,1} = {'Sex'};         % Sex of the beetle
% T{5,1} = {'Order'};       % Order in which the samples were run
for i = 1:size(factor_names,2)
    T{i+1,1} = {factor_names(i)};
end
% disp(T)

%% build model
disp("Creating ASCA model...")
ascao = asca(parglmo);
ascao.nFactors = ascao.n_factors;
%% ASCA Visualization - Scores
if plot_bool
    disp("Plotting ASCA scores...")
    % for factor_id = 1:ascao.nFactors
    for factor_id = 1:2
        % factor_name = string(row_label_names(factor_id));
        factor_name = factor_names{factor_id};
        factor_model = ascao.factors{factor_id};

        lvs = min([max(factor_model.lvs), 2]);
        % lvs = 1;
        factor_model.lvs = 1:lvs;
        class = F(:,factor_id);
        labels = F(:,factor_id);
        if factor_id ==2
            labels = repelem("",length(labels));
        end

        scores(factor_model, "ObsLabel", labels, "ObsClass", class, "Color", "okabeIto", 'BlurIndex', 1e6); title("Factor " + factor_name + " Scores - Color: " + factor_name);
        ax = gca;   
        if factor_id ==1
            lgd = legend("24h", "72h");
            set(gcf, 'Position', [100, 100, 1100, 450]);    
            title("Factor Inoculation Scores")
            ax.Box = "off";                           
        end
        if factor_id ==2
            lgd = legend("Healthy", "Primed", "Wounded");
            title("Factor Treatment Scores")
            set(gcf, 'Position', [100, 100, 1100, 600]);  
        end
        lgd.FontSize = 25;
        ax.FontSize = 20;

    end
end

%% ASCA Visualization - Loadings
% if plot_bool
%     disp("Plotting ASCA loadings...")
%     % for factor_id = 1:ascao.nFactors
%     for factor_id = 1:2
%         % factor_name = string(row_label_names(factor_id));
%         factor_name = factor_names{factor_id};
%         factor_model = ascao.factors{factor_id};

%         lvs = min([max(factor_model.lvs), 2]);
%         factor_model.lvs = 1:lvs;
%         class = F(:,factor_id);

%         loadings(factor_model, "VarsLabel", times, "VarsClass", times, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Time");
%         loadings(factor_model, "VarsLabel", freqs, "VarsClass", freqs, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: Frequencies");
%     end
% end
%%
SS_time      = T{2,3};
SS_treatment = T{3,3};
SS_sex       = T{4,3};
SS_order     = T{5,3};
SS_residuals = T{6,3};

p_time      = T{2,7};
p_treatment = T{3,7};
p_sex       = T{4,7};
p_order     = T{5,7};

%%
distances = zeros(ascao.nFactors, 1);
for factor_id = 1:ascao.nFactors
    classes = F(:,factor_id);
    levels = unique(classes);
    
    pcs = max(ascao.factors{factor_id}.lvs);
    means = zeros(length(levels), pcs);
    for level_id = 1:size(levels,1)
        idx = (classes == levels(level_id));
        values = ascao.factors{factor_id}.scores(idx,:);
        means(level_id,:) = mean(values);
    end

    distance = 0;
    num_groups = size(means, 1);
    
    % Doble bucle para comparar cada grupo con los siguientes
    for i = 1:num_groups - 1
        for j = i + 1:num_groups
            % Calcula la distancia euclidiana entre la media del grupo i y el grupo j
            diff = means(i, :) - means(j, :);
            dist_ij = sqrt(sum(abs(diff).^2)); 
            
            % Acumula la distancia
            distance = distance + dist_ij;
        end
    end
    distances(factor_id) = distance;

end

dist_time = distances(1);
dist_treatment = distances(2);
dist_sex = distances(3);
dist_order = distances(4);


%%
T

%% Null factor


% %% Peak table
% [T_peak, parglmo_peak] = parglm(X_peak, F_peak, 'Preprocessing', 1, 'Model', {[1,2]});
% T_peak
% %% build model
% disp("Creating ASCA model...")
% ascao_peak = asca(parglmo_peak);
% ascao_peak.nFactors = ascao_peak.n_factors;
% factor_names = ["Time", "Treatment", "Sex", "Order"];
% %% ASCA Visualization Peaks  - Scores
% disp("Plotting ASCA scores...")
% for factor_id = 1:ascao_peak.nFactors
%     % factor_name = string(row_label_names(factor_id));
%     factor_name = factor_names{factor_id};
%     factor_model = ascao_peak.factors{factor_id};

%     lvs = min([max(factor_model.lvs), 2]);
%     factor_model.lvs = 1:lvs;
%     class = F_peak(:,factor_id);

%     scores(factor_model, "ObsLabel", class, "ObsClass", class, "Color", "parula"); title("Factor " + factor_name + " Scores - Color: " + factor_name);
% end

% %%
