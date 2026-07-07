function [Mean_SS, Mean_perc_SS, SS_time, SS_treatment, SS_sex, SS_null, ...
    SS_order, SS_residuals,F_time, F_treatment, F_sex, F_order, F_null,...
    F_residuals, p_time, p_treatment, p_sex, p_order, p_null, ...
    dist_time, dist_treatment, dist_sex, dist_order, dist_null] = GC_FID_optimization()
%% load data
close all;
load external/examples/GC-FID/GC-FID.mat

%% Preprocess
X = preprocess2D(data, 'Preprocessing', 0);
%% ASCA 
%% parglm
F = F_data; 

[T, parglmo] = parglm(X, F, 'Preprocessing', 2, 'Model', 'linear');
factor_names = ["Time", "Treatment", "Sex", "Order"];
for i = 1:size(factor_names,2)
    T{i+1,1} = {factor_names(i)};
end
%% build model
ascao = asca(parglmo);
ascao.nFactors = ascao.n_factors;
%%
SS_time      = T{2,3};
SS_treatment = T{3,3};
SS_sex       = T{4,3};
SS_order     = T{5,3};
SS_null      = T{6,3};
SS_residuals = T{7,3};

F_time      = T{2,6};
F_treatment = T{3,6};
F_sex       = T{4,6};
F_order     = T{5,6};
F_null      = T{6,6};
F_residuals = T{7,6};

p_time      = T{2,7};
p_treatment = T{3,7};
p_sex       = T{4,7};
p_order     = T{5,7};
p_null      = T{6,7};

Mean_SS = T{1,2};
Mean_perc_SS = T{1,3};


% distances = zeros(ascao.nFactors, 1);
% for factor_id = 1:ascao.nFactors
%     classes = F(:,factor_id);
%     levels = unique(classes);
    
%     pcs = max(ascao.factors{factor_id}.lvs);
%     means = zeros(length(levels), pcs);
%     for level_id = 1:size(levels,1)
%         idx = (classes == levels(level_id));
%         values = ascao.factors{factor_id}.scoresV(idx,:);
%         means(level_id,:) = mean(values);
%     end

%     distance = 0;
%     num_groups = size(means, 1);
    
%     % Doble bucle para comparar cada grupo con los siguientes
%     for i = 1:num_groups - 1
%         for j = i + 1:num_groups
%             % Calcula la distancia euclidiana entre la media del grupo i y el grupo j
%             diff = means(i, :) - means(j, :);
%             dist_ij = sqrt(sum(abs(diff).^2)); 
            
%             % Acumula la distancia
%             distance = distance + dist_ij;
%         end
%     end
%     distances(factor_id) = distance;

% end
% % Normalization according to number of variables
% M = size(X, 2);
% dist_time = distances(1)/M;
% dist_treatment = distances(2)/M;
% dist_sex = distances(3)/M;
% dist_order = distances(4)/M;
% dist_null = distances(5)/M;

%% Shilouette
distances = zeros(ascao.nFactors, 1);

for factor_id = 1:ascao.nFactors
    classes = F(:, factor_id);
    pcs = max(ascao.factors{factor_id}.lvs);
    scores = ascao.factors{factor_id}.scoresV(:, 1:pcs);
    scores = abs(scores);

    sil_values = silhouette(scores, classes, 'Euclidean');
    
    distances(factor_id) = mean(sil_values);
end

% Silhouette is already normalized by definition between -1 and 1, 
% so dividing by M (number of variables) is likely no longer necessary.
dist_time      = distances(1);
dist_treatment = distances(2);
dist_sex       = distances(3);
dist_order     = distances(4);
dist_null      = distances(5);


end