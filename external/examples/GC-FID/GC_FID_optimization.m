function [Mean_SS, Mean_perc_SS, SS_time, SS_treatment, SS_sex, SS_order, SS_residuals,F_time, F_treatment, F_sex, F_order, F_residuals, p_time, p_treatment, p_sex, p_order] = GC_FID_optimization()
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
SS_residuals = T{6,3};

F_time      = T{2,6};
F_treatment = T{3,6};
F_sex       = T{4,6};
F_order     = T{5,6};
F_residuals = T{6,6};

p_time      = T{2,7};
p_treatment = T{3,7};
p_sex       = T{4,7};
p_order     = T{5,7};

Mean_SS = T{1,2};
Mean_perc_SS = T{1,3};

end