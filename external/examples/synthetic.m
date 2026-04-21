%% load data
load external/examples/trumpet.mat

%%
row_label_names = string(label_names{1});
col_label_names = string(label_names{2});

%%
times = obs_l;
sensors  = [var_l{1,:}];
channels = string(var_l(2,:));
freqs    = [var_l{3,:}];