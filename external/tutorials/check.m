load data.mat
close all

prep = 2;
X = preprocess2D(spectrogram', 'Preprocessing', prep);

varPca(X, 'Preprocessing', 0, 'PCs', 1:5,'PlotCkf', true);
model = pcaEig(X, 'PCs', 1:2);
biplot(model)