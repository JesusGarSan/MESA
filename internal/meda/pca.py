from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import scipy

def preprocess(data, method="autoscale"):
    if method is None:
        return data.copy()  # Devolvemos una copia para evitar modificaciones in-place

    scaler = StandardScaler()

    if method == "demean":
        scaler.scale_ = np.ones(data.shape[1])  # Forzamos que la escala sea 1 (solo centrado)
        scaler.mean_ = np.mean(data, axis=0)
        data_processed = scaler.transform(data)
    elif method == "autoscale":
        data_processed = scaler.fit_transform(data)
    else:
        raise ValueError(f"Método de preprocesado '{method}' no reconocido. Debe ser 'demean', 'autoscale' o None.")

    return data_processed

def pca(data, n_components):
    pca_model = PCA(n_components=n_components)
    pca_model.fit(data)
    scores = pca_model.transform(data)  # Obtener la matriz de scores
    loadings = pca_model.components_.T   # Obtener la matriz de loadings (transpuesta de components_)
    return scores, loadings, pca_model.explained_variance_ratio_

def scores_plot(scores, exp_var = None, color = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    fig.suptitle("Scores plot")
    if color is not None:
        scatter = ax.scatter(scores[:,0], scores[:,1],c=color, cmap = 'viridis')
    else:
        scatter = ax.scatter(scores[:,0], scores[:,1], color="royalblue")

    ax.grid()
    cbar = fig.colorbar(scatter)
    cbar.set_label("Time") # Añadimos una etiqueta al colorbar
    if exp_var is not None:
        ax.set_xlabel(f"PC 1 ({exp_var[0]*100:.2f}%)")
        ax.set_ylabel(f"PC 2 ({exp_var[1]*100:.2f}%)")
    else:
        ax.set_xlabel(f"PC 1")
        ax.set_ylabel(f"PC 2")

    return fig, ax

def loadings_plot(loadings, exp_var = None, color = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    fig.suptitle("Loadings plot")
    if color is not None:
        scatter = ax.scatter(loadings[:,0], loadings[:,1],c=color, cmap = 'viridis')
    else:
        scatter = ax.scatter(loadings[:,0], loadings[:,1], color="royalblue")

    ax.grid()
    cbar = fig.colorbar(scatter)
    # cbar.set_label("") # Añadimos una etiqueta al colorbar
    if exp_var is not None:
        ax.set_xlabel(f"PC 1 ({exp_var[0]*100:.2f}%)")
        ax.set_ylabel(f"PC 2 ({exp_var[1]*100:.2f}%)")
    else:
        ax.set_xlabel(f"PC 1")
        ax.set_ylabel(f"PC 2")

    return fig, ax

if __name__ =="__main__":
    pass
    # matrix, variables = load("tests/data/matrix.mat")

    # print(matrix)
    # print(variables)