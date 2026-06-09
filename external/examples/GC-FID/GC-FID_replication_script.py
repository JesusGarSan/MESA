# %% Import libraries
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.features import stft, stft_labels
from mesa.visualization import plot_signal, dual_plot, spectrogram
import librosa
from airPLS import airPLS

# Experiment parameters
correct_pipeline = False # Run the corrected pipeline or replicate the incorrect one
filter = True # Delete or keep high frequencies
slip = 0 # Arificial slip for testing different samples as blanks. Keep at 0 for true replication of the original pipeline

subtitle = "Replicated incorrect pipeline"
if correct_pipeline:
        subtitle="Corrected pipeline"

# %% Read data
F_data = pd.read_csv("external/data/GC-FID/Y_data.csv", header=None)
F_data_matrix = F_data.to_numpy()

idx = np.where(np.sum(F_data_matrix, axis=1)==0)
idx_blank_24h = [0, 1, 2, 3 ]
idx_blank_72h = [52,53,54,55]
print(f"Identified blanks: {idx[0]}")
print(f"24h blank ids: {idx_blank_24h}")
print(f"72h blank ids: {idx_blank_72h}\n")

df = pd.read_csv("external/data/GC-FID/X_data.csv", header=None)
X = df.to_numpy().T

print(f"Data   matrix X.shape: {X.shape}")
print(f"Design matrix F.shape: {F_data_matrix.shape}")

# %% Delete blanks
delete = not(correct_pipeline) # True: incorrect pipeline. False: Corrected pipeline
if delete:
    print(f"Deleting entries with ids: {idx[0]}\n")
    X = np.delete(X, idx, axis=0)
    F_data_matrix = np.delete(F_data_matrix, idx, axis=0)
    
    print(f"New Data   matrix X.shape: {X.shape}")
    print(f"New Design matrix F.shape: {F_data_matrix.shape}\n")


# %% FFT calculation
sr = 200
n_fft = 45001 // 1
FFT = (np.fft.fft(X, n = n_fft)[:,0:n_fft//2+1])
# FFT = FFT[:,:,np.newaxis]
freqs = np.fft.fftfreq(n_fft, 1/sr)[0:n_fft//2+1]

print(f"FFT.shape: {FFT.shape} # samples, freqs, times")
print(f"freqs.shape: {freqs.shape}\n")
# %% Delete high frequencies
if filter:
    print("Filtering frequencies...")
    mask = np.ones(len(freqs), dtype=bool)

    mask[freqs >  5.0] = False
    mask[freqs <  0.0] = False
    # id_50Hz = np.argmin(np.abs(freqs - 50.0))
    # mask[id_50Hz] = False


    freqs = freqs[mask]
    FFT = FFT[:, mask]
    n_frequencies = len(freqs)

    print(f"New FFT.shape: {FFT.shape} # samples, freqs, times")
    print(f"New freqs.shape: {freqs.shape}\n")

# Calculate averages of groups

idx_24h = np.where(F_data_matrix[:, 0] == 1)[0]
idx_72h = np.where(F_data_matrix[:, 0] == 2)[0]

print(f"ids of group 24h:\n{idx_24h}")
print(f"ids of group 72h:\n{idx_72h}\n")

# Averages across freqs before removing baseline
avg_24h = np.mean(FFT[idx_24h], axis = 0)
avg_72h = np.mean(FFT[idx_72h], axis = 0)  

# Arificial slip for testing
idx_blank_24h = np.array(idx_blank_24h) + slip
idx_blank_72h = np.array(idx_blank_72h) + slip

# Blanks
avg_blanks_24h = np.mean(FFT[idx_blank_24h], axis = 0)
avg_blanks_72h = np.mean(FFT[idx_blank_72h], axis = 0)

FFT[idx_24h] -= avg_blanks_24h
FFT[idx_72h] -= avg_blanks_72h

# Averages across freqs after removing baseline
avg_B_24h = np.mean(FFT[idx_24h], axis = 0)
avg_B_72h = np.mean(FFT[idx_72h], axis = 0)

#%%
print("Plotting FFT group averages...\n")

fig, ax = plt.subplots()
ax.plot(np.abs(avg_24h), label ="Sxx average 24h", color="royalblue");
ax.plot(np.abs(avg_72h), label ="Sxx average 72h", color = "orange");
ax.legend()
fig.suptitle(f"FFT average before removing blank averages\n{subtitle}");
fig.show()

fig, ax = plt.subplots()
ax.plot(np.abs(avg_blanks_24h), label ="24h blanks", color="royalblue");
ax.plot(np.abs(avg_blanks_72h), label ="72h blanks", color="orange");
ax.legend()
fig.suptitle(f"Blank FFT averages across frequencies\n{subtitle}");
fig.show()

fig, ax = plt.subplots()
ax.plot(np.abs(avg_B_24h), label ="Sxx average 24h", color="royalblue");
ax.plot(np.abs(avg_B_72h), label ="Sxx average 72h", color="orange");
ax.legend()
fig.suptitle(f"FFT average after removing blank averages\n{subtitle}");
fig.show()



# %% Save the data
if FFT.shape[0] > 96: # Checks if the blanks have been deleted
    print("Deleting blanks after blank average removal...")
    print(f"Blank ids: {idx}")
    FFT = np.delete(FFT, idx, axis=0)
    F_data_matrix = np.delete(F_data_matrix, idx, axis=0)

path = "external/examples/GC-FID/GC-FID.mat"
label_names = ["subjects", "freq", "time"]
subjects = np.arange(96)+1
savemat(
    path, {"data": FFT,
                "label_names": np.array(label_names, dtype=object),
                "obs_l": np.array(subjects,  dtype=object),
                "var_l": np.array(freqs,  dtype=object),
                "F_data": F_data_matrix}
)

print(f"Data saved at {path}.")

input()