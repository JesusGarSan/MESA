import librosa
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mesa.features import stft, stft_labels
from mesa.fusion import unfold_2D
from parglm_torch.parglm import parglm
from airPLS import airPLS

def load_gc_fid_data():
    """Loads GC-FID data, removes empty samples, and performs airPLS baseline correction."""
    # Load signal data and labels
    # Paths are relative to the notebook's environment
    Y_labels = pd.read_csv("external/data/GC-FID/Y_data.csv", header=None).to_numpy()
    X_raw = pd.read_csv("external/data/GC-FID/X_data.csv", header=None).to_numpy()


    idx = np.where(np.sum(Y_labels, axis=1)==0)
    X = np.delete(X_raw, idx, axis=1).T # Shape: (samples, time_points)
    F_data = np.delete(Y_labels, idx, axis = 0)

    print(f"Removing baseline for {X.shape[0]} samples...")
    for i in range(X.shape[0]):
        correction = airPLS(X[i])
        X[i] -= correction
        
    return X, F_data

def run_stft_parglm_pipeline(X, F_data, window_length, hop_fraction, window_func):
    """Executes STFT, unfolding, and parglm for a specific parameter set."""
    sr = 200 # Sample rate as per notebook
    window_length = window_length
    # Ensure window length doesn't exceed signal length
    window_length = min(window_length, X.shape[1])
    n_fft = window_length 
    hop = int(window_length * hop_fraction)
    hop = max(1, hop)

    # STFT and dB conversion
    D = stft(X, window_length, hop, window=window_func, center=True, n_fft=n_fft)
    Sxx = librosa.amplitude_to_db(np.abs(D))
    times, freqs = stft_labels(D, sr, hop, n_fft=n_fft)

    # Frequency masking (0.0 to 15.0 Hz as per notebook)
    mask = (freqs >= 0.0) & (freqs <= 15.0)
    Sxx = Sxx[:, mask, :]
    freqs = freqs[mask]

    # Unfolding
    subjects = np.arange(Sxx.shape[0]) + 1
    labels = [subjects, freqs, times]
    label_names = ["subjects", "freq", "time"]

    X_unfold, labels_unfold, _ = unfold_2D(Sxx, rows=[0], cols=[1, 2], 
                                           labels=labels, label_names=label_names)
    
    # We use F_data from Y_data.csv as the design matrix for GLM
    # Assuming we analyze the effect of the first factor in F_data
    # You might need to adjust which column of F_data you are testing
    T, _ = parglm(X_unfold, F_data, Preprocessing=1)
    
    # Returning PercSS and P-value for the first factor (index 2 in parglm output usually)
    # Adjust index if F_data has multiple columns and you want a specific one
    return T["PercSumSq"][2], T["Pvalue"][2]

def plot_heatmaps_shared_scale(csv_file, output_dir):
    """Generates heatmaps with shared color scale for the parameter sweep."""
    df = pd.read_csv(csv_file)
    df['-log10(p)'] = -np.log10(df['p_value'] + 1e-15)
    
    window_functions = df['window_function'].unique()
    vmin, vmax = df['PercSS'].min(), df['PercSS'].max()

    for win in window_functions:
        subset = df[df['window_function'] == win]
        perc_pivot = subset.pivot(index="window_length", columns="hop_fraction", values="PercSS")
        logp_pivot = subset.pivot(index="window_length", columns="hop_fraction", values="-log10(p)")
        
        annot_matrix = []
        for i in range(len(perc_pivot)):
            row = []
            for j in range(len(perc_pivot.columns)):
                row.append(f"{perc_pivot.iloc[i, j]:.3f}\n({logp_pivot.iloc[i, j]:.2f})")
            annot_matrix.append(row)
        
        plt.figure(figsize=(12, 9))
        mask = logp_pivot < 2 # Highlight significant results
        
        sns.heatmap(perc_pivot, mask=~mask, cmap=mcolors.ListedColormap(['#C0C0C0']), 
                    cbar=False, annot=np.array(annot_matrix), fmt="", annot_kws={"color": "black"})
        sns.heatmap(perc_pivot, mask=mask, annot=np.array(annot_matrix), fmt="",
                    cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)

        plt.title(f'GC-FID: PercSS & -log10(p) | Window: {win}\n(Grey cells: p > 0.01)')
        plt.xlabel('Hop Fraction')
        plt.ylabel('Window Length (s)')

        plt.savefig(os.path.join(output_dir, f'gc_fid_heatmap_{win}.png'))
        plt.close()

if __name__ == "__main__":
    windows_length = [10, 20, 40, 60, 100, 200, 400, 800, 1200, 2000]
    hops_fraction = [1.0, 0.75, 0.5, 0.25, 0.10]
    window_functions = ["boxcar", "hann", "hamming"]

    X_data, F_labels = load_gc_fid_data()
    results = []

    print(f"\nStarting parameter sweep...")
    print(f"{'Win Fnc':<10} | {'Win (samples)':<13} | {'Hop':<6} | {'PercSS':<10} | {'p-value':<10}")
    print("-" * 60)

    for wf in window_functions:
        for w in windows_length:
            for h in hops_fraction:
                try:
                    percss, p_val = run_stft_parglm_pipeline(X_data, F_labels, w, h, wf)
                    results.append({
                        "window_function": wf, "window_length": w,
                        "hop_fraction": h, "PercSS": percss, "p_value": p_val
                    })
                    print(f"{wf:<10} | {w:<13.2f} | {h:<6.2f} | {percss:<10.6f} | {p_val:<10.4e}")
                except Exception as e:
                    print(f"Error in {wf} W:{w}, H:{h} -> {e}")

    output_path = "results_gc_fid/"
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(output_path, "gc_fid_params.csv"), index=False)
    plot_heatmaps_shared_scale(os.path.join(output_path, "gc_fid_params.csv"), output_path)