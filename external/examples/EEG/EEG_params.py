import mne
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

def load_eeg_data():
    """Loads raw EDF data once to avoid redundant disk I/O operations."""
    subject_info = pd.read_csv("external/data/EEG/subject-info.csv")
    subject_ids = subject_info["Subject"].to_numpy(dtype=str)
    operations = [1, 2]
    channels_names = None
    
    sample_file = f"external/data/EEG/{subject_ids[0]}_1.edf"
    sample_data = mne.io.read_raw_edf(sample_file, verbose=False)
    n_channels = len(sample_data.ch_names)
    
    Y = np.full((len(subject_ids), len(operations), n_channels, int(1e5)), np.nan)

    print("Loading EDF files...")
    for i, op in enumerate(operations):
        for j, sid in enumerate(subject_ids):
            file = f"external/data/EEG/{sid}_{op}.edf"
            if os.path.exists(file):
                data = mne.io.read_raw_edf(file, verbose=False)
                aux = data.get_data()
                last_id = aux.shape[-1] - 1000
                Y[j, i, :, 0:last_id] = aux[:, 0:last_id]
                if channels_names is None:
                    channels_names = [x[4:] for x in data.ch_names]

    mask = ~np.isnan(Y).any(axis=(0, 1, 2))
    Y = Y[:, :, :, mask]
    return Y, subject_ids, channels_names

def run_stft_parglm_pipeline(Y, channels, window_sec, hop_fraction, window_func):
    """Executes the processing pipeline and returns PercSS and P-value."""
    original_sr = 500
    target_sr = 100
    Y_resampled = librosa.resample(Y, orig_sr=original_sr, target_sr=target_sr)

    sr = target_sr
    window_length = int(window_sec * sr)
    n_fft = window_length
    hop = int(window_length * hop_fraction)
    hop = max(1, hop)

    D = stft(Y_resampled, window_length, hop, window=window_func, center=True, n_fft=n_fft)
    Sxx = librosa.amplitude_to_db(np.abs(D))
    times, freqs = stft_labels(D, sr, hop, n_fft=n_fft)

    mask = (freqs >= 0.5) & (freqs <= 45.0)
    Sxx = Sxx[:, :, :, mask, :]
    freqs = freqs[mask]

    subjects = np.arange(Sxx.shape[0]) + 1
    operations = np.arange(2)
    Sxx = Sxx[:, :, 0:20]
    current_channels = channels[0:20]

    labels = [subjects, operations, current_channels, freqs, times]
    label_names = ["subjects", "operations", "channels", "freq", "time"]

    Sxx_mean = np.nanmean(Sxx, axis=2)
    labels.pop(2)
    label_names.pop(2)

    X_unfold, labels_unfold, _ = unfold_2D(Sxx_mean, rows=[0, 1], cols=[2, 3], 
                                           labels=labels, label_names=label_names)
    
    su = labels_unfold[0][0]
    op = labels_unfold[0][1]
    F = np.vstack([su, op]).T

    T, _ = parglm(X_unfold, F, Preprocessing=1)
    
    return T["PercSumSq"][2], T["Pvalue"][2]

def plot_heatmaps_shared_scale(csv_file, output_dir):
    """Generates heatmaps with shared color scale and conditional formatting."""
    df = pd.read_csv(csv_file)
    df['-log10(p)'] = -np.log10(df['p_value'] + 1e-15) # epsilon to avoid log(0)
    
    window_functions = df['window_function'].unique()
    
    # Calculate global min/max for PercSS to share color scale
    vmin = df['PercSS'].min()
    vmax = df['PercSS'].max()

    for win in window_functions:
        subset = df[df['window_function'] == win]
        
        # Pivot data for heatmap and annotations
        perc_pivot = subset.pivot(index="window_length", columns="hop_fraction", values="PercSS")
        logp_pivot = subset.pivot(index="window_length", columns="hop_fraction", values="-log10(p)")
        
        # Create custom annotation matrix: "PercSS\n(-log10p)"
        annot_matrix = []
        for i in range(len(perc_pivot)):
            row = []
            for j in range(len(perc_pivot.columns)):
                p_val = perc_pivot.iloc[i, j]
                lp_val = logp_pivot.iloc[i, j]
                row.append(f"{p_val:.3f}\n({lp_val:.2f})")
            annot_matrix.append(row)
        
        plt.figure(figsize=(12, 9))
        
        # Create a mask for cells where -log10(p) < 2
        mask = logp_pivot < 2
        
        # Plot heatmap. We use two layers: one for the grey background, one for data
        # Layer 1: Grey background for masked values
        sns.heatmap(perc_pivot, mask=~mask, cmap=mcolors.ListedColormap(['#C0C0C0']), 
                    cbar=False, annot=np.array(annot_matrix), fmt="", annot_kws={"color": "black"})
        
        # Layer 2: Main data with shared color scale
        sns.heatmap(perc_pivot, mask=mask, annot=np.array(annot_matrix), fmt="",
                    cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)

        plt.title(f'PercSS and -log10(p) | Window: {win}\n(Grey cells indicate -log10(p) < 2)')
        plt.xlabel('Hop Fraction')
        plt.ylabel('Window Length (s)')

        plot_path = os.path.join(output_dir, f'eeg_heatmap_{win}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved heatmap for {win} to {plot_path}")

if __name__ == "__main__":
    windows_sec = [0.05, 0.10, 0.25, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 60.0]
    hops_fraction = [1.0, 0.90, 0.80, 0.75, 0.5, 0.25, 0.20, 0.10]
    window_functions = ["boxcar", "hann", "hamming", "blackman"]

    Y_raw, _, channels_list = load_eeg_data()
    results = []

    print(f"\nStarting parameter sweep...")
    print(f"{'Win Fnc':<10} | {'Win (s)':<8} | {'Hop':<6} | {'PercSS':<10} | {'p-value':<10}")
    print("-" * 60)

    for wf in window_functions:
        for w in windows_sec:
            for h in hops_fraction:
                try:
                    percss, p_val = run_stft_parglm_pipeline(Y_raw, channels_list, w, h, wf)
                    results.append({
                        "window_function": wf,
                        "window_length": w,
                        "hop_fraction": h,
                        "PercSS": percss,
                        "p_value": p_val
                    })
                    print(f"{wf:<10} | {w:<8.2f} | {h:<6.2f} | {percss:<10.6f} | {p_val:<10.4e}")
                except Exception as e:
                    print(f"Error in {wf} W:{w}, H:{h} -> {e}")

    output_path = "external/examples/EEG/"
    os.makedirs(output_path, exist_ok=True)
    
    csv_file = os.path.join(output_path, "eeg_params.csv")
    pd.DataFrame(results).to_csv(csv_file, index=False)
    
    print(f"\nResults saved to {csv_file}")
    plot_heatmaps_shared_scale(csv_file, output_path)