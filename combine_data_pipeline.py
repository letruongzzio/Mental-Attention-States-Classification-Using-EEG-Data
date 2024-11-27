import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, iirnotch
import matplotlib.pyplot as plt
import kagglehub
import pywt

print("Kaggle hub version:", kagglehub.__version__)

# Download latest version
# path = kagglehub.dataset_download("inancigdem/eeg-data-for-mental-attention-state-detection")
path = "/home/thangquang/.cache/kagglehub/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/versions/1"
print("Path to dataset files:", path)

# Join file paths and sort
data_path = os.path.join(path, 'EEG Data')
data_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
data_files.sort(key=lambda x: int(x.split('eeg_record')[1].split('.mat')[0]))

# Verify sorted paths
for file in data_files[:3]:
    print(file)

# Configuration parameters
fs = 128  # Sampling frequency
lowcut = 0.5  # Low cut-off frequency for band-pass filter
highcut = 40  # High cut-off frequency for band-pass filter

# Define helper functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

def notch_filter(data, fs, freq=50.0, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return lfilter(b, a, data, axis=0)

def get_state(t):
    if t <= 10 * 128 * 60:
        return 'focused'
    elif t > 20 * 128 * 60:
        return 'drowsy'
    else:
        return 'unfocused'

def preprocess_eeg_dataframe(df, channel_cols):
    eeg_data = df[channel_cols].values
    eeg_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)
    eeg_data = notch_filter(eeg_data, fs, freq=50.0)
    mean_ref = np.mean(eeg_data, axis=1, keepdims=True)
    eeg_data = eeg_data - mean_ref
    df[channel_cols] = eeg_data
    return df

def extract_data(file_path):
    mat_data = loadmat(file_path)
    data = mat_data['o'][0][0]['data']
    columns = ['ED_COUNTER', 'ED_INTERPOLATED', 'ED_RAW_CQ', 'ED_AF3', 'ED_F7', 'ED_F3', 'ED_FC5', 'ED_T7', 'ED_P7', 'ED_O1',
               'ED_O2', 'ED_P8', 'ED_T8', 'ED_FC6', 'ED_F4', 'ED_F8', 'ED_AF4', 'ED_GYROX', 'ED_GYROY', 'ED_TIMESTAMP',
               'ED_ES_TIMESTAMP', 'ED_FUNC_ID', 'ED_FUNC_VALUE', 'ED_MARKER', 'ED_SYNC_SIGNAL']

    df = pd.DataFrame(data, columns=columns)
    df = df.reset_index().rename(columns={'index': 't'})
    channel_columns = ['t'] + list(df.columns[4:18])
    df_channels = df[channel_columns].copy()
    df_channels['state'] = df_channels['t'].apply(get_state)
    filtered_df = preprocess_eeg_dataframe(df_channels, channel_columns[1:])
    useful_columns = ['t', 'ED_F7', 'ED_F3', 'ED_P7', 'ED_O1', 'ED_O2', 'ED_P8', 'ED_AF4', 'state']
    return filtered_df[useful_columns]

def skewness(x):
    n = len(x)
    mean_x = np.mean(x)
    numerator = np.sum((x - mean_x) ** 3) / n
    denominator = (np.sum((x - mean_x) ** 2) / n) ** 1.5
    return numerator / denominator if denominator != 0 else 0

def kurtosis(x):
    n = len(x)
    mean_x = np.mean(x)
    numerator = np.sum((x - mean_x) ** 4) / n
    denominator = (np.sum((x - mean_x) ** 2) / n) ** 2
    return numerator / denominator if denominator != 0 else 0

def compute_spectral_entropy(power_spectrum):
    num_channels = power_spectrum.shape[1]
    spectral_entropies = []
    for ch in range(num_channels):
        psd_ch = power_spectrum[:, ch]
        psd_sum = np.sum(psd_ch)
        if psd_sum == 0:
            spectral_entropy = 0
        else:
            P_i = psd_ch / psd_sum
            P_i = P_i[P_i > 0]
            spectral_entropy = -np.sum(P_i * np.log2(P_i))
        spectral_entropies.append(spectral_entropy)
    return spectral_entropies

def compute_band_powers(power_spectrum, freqs_in_band):
    band_powers = {}
    for band_name, band_mask in freqs_in_band.items():
        band_power = np.sum(power_spectrum[band_mask, :], axis=0)
        band_powers[band_name] = band_power
    return band_powers

def compute_relative_powers(band_powers):
    total_power = np.sum(list(band_powers.values()), axis=0)
    relative_powers = {}
    for band_name in band_powers.keys():
        relative_powers[band_name] = band_powers[band_name] / total_power
    return relative_powers

def compute_peak_frequencies(positive_fft_data, positive_freqs):
    freq_indices = np.argmax(positive_fft_data, axis=0)
    peak_freqs = positive_freqs[freq_indices]
    return peak_freqs

def compute_statistics(positive_fft_data):
    mean_vals = np.mean(positive_fft_data, axis=0)
    std_vals = np.std(positive_fft_data, axis=0)
    median_vals = np.median(positive_fft_data, axis=0)
    min_vals = np.min(positive_fft_data, axis=0)
    max_vals = np.max(positive_fft_data, axis=0)
    return mean_vals, std_vals, median_vals, min_vals, max_vals

def compute_skewness_kurtosis(positive_fft_data):
    num_channels = positive_fft_data.shape[1]
    skewness_vals = []
    kurtosis_vals = []
    for ch in range(num_channels):
        ch_data = positive_fft_data[:, ch]
        skewness_vals.append(skewness(ch_data))
        kurtosis_vals.append(kurtosis(ch_data))
    return skewness_vals, kurtosis_vals

def compute_wavelet_features(window_data, wavelet_name='db4', level=5):
    num_channels = window_data.shape[1]
    wavelet_features = {}
    for ch in range(num_channels):
        coeffs = pywt.wavedec(window_data[:, ch], wavelet_name, level=level)
        detail_energies = [np.sum(np.square(coeff)) for coeff in coeffs[1:]]
        total_energy = sum(detail_energies) if sum(detail_energies) != 0 else 1
        relative_energies = [e / total_energy for e in detail_energies]
        wavelet_features[ch] = {
            'detail_energies': detail_energies,
            'relative_energies': relative_energies
        }
    return wavelet_features

def feature_extraction(df, fs, window_length, step_size):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    len_data = len(df)
    feature_list = []
    labels = []
    num_channels = df.shape[1] - 2
    channel_names = df.columns[1:-1]

    eeg_data = df.iloc[:, 1:num_channels+1].values
    states = df.iloc[:, -1].values

    window_samples = window_length
    step_samples = step_size

    freqs = np.fft.fftfreq(window_samples, d=1/fs)
    positive_freqs = freqs[freqs >= 0]

    freqs_in_band = {}
    for band_name, (low, high) in bands.items():
        freqs_in_band[band_name] = (positive_freqs >= low) & (positive_freqs < high)

    for start in range(0, len_data - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = eeg_data[start:end, :]

        if window_data.shape[0] != window_samples:
            continue

        window_labels = states[start:end]
        unique_labels, counts = np.unique(window_labels, return_counts=True)

        if len(unique_labels) > 1:
            continue
        else:
            label = unique_labels[0]

        fft_data = np.fft.fft(window_data, axis=0)
        positive_fft_data = np.abs(fft_data[freqs >= 0, :])

        power_spectrum = positive_fft_data ** 2

        spectral_entropies = compute_spectral_entropy(power_spectrum)
        band_powers = compute_band_powers(power_spectrum, freqs_in_band)
        relative_powers = compute_relative_powers(band_powers)
        peak_freqs = compute_peak_frequencies(positive_fft_data, positive_freqs)
        mean_vals, std_vals, median_vals, min_vals, max_vals = compute_statistics(positive_fft_data)
        skewness_vals, kurtosis_vals = compute_skewness_kurtosis(positive_fft_data)
        wavelet_features = compute_wavelet_features(window_data)

        feature_dict = {}
        for idx, ch_name in enumerate(channel_names):
            feature_dict[f'{ch_name}_spectral_entropy'] = spectral_entropies[idx]
            feature_dict[f'{ch_name}_peak_frequency'] = peak_freqs[idx]
            feature_dict[f'{ch_name}_mean'] = mean_vals[idx]
            feature_dict[f'{ch_name}_std'] = std_vals[idx]
            feature_dict[f'{ch_name}_median'] = median_vals[idx]
            feature_dict[f'{ch_name}_min'] = min_vals[idx]
            feature_dict[f'{ch_name}_max'] = max_vals[idx]
            feature_dict[f'{ch_name}_skewness'] = skewness_vals[idx]
            feature_dict[f'{ch_name}_kurtosis'] = kurtosis_vals[idx]

            for band_name in bands.keys():
                feature_dict[f'{ch_name}_power_{band_name}'] = band_powers[band_name][idx]
                feature_dict[f'{ch_name}_relative_power_{band_name}'] = relative_powers[band_name][idx]

            for level_idx, energy in enumerate(wavelet_features[idx]['detail_energies'], start=1):
                feature_dict[f'{ch_name}_wavelet_d{level_idx}_energy'] = energy
            for level_idx, rel_energy in enumerate(wavelet_features[idx]['relative_energies'], start=1):
                feature_dict[f'{ch_name}_wavelet_d{level_idx}_relative_energy'] = rel_energy

        feature_dict['state'] = label
        feature_list.append(feature_dict)

    feature_df = pd.DataFrame(feature_list)
    return feature_df

print("Processing Stage ....")
window_length = 256
step_rate = 0.25
step_size = int(window_length * step_rate)

final_df = None

for idx, file in enumerate(data_files):
    new_df = extract_data(file)
    new_df = feature_extraction(new_df, fs, window_length, step_size)
    if final_df is None:
        final_df = new_df
    else:
        final_df = pd.concat([final_df, new_df], ignore_index=True)

final_df.reset_index(drop=True, inplace=True)
print("Number of samples", len(final_df))
final_df.to_csv("data/full_eeg.csv", index=False)
