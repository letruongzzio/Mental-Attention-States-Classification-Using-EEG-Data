import numpy as np
import pandas as pd
import pywt  # Library for Wavelet transform
from constants import FS

def compute_statistics(positive_fft_data):
    """Compute statistical measures for each channel."""
    mean_vals = np.mean(positive_fft_data, axis=0)
    std_vals = np.std(positive_fft_data, axis=0)
    median_vals = np.median(positive_fft_data, axis=0)
    min_vals = np.min(positive_fft_data, axis=0)
    max_vals = np.max(positive_fft_data, axis=0)
    return mean_vals, std_vals, median_vals, min_vals, max_vals


def compute_band_powers(power_spectrum, freqs_in_band):
    """Compute band powers for each frequency band and each channel."""
    band_powers = {}
    for band_name, band_mask in freqs_in_band.items():
        band_power = np.sum(power_spectrum[band_mask, :], axis=0)
        band_powers[band_name] = band_power
    return band_powers


def compute_relative_powers(band_powers):
    """Compute relative powers for each frequency band and each channel."""
    total_power = np.sum(list(band_powers.values()), axis=0)
    relative_powers = {}
    for band_name in band_powers.keys():
        relative_powers[band_name] = band_powers[band_name] / total_power
    return relative_powers


def compute_peak_frequencies(positive_fft_data, positive_freqs):
    """Find peak frequencies for each channel."""
    freq_indices = np.argmax(positive_fft_data, axis=0)
    peak_freqs = positive_freqs[freq_indices]
    return peak_freqs


def compute_spectral_entropy(power_spectrum):
    """Compute spectral entropy for each channel."""
    num_channels = power_spectrum.shape[1]
    spectral_entropies = []
    for ch in range(num_channels):
        psd_ch = power_spectrum[:, ch]
        psd_sum = np.sum(psd_ch)
        if psd_sum == 0:
            spectral_entropy = 0
        else:
            P_i = psd_ch / psd_sum
            P_i = P_i[P_i > 0]  # Avoid log(0)
            spectral_entropy = -np.sum(P_i * np.log2(P_i))
        spectral_entropies.append(spectral_entropy)
    return spectral_entropies


def skewness(x):
    """Calculate the skewness of the sequence x."""
    n = len(x)
    mean_x = np.mean(x)
    numerator = np.sum((x - mean_x) ** 3) / n
    denominator = (np.sum((x - mean_x) ** 2) / n) ** 1.5
    return numerator / denominator if denominator != 0 else 0


def kurtosis(x):
    """Calculate the kurtosis of the sequence x."""
    n = len(x)
    mean_x = np.mean(x)
    numerator = np.sum((x - mean_x) ** 4) / n
    denominator = (np.sum((x - mean_x) ** 2) / n) ** 2
    return numerator / denominator if denominator != 0 else 0


def compute_skewness_kurtosis(positive_fft_data):
    """Compute skewness and kurtosis for each channel."""
    num_channels = positive_fft_data.shape[1]
    skewness_vals = []
    kurtosis_vals = []
    for ch in range(num_channels):
        ch_data = positive_fft_data[:, ch]
        skewness_vals.append(skewness(ch_data))
        kurtosis_vals.append(kurtosis(ch_data))
    return skewness_vals, kurtosis_vals


def compute_wavelet_features(window_data, wavelet_name="db4", level=5):
    """Compute wavelet features for each channel."""
    num_channels = window_data.shape[1]
    wavelet_features = {}
    for ch in range(num_channels):
        coeffs = pywt.wavedec(window_data[:, ch], wavelet_name, level=level)
        detail_energies = [
            np.sum(np.square(coeff)) for coeff in coeffs[1:]
        ]  # Skip approximation coeffs
        total_energy = sum(detail_energies) if sum(detail_energies) != 0 else 1
        relative_energies = [e / total_energy for e in detail_energies]
        wavelet_features[ch] = {
            "detail_energies": detail_energies,
            "relative_energies": relative_energies,
        }
    return wavelet_features


def feature_extraction(df, window_length=256, step_size=64):
    """
    Extract features from EEG data and return a DataFrame.
    """
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    len_data = len(df)
    feature_list = []
    labels = []
    num_channels = df.shape[1] - 2  # Exclude time and state columns
    channel_names = df.columns[1:-1]  # Get channel names

    # Convert data to numpy array for performance improvement
    eeg_data = df.iloc[:, 1 : num_channels + 1].values  # EEG data
    states = df.iloc[:, -1].values  # State labels

    window_samples = window_length
    step_samples = step_size

    # Precompute frequencies for FFT
    freqs = np.fft.fftfreq(window_samples, d=1 / FS)
    positive_freqs = freqs[freqs >= 0]

    # Precompute masks for frequency bands
    freqs_in_band = {}
    for band_name, (low, high) in bands.items():
        freqs_in_band[band_name] = (positive_freqs >= low) & (positive_freqs < high)

    # Iterate over windows
    for start in range(0, len_data - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = eeg_data[start:end, :]

        # Check if the window is of sufficient length
        if window_data.shape[0] != window_samples:
            continue

        window_labels = states[start:end]
        unique_labels, counts = np.unique(window_labels, return_counts=True)

        # Check if the window contains multiple classes
        if len(unique_labels) > 1:
            continue
        else:
            label = unique_labels[0]

        # Apply FFT to the window
        fft_data = np.fft.fft(window_data, axis=0)
        positive_fft_data = np.abs(fft_data[freqs >= 0, :])

        # Calculate power spectrum
        power_spectrum = positive_fft_data**2

        # Compute features
        mean_vals, std_vals, median_vals, min_vals, max_vals = compute_statistics(
            positive_fft_data
        )
        band_powers = compute_band_powers(power_spectrum, freqs_in_band)
        relative_powers = compute_relative_powers(band_powers)
        peak_freqs = compute_peak_frequencies(positive_fft_data, positive_freqs)
        spectral_entropies = compute_spectral_entropy(power_spectrum)
        skewness_vals, kurtosis_vals = compute_skewness_kurtosis(positive_fft_data)
        wavelet_features = compute_wavelet_features(window_data)

        # Build feature dictionary for this window
        feature_dict = {}
        for idx, ch_name in enumerate(channel_names):
            # Features per channel
            feature_dict[f"{ch_name}_spectral_entropy"] = spectral_entropies[idx]
            feature_dict[f"{ch_name}_peak_frequency"] = peak_freqs[idx]
            feature_dict[f"{ch_name}_mean"] = mean_vals[idx]
            feature_dict[f"{ch_name}_std"] = std_vals[idx]
            feature_dict[f"{ch_name}_median"] = median_vals[idx]
            feature_dict[f"{ch_name}_min"] = min_vals[idx]
            feature_dict[f"{ch_name}_max"] = max_vals[idx]
            feature_dict[f"{ch_name}_skewness"] = skewness_vals[idx]
            feature_dict[f"{ch_name}_kurtosis"] = kurtosis_vals[idx]

            # Band powers and relative powers
            for band_name in bands.keys():
                feature_dict[f"{ch_name}_power_{band_name}"] = band_powers[band_name][
                    idx
                ]
                feature_dict[f"{ch_name}_relative_power_{band_name}"] = relative_powers[
                    band_name
                ][idx]

            # Wavelet features
            # Detail energies
            for level_idx, energy in enumerate(
                wavelet_features[idx]["detail_energies"], start=1
            ):
                feature_dict[f"{ch_name}_wavelet_d{level_idx}_energy"] = energy
            # Relative energies
            for level_idx, rel_energy in enumerate(
                wavelet_features[idx]["relative_energies"], start=1
            ):
                feature_dict[f"{ch_name}_wavelet_d{level_idx}_relative_energy"] = (
                    rel_energy
                )

        feature_dict["state"] = label
        feature_list.append(feature_dict)

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list)
    return feature_df
