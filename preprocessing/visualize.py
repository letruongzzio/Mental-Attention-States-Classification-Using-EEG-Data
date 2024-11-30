import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from mne.io.array.array import RawArray
from mne.time_frequency import psd_array_welch
from mne.preprocessing import ICA

from constants import FS, LOWCUT, HIGHCUT


def plot_channel_signal(df, channel):
    focused_df = df[df["state"] == "focused"]
    unfocused_df = df[df["state"] == "unfocused"]
    drownsy_df = df[df["state"] == "drownsy"]

    plt.plot(focused_df["t"], focused_df[channel])
    plt.plot(unfocused_df["t"], unfocused_df[channel])
    plt.plot(drownsy_df["t"], drownsy_df[channel])
    plt.title(f"{channel} signal by time")
    plt.xlabel("Time")
    plt.ylabel(f"{channel} Signal")
    plt.legend(["Focused", "Unfocused", "Drownsy"])
    plt.tight_layout()


def viz_fft(df, channel, state, start_idx, fs=128, epoch_length=128):

    # Get data for the specific channel and state
    data = df[(df["state"] == state)][channel].values

    # Check if there is enough data to get an epoch
    if start_idx + epoch_length > len(data):
        print(f"Not enough data for epoch starting at position {start_idx}.")
        return

    # Get data for the epoch
    epoch_data = data[start_idx : start_idx + epoch_length]

    # Check for empty data
    if len(epoch_data) == 0:
        print(f"Empty data for state '{state}' from position {start_idx}.")
        return

    # Apply FFT
    fft_data = np.fft.fft(epoch_data)
    freqs = np.fft.fftfreq(len(epoch_data), d=1 / fs)

    # Get positive frequencies
    positive_freqs = freqs[freqs >= 0]
    positive_fft_data = np.abs(fft_data[freqs >= 0])

    # Plot the graph
    plt.plot(positive_freqs, positive_fft_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of {channel}, State: {state}, Position: {start_idx}")
    plt.grid(True)


def viz_nepochs_state(
    df, channel, num_epochs, state="focused", fs=128, epoch_length=128
):
    plt.figure(figsize=(12, num_epochs * 3))
    current_idx = 0  # Start from the first sample
    epochs_plotted = 0

    # Get data for the specific state
    state_data = df[df["state"] == state]
    total_samples = len(state_data)

    while epochs_plotted < num_epochs and current_idx + epoch_length <= total_samples:
        plt.subplot(num_epochs, 1, epochs_plotted + 1)
        viz_fft(df, channel, state, current_idx, fs, epoch_length)
        epochs_plotted += 1
        current_idx += epoch_length

    if epochs_plotted == 0:
        print(f"No data to display for state '{state}'.")
    else:
        plt.tight_layout()


def plot_spectrogram(df, channel, fs=128, nperseg=256, noverlap=128):
    data = df[channel].values

    # Compute the spectrogram
    f, t, Sxx = spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)

    # Plot the spectrogram
    # plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    plt.colorbar(label="Power/Frequency (dB/Hz)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"Spectrogram of {channel}")


def plot_power_spectral_density(ica: ICA, raw: RawArray):
    # Extract signal from ICA
    ica_sources = ica.get_sources(raw).get_data()
    sources_length = len(ica_sources)
    fig, axes = plt.subplots(sources_length, 1, figsize=(20, sources_length * 5))
    axes = axes.flatten()

    for i, source in enumerate(ica_sources):
        psds, freqs = psd_array_welch(
            source, sfreq=FS, fmin=int(LOWCUT), fmax=80, n_fft=4096
        )

        # Change PSD to log (dB)
        psds_log = 10 * np.log10(psds)

        ax = axes[i]
        ax.plot(freqs, psds_log)
        ax.set_title(f"Component {i}", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(
            r"Log Power Spectral Density $10 \cdot \log_{10}(\mu V^2 / \mathrm{Hz})$"
        )
        ax.grid(True)

    # Tinh chỉnh khoảng cách giữa các plot
    plt.tight_layout()
