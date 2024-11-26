import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos: np.ndarray = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfilt(sos, data, axis=0)

def notch_filter(data, fs, freq=50.0, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return lfilter(b, a, data, axis=0)
