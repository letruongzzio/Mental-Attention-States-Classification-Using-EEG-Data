import pandas as pd
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, zpk2sos, tf2zpk

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
    # Thiết kế bộ lọc Notch
    b, a = iirnotch(freq / (0.5 * fs), quality)
    # Chuyển đổi sang dạng SOS
    sos = zpk2sos(*tf2zpk(b, a))
    # Áp dụng bộ lọc
    return sosfilt(sos, data, axis=0)
