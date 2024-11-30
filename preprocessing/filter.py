import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt


def butter_highpass(lowcut, fs, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos: np.ndarray = butter(order, low, btype="highpass", output="sos")
    return sos


def butter_highpass_filter(data, lowcut, fs, order=4):
    sos = butter_highpass(lowcut, fs, order=order)
    return sosfilt(sos, data, axis=0)
