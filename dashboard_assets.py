# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:35:16 2025

@author: Admin
"""

import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal import fftconvolve
from math import floor
import matplotlib.pyplot as plt

# === Custom Filters ===
def bandpass_filter(signal, fs, lowcut=0.8, highcut=16, order=4):
    """
    Applies a butterworth bandpass filter to the given signal.
    
    Parameters:
    - signal (np.ndarray): The signal to filter.
    - lowcut (float): The low cutoff frequency of the filter (Hz)
    - highcut (float): The high cutoff frequency of the filter (Hz)
    - fs (float, optional): The sampling frequency (default is fs) (Hz)
    - order (int, optional): The order of the filter (default is 4).
    
    Returns:
    - np.ndarray: The filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: {low}, {high}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def bandpass_filter_2(signal, fs, highWindow=30):
    # moving average
    # fs isnt actually used in this function, it is only present to follow the same formatting as the other bandpass filters
    signal = np.copy(signal)
    signal -= np.mean(signal)
    for i in range(0, len(signal), highWindow):
        signal[i:i+highWindow] -= np.mean(signal[i:i+highWindow])
    return signal

def bandpass_filter_3(signal, fs, highCut=16):
    # moving average with automatic frequency cutoff choice
    order = floor(fs / highCut)
    signal = np.copy(signal)
    signal -= np.mean(signal)
    for i in range(0, len(signal), order):
        signal[i:i+order] -= np.mean(signal[i:i+order])
    return signal

def bandpass_filter_4_legacy(signal, fs, highCut=16):
    # weighted moving average
    # before cosh adjustement 
    order = floor(fs / highCut)
    signal = np.copy(signal)
    delta = np.floor(order / 2).astype(int)
    signal = np.concatenate([signal[delta-1::-1], signal, signal[:-delta-1:-1]])
    signal -= np.mean(signal)
    result = np.zeros(len(signal) - 2 * delta)
    for i in range(len(result)):
        result[i] = np.mean(signal[i:i+2*delta])
    return result

def bandpass_filter_4(signal, fs, K = 2):
    # weighted moving average
    # TODO: correct the effect of the offset due to the 'same' convolution (negligeable if signal is long enough)
    # normalization
    norm_constant = 4 * K - np.sinh(1 / (2 * K) + 1) / np.sinh(1 / (2 * K)) + 2
    signal -= np.mean(signal)
    avg_filter = np.array([2-np.cosh(k/K) for k in range(-K,K+1)])/norm_constant
    avg_filter = np.maximum(0,avg_filter)
    filtered_signal = fftconvolve(signal, avg_filter, mode='same')
    offset = len(avg_filter)//2
    return filtered_signal[offset::-offset]

def bandpass_filter_5(signal, fs, highCut=16):
    # hamming window filter
    order = floor(fs / highCut)
    signal = np.copy(signal)
    signal -= np.mean(signal)
    # normalization
    K = order; alpha, beta = 0.54, 0.46
    norm_constant = alpha * K + (beta * np.sin(np.pi * K / (1 - K))) / np.sin(np.pi / (1 - K))
    
    avg_filter = np.hamming(order)/norm_constant
    filtered_signal = fftconvolve(signal, avg_filter)
    offset = len(avg_filter)//2
    return filtered_signal[offset::-offset]

def compute_fft(signal, fs):
    N = 2048
    fft_vals = rfft(signal,n=N)
    freqs = rfftfreq(N, d=1/fs)
    return freqs, np.abs(fft_vals)

def compute_sampling_frequency(time_array):
    fs = len(time_array) /(time_array[-1]-time_array[0])
    # fs *= 1000 # ms to s conversion
    return fs
# %%

def timeToIndex(timeList, maxTime, maxIndex):
    return [floor(x*maxIndex/maxTime) for x in timeList]
    
def plot(x,y):
    plt.scatter(x,y,label = "plot")
    plt.xlabel("x")
    plt.ylabel("y")
    return

# def data_parser(csv_file):
#     return
# %%

# csv_file =  r"C:\Users\Admin\Documents\1 - School\4A\Internship\VTEC\work\Code\Vizualisation dashboard\gas_data_visualizer\data\HVAMMONIA03-2025-06-13.csv"
# df = pd.read_csv(csv_file,delimiter = ",")

# t = df['Unix Timestamp'].to_numpy()

# def data_parser(csvname):
#     return array_data


