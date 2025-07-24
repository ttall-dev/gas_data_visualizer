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
from scipy.stats import linregress

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
    avg_filter = np.maximum(0,avg_filter) # removing eventual negative values
    filtered_signal = fftconvolve(signal, avg_filter, mode='same')
    offset = len(avg_filter)//2
    return filtered_signal[offset:-offset]

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
    return filtered_signal[offset:-offset]

def compute_fft(signal, fs):
    N = 4096
    fft_vals = rfft(signal,n=N)
    freqs = rfftfreq(N, d=1/fs)
    return freqs, fft_vals


# TODO: Ts does not seem to be consistent for temperature, ask out to know how it is determined and used for FFT
def compute_sampling_frequency(time_array,nSamples=3):
    # computes the sampling frequency of an array over nSamples samples
    assert(len(time_array)>= nSamples), "Cannot select more samples than available in the array"
    fs = nSamples /(time_array[nSamples]-time_array[0])
    # fs *= 1000 # ms to s conversion
    return fs

def detect_slope_portions(slopes,threshold=1):
    # detects which portions are relevant for the computation of temperature fs
    # we exclude any portion with a slope too different from the majority
    slp = np.abs(slopes)
    norm_slopes = (slp-np.mean(slp))/np.std(slp)
    slope_mask = np.abs(norm_slopes) < threshold
    return slope_mask

def diff(a):    
    return np.abs(a[:-1]-a[1:])

def extract_temperature_deltas(time, data, N, plotPortions = False):
    # performs N linear regressions after dividing the signal in N portions
    # the coefficients obtain allow for adaptive selection of portions relevant to the calculation of fs
    # returns the difference between successive samples in relvant portions for an estimation f the sampling period
    portion_size = len(data) // N
    slopes = []
    intercepts = []
    r_squared_values = []
    data_portions = []
    data = data
    for i in range(N):
        start_index = i * portion_size
        end_index = (i + 1) * portion_size if i < N - 1 else len(data)
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = linregress(time[start_index:end_index], data[start_index:end_index])
        
        slopes.append(slope)
        intercepts.append(intercept)
        r_squared_values.append(r_value**2)
        
        data_portions.append(data[start_index:end_index].to_numpy())
        if plotPortions:
            # Plot the fitted line
            plt.plot(time[start_index:end_index], slope * time[start_index:end_index] + intercept, color='orange', linewidth=2)
    
    
    # data_portions = data_portions
    slope_mask = detect_slope_portions(slopes, threshold = 1) # we detect which of the portions corrspond to ramp slopes
    deltas =[diff(data_portions[i]) for i in range(N) if slope_mask[i]]
    # we could study portion by portion if need be
    deltas = np.concatenate(deltas)
    
    if plotPortions:
        plt.scatter(time, data, color='blue', label='Data Points', alpha=0.5)
        plt.xlabel("t")
        plt.ylabel("Data")
        plt.title("Linear Regressions on Time Series Data")
        plt.grid()
        # plt.legend()
    return deltas

def compute_temperature_fs(time,temp,N=12):
    # Computes an estimate of the sampling frequency for temperature
    # N needs to be properly defined for efficient computation
    # TODO: add automatic computing for the appropriate N value
    return 1/np.mean(extract_temperature_deltas(time, temp, N))
# %%

def timeToIndex(timeList, maxTime, maxIndex):
    return [floor(x*maxIndex/maxTime) for x in timeList]
    
def plot(x,y):
    plt.scatter(x,y,label = "plot")
    plt.xlabel("x")
    plt.ylabel("y")
    return

