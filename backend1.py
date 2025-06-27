# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 20:12:23 2025

@author: Admin

TODO: 
    check the signal proocessing units in matlab to verify is theydo the right thing
    ask Jan about the pd1 vs pd2 display requirements
    data loading: see how to consistently load the data and gather it in one csv+ make the algorithm worka cross all devices    
    add snr
"""
# %% IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, fftconvolve
from data_synthesis import generate_synthetic_data, save_to_csv
import os
from math import floor
# %% PARAMETERS

data_path = "./data/test_data_cleaned.csv"
reference_labels = sorted(['time', 'temp', 'pd1', 'pd2'])
# fft_sampling_freq = 1.

# %% Temporary: generation of synthetic data to be able to test the code
# if os.path.exists(data_path):
#     pass
# else:
#     save_to_csv(generate_synthetic_data(num_samples=1000))

# %% Formatting train_data.csv to fit our application

# df = pd.read_csv("data/train_data.csv")
# df2 = df.iloc[:, -4:].copy()
# # df2.rename(columns = {'A':'temp','B':'pd1','C':'pd2',"D":'time'},inplace = True)


# # df2.to_csv("data/test_data.csv", index=False)
# df = pd.read_csv("data/test_data.csv", decimal=',')

# df['time'] = pd.to_datetime(df['time'])
# relative_time = (df['time'] - df['time'].iloc[0]).dt.total_seconds().to_numpy()
# df['time'] = relative_time

# df.to_csv("data/test_data_cleaned.csv", index = False)
# for label in list(df.head(0)):
#     if label!="time":
#         df[label] = pd.to_numeric(df[label]) 
#     else: 
#         df[label] = pd.to_timedelta(df[label])
# %% DATA LOADING

# MEMO: probably will have to add some kind of interpolation to account for eventual holes in certain signals
# The data loading part will have to be adapted to run properly on all devices
# data = pd.read_csv(data_path,decimal = ",",delimiter=";")


data = pd.read_csv(data_path,decimal = ",")

# ======================================================================
# Column selection
# ======================================================================
labels = sorted(list(data.columns))
assert labels == reference_labels, "Labels mismatch between data and reference."

# Store data as numpy arrays in a dictionary
array_data = {label: data[label].to_numpy(dtype=np.float64) for label in labels}

# MEMO: there might be problems with the estimation of the frequency, as some axis are apparently not linear (time, temp)
# we estimate the number of datapoints per second to get the sampling frequency

fs = len(array_data['time'])/(array_data['time'][-1]-array_data['time'][0])
# fs = 1000
# %% SIGNAL PROCESSING

def portion_selector(signal_label: str, start: int, end: int):
    """
    Selects a portion of the signal based on the provided label and indices.
    
    Parameters:
    - signal_label (str): Label of the signal to select (e.g., 'pd1', 'pd2').
    - start (int): starting index of the portion to select.
    - end (int): ending index of the portion to select.
    
    Returns:
    - tuple: (list of indices, corresponding data portion from array_data).
    """
    # TODO: add cerrespondance between user input (date/time) and dataframe representation (indices)
    assert signal_label in labels, f"{signal_label} not found in labels."
    portion_indices = list(range(start, end))
    return portion_indices, array_data[signal_label][start:end]

# band pass: ~[0.8 16]Hz based on Ime's documentation (section 1.2)
def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float = fs, order: int = 4):
    """
    Applies a bandpass filter to the given signal.
    
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
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def bandpass_filter_2(signal: np.ndarray, highWindow:int):
    """
    handmade bandpass filter
    
    """
    # for i in range(0,len(signal),lowWindow):
    #     signal[i:i+lowWindow] -= np.mean(signal[i:i+lowWindow])
    signal = np.copy(signal)
    signal -= np.mean(signal) # removes DC component
    for i in range(0,len(signal),highWindow):
        signal[i:i+highWindow] -= np.mean(signal[i:i+highWindow])
    return signal

def bandpass_filter_3(signal: np.ndarray, highCut = 16):
    """
    handmade bandpass filter
    NOTICE: This is not a clean bandpass filter
    Removes the DC component from the signal
    averaging filter:
    averages the points only after each sample
    Cuts frequencies above highCut, though with some non negligeable residuals above highcut
    TODO: write doc to show the frequency response of the filter (sort of cardinal sine)
    highCut: frequency above which we desire cutting the signal
    the necessary amount of coefficients 
    """
    order=floor(fs/highCut)
    print(order)
    # for i in range(0,len(signal),lowWindow):
    #     signal[i:i+lowWindow] -= np.mean(signal[i:i+lowWindow])
    signal = np.copy(signal)
    signal -= np.mean(signal) # removes DC component
    # roughly removes frequencies above highCut
    for i in range(0,len(signal),order):
        signal[i:i+order] -= np.mean(signal[i:i+order])
    return signal

def bandpass_filter_4(signal: np.ndarray, highCut = 16):
    """
    handmade bandpass filter
    NOTICE: This is not a clean bandpass filter
    Removes the DC component from the signal
    averaging filter:
    averages the points around each sample
    Cuts frequencies above highCut, though with some non negligeable residuals above highcut
    TODO: write doc to show the frequency response of the filter (sort of cardinal sine)
    highCut: frequency above which we desire cutting the signal
    the necessary amount of coefficients 
    """
    order=floor(fs/highCut)
    # print(order)
    # for i in range(0,len(signal),lowWindow):
    #     signal[i:i+lowWindow] -= np.mean(signal[i:i+lowWindow])
    
    signal = np.copy(signal)
    # artificially add data before and after signal to avoid edge effects
    delta = np.floor(order / 2).astype(int)
    reversed_part = signal[delta-1::-1]
    signal = np.concatenate((reversed_part, signal))
    reversed_end_part = signal[:-delta-1:-1]
    signal = np.concatenate((signal, reversed_end_part))
    signal -= np.mean(signal) # removes DC component
    # roughly removes frequencies above highCut
    resultSignal = np.array([0 for i in range(len(signal)-2*delta)])
    # resultSignal = np.zeros(1,len(signal)-2*delta)
    resultSignal = np.zeros(len(signal) - 2 * delta)
    for i in range(len(resultSignal)):
        avg = np.mean(signal[i:i+2*delta])
        # print(avg)
        resultSignal[i] = avg
        # print(resultSignal[i])
    return resultSignal

def bandpass_filter_5(signal: np.ndarray, highCut = 16):
    """
    handmade bandpass filter
    NOTICE: This is not a clean bandpass filter
    Removes the DC component from the signal
    averaging filter:
    averages the points around each sample
    Cuts frequencies above highCut, though with some non negligeable residuals above highcut
    TODO: Automated order setting isnt calibrated properly
    highCut: frequency above which we desire cutting the signal
    the necessary amount of coefficients 
    """
    order=floor(fs/highCut)
    signal = np.copy(signal)
    signal-=np.mean(signal)
    avgFilter = np.hamming(order)
    return fftconvolve(signal, avgFilter, mode='same')
 
def compute_fft(signal: np.ndarray, fs: float = fs):
    """
    Computes the Fast Fourier Transform (FFT) of the given signal.
    
    Parameters:
    - signal (np.ndarray): The signal to compute the FFT for
    - fs (float, optional): The sampling frequency (default is fs) (Hz)
    
    Returns:
    - tuple: (frequencies, absolute values of the FFT).
    """
    N = len(signal)
    fft_values = rfft(signal)
    freqs = rfftfreq(N, d=1/fs)
    return freqs, np.abs(fft_values)

# %% PLOTTING
"""
MEMO - Graphs to be plotted:
    time domain signal (Pd1, Pd2) (subplot): 
        before and after BP filter)
    Pd1 Vs Pd2 (one graph)
    FFT of Pd1 and Pd2 (in one graph)
"""

def plotTimeDomain(timePortion, signalPortion, filteredSignalPortion, showOriginal=True, showFiltered=True):
    plt.figure()
    if showFiltered:
        plt.plot(timePortion, filteredSignalPortion, label='Filtered Signal')
    if showOriginal:
        plt.plot(timePortion, signalPortion, label='Original Signal', linestyle='-.')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time Domain Signal")
    plt.legend()
    # plt.yscale('log')
    plt.grid()
    plt.show()

def plotPd1VSPd2(pd1, pd2):
    plt.figure()
    # tmp = [(pd1Portion[i],pd2Portion[i]) for i in range(len(pd1Portion))]
    # tmp = sorted(tmp, key=lambda x: x[0])
    # x = [u[0] for u in tmp]
    # y = [u[1] for u in tmp] 
    # plt.plot(x,y, color='blue')

    # appears completely entangled, check with Jan to see what he plotted in his comparisons

    plt.plot(pd1, pd2, 'o', color='blue',markersize=3)
    plt.xlabel("Pd1")
    plt.ylabel("Pd2")
    plt.title("Pd1 VS Pd2")
    plt.grid()
    plt.show()

def plotFFT(fPd1, fftPd1, fPd2, fftPd2):
    plt.figure()
    plt.plot(fPd1, fftPd1, '-b', label='FFT of Pd1')
    plt.plot(fPd2, fftPd2, '-r', label='FFT of Pd2')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Amplitude")
    plt.legend()
    plt.title("Frequency Domain Signal")
    plt.grid()
    plt.show()

# %% EXAMPLE USAGE

# ~~~ Signal processing
# Select a portion of the signal
start_sample, end_sample = 0,len(data['temp']) # we select the whole
portionIndicesPd1, pd1Portion = portion_selector('pd1', start_sample, end_sample)
portionIndicesPd2, pd2Portion = portion_selector('pd2', start_sample, end_sample)

# time_portion = array_data['time'][start_sample:end_sample]
 
portionIndicesTime , timePortion = portion_selector('time', start_sample, end_sample)

# Apply bandpass filter
low,high = 0.8,16

# filteredPd1 = bandpass_filter(pd1Portion, lowcut=low, highcut=high)
# filteredPd2 = bandpass_filter(pd2Portion, lowcut=low, highcut=high)

filteredPd1 = bandpass_filter_4(pd1Portion)
filteredPd2 = bandpass_filter_4(pd2Portion)

# Compute FFT
fPd1, fftPd1 = compute_fft(filteredPd1)
fPd2, fftPd2 = compute_fft(filteredPd2)

# ~~~ Plotting
# Plot time domain signals
plotTimeDomain(timePortion, pd1Portion, filteredPd1,showOriginal=False)

# Plot Pd1 vs Pd2
plotPd1VSPd2(pd1Portion, pd2Portion)

# Plot FFT of Pd1 and Pd2
plotFFT(fPd1, fftPd1, fPd2, fftPd2)
