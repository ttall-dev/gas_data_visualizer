# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:52:15 2025

@author: Admin
"""
# imports 1
import streamlit as st
import plotly.graph_objects as go
from dashboard_assets import *
# imports 2
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal import fftconvolve
from math import floor
import matplotlib.pyplot as plt


# st.set_page_config(layout="wide")
# st.title("🧪 Signal Filter Dashboard (Pd1 / Pd2)")

# === Merge CSVs ===
def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter = ';')
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
            # st.error(f"❌ Failed to read {file.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

# %% Filter testing (temporary section)
# f0 = 510 #Hz
# Fe = 50 #Hz

# test_time_axis = np.linspace(0,3,3*Fe).astype('float64')
# x = np.sin(2 * np.pi*test_time_axis*f0) + np.sin(2 * np.pi*test_time_axis*f0/2)
# x_filtered = bandpass_filter_3(x,fs=Fe)
# plt.figure()
# plt.plot(test_time_axis,x,label='x',linestyle = "dotted")
# plt.plot(test_time_axis,x_filtered,label='x_filtered')
# plt.legend()
# plt.grid()
# plt.hlines()

# %% # === Upload Files ===
uploaded_files = ["./data/2025-04-23T00_08_12.000Z.merged.csv"]

df = merge_uploaded_csvs(uploaded_files)
df['time'] = df['timeStamp'].astype(str)

# %% # === Time parsing ===
"""
    # === Time parsing ===
    # df['time'] = df['timeStamp'].astype(str)
    # if df['time'].str.contains(",").any():
    #     df[['ts_sec', 'ts_micro']] = df['time'].str.split(",", expand=True)
    #     df['ts_sec'] = pd.to_numeric(df['ts_sec'], errors='coerce')
    #     df['ts_micro'] = pd.to_numeric(df['ts_micro'], errors='coerce')
    #     df['time'] = df['ts_sec'] + df['ts_micro'] * 1e-6
    #     df.drop(columns=['ts_sec', 'ts_micro'], inplace=True)
    # else:
    #     df['time'] = pd.to_numeric(df['time'], errors='coerce')

    
    # df = df[['time', 'rawPd1', 'rawPd2']].dropna()
    # labels = df.columns.tolist()
    # array_data = {label: df[label].to_numpy() for label in labels}
    # fs = len(df) / (df['time'].iloc[-1] - df['time'].iloc[0])
"""

# df['time'] = df['timeStamp'].astype(str)
if df['time'].str.contains(",").any():
    # df[['ts_sec', 'ts_micro']] = df['time'].str.split(",", expand=True)
    # df['ts_sec'] = pd.to_numeric(df['ts_sec'], errors='coerce')
    # df['ts_micro'] = pd.to_numeric(df['ts_micro'], errors='coerce')
    # df['time'] = df['ts_sec'] + df['ts_micro'] * 1e-6
    # df.drop(columns=['ts_sec', 'ts_micro'], inplace=True)
    df['time'] = df["time"].str.replace(",",".")
else:
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

df = df[['time', 'rawPd1', 'rawPd2','ntc_1530']].dropna()
labels = df.columns.tolist()
array_data = {label: df[label].str.replace(",",".").to_numpy(dtype=np.float64) for label in labels}

# %% # === Signal Range Selector ===
# st.write("### 🔧 Select Signal Range")
# start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)
start, end = 0, (len(df)-1)
time = array_data['time'][start:end]


fs = compute_sampling_frequency(array_data['time'])

pd1 = array_data['rawPd1'][start:end]
pd2 = array_data['rawPd2'][start:end]
temp = array_data['ntc_1530'][start:end]

# %%


# === Filter choice ===
# st.write("### 🎛️ Choose a Filter")
filter_types = ['bandpass_filter','bandpass_filter_2', 'bandpass_filter_3', 'bandpass_filter_4', 'bandpass_filter_5']

filter_type = filter_types[0]

try:
    if filter_type == 'bandpass_filter':
        filtered_pd1 = bandpass_filter(pd1, fs=fs)
        filtered_pd2 = bandpass_filter(pd2, fs=fs) 
    if filter_type == 'bandpass_filter_2':
        filtered_pd1 = bandpass_filter_2(pd1, highWindow=30)
        filtered_pd2 = bandpass_filter_2(pd2, highWindow=30)
    elif filter_type == 'bandpass_filter_3':
        filtered_pd1 = bandpass_filter_3(pd1, fs=fs)
        filtered_pd2 = bandpass_filter_3(pd2, fs=fs)
    elif filter_type == 'bandpass_filter_4':
        filtered_pd1 = bandpass_filter_4(pd1, fs=fs)
        filtered_pd2 = bandpass_filter_4(pd2, fs=fs)
    elif filter_type == 'bandpass_filter_5':
        filtered_pd1 = bandpass_filter_5(pd1, fs=fs)
        filtered_pd2 = bandpass_filter_5(pd2, fs=fs)
except Exception as e:
    print(f"⚠️ Filter error: {e}")
    filtered_pd1 = pd1
    filtered_pd2 = pd2

#%% #=== Plots ===
# Time Domain Plot
plt.figure(figsize=(12, 6))
plt.plot(temp, pd1, label="Raw Pd1", linestyle='dotted')
plt.plot(temp, filtered_pd1, label="Filtered Pd1")
plt.plot(temp, pd2, label="Raw Pd2", linestyle='dotted')
plt.plot(temp, filtered_pd2, label="Filtered Pd2")
plt.title("📉 Temperature Domain")
plt.xlabel("Temperature")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# # Pd1 vs Pd2 Scatter Plot
# plt.figure(figsize=(8, 8))
# plt.scatter(pd1, pd2, alpha=0.6, label="Raw Pd1 vs Pd2")
# plt.title("🧪 Pd1 vs Pd2")
# plt.xlabel("Raw Pd1")
# plt.ylabel("Raw Pd2")
# plt.legend()
# plt.grid()
# plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(pd1, pd2, c=temp, alpha=0.6, cmap='viridis')
plt.title(" Pd1 vs Pd2")
plt.xlabel("Raw Pd1")
plt.ylabel("Raw Pd2")
plt.colorbar(label='Color Values')
plt.grid()
plt.show()


# FFT Plot
print(filtered_pd1[:10])

f1, fft1 = compute_fft(filtered_pd1, fs)
f2, fft2 = compute_fft(filtered_pd2, fs)
plt.figure(figsize=(12, 6))
plt.plot(f1, fft1, label="FFT Pd1")
plt.plot(f2, fft2, label="FFT Pd2")
plt.title("📊 FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
# %%

dfs = []
for file in uploaded_files:
    try:
        df = pd.read_csv(file, delimiter = ';')
        dfs.append(df)
    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")
        # st.error(f"❌ Failed to read {file.name}: {e}")
res = pd.concat(dfs, ignore_index=True) if dfs else None


# %%

refArray = array_data['ntc_1530']
offsetArray = np.array([0]+ [x for x in refArray[:-1]])
diffArray = refArray - offsetArray
maxIndex = 200
plt.plot(refArray[:maxIndex],'or',label= "reference")
plt.plot(offsetArray[:maxIndex],'ob',label= "offset")
plt.legend()
plt.yscale('log')

plt.figure()

plt.plot(diffArray[1:500])

plt.legend()
plt.yscale('log')


plt.figure()
# plt.plot(refArray[:1000])


