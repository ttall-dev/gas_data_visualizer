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
# imports 3

import torch

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
# uploaded_files = ["./data/2025-04-23T00_08_12.000Z.merged.csv"]
uploaded_files = ["./data/merged-data-HVAMMONIA03-2025-07-10.csv"]

df = merge_uploaded_csvs(uploaded_files)
df['time'] = df["timeStamp"].astype(str)

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



def diff(refArray):
    offsetArray = np.array([0] + [x for x in refArray[:-1]])
    diffArray = refArray - offsetArray
    return diffArray

# Assuming array_data['ntc_1530'] is defined
refArray = array_data['ntc_1530'][::15 ]

diffArray = diff(diff(refArray))
epsilon = 1e-1000

# Create a mask for values where diffArray is less than epsilon
mask = diffArray < epsilon
labeled_temp = np.where(mask, refArray, np.nan)

# maxIndex = len(refArray) // 10 * 2
maxIndex = 750
offset = 500

plt.figure(1)

# Plot the reference array
plt.plot(refArray[maxIndex-offset:maxIndex], '-r', label="reference")
plt.plot(labeled_temp[maxIndex-offset:maxIndex],'-ob',label="interval with criteria")

# Plot the masked values in red
# plt.plot(np.where(mask[:maxIndex], refArray[:maxIndex], np.nan), 'or', label="masked values (diff < epsilon)")

plt.legend()
# plt.yscale('log')
plt.title("Reference Array with Masked Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

plt.figure(2)

second_derivative_with_criteria = np.where(mask, diffArray, np.nan)

plt.plot(diffArray[1:maxIndex], label="diff array")
plt.plot(second_derivative_with_criteria[maxIndex-offset:maxIndex], 'or', label="masked diff array")
plt.legend()
plt.title("Temperature Deltas")
plt.yscale('log')
plt.xlabel("Index")
plt.ylabel("Difference")
plt.show()

plt.figure(3)
plt.plot(mask[:10],'ob')
# %%
# Calculate time differences
t = time - time[0]  # Normalize time
dt = np.diff(t)

# Calculate temperature differences
dT = np.diff(temp)
ddT = np.diff(dT)

# Calculate first and second derivatives
dv = dT / dt
ddv = ddT / dt[1:]  # Adjust for the size difference

# Plot first and second derivatives
plt.figure(figsize=(10, 5))
plt.plot(t[1:], dv, label='1st Derivative (dv/dt)', marker='o')
plt.plot(t[2:], ddv, label='2nd Derivative (d²v/dt²)', marker='x')
plt.title('First and Second Derivatives')
plt.xlabel('Time (s)')
plt.ylabel('Derivative Values')
plt.legend()
plt.grid()
plt.tight_layout()


# Plot temperature with masking
plt.figure(figsize=(10, 5))
eps0 = 1e-5 
mask = abs(ddv) < eps0
temp1 = np.where(mask, temp[2:], np.nan)  # Masking values based on condition
plt.plot(temp1, label='Masked Temperature', marker='o')
plt.title('Masked Temperature Values')
plt.xlabel('Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.tight_layout()

# Show plots
plt.show()

# %%


# plt.figure()
# # plt.plot(refArray[:1000])

# y = torch.from_numpy(refArray)
# y.requires_grad_()
# z= 2*y
# y.backward()
# gradient = y.grad.item()
# plt.plot(gradient)

# plt.title("gradient")
# %%

import numpy as np
import torch
import matplotlib.pyplot as plt

# Step 1: Create a NumPy array (example data)
refArray = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# Step 2: Convert the NumPy array to a PyTorch tensor
y = torch.from_numpy(refArray)

# Step 3: Set requires_grad to True
y.requires_grad_()  # Correct way to set requires_grad

# Step 4: Perform some operations (example: sum)
output = y.sum()  # You need to perform some operation to compute gradients

# Step 5: Backpropagate to compute gradients
output.backward()

# Step 6: Get the gradient
gradient = y.grad

# Display the gradient
print("Gradient:")
print(gradient)

# Plot the gradient (if you want to visualize it)
plt.plot(gradient.numpy())
plt.title("Gradient")
plt.xlabel("Index")
plt.ylabel("Gradient Value")
plt.grid()
plt.show()

# %%

