import streamlit as st
from dashboard_assets import *
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧪 Signal Filter Dashboard (Pd1 / Pd2)")

# === Merge CSVs ===
def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter=';')
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to read {file.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

# === Upload UI ===
uploaded_files = st.file_uploader("📁 Upload CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded")

    # === Time Parsing ===
    if df['time'].str.contains(",").any():
        df['time'] = df["time"].str.replace(",", ".")
    else:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')

    df = df[['time', 'rawPd1', 'rawPd2', 'ntc_1530']].dropna()
    labels = df.columns.tolist()
    array_data = {label: df[label].astype(str).str.replace(",", ".").astype(np.float64) for label in labels}

    st.write("### 🔧 Select Signal Range")
    start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)

    time = array_data['time'][start:end]
    pd1 = array_data['rawPd1'][start:end]
    pd2 = array_data['rawPd2'][start:end]
    temp = array_data['ntc_1530'][start:end]
    fs = compute_sampling_frequency(list(array_data['time']))


    # === Filter selection ===
    st.write("### 🎛️ Choose a Filter")
    filter_types0 = ['bandpass_filter', 'bandpass_filter_2', 'bandpass_filter_3', 'bandpass_filter_4', 'bandpass_filter_5']
    
    filter_types  = {"butterworth":             bandpass_filter,
                     "moving average":          bandpass_filter_3,
                     "weighted moving average": bandpass_filter_4,
                     "hamming window":          bandpass_filter_5}
    filter_type = st.selectbox("Select filter", list(filter_types.keys()))
        
    try:
        if filter_type == "butterworth":
            filtered_pd1 = bandpass_filter(pd1, fs=fs)
            filtered_pd2 = bandpass_filter(pd2, fs=fs)
        # elif filter_type == filter_types[1]:
        #     filtered_pd1 = bandpass_filter_2(pd1, fs=fs)
        #     filtered_pd2 = bandpass_filter_2(pd2, fs=fs)
        elif filter_type == "moving average":
            filtered_pd1 = bandpass_filter_3(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_3(pd2, fs=fs)
        elif filter_type == "weighted moving average":
            filtered_pd1 = bandpass_filter_4(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_4(pd2, fs=fs)
        elif filter_type == "hamming window":
            filtered_pd1 = bandpass_filter_5(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_5(pd2, fs=fs)
    except Exception as e:
        st.warning(f"⚠️ Filter error: {e}")
        filtered_pd1 = pd1
        filtered_pd2 = pd2
    
    # === Time domain ===
    st.subheader("📉 Time Domain")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time, y=pd1, name="Raw Pd1", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd1, name="Filtered Pd1"))
    fig_time.add_trace(go.Scatter(x=time, y=pd2, name="Raw Pd2", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd2, name="Filtered Pd2"))
    st.plotly_chart(fig_time, use_container_width=True)

    # === Pd1 vs Pd2 Scatter ===
    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=pd1,
        y=pd2,
        mode='markers',
        name="Raw Pd1 vs Pd2",
        marker=dict(color=temp, colorscale='Viridis'),
        opacity=0.6
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # === Pd1 Range Zoom Slider ===
    st.write("### 🔍 Zoom Range on Pd1 for FFT")
    pd1_min, pd1_max = float(np.min(pd1)), float(np.max(pd1))
    zoom_start, zoom_end = st.slider("Select Pd1 range to recompute FFT", pd1_min, pd1_max, (pd1_min, pd1_max), step=0.0001)

    zoom_mask = (pd1 >= zoom_start) & (pd1 <= zoom_end)

    zoomed_pd1 = filtered_pd1[zoom_mask]
    zoomed_pd2 = filtered_pd2[zoom_mask]
    zoomed_fs = fs  # Sampling frequency remains same (you can recompute on time if needed)


    # === FFT Plot (Zoomed) ===
    st.subheader("📊 FFT (on Zoomed Pd1 Range)")
    f1, fft1 = compute_fft(zoomed_pd1, zoomed_fs)
    f2, fft2 = compute_fft(zoomed_pd2, zoomed_fs)

    fig_fft = go.Figure()
    
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, name="FFT Pd1", mode='lines+markers',
                              marker=dict(color=temp, size=5)))

    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, name="FFT Pd2", mode='lines+markers',
                              marker=dict(color=temp, size=5)))
    
    fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)

