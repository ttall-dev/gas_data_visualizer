import streamlit as st
from dashboard_assets import *
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.fft import rfft, rfftfreq
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧪 Signal Filter Dashboard (Interpolated Pd1 / Pd2)")

# === Merge CSVs ===
def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter=',')
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to read {file.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

# === Segment Detection ===
def extract_segments(mask, min_len=10):
    segments = []
    in_segment = False
    for i, val in enumerate(mask):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            end = i
            if (end - start) >= min_len:
                segments.append((start, end))
            in_segment = False
    if in_segment:
        segments.append((start, len(mask)))
    return segments

# === Upload UI ===
uploaded_files = st.file_uploader("📁 Upload CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded")

    # === Display Metadata Column ===
    if 'metadata' in df.columns:
        st.subheader("📋 Metadata")
        st.dataframe(df[['metadata']].dropna().drop_duplicates().reset_index(drop=True))
    else:
        st.info("ℹ️ No 'metadata' column found in the file.")

    # === Clean columns ===
    if df['timeStamp'].astype(str).str.contains(",").any():
        df['timeStamp'] = df["timeStamp"].astype(str).str.replace(",", ".")
    else:
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')

    df = df[['timeStamp', 'intpl_rawPd1', 'intpl_rawPd2', 'intpl_ntc_1530']].dropna()
    labels = df.columns.tolist()
    array_data = {label: df[label].astype(str).str.replace(",", ".").astype(np.float64) for label in labels}

    # === Range Selector ===
    st.write("### 🔧 Select Signal Range")
    start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)
    time = df['timeStamp'].iloc[start:end]
    pd1 = array_data['intpl_rawPd1'][start:end]
    pd2 = array_data['intpl_rawPd2'][start:end]
    temp = array_data['intpl_ntc_1530'][start:end]
    fs = compute_sampling_frequency(temp)

    # === Filter selection ===
    st.write("### 🎛️ Choose a Filter")
    filter_types = {
        "butterworth": bandpass_filter,
        "moving average": bandpass_filter_3,
        "weighted moving average": bandpass_filter_4,
        "hamming window": bandpass_filter_5
    }
    filter_type = st.selectbox("Select filter", list(filter_types.keys()))

    try:
        filter_func = filter_types[filter_type]
        filtered_pd1 = filter_func(pd1, fs=fs)
        filtered_pd2 = filter_func(pd2, fs=fs)
    except Exception as e:
        st.warning(f"⚠️ Filter error: {e}")
        filtered_pd1 = pd1
        filtered_pd2 = pd2

    # === Time Domain Plot ===
    st.subheader("📉 Time Domain")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time, y=pd1, name="Raw Pd1", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd1, name="Filtered Pd1"))
    fig_time.add_trace(go.Scatter(x=time, y=pd2, name="Raw Pd2", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd2, name="Filtered Pd2"))
    fig_time.update_layout(title='Time Domain Plot', xaxis_title='Time (s)', yaxis_title='Amplitude (a.u.)')
    st.plotly_chart(fig_time, use_container_width=True)
  # === Cycle-wise Plots of Pd1/Pd2 vs Temperature ===
    st.subheader("🔄 Pd1 & Pd2 vs Temperature (by ramp segment)")

    temp_full = df['intpl_ntc_1530'].values
    smoothed_temp = gaussian_filter1d(temp_full, sigma=5)
    slope = np.gradient(smoothed_temp)
    rising_mask = slope > 0.002
    falling_mask = slope < -0.002
    rising_segments = extract_segments(rising_mask)
    falling_segments = extract_segments(falling_mask)

    tab_up, tab_down = st.tabs(["Ramp Up", "Ramp Down"])

    with tab_up:
        st.write("📈 **Ramp Up Cycles**")
        fig_up = go.Figure()
        for i, (start, end) in enumerate(rising_segments):
            fig_up.add_trace(go.Scatter(
                x=temp_full[start:end],
                y=df['intpl_rawPd1'][start:end],
                mode='lines', name=f"Pd1 Up #{i}", line=dict(color='blue')))
            fig_up.add_trace(go.Scatter(
                x=temp_full[start:end],
                y=df['intpl_rawPd2'][start:end],
                mode='lines', name=f"Pd2 Up #{i}", line=dict(color='orange')))
        fig_up.update_layout(title="Ramp Up Cycles", xaxis_title="Temperature (°C)", yaxis_title="Signal")
        st.plotly_chart(fig_up, use_container_width=True)

    with tab_down:
        st.write("📉 **Ramp Down Cycles**")
        fig_down = go.Figure()
        for i, (start, end) in enumerate(falling_segments):
            fig_down.add_trace(go.Scatter(
                x=temp_full[start:end],
                y=df['intpl_rawPd1'][start:end],
                mode='lines', name=f"Pd1 Down #{i}", line=dict(color='green')))
            fig_down.add_trace(go.Scatter(
                x=temp_full[start:end],
                y=df['intpl_rawPd2'][start:end],
                mode='lines', name=f"Pd2 Down #{i}", line=dict(color='red')))
        fig_down.update_layout(title="Ramp Down Cycles", xaxis_title="Temperature (°C)", yaxis_title="Signal")
        st.plotly_chart(fig_down, use_container_width=True)

    # === Pd1 vs Pd2 Scatter ===
    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=pd1, y=pd2, mode='markers', name="Raw Pd1 vs Pd2",
        marker=dict(color=temp, colorscale='Viridis'), opacity=0.6
    ))
    fig_scatter.update_layout(title='Pd1 vs Pd2', xaxis_title='Pd1 (a.u.)', yaxis_title='Pd2 (a.u.)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # === FFT Plot ===
    st.subheader("📊 FFT")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, name="FFT Pd1"))
    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, name="FFT Pd2"))
    fig_fft.update_layout(title='FFT of Filtered Signals', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    st.plotly_chart(fig_fft, use_container_width=True)

    
