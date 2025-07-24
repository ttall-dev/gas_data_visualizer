import streamlit as st
from dashboard_assets import *
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import json
import ast
st.set_page_config(layout="wide")
st.title("🧪 Signal Filter Dashboard (Interpolated Pd1 / Pd2)")
# %%


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


# === Upload UI ===
uploaded_files = st.file_uploader("📁 Upload CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded")
    # === Display Metadata Column ===
    if 'metadata' in df.columns:
        st.subheader("📋 Metadata")
        # st.dataframe(df[['metadata']].dropna().drop_duplicates().reset_index(drop=True))
        metadata = json.loads(df['metadata'][0]) # metadata in dictionary format
        metadata["date"] = pd.to_datetime(df['date'][0])
        # metadata.drop("sw", axis = 1)
        st.dataframe(metadata)
        
    else:
        st.info("ℹ️ No 'metadata' column found in the file.")
   

    # === Display Device Config ===
    if 'device_config' in df.columns:
        st.subheader("🧾 Device Configuration")
        try:
        # Parse the first row (safely handles Python-style dict)
            device_config = ast.literal_eval(df['device_config'].iloc[0])
            device_config["date"] = pd.to_datetime(df['date'].iloc[0])

        # Display top-level keys if flat
            st.markdown("### 🧩 Top-Level Config")
            st.dataframe(pd.DataFrame.from_dict({k: v for k, v in device_config.items() if not isinstance(v, dict)}, orient='index', columns=["Value"]))

        # Display nested 'config_data' if present
            if "config_data" in device_config:
                st.markdown("### ⚙️ config_data")
                st.dataframe(pd.DataFrame.from_dict(device_config["config_data"], orient='index', columns=["Value"]))

        

        except Exception as e:
            st.warning(f"⚠️ Could not parse 'device_config': {e}")

    # === Time Parsing ===
    if df['timeStamp'].astype(str).str.contains(",").any():
        df['timeStamp'] = df["timeStamp"].astype(str).str.replace(",", ".")
    else:
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')

    df = df[['timeStamp', 'intpl_rawPd1', 'intpl_rawPd2', 'intpl_ntc_1530']].dropna()


    
    labels = df.columns.tolist()
    array_data = {label: df[label].astype(str).str.replace(",", ".").astype(np.float64) for label in labels}

    st.write("### 🔧 Select Signal Range")
    
    
    # ~~~ Time Slider: time to index conversion
    test = df['timeStamp'].iloc[:]
    timeSelector = np.array(df['timeStamp'].iloc[:])
    timeSelector -= timeSelector[0]
    time_float_start, time_float_end = st.slider("Select time range (s):", min_value=timeSelector[0], max_value=timeSelector[-1], value=(timeSelector[0], timeSelector[-1]), step=0.1)
    
    # start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)
    start, end = timeToIndex([time_float_start,time_float_end], timeSelector[-1], len(df))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    relative_timetime = df['timeStamp'].iloc[start:end]
    pd1 = array_data['intpl_rawPd1'][start:end]
    pd2 = array_data['intpl_rawPd2'][start:end]
    temp = array_data['intpl_ntc_1530'][start:end]
    # === Define thresholds ===
    rise_thresh = 0.002
    fall_thresh = -0.002
    min_segment_length = 10

# === Helper function for segment detection ===
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

# === Detect ramp segments ===
    smoothed_temp = gaussian_filter1d(temp, sigma=5)
    slope = np.gradient(smoothed_temp)

    rising_mask = slope > rise_thresh
    falling_mask = slope < fall_thresh

    rising_segments = extract_segments(rising_mask, min_segment_length)
    falling_segments = extract_segments(falling_mask, min_segment_length)
    # fs = compute_sampling_frequency(list(array_data['intpl_ntc_1530'])) # vs temperature
    fs = compute_temperature_fs(array_data['timeStamp'], array_data['intpl_ntc_1530'])
    # st.text_area(str(fs))
    # === Filter selection ===
    st.write("### 🎛️ Choose a Filter")
    filter_types = {
        "butterworth": bandpass_filter,
        "moving average": bandpass_filter_3,
        "weighted moving average": bandpass_filter_4,
        "hamming window": bandpass_filter_5
    }
    filter_type = st.selectbox("Select filter", list(filter_types.keys()))
    st.write('<a href="https://github.com/ttall-dev/gas_data_visualizer/blob/main/documentation/Filter_Documentation.pdf" target="_blank">More documentation about the filtering methods</a>', unsafe_allow_html=True)

    # st.button("More documentation about the filtering methods", on_click=lambda: open("https://www.google.com", "_blank"))
    try:
        filter_func = filter_types[filter_type]
        if filter_type == "butterworth":
            minCut, maxCut = 0.,50.
            lowCut,highCut = st.slider("Bandpass frequency range (Hz)", min_value=0.1, max_value=maxCut, value=(minCut,maxCut), step=0.1)
            filtered_pd1 = filter_func(pd1, fs=fs, lowcut=lowCut, highcut=highCut,order=4)
            filtered_pd2 = filter_func(pd2, fs=fs, lowcut=lowCut, highcut=highCut,order=4)
        else:
            filtered_pd1 = filter_func(pd1, fs=fs)
            filtered_pd2 = filter_func(pd2, fs=fs)
    except Exception as e:
        st.warning(f"⚠️ Filter error: {e}")
        filtered_pd1 = pd1
        filtered_pd2 = pd2

    # === Time Domain Plot ===
    st.subheader("📉 Time Domain")
    fig_time = go.Figure()
    fig_time.update_layout(
        title='Time Domain Plot',
        xaxis_title='t (s)',     # or your real x-axis meaning
        yaxis_title='Amplitude (a.u.)'  # update to real units if you have them
    )
    relative_time = relative_timetime - array_data["timeStamp"][0]
    fig_time.add_trace(go.Scatter(x=relative_time, y=pd1, name="Raw Pd1", line=dict(dash='dot'),visible='legendonly'))
    fig_time.add_trace(go.Scatter(x=relative_time, y=filtered_pd1, name="Filtered Pd1"))
    fig_time.add_trace(go.Scatter(x=relative_time, y=pd2, name="Raw Pd2", line=dict(dash='dot'),visible='legendonly'))
    fig_time.add_trace(go.Scatter(x=relative_time, y=filtered_pd2, name="Filtered Pd2",visible='legendonly'))
    fig_time.update_layout(title='Time domain Plot', xaxis_title='t (s)', yaxis_title='Amplitude')
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.subheader("📈 Pd1 / Pd2 vs Temperature by Ramp Type")
    
# Plot ramp-up segments
    fig_ramp_up = go.Figure()
    for i, (start, end) in enumerate(rising_segments):
        if i == start:    
            visibility = True
        else:
            visibility = 'legendonly'
            
        fig_ramp_up.add_trace(go.Scatter(x=temp[start:end], y=pd1[start:end], mode='lines', name=f'Pd1 Ramp↑ {i+1}', visible = visibility))
        fig_ramp_up.add_trace(go.Scatter(x=temp[start:end], y=pd2[start:end], mode='lines', name=f'Pd2 Ramp↑ {i+1}', visible = visibility))
        

    fig_ramp_up.update_layout(
        title='Ramp Up: Pd1 & Pd2 vs Temperature',
        xaxis_title='Temperature (°C)',
        yaxis_title='Amplitude (a.u.)'
    )
    

# Plot ramp-down segments
    fig_ramp_down = go.Figure()
    for i, (start, end) in enumerate(falling_segments):
        if i == start:    
            visibility = True
        else:
            visibility = 'legendonly'
            
        fig_ramp_down.add_trace(go.Scatter(x=temp[start:end], y=pd1[start:end], mode='lines', name=f'Pd1 Ramp↓ {i+1}', visible = visibility))
        fig_ramp_down.add_trace(go.Scatter(x=temp[start:end], y=pd2[start:end], mode='lines', name=f'Pd2 Ramp↓ {i+1}', visible = visibility))

    fig_ramp_down.update_layout(
        title='Ramp Down: Pd1 & Pd2 vs Temperature',
        xaxis_title='Temperature (°C)',
        yaxis_title='Amplitude (a.u.)'
    )
    # Side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔼 Ramp Up")
        st.plotly_chart(fig_ramp_up, use_container_width=True)

    with col2:
        st.markdown("#### 🔽 Ramp Down")
        st.plotly_chart(fig_ramp_down, use_container_width=True)


    # === Pd1 vs Pd2 Scatter ===
    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.update_layout(
        title='Pd1 vs Pd2',
        xaxis_title='Pd1 (a.u.)',
        yaxis_title='Pd2 (a.u.)',
        coloraxis_colorbar_title="Temperature"
    )
    
    fig_scatter.add_trace(go.Scatter(
        x=pd1, y=pd2, mode='markers', name="Raw Pd1 vs Pd2",
        marker=dict(color=temp, colorscale='Viridis', colorbar=dict(title="Temperature"))
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)
    

    # === FFT Plot ===
    st.subheader("📊 FFT")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.update_layout(
        title='FFT of Filtered Signals',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude'
    )

    fig_fft.add_trace(go.Scatter(x=f1, y=np.abs(fft1), name="FFT Pd1"))
    fig_fft.add_trace(go.Scatter(x=f2, y=np.abs(fft2), name="FFT Pd2"))
    fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)
    # === Temperature plot ===
    plotTemp = st.radio("Plot temperature", ["yes", "no"])
    if plotTemp == "yes":
        st.subheader("Temperature")
        fig_fft = go.Figure()
        fig_fft.update_layout(
            title='Temp = f(time)',
            xaxis_title='time (s)',
            yaxis_title='temperature (°C)'
        )
        timeAxis = relative_timetime-df['timeStamp'].iloc[0]
        fig_fft.add_trace(go.Scatter(x=timeAxis, y=temp, name="tempearature"))
        fig_fft.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_fft, use_container_width=True)
    
    
