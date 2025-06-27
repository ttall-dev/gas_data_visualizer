# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:55:25 2025

@author: Admin
"""

import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000, start_time=0, time_step=1):
    """Generate synthetic data for testing."""
    # Generate absolute time
    fs = 100 # number of datapoints per second
    time = np.arange(start_time, start_time + num_samples * time_step, time_step)/fs

    # Generate synthetic temperature data (e.g., sinusoidal variation)
    temperature = 20 + 5 * np.sin(2 * np.pi * time / 100)  # Temperature oscillating around 20°C

    # Generate synthetic power data (e.g., random noise with a trend)
    power1 = 50 + 10 * np.random.randn(num_samples) + 0.05 * time  # Power 1 with noise and trend
    power2 = 30 + 5 * np.random.randn(num_samples) + 0.03 * time   # Power 2 with noise and trend

    # Create a DataFrame
    data = pd.DataFrame({
        'time': time,
        'temp': temperature,
        'pd1': power1,
        'pd2': power2
    })

    return data

def save_to_csv(data, file_name='data/synthetic_data.csv'):
    """Save the DataFrame to a CSV file."""
    data.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(num_samples=1000)

    # Save to CSV
    save_to_csv(synthetic_data)
