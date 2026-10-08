"""
worker_rul.py - Standalone RUL prediction worker
Reads CSV paths from files written by app.py, outputs JSON result to stdout.
Usage: python worker_rul.py <temp_csv_path> <vib_csv_path> <model_type>
"""

import sys
import json
import numpy as np
import pandas as pd
from scipy import signal, stats
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def resample_signal(signal_data, original_freq, target_freq):
    num_samples = len(signal_data)
    duration = num_samples / original_freq
    target_samples = int(duration * target_freq)
    return signal.resample(signal_data, target_samples)


def preprocess_dataframe(df, original_freq=25600, target_freq=1000):
    df_processed = df.copy()
    if 'timestamp' in df_processed.columns:
        df_processed = df_processed.drop('timestamp', axis=1)
    resampled_data = {}
    for col in df_processed.columns:
        resampled_data[col] = resample_signal(
            df_processed[col].values, original_freq, target_freq
        )
    df_resampled = pd.DataFrame(resampled_data)
    df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
    df_resampled = df_resampled.ffill().bfill()
    return df_resampled


def extract_time_domain_features(window):
    features = {}
    features['mean']         = np.mean(window)
    features['std']          = np.std(window)
    features['rms']          = np.sqrt(np.mean(window ** 2))
    features['peak']         = np.max(np.abs(window))
    features['peak_to_peak'] = np.ptp(window)
    features['crest_factor'] = features['peak'] / (features['rms'] + 1e-8)
    features['kurtosis']     = stats.kurtosis(window)
    features['skewness']     = stats.skew(window)
    return features


def extract_frequency_domain_features(window, fs=1000):
    features = {}
    fft_vals  = np.fft.fft(window)
    fft_freq  = np.fft.fftfreq(len(window), 1 / fs)
    fft_power = np.abs(fft_vals) ** 2
    pos_mask      = fft_freq > 0
    fft_freq_pos  = fft_freq[pos_mask]
    fft_power_pos = fft_power[pos_mask]
    features['spectral_centroid'] = np.sum(fft_freq_pos * fft_power_pos) / (np.sum(fft_power_pos) + 1e-8)
    features['spectral_variance'] = np.sqrt(
        np.sum(((fft_freq_pos - features['spectral_centroid']) ** 2) * fft_power_pos)
        / (np.sum(fft_power_pos) + 1e-8)
    )
    misalign_range = (fft_freq_pos >= 100) & (fft_freq_pos <= 200)
    features['misalign_energy'] = np.sum(fft_power_pos[misalign_range])
    return features


def create_windowed_features(df_temp, df_vib, window_size=2000, overlap=0.5):
    step_size   = int(window_size * (1 - overlap))
    num_windows = (len(df_temp) - window_size) // step_size + 1
    features_list = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx   = start_idx + window_size
        if end_idx > len(df_temp):
            break
        window_features = {}
        for col in df_temp.columns:
            temp_window = df_temp[col].values[start_idx:end_idx]
            for k, v in extract_time_domain_features(temp_window).items():
                window_features[f'temp_{col}_{k}'] = v
        for col in df_vib.columns:
            vib_window = df_vib[col].values[start_idx:end_idx]
            all_feats  = {
                **extract_time_domain_features(vib_window),
                **extract_frequency_domain_features(vib_window)
            }
            for k, v in all_feats.items():
                window_features[f'vib_{col}_{k}'] = v
        features_list.append(window_features)
    return pd.DataFrame(features_list)


def run(temp_path, vib_path, model_type):
    import tensorflow as tf
    from tensorflow import keras

    df_temp = pd.read_csv(temp_path)
    df_vib  = pd.read_csv(vib_path)

    if model_type == 'misalign':
        model_path    = Path('misalign') / 'misalign_predictive_maintenance_model.h5'
        scaler_x_path = Path('misalign') / 'scaler_X.pkl'
        scaler_y_path = Path('misalign') / 'scaler_y.pkl'
    else:
        model_path    = Path('BPFI') / 'predictive_maintenance_model.h5'
        scaler_x_path = Path('BPFI') / 'scaler_X.pkl'
        scaler_y_path = Path('BPFI') / 'scaler_y.pkl'

    model    = keras.models.load_model(model_path, compile=False)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    df_temp_r  = preprocess_dataframe(df_temp)
    df_vib_r   = preprocess_dataframe(df_vib)
    feature_df = create_windowed_features(df_temp_r, df_vib_r)

    # Reconcile feature count
    if feature_df.shape[1] > scaler_X.n_features_in_:
        feature_df = feature_df.iloc[:, :scaler_X.n_features_in_]
    elif feature_df.shape[1] < scaler_X.n_features_in_:
        padding = pd.DataFrame(
            np.zeros((feature_df.shape[0], scaler_X.n_features_in_ - feature_df.shape[1])),
            columns=[f'pad_{i}' for i in range(scaler_X.n_features_in_ - feature_df.shape[1])]
        )
        feature_df = pd.concat([feature_df, padding], axis=1)

    seq_len     = 10
    X           = feature_df.values
    X_sequences = np.array([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)])

    X_reshaped    = X_sequences.reshape(-1, X_sequences.shape[-1])
    X_scaled      = scaler_X.transform(X_reshaped)
    X_test_scaled = X_scaled.reshape(X_sequences.shape)

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred        = scaler_y.inverse_transform(y_pred_scaled).flatten()

    return y_pred.tolist()


if __name__ == '__main__':
    temp_path  = sys.argv[1]
    vib_path   = sys.argv[2]
    model_type = sys.argv[3]

    try:
        result = run(temp_path, vib_path, model_type)
        # IMPORTANT: print JSON as the very last line to stdout
        print(json.dumps({'success': True, 'predictions': result}))
    except Exception as e:
        import traceback
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)