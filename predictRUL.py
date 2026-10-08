import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from scipy import signal, stats
from classify import classify_bearing


# ================= CONFIG VALUES =================

ORIGINAL_FREQ = 25600
TARGET_FREQ = 1000
WINDOW_SIZE = 2000
OVERLAP = 0.5
SEQUENCE_LENGTH = 10

MISALIGN_MODEL_PATH = "misalign/misalign_predictive_maintenance_model.h5"
MISALIGN_SCALER_X_PATH = "misalign/scaler_X.pkl"
MISALIGN_SCALER_Y_PATH = "misalign/scaler_y.pkl"

BPFI_MODEL_PATH = "BPFI/predictive_maintenance_model.h5"
BPFI_SCALER_X_PATH = "BPFI/scaler_X.pkl"
BPFI_SCALER_Y_PATH = "BPFI/scaler_y.pkl"


# ================= INPUT FILE PATHS =================

TEMP_CSV_PATH = "temp_data.csv"
VIBRATION_CSV_PATH = "vibration_data.csv"
MODEL_TYPE = "misalign"   # or "bpfi" select the model jo test karna hai 


# ================= LOAD MODELS =================

def load_models():

    models = {}
    scalers = {}

    models["misalign"] = keras.models.load_model(MISALIGN_MODEL_PATH, compile=False)
    scalers["misalign_X"] = joblib.load(MISALIGN_SCALER_X_PATH)
    scalers["misalign_y"] = joblib.load(MISALIGN_SCALER_Y_PATH)

    models["bpfi"] = keras.models.load_model(BPFI_MODEL_PATH, compile=False)
    scalers["bpfi_X"] = joblib.load(BPFI_SCALER_X_PATH)
    scalers["bpfi_y"] = joblib.load(BPFI_SCALER_Y_PATH)

    return models, scalers


# ================= HELPER FUNCTIONS =================

def resample_signal(signal_data):

    duration = len(signal_data) / ORIGINAL_FREQ
    target_samples = int(duration * TARGET_FREQ)

    return signal.resample(signal_data, target_samples)


def preprocess_dataframe(df):

    if "timestamp" in df.columns:
        df = df.drop("timestamp", axis=1)

    resampled = {}

    for col in df.columns:
        resampled[col] = resample_signal(df[col].values)

    df_resampled = pd.DataFrame(resampled)

    df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
    df_resampled = df_resampled.ffill().bfill()

    return df_resampled


def extract_time_features(window):

    mean = np.mean(window)
    std = np.std(window)
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": np.ptp(window),
        "crest_factor": peak / (rms + 1e-8),
        "kurtosis": stats.kurtosis(window),
        "skewness": stats.skew(window)
    }


def extract_frequency_features(window):

    fft_vals = np.fft.fft(window)
    fft_freq = np.fft.fftfreq(len(window), 1 / TARGET_FREQ)
    fft_power = np.abs(fft_vals) ** 2

    mask = fft_freq > 0

    freq = fft_freq[mask]
    power = fft_power[mask]

    centroid = np.sum(freq * power) / (np.sum(power) + 1e-8)

    misalign_range = (freq >= 100) & (freq <= 200)

    return {
        "spectral_centroid": centroid,
        "spectral_variance": np.sqrt(
            np.sum(((freq - centroid) ** 2) * power) / (np.sum(power) + 1e-8)
        ),
        "misalign_energy": np.sum(power[misalign_range])
    }


def create_features(df_temp, df_vib):

    step = int(WINDOW_SIZE * (1 - OVERLAP))
    windows = (len(df_temp) - WINDOW_SIZE) // step + 1

    features = []

    for i in range(windows):

        start = i * step
        end = start + WINDOW_SIZE

        row = {}

        for col in df_temp.columns:

            window = df_temp[col].values[start:end]

            feats = extract_time_features(window)

            for k, v in feats.items():
                row[f"temp_{col}_{k}"] = v

        for col in df_vib.columns:

            window = df_vib[col].values[start:end]

            feats = extract_time_features(window)
            feats.update(extract_frequency_features(window))

            for k, v in feats.items():
                row[f"vib_{col}_{k}"] = v

        features.append(row)

    return pd.DataFrame(features)


def classify_health(rul):

    if rul > 60:
        return "Healthy"
    elif rul > 40:
        return "Warning"
    elif rul > 20:
        return "Severe"
    else:
        return "Critical"


# ================= MAIN PIPELINE =================

def predict_rul(temp_path, vib_path, model_type):

    models, scalers = load_models()

    model = models[model_type]
    scaler_X = scalers[model_type + "_X"]
    scaler_y = scalers[model_type + "_y"]

    df_temp = pd.read_csv(temp_path)
    df_vib = pd.read_csv(vib_path)

    df_temp = preprocess_dataframe(df_temp)
    df_vib = preprocess_dataframe(df_vib)

    feature_df = create_features(df_temp, df_vib)

    X = feature_df.values

    sequences = []

    for i in range(len(X) - SEQUENCE_LENGTH + 1):
        sequences.append(X[i:i + SEQUENCE_LENGTH])

    sequences = np.array(sequences)

    X_reshaped = sequences.reshape(-1, sequences.shape[-1])
    X_scaled = scaler_X.transform(X_reshaped)

    X_scaled = X_scaled.reshape(sequences.shape)

    y_scaled = model.predict(X_scaled, verbose=0)

    y_pred = scaler_y.inverse_transform(y_scaled).flatten()

    health = [classify_health(v) for v in y_pred]

    classification = classify_bearing(df_temp, df_vib)

    result = {
        "rul_minutes": y_pred.tolist(),
        "health_state": health,
        "mean_rul": float(np.mean(y_pred)),
        "min_rul": float(np.min(y_pred)),
        "max_rul": float(np.max(y_pred)),
        "classification": classification
    }

    return result


# ================= CALL FUNCTION =================

output = predict_rul(
    TEMP_CSV_PATH,
    VIBRATION_CSV_PATH,
    MODEL_TYPE
)

print(output)