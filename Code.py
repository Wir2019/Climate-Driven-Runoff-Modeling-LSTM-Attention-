import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Softmax, Multiply, Lambda

tf.random.set_seed(42)
np.random.seed(42)

INPUT_FILE = "runoff_dataset.xlsx"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL_IDX = 0
FEATURE_COL_IDXS = list(range(1, 9))   # B..I
TARGET_COL_IDX = 9                     # J

TRAIN_START, TRAIN_END = 1980, 1996
VAL_START, VAL_END = 1997, 2000
SIM_START, SIM_END = 2001, 2020

N_IN = 6
N_OUT = 3
STRIDE = 1

EXPECTED_COLS = [
    "data",
    "snowmelt_sum_mm",
    "pet",
    "total_precipitation_sum_mm",
    "surface_sensible_heat_flux_sum_MJ",
    "snowfall_sum_mm",
    "skin_temperature_celsius",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "runoff",
]

def _norm_col(x):
    return str(x).strip()

def create_supervised_sequences(feature_arr, target_arr, n_in, n_out, stride=1):
    feature_arr = np.asarray(feature_arr, dtype=np.float32)
    target_arr = np.asarray(target_arr, dtype=np.float32)

    n = len(feature_arr)
    max_i = n - n_in - n_out + 1
    if max_i <= 0:
        return (
            np.empty((0, n_in, feature_arr.shape[1]), dtype=np.float32),
            np.empty((0, n_out), dtype=np.float32),
        )

    X_list, Y_list = [], []
    for i in range(0, max_i, stride):
        X = feature_arr[i : i + n_in, :]
        Y = target_arr[i + n_in : i + n_in + n_out]
        X_list.append(X)
        Y_list.append(Y)

    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)

def build_lstm_attention_model(n_in, n_features, n_out):
    inputs = Input(shape=(n_in, n_features))
    x = LSTM(64, return_sequences=True)(inputs)

    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    weighted = Multiply()([x, weights])
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)

    context = Dropout(0.3)(context)
    context = Dense(64, activation="relu")(context)
    outputs = Dense(n_out)(context)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

dataset = pd.read_excel(INPUT_FILE)

actual_cols = [_norm_col(c) for c in list(dataset.columns[: len(EXPECTED_COLS)])]
expected_cols = [_norm_col(c) for c in EXPECTED_COLS]
if actual_cols != expected_cols:
    raise ValueError(
        "Input file columns (A..J) do not match the expected names/order.\n"
        f"Expected: {expected_cols}\n"
        f"Actual:   {actual_cols}\n"
        "Please ensure the Excel file has exactly these 10 columns in this order."
    )

date_col_name = dataset.columns[DATE_COL_IDX]
dataset[date_col_name] = pd.to_datetime(dataset[date_col_name])
dataset = dataset.sort_values(by=date_col_name).reset_index(drop=True)
dataset["Year"] = dataset[date_col_name].dt.year
dataset["Month"] = dataset[date_col_name].dt.month

train_df = dataset[(dataset["Year"] >= TRAIN_START) & (dataset["Year"] <= TRAIN_END)].copy()
val_df   = dataset[(dataset["Year"] >= VAL_START)   & (dataset["Year"] <= VAL_END)].copy()
sim_df   = dataset[(dataset["Year"] >= SIM_START)   & (dataset["Year"] <= SIM_END)].copy()

train_df = train_df.dropna(subset=[dataset.columns[i] for i in FEATURE_COL_IDXS + [TARGET_COL_IDX]]).copy()
val_df   = val_df.dropna(subset=[dataset.columns[i] for i in FEATURE_COL_IDXS + [TARGET_COL_IDX]]).copy()
sim_df   = sim_df.dropna(subset=[dataset.columns[i] for i in FEATURE_COL_IDXS]).copy()

X_train_raw = train_df.iloc[:, FEATURE_COL_IDXS].values
y_train_raw = train_df.iloc[:, TARGET_COL_IDX].values
X_val_raw   = val_df.iloc[:, FEATURE_COL_IDXS].values
y_val_raw   = val_df.iloc[:, TARGET_COL_IDX].values

X_train, Y_train = create_supervised_sequences(X_train_raw, y_train_raw, N_IN, N_OUT, STRIDE)
X_val,   Y_val   = create_supervised_sequences(X_val_raw,   y_val_raw,   N_IN, N_OUT, STRIDE)

if X_train.shape[0] == 0:
    raise ValueError("Not enough training data to build supervised samples. Check TRAIN period or N_IN/N_OUT.")
if X_val.shape[0] == 0:
    raise ValueError("Not enough validation data to build supervised samples. Check VAL period or N_IN/N_OUT.")

n_features = X_train.shape[2]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_2d = X_train.reshape((X_train.shape[0], N_IN * n_features))
X_val_2d   = X_val.reshape((X_val.shape[0],   N_IN * n_features))

X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape((X_train.shape[0], N_IN, n_features))
X_val_scaled   = x_scaler.transform(X_val_2d).reshape((X_val.shape[0], N_IN, n_features))

Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_val_scaled   = y_scaler.transform(Y_val)

model = build_lstm_attention_model(N_IN, n_features, N_OUT)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
]

history = model.fit(
    X_train_scaled, Y_train_scaled,
    validation_data=(X_val_scaled, Y_val_scaled),
    epochs=150,
    batch_size=16,
    shuffle=True,
    verbose=1,
    callbacks=callbacks
)

model_path = os.path.join(OUTPUT_DIR, "lstm_attention_runoff.keras")
model.save(model_path)

with open(os.path.join(OUTPUT_DIR, "x_scaler.pkl"), "wb") as f:
    pickle.dump(x_scaler, f)
with open(os.path.join(OUTPUT_DIR, "y_scaler.pkl"), "wb") as f:
    pickle.dump(y_scaler, f)

sim_start_date = pd.Timestamp(f"{SIM_START}-01-01")
base_hist = dataset[dataset[date_col_name] < sim_start_date].dropna(
    subset=[dataset.columns[i] for i in FEATURE_COL_IDXS]
).tail(N_IN)

if len(base_hist) < N_IN:
    raise ValueError("Not enough history before simulation start to build the initial input window.")

current_input = base_hist.iloc[:, FEATURE_COL_IDXS].values.astype(np.float32)
future_features = sim_df.iloc[:, FEATURE_COL_IDXS].values.astype(np.float32)

future_preds = []
for i in range(0, len(future_features), N_OUT):
    if len(current_input) < N_IN:
        pad = np.zeros((N_IN - len(current_input), n_features), dtype=np.float32)
        window = np.vstack([pad, current_input])
    else:
        window = current_input[-N_IN:, :]

    x_in = window.reshape(1, N_IN * n_features)
    x_in_scaled = x_scaler.transform(x_in).reshape(1, N_IN, n_features)

    y_pred_scaled = model.predict(x_in_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)[0]

    for v in y_pred:
        if len(future_preds) < len(future_features):
            future_preds.append(float(v))

    block_feats = future_features[i : i + N_OUT]
    if len(block_feats) == 0:
        break
    current_input = np.vstack([current_input[len(block_feats):], block_feats])

sim_out = sim_df[[date_col_name, "Year", "Month"]].copy()
sim_out["runoff_pred"] = future_preds[: len(sim_out)]

sim_out_path = os.path.join(OUTPUT_DIR, "runoff_simulation_2001_2020.xlsx")
sim_out.to_excel(sim_out_path, index=False)

print("Done.")
print(f"Model saved to: {model_path}")
print(f"Simulation output saved to: {sim_out_path}")
