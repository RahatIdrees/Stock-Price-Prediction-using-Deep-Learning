# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

df = pd.read_csv('Netflix.csv')
df.head(5)

"""**PREPROCESSING**"""

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Select features (multivariate)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features]

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

"""**CREATE SEQUENCES**"""

def create_dataset(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        target_index = features.index('Close')

        y.append(data[i + time_step, target_index])
    return np.array(X), np.array(y)

time_step = 50
X, y = create_dataset(data_scaled, time_step)

""" **TRAIN / VAL / TEST SPLIT**




"""

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

"""**RNN**"""

def build_rnn():
    model = Sequential([
        SimpleRNN(50, activation='tanh', input_shape=(time_step, len(features))),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

"""**LSTM**"""

def build_lstm():
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(time_step, len(features))),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

"""**GRU**"""

def build_gru():
    model = Sequential([
        GRU(50, activation='tanh', input_shape=(time_step, len(features))),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

"""**TRAIN MODELS**"""

rnn_model = build_rnn()

history_rnn = rnn_model.fit(X_train, y_train,
                           validation_data=(X_val, y_val),
                           epochs=20, batch_size=32)

lstm_model = build_lstm()
history_lstm = lstm_model.fit(X_train, y_train,
                             validation_data=(X_val, y_val),
                             epochs=20, batch_size=32)

gru_model = build_gru()
history_gru = gru_model.fit(X_train, y_train,
                           validation_data=(X_val, y_val),
                           epochs=20, batch_size=32)

"""**EVALUATION FUNCTION**"""

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    # Rebuild full array for inverse scaling
    temp = np.zeros((len(pred), len(features)))
    temp[:, 3] = pred[:, 0]   # Close index = 3
    pred_inv = scaler.inverse_transform(temp)[:, 3]

    temp2 = np.zeros((len(y_test), len(features)))
    temp2[:, 3] = y_test
    y_test_inv = scaler.inverse_transform(temp2)[:, 3]

    mse = mean_squared_error(y_test_inv, pred_inv)
    mae = mean_absolute_error(y_test_inv, pred_inv)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_inv - pred_inv) / y_test_inv)) * 100

    return mse, mae, rmse, mape, pred_inv, y_test_inv

"""**RESULTS**"""

rnn_res = evaluate(rnn_model, X_test, y_test)
lstm_res = evaluate(lstm_model, X_test, y_test)
gru_res = evaluate(gru_model, X_test, y_test)

print("RNN:", rnn_res[:4])
print("LSTM:", lstm_res[:4])
print("GRU:", gru_res[:4])

"""**PLOTS ALL MODELS**"""

pred = rnn_res[4]
actual = rnn_res[5]

plt.figure(figsize=(10,5))
plt.plot(actual, label='Actual')
plt.plot(pred, label='Predicted')
plt.title("RNN Prediction vs Actual")
plt.legend()
plt.show()

# ================================
pred = lstm_res[4]
actual = lstm_res[5]

plt.figure(figsize=(10,5))
plt.plot(actual, label='Actual')
plt.plot(pred, label='Predicted')
plt.title("LSTM Prediction vs Actual")
plt.legend()
plt.show()

pred = gru_res[4]
actual = gru_res[5]

plt.figure(figsize=(10,5))
plt.plot(actual, label='Actual')
plt.plot(pred, label='Predicted')
plt.title("GRU Prediction vs Actual")
plt.legend()
plt.show()

plt.plot(history_rnn.history['loss'], label='Train Loss')
plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss (RNN)")
plt.show()

plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss (LSTM)")
plt.show()

plt.plot(history_gru.history['loss'], label='Train Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss (GRU)")
plt.show()