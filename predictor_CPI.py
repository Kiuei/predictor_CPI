import yfinance as yf
import pandas as pd
import numpy as np
import fredapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# API Key für FRED (Falls benötigt, setze deine API ein)
FRED_API_KEY = "YOUR_FRED_API_KEY"
fred = fredapi.Fred(api_key=FRED_API_KEY)


# 1. Laden der historischen Aktienkurse
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)[["Adj Close"]]
    df.columns = ["Stock Price"]
    return df


# 2. Laden der CPI-Daten von der FRED API
def get_cpi_data():
    cpi_data = fred.get_series("CPIAUCSL")
    cpi_data = pd.DataFrame(cpi_data, columns=["CPI"])
    return cpi_data


# 3. Daten zusammenführen
def merge_data(stock_df, cpi_df):
    df = stock_df.join(cpi_df, how="left")
    df = df.fillna(method="ffill")
    return df


# 4. Datenvorverarbeitung (Normierung)
def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler


# 5. Erstellen der Trainings- und Test-Daten
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Vorhersage der Aktienpreise
    return np.array(X), np.array(y)


# 6. LSTM Modell erstellen
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# 7. Simulation durchführen
def simulate_scenario(df, model, scaler, cpi_factor=1.05):
    df_sim = df.copy()
    df_sim["CPI"] *= cpi_factor  # Erhöht den CPI um 5%
    df_sim_scaled = scaler.transform(df_sim)

    X_sim = df_sim_scaled[-60:].reshape(1, 60, -1)
    predicted_price = model.predict(X_sim)
    return scaler.inverse_transform([[predicted_price[0][0], df_sim.iloc[-1, 1]]])[0][0]


# Hauptprogramm
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2024-01-01"

stock_data = get_stock_data(ticker, start_date, end_date)
cpi_data = get_cpi_data()
df = merge_data(stock_data, cpi_data)
df_scaled, scaler = preprocess_data(df)

X, y = create_sequences(df_scaled)
model = build_model((X.shape[1], X.shape[2]))

# Modell Training
model.fit(X, y, epochs=10, batch_size=32)

# Simulation
predicted_price = simulate_scenario(df, model, scaler, cpi_factor=1.05)
print(f"Predizierter Preis nach CPI-Änderung: {predicted_price}")
