import yfinance as yf
import pandas as pd
import numpy as np


def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Holt historische Aktiendaten für ein gegebenes Symbol.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        return data
    except Exception as e:
        print(f"Fehler beim Herunterladen der Daten: {e}")
        return pd.DataFrame()


def strategy_moving_average_crossover(data: pd.DataFrame) -> tuple:
    """
    Strategie: Moving Average Crossover
    """
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    crossover_data = data[(data['MA50'] > data['MA200'])]

    if crossover_data.empty:
        return None, None

    entry_price = crossover_data['Close'].min()
    exit_price = crossover_data['Close'].max()

    return float(entry_price), float(exit_price)


def strategy_rsi(data: pd.DataFrame) -> tuple:
    """
    Strategie: Relative Strength Index (RSI)
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    oversold = data[rsi < 30]
    overbought = data[rsi > 70]

    if oversold.empty or overbought.empty:
        return None, None

    entry_price = float(oversold['Close'].min().iloc[0])
    exit_price = float(overbought['Close'].max().iloc[0])

    return entry_price, exit_price


def strategy_bollinger_bands(data: pd.DataFrame) -> tuple:
    """
    Strategie: Bollinger Bands
    """
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STD'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + 2 * data['STD']
    data['Lower_Band'] = data['MA20'] - 2 * data['STD']

    lower_band_data = data[data['Close'] < data['Lower_Band']]
    upper_band_data = data[data['Close'] > data['Upper_Band']]

    if lower_band_data.empty or upper_band_data.empty:
        return None, None

    entry_price = float(lower_band_data['Close'].min())
    exit_price = float(upper_band_data['Close'].max())

    return entry_price, exit_price


def strategy_macd(data: pd.DataFrame) -> tuple:
    """
    Strategie: Moving Average Convergence Divergence (MACD)
    """
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    macd_signal_data = data[macd > signal]

    if macd_signal_data.empty:
        return None, None

    entry_price = float(macd_signal_data['Close'].min().iloc[0])
    exit_price = float(macd_signal_data['Close'].max().iloc[0])

    return entry_price, exit_price


def strategy_fibonacci_retracement(data: pd.DataFrame) -> tuple:
    """
    Strategie: Fibonacci Retracement
    """
    high = float(data['High'].max().iloc[0])
    low = float(data['Low'].min().iloc[0])

    diff = high - low
    levels = [
        low + 0.236 * diff,  # 23.6% Retracement
        low + 0.382 * diff,  # 38.2% Retracement
        low + 0.5 * diff,  # 50% Retracement
        low + 0.618 * diff,  # 61.8% Retracement
    ]

    entry_price = min(levels)
    exit_price = max(levels)

    return entry_price, exit_price


def strategy_stochastic_oscillator(data: pd.DataFrame) -> tuple:
    """
    Strategie: Stochastischer Oszillator
    """
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()

    k_percent = 100 * (data['Close'] - low_14) / (high_14 - low_14)
    d_percent = k_percent.rolling(window=3).mean()

    oversold = data[k_percent < 20]
    overbought = data[k_percent > 80]

    if oversold.empty or overbought.empty:
        return None, None

    entry_price = float(oversold['Close'].min().iloc[0])
    exit_price = float(overbought['Close'].max().iloc[0])

    return entry_price, exit_price


def strategy_average_true_range(data: pd.DataFrame) -> tuple:
    """
    Strategie: Average True Range (ATR)
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    low_volatility = data[atr < atr.median()]
    high_volatility = data[atr > atr.median()]

    if low_volatility.empty or high_volatility.empty:
        return None, None

    entry_price = float(low_volatility['Close'].min().iloc[0])
    exit_price = float(high_volatility['Close'].max().iloc[0])

    return entry_price, exit_price


def strategy_volume_breakout(data: pd.DataFrame) -> tuple:
    """
    Strategie: Volume Breakout
    """
    avg_volume = data['Volume'].rolling(window=20).mean()

    high_volume = data[data['Volume'] > 2 * avg_volume]

    if high_volume.empty:
        return None, None

    entry_price = float(high_volume['Close'].min().iloc[0])
    exit_price = float(high_volume['Close'].max().iloc[0])

    return entry_price, exit_price


def strategy_on_balance_volume(data: pd.DataFrame) -> tuple:
    """
    Strategie: On-Balance Volume (OBV)
    """
    close_change = data['Close'].diff()
    volume_signal = np.where(close_change > 0, data['Volume'],
                             np.where(close_change < 0, -data['Volume'], 0))

    obv = pd.Series(volume_signal).cumsum()
    data['OBV'] = obv

    above_avg_obv = data[data['OBV'] > data['OBV'].rolling(window=20).mean()]

    if above_avg_obv.empty:
        return None, None

    entry_price = float(above_avg_obv['Close'].min().iloc[0])
    exit_price = float(above_avg_obv['Close'].max().iloc[0])

    return entry_price, exit_price


def analyze_trading_strategies(symbol: str, start_date: str, end_date: str) -> None:
    """
    Hauptfunktion zur Analyse von Handelsstrategien.
    """
    # Historische Daten herunterladen
    historical_data = fetch_historical_data(symbol, start_date, end_date)

    if historical_data.empty:
        print("Keine Daten verfügbar.")
        return

    # Liste der Strategien
    strategies = [
        ("Moving Average Crossover", strategy_moving_average_crossover),
        ("RSI", strategy_rsi),
        ("Bollinger Bands", strategy_bollinger_bands),
        ("MACD", strategy_macd),
        ("Fibonacci Retracement", strategy_fibonacci_retracement),
        ("Stochastischer Oszillator", strategy_stochastic_oscillator),
        ("Average True Range", strategy_average_true_range),
        ("Volume Breakout", strategy_volume_breakout),
        ("On-Balance Volume", strategy_on_balance_volume)
    ]

    print(f"Strategieanalyse für {symbol} von {start_date} bis {end_date}:\n")

    # Strategien durchlaufen und Ergebnisse ausgeben
    for name, strategy in strategies:
        try:
            entry, exit = strategy(historical_data)
            if entry is not None and exit is not None:
                print(f"{name}:")
                print(f"  Einstiegspreis: {entry:.2f}")
                print(f"  Ausstiegspreis: {exit:.2f}\n")
            else:
                print(f"{name}: Keine passenden Daten gefunden.\n")
        except Exception as e:
            print(f"{name}: Fehler bei der Berechnung - {e}\n")


# Beispielaufruf
if __name__ == "__main__":
    analyze_trading_strategies("AAPL", "2022-01-01", "2023-01-01")