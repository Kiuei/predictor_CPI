import yfinance as yf
import pandas as pd


def get_historical_data(symbol, start, end):
    """Lädt historische Daten von Yahoo Finance."""
    df = yf.download(symbol, start=start, end=end)
    return df


def macd_strategy(df, short=12, long=26, signal=9):
    """
    MACD Strategie zur Ermittlung von Einstiegs- und Ausstiegspunkten.

    :param df: Historische Marktdaten (DataFrame)
    :param short: Periode für die schnelle EMA (Exponential Moving Average)
    :param long: Periode für die langsame EMA
    :param signal: Periode für die Signal-Linie
    :return: Einstiegspunkte (Preis), Ausstiegspunkte (Preis)
    """
    # Berechnung der EMAs
    df['EMA_Short'] = df['Close'].ewm(span=short, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long, adjust=False).mean()

    # Berechnung des MACD (Differenz der beiden EMAs)
    df['MACD'] = df['EMA_Short'] - df['EMA_Long']

    # Berechnung der Signal-Linie (Exponentiell gleitender Durchschnitt des MACD)
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # Einstiegspunkte: MACD über Signal-Linie
    entry_points = df[df['MACD'] > df['Signal']].index

    # Ausstiegspunkte: MACD unter Signal-Linie
    exit_points = df[df['MACD'] < df['Signal']].index

    # Extrahiere die Preise an den Einstiegspunkten und Ausstiegspunkten
    entry_prices = df.loc[entry_points, 'Close']
    exit_prices = df.loc[exit_points, 'Close']

    return entry_prices, exit_prices


# Hauptfunktion
def main():
    symbol = "NVDA"  # Beispiel-Symbol (NVIDIA)
    start = "2025-01-01"
    end = "2025-04-01"

    # Daten herunterladen
    df = get_historical_data(symbol, start, end)

    # MACD Strategie anwenden
    entry_prices, exit_prices = macd_strategy(df)

    # Ergebnisse anzeigen
    print("Einstiegspreise:")
    print(entry_prices)

    print("\nAusstiegspreise:")
    print(exit_prices)


if __name__ == "__main__":
    main()
