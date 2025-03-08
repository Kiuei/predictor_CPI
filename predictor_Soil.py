## Writen by Claude Sonet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf


class AgriStockSimulator:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_stock_data(self, ticker, start_date, end_date):
        """Load historical stock data using yfinance"""
        try:
            self.stock_data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Successfully loaded data for {ticker}")
            return self.stock_data
        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None

    def prepare_agricultural_data(self, soil_quality, rainfall, temperature, harvest_yield=None):
        """Create a DataFrame with agricultural factors"""
        # Scale values between 0 and 1
        soil_scaled = soil_quality / 10  # Assuming soil quality is rated 0-10
        rainfall_scaled = rainfall / 200  # Normalizing based on average rainfall
        temp_scaled = (temperature - 10) / 30  # Normalizing temperature range 10-40C

        if harvest_yield is None:
            # Simple model to estimate harvest yield based on agricultural factors
            harvest_yield = (soil_scaled * 0.5 + rainfall_scaled * 0.3 + temp_scaled * 0.2) * 100

        # Create DataFrame with these factors
        dates = self.stock_data.index
        agri_data = pd.DataFrame({
            'soil_quality': [soil_quality] * len(dates),
            'rainfall': [rainfall] * len(dates),
            'temperature': [temperature] * len(dates),
            'harvest_yield': [harvest_yield] * len(dates)
        }, index=dates)

        return agri_data

    def merge_data(self, agri_data):
        """Merge stock and agricultural data"""
        if self.stock_data is None:
            print("Stock data not loaded. Use load_stock_data first.")
            return None

        # Combine the datasets
        combined_data = pd.concat([self.stock_data, agri_data], axis=1)
        combined_data = combined_data.dropna()
        self.data = combined_data

        return self.data

    def build_model(self):
        """Build prediction model using ARIMA with exogenous variables"""
        if self.data is None:
            print("Data not prepared. Prepare data first.")
            return None

        # Scale data
        features = self.data[['soil_quality', 'rainfall', 'temperature', 'harvest_yield']].values
        price = self.data['Close'].values.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(features)
        y_scaled = self.scaler.fit_transform(price)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

        # Build ARIMAX model (ARIMA with exogenous variables)
        try:
            model = sm.tsa.statespace.SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = model.fit(disp=False)
            print("Model successfully built")
            return self.model
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def predict_future(self, future_soil, future_rainfall, future_temp, future_harvest=None, periods=30):
        """Predict future stock prices based on agricultural inputs"""
        if self.model is None:
            print("Model not built. Build model first.")
            return None

        # Create future exogenous variables
        if future_harvest is None:
            soil_scaled = future_soil / 10
            rainfall_scaled = future_rainfall / 200
            temp_scaled = (future_temp - 10) / 30
            future_harvest = (soil_scaled * 0.5 + rainfall_scaled * 0.3 + temp_scaled * 0.2) * 100

        future_features = np.array([[future_soil, future_rainfall, future_temp, future_harvest]] * periods)
        future_scaled = self.scaler.transform(future_features)

        # Make prediction
        forecast = self.model.forecast(steps=periods, exog=future_scaled)

        # Transform back to original scale
        forecast_transformed = self.scaler.inverse_transform(forecast.reshape(-1, 1))

        # Create date range for forecast
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': forecast_transformed.flatten()
        })
        forecast_df.set_index('Date', inplace=True)

        return forecast_df

    def plot_results(self, forecast_df):
        """Plot historical and predicted stock prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Historical Prices')
        plt.plot(forecast_df.index, forecast_df['Predicted_Price'], label='Predicted Prices', color='red')
        plt.title('Stock Price Prediction with Agricultural Factors')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage
def run_simulation(ticker, start_date, end_date, soil_quality, rainfall, temperature, harvest_yield=None):
    simulator = AgriStockSimulator()

    # Load stock data
    simulator.load_stock_data(ticker, start_date, end_date)

    # Prepare agricultural data
    agri_data = simulator.prepare_agricultural_data(soil_quality, rainfall, temperature, harvest_yield)

    # Merge data
    simulator.merge_data(agri_data)

    # Build model
    simulator.build_model()

    # Make prediction for the next 30 days
    forecast = simulator.predict_future(soil_quality, rainfall, temperature, harvest_yield)

    # Plot results
    simulator.plot_results(forecast)

    return forecast


# Example: Olive oil stock prediction based on Italian soil fertility
if __name__ == "__main__":
    # Example for an olive oil company (using a sample ticker)
    ticker = "OLIV.MI"  # Example ticker for an Italian olive oil company

    # Parameters
    soil_quality = 8.5  # High soil fertility (0-10 scale)
    rainfall = 120  # mm of rainfall
    temperature = 24  # Average temperature in Celsius

    # Run simulation
    forecast = run_simulation(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2023-12-31",
        soil_quality=soil_quality,
        rainfall=rainfall,
        temperature=temperature
    )

    print("Forecast for the next 30 days:")
    print(forecast.head())