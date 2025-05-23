import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

# Load the .kv UI layout manually
Builder.load_file("Amanda_White_Stock_Prediction_Plan_UI.kv")

# Use US Central Time for consistency
CENTRAL_TZ = pytz.timezone('US/Central')

class StockScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_future = True  # Start in future mode

    def toggle_view(self):
        """Switch between future forecast and past evaluation"""
        self.show_future = not self.show_future
        self.ids.toggle_button.text = "Switch to Past View" if self.show_future else "Switch to Future View"

    def extract_features(self, df):
        """Generate time-based features for Random Forest"""
        X = pd.DataFrame()
        X['idx'] = np.arange(len(df))  # sequential index
        dt = pd.to_datetime(df.index)
        X['hour'] = dt.hour if hasattr(dt, 'hour') else 0
        X['dayofweek'] = dt.dayofweek if hasattr(dt, 'dayofweek') else 0
        return X

    def predict_stock(self):
        """Main method to train models and generate predictions + plot"""

        # Get selected inputs
        ticker = self.ids.selected_ticker.text
        time_frame = self.ids.selected_time_frame.text

        if ticker == 'Select a Ticker' or time_frame == 'Select a time frame':
            self.ids.result_label.text = "Please select both Ticker and Time Frame."
            return

        # Choose interval and data range
        interval = '1h' if time_frame == '1 Day' else '1d'
        period = '60d' if time_frame == '30 Days' else '30d'

        # Download historical data
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            self.ids.result_label.text = "No data returned."
            return

        # Prepare features and labels
        data['Close'] = data['Close'].fillna(method='ffill')
        data.index = pd.to_datetime(data.index)
        y = data['Close'].values
        X = self.extract_features(data)

        # Train models
        model_lr = LinearRegression().fit(X[['idx']], y)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

        # Set number of steps and time frequency
        now = datetime.now(tz=CENTRAL_TZ)
        if time_frame == '1 Day':
            future_steps = 24
            freq = 'H'
        elif time_frame == '7 Days':
            future_steps = 7
            freq = 'D'
        elif time_frame == '30 Days':
            future_steps = 30
            freq = 'D'
        else:
            self.ids.result_label.text = "Unknown timeframe"
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set dark theme plot styling
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('lime')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        if self.show_future:
            # Future forecast mode
            future_dates = pd.date_range(now, periods=future_steps, freq=freq, tz=CENTRAL_TZ)
            future_df = pd.DataFrame()
            future_df['idx'] = np.arange(len(data), len(data) + future_steps)
            future_df['hour'] = future_dates.hour
            future_df['dayofweek'] = future_dates.dayofweek

            # Make predictions
            pred_lr = model_lr.predict(future_df[['idx']])
            pred_rf = model_rf.predict(future_df[['idx', 'hour', 'dayofweek']])

            # Plot predictions
            ax.plot(future_dates, pred_lr, label='Linear Regression', linestyle='--', color='lime')
            ax.plot(future_dates, pred_rf, label='Random Forest', linestyle='-.', color='cyan')
            ax.set_title(f"{ticker} Forecast ({time_frame}) - Future")

            self.ids.result_label.text = f"Final Forecasted Value (LR): ${float(pred_lr[-1]):.3f}"

        else:
            # Past evaluation mode
            split_idx = int(len(X) * 0.7)
            X_train, y_train = X.iloc[:split_idx], y[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y[split_idx:]

            # Retrain models on training data
            model_lr = LinearRegression().fit(X_train[['idx']], y_train)
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

            # Predict on test
            y_pred_lr = model_lr.predict(X_test[['idx']])
            y_pred_rf = model_rf.predict(X_test)

            test_dates = data.index[split_idx:]

            # Plot actual and predicted values
            ax.plot(test_dates, y_test, label='Actual', color='white')
            ax.plot(test_dates, y_pred_lr, label='Linear Regression', linestyle='--', color='lime')
            ax.plot(test_dates, y_pred_rf, label='Random Forest', linestyle='-.', color='cyan')
            ax.set_title(f"{ticker} Evaluation ({time_frame}) - Past")

            self.ids.result_label.text = f"Last Actual Value: ${float(y_test[-1]):.3f}"

        ax.set_xlabel('Time (Central)')
        ax.set_ylabel('Price')
        ax.legend()
        fig.autofmt_xdate()

        # Display updated plot
        self.ids.graph_layout.clear_widgets()
        self.ids.graph_layout.add_widget(FigureCanvasKivyAgg(fig))

class AmandaWhiteStockPredictionPlanApp(App):
    def build(self):
        return StockScreen()

if __name__ == '__main__':
    AmandaWhiteStockPredictionPlanApp().run()
