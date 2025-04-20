import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Settings
TICKERS = ['AAPL', 'BTC-USD', 'XRP-USD', 'META', 'AMZN']
LOOKBACK = 60
PRED_DAYS = 7

# Data Handling
def get_data(ticker):
    df = yf.download(ticker, period='6mo', interval='1d')
    df = df[['Close']].dropna()
    return df

def prepare_lstm_data(series, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_random_forest(data):
    data = data.copy()
    data['Target'] = data['Close'].shift(-PRED_DAYS)
    data.dropna(inplace=True)
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Target'].values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    future_X = np.arange(len(data), len(data) + PRED_DAYS).reshape(-1, 1)
    prediction = model.predict(future_X)
    actual = data['Close'].values[-PRED_DAYS:]
    return prediction, actual

def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = prepare_lstm_data(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    predictions = []
    last_sequence = scaled_data[-LOOKBACK:]

    for _ in range(PRED_DAYS):
        input_seq = last_sequence.reshape(1, LOOKBACK, 1)
        pred_scaled = model.predict(input_seq)[0][0]
        predictions.append(pred_scaled)
        last_sequence = np.append(last_sequence[1:], [[pred_scaled]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actual = data['Close'].values[-PRED_DAYS:]
    return predictions, actual

# UI Logic
class StockPredictorUI(BoxLayout):
    def predict_model(self, model_type):
        ticker = self.ids.ticker_spinner.text
        if ticker not in TICKERS:
            self.ids.result_label.text = "Please select a valid ticker."
            return

        self.ids.result_label.text = f"Running {model_type} for {ticker}..."
        df = get_data(ticker)

        if model_type == "Random Forest":
            prediction, actual = train_random_forest(df)
        else:
            prediction, actual = train_lstm_model(df)

        self.plot_results(prediction, actual, model_type, ticker)

    def plot_results(self, prediction, actual, model_type, ticker):
        days = list(range(1, len(prediction)+1))
        fig, ax = plt.subplots()
        ax.plot(days, actual, label='Actual', marker='o')
        ax.plot(days, prediction, label='Predicted', marker='x')
        ax.set_title(f'{model_type} Prediction - {ticker}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Price ($)')
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        img = Image()
        img.texture = self.load_texture_from_buffer(buf)
        popup = Popup(title='Prediction Chart', content=img, size_hint=(0.9, 0.9))
        popup.open()

    def load_texture_from_buffer(self, buf):
        buf.seek(0)
        img_data = plt.imread(buf)
        img_data = (img_data[:, :, :3] * 255).astype(np.uint8)

        texture = Texture.create(size=(img_data.shape[1], img_data.shape[0]), colorfmt='rgb')
        texture.blit_buffer(img_data.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture

class Amanda_White_Stock_Prediction_Plan_UIApp(App):
    def build(self):
        return StockPredictorUI()

if __name__ == '__main__':
    Amanda_White_Stock_Prediction_Plan_UIApp().run()
