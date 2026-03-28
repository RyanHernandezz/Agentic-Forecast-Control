import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings("ignore")

class DataAgent:
    def __init__(self, ticker="CL=F"):
        self.ticker = ticker

    def fetch_data(self, period="1y"):
        prices = yf.download(self.ticker, period=period, progress=False)
        # Flatten columns if yfinance returns multi-index
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
            
        df = prices[['Close']].copy()
        if 'Close' not in df.columns and len(df.columns) > 0:
            df.columns = ['Close'] # fallback
            
        df.dropna(inplace=True)
        return df

class ModelerAgent:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "ARIMA": (ARIMA, (5, 1, 0))
        }

    def train_and_forecast(self, train_data, test_data_length):
        predictions = {}
        df = train_data.copy()
        for i in range(1, 4):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        df.dropna(inplace=True)
        
        if len(df) < 5:
            # Fallback if not enough data
            return {k: [train_data['Close'].iloc[-1]] * test_data_length for k in self.models.keys()}
            
        X_train = df.drop(columns=['Close'])
        y_train = df['Close'].values

        # ML Models
        for name in ["LinearRegression", "RandomForest"]:
            model = self.models[name]
            model.fit(X_train.values, y_train)
            
            preds = []
            current_features = X_train.iloc[-1:].values.copy()
            
            lag1_idx = X_train.columns.get_loc('Lag_1')
            lag2_idx = X_train.columns.get_loc('Lag_2')
            lag3_idx = X_train.columns.get_loc('Lag_3')
            
            for _ in range(test_data_length):
                p = model.predict(current_features)[0]
                preds.append(p)
                # Shift for next prediction without erasing static exogenous features (like ^VIX)
                current_features[0, lag3_idx] = current_features[0, lag2_idx]
                current_features[0, lag2_idx] = current_features[0, lag1_idx]
                current_features[0, lag1_idx] = p
            predictions[name] = preds

        # ARIMA
        arima_class, order = self.models["ARIMA"]
        try:
            model_fit = arima_class(train_data['Close'].values, order=order).fit()
            predictions["ARIMA"] = list(model_fit.forecast(steps=test_data_length))
        except Exception as e:
            predictions["ARIMA"] = predictions["LinearRegression"]
            
        return predictions

class ChallengerAgent:
    def __init__(self, alpha=0.3):
        """Exponential smoothing factor alpha. Higher favors recent data points."""
        self.alpha = alpha  
        self.history_rmse = {}
        
    def evaluate(self, actuals, predictions_dict):
        """Calculates exponentially smoothed RMSE."""
        smoothed_scores = {}
        for name, preds in predictions_dict.items():
            rmse = root_mean_squared_error(actuals, preds)
            if name not in self.history_rmse:
                self.history_rmse[name] = [rmse]
            else:
                self.history_rmse[name].append(rmse)
            
            # EMA over historical errors
            ema = 0
            weight_sum = 0
            # iterate backwards so recent has index i=0 (weight 1)
            for i, r in enumerate(reversed(self.history_rmse[name])):
                weight = (1 - self.alpha)**i
                ema += r * weight
                weight_sum += weight
            smoothed_scores[name] = ema / weight_sum
            
        return smoothed_scores

class EnsembleAgent:
    def mix(self, predictions_dict, challenger_scores):
        """Mixes models inversely proportional to their smoothed RMSE."""
        weights = {}
        target_sum = 0
        for name, score in challenger_scores.items():
            w = 1.0 / (score + 1e-6)
            weights[name] = w
            target_sum += w
            
        for name in weights:
            weights[name] /= target_sum
            
        n_steps = len(list(predictions_dict.values())[0])
        combined = []
        for step in range(n_steps):
            step_val = 0
            for name, preds in predictions_dict.items():
                step_val += weights[name] * preds[step]
            combined.append(step_val)
            
        return combined, weights
