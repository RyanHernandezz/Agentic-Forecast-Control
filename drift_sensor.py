import pandas as pd
import numpy as np

class DriftSensor:
    def __init__(self, features_df: pd.DataFrame, window: int = 60):
        self.df = features_df
        self.window = window
        # EXCLUDE Close, regime, and any ML Lags ensuring anomaly maps purely to statistical exogenous metrics.
        self.macro_features = [
            c for c in features_df.columns 
            if c not in ["Close", "CL=F", "regime"] and not c.startswith("Lag_")
        ]
        self._compute_z_scores()
        
    def _compute_z_scores(self) -> None:
        self.z_scores = pd.DataFrame(index=self.df.index)
        for col in self.macro_features:
            rolling_mean = self.df[col].rolling(window=self.window).mean()
            rolling_std = self.df[col].rolling(window=self.window).std()
            self.z_scores[f"{col}_zscore"] = (self.df[col] - rolling_mean) / rolling_std
            
        self.z_scores.bfill(inplace=True)
        self.z_scores.fillna(0, inplace=True)
            
    def drift_score(self) -> float:
        recent_z = self.z_scores.iloc[-1].abs()
        return float(recent_z.mean()) if not recent_z.empty else 0.0
        
    def is_drifting(self) -> bool:
        return self.drift_score() > 2.0
        
    def alert_message(self) -> str:
        recent_z = self.z_scores.iloc[-1].abs().sort_values(ascending=False)
        top_2 = recent_z.head(2)
        if len(top_2) < 2:
            return "Insufficient drift data."
        return f"Drift Alert! Top anomalies: {top_2.index[0]} ({top_2.iloc[0]:.2f}z), {top_2.index[1]} ({top_2.iloc[1]:.2f}z)."
