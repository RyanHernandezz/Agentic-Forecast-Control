import pandas as pd
from statsmodels.tsa.seasonal import STL
import logging

class SeasonalityAgent:
    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self._decompose()
        
    def _decompose(self) -> None:
        period = 252
        if len(self.series) < 2 * period:
            period = max(2, len(self.series) // 2)
            if period % 2 == 0: period -= 1
            logging.warning(f"Series too short for 252 period. Auto-reducing period to {period}.")
        else:
            # Assure strict oddness natively for edge-cases in specific Statsmodels versions
            if period % 2 == 0: period -= 1
            
        try:
            self.stl = STL(self.series, period=period, robust=True).fit()
            self._seasonal = self.stl.seasonal
            self._trend = self.stl.trend
            self._resid = self.stl.resid
        except Exception as e:
            logging.error(f"STL Error: {e}. Defaulting arrays.")
            self._seasonal = pd.Series(0.0, index=self.series.index)
            self._trend = self.series
            self._resid = pd.Series(0.0, index=self.series.index)
            
    def seasonal_component(self) -> pd.Series: return self._seasonal
    def trend_component(self) -> pd.Series: return self._trend
    def residual(self) -> pd.Series: return self._resid
    def current_residual(self) -> float: return float(self._resid.iloc[-1])
    
    def seasonal_zscore(self) -> float:
        rolling_std = self._resid.rolling(60).std().iloc[-1]
        if pd.isna(rolling_std) or rolling_std == 0: return 0.0
        return float(self.current_residual() / rolling_std)
