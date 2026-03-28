import pandas as pd

class RegimeAgent:
    def __init__(self, features_df: pd.DataFrame):
        self.df = features_df.copy()
        self._classify()
        
    def _classify(self) -> None:
        def get_regime(vix_val: float) -> int:
            if pd.isna(vix_val) or vix_val <= 15: return 0
            elif vix_val <= 25: return 1
            else: return 2
        
        if "^VIX" in self.df.columns:
            self.df["regime"] = self.df["^VIX"].apply(get_regime)
        else:
            self.df["regime"] = 0
            
    def current_regime(self) -> int:
        return int(self.df["regime"].iloc[-1])
        
    def regime_label(self) -> str:
        r = self.current_regime()
        if r == 0: return "trending"
        elif r == 1: return "transition"
        else: return "mean-reverting"
