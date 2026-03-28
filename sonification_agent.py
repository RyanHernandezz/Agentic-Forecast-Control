import json
import datetime

class SonificationAgent:
    def __init__(self, ensemble_weights: dict, regime_label: str, seasonal_zscore: float, drift_score: float):
        self.weights = ensemble_weights
        self.regime = regime_label
        self.seasonal_zscore = seasonal_zscore
        self.drift_score = drift_score
        
    def calendar_month_factor(self) -> float:
        month = datetime.datetime.now().month
        mapping = {1:0.92, 2:0.95, 3:1.06, 4:1.08, 5:1.10, 6:1.05, 7:1.03, 8:1.04, 9:1.02, 10:0.97, 11:0.91, 12:0.93}
        return mapping.get(month, 1.0)
        
    def audio_params(self) -> dict:
        ch1_freq = 220.0 * self.calendar_month_factor()
        ch1_freq = max(180.0, min(280.0, ch1_freq))
        
        ch2_freq = ch1_freq + (self.seasonal_zscore * 15.0)
        
        dom_weight = max(self.weights.values()) if self.weights else 0.5
        ch2_amplitude = max(0.0, min(1.0, float(dom_weight)))
        
        bpm = 30 + (self.drift_score * 20)
        ch3_tempo_bpm = min(180.0, float(bpm))
        
        if self.regime == "trending": ch3_timbre = "sine"
        elif self.regime == "transition": ch3_timbre = "triangle"
        else: ch3_timbre = "sawtooth"
        
        dissonance = max(0.0, abs(ch2_freq - ch1_freq))
        
        return {
            "ch1_freq": round(float(ch1_freq), 2),
            "ch2_freq": round(float(ch2_freq), 2),
            "ch2_amplitude": round(float(ch2_amplitude), 2),
            "ch3_tempo_bpm": round(float(ch3_tempo_bpm), 2),
            "ch3_timbre": ch3_timbre,
            "dissonance_beating_hz": round(float(dissonance), 2)
        }
        
    def to_json(self) -> str:
        return json.dumps(self.audio_params())
