import numpy as np

PIPELINE_STATE = {}

def get_ensemble_weights() -> dict:
    """Returns the ensemble weights dictionary, dominant model name, and current market regime label."""
    if not PIPELINE_STATE: return {"error": "Pipeline state uninitialized."}
    w = PIPELINE_STATE.get("structural_weights", {})
    r = PIPELINE_STATE.get("regime", "unknown")
    dom = max(w, key=w.get) if w else "unknown"
    return {"weights": w, "dominant_model": dom, "current_regime": r}
    
def get_seasonal_status() -> dict:
    """Returns seasonal z-score, current stl residual, calendar month, and expected direction per seasonality."""
    if not PIPELINE_STATE: return {"error": "Pipeline state uninitialized."}
    z = PIPELINE_STATE.get("seasonal_zscore", 0.0)
    cr = PIPELINE_STATE.get("current_residual", 0.0)
    mo = PIPELINE_STATE.get("calendar_month_factor", 1.0)
    exp = "upward pull" if mo > 1.0 else "downward pull" if mo < 1.0 else "neutral"
    
    return {"seasonal_zscore": z, "current_residual": cr, "month_factor": mo, "expected_direction": exp}

def get_drift_alert() -> dict:
    """Returns drift alerts, drift score, alert message, and top anomalous features."""
    if not PIPELINE_STATE: return {"error": "Pipeline state uninitialized."}
    return {
        "is_drifting": PIPELINE_STATE.get("is_drifting", False),
        "drift_score": PIPELINE_STATE.get("drift_score", 0.0),
        "alert_message": PIPELINE_STATE.get("drift_alert", "None")
    }
    
def get_audio_params() -> dict:
    """Returns the JSON serialized parameters from the Sonification Agent."""
    if not PIPELINE_STATE: return {"error": "Pipeline state uninitialized."}
    return PIPELINE_STATE.get("sonification_params", {})
    
def get_full_state() -> dict:
    """Returns a merged state of weights, seasonals, drifts, and audio parameters for spoken summary."""
    if not PIPELINE_STATE: return {"error": "Pipeline state uninitialized."}
    return {
        "ensemble": get_ensemble_weights(),
        "seasonality": get_seasonal_status(),
        "drift": get_drift_alert(),
        "audio": get_audio_params()
    }

def simulate_geiger_audio(volatility: str) -> dict:
    """Returns a geiger counter audio simulation payload based on volatility ('high', 'medium', 'low')."""
    vol_level = volatility.lower()
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = 300.0
    
    ticks = np.zeros_like(t)
    num_ticks = 150 if vol_level == 'high' else (40 if vol_level == 'medium' else 8)
    
    import random
    tick_indices = random.sample(range(len(ticks)), num_ticks)
    
    timbre = 'sawtooth' if vol_level == 'high' else 'sine'
    for idx in tick_indices:
        blip_len = int(sr * (0.05 if vol_level == 'high' else 0.1))
        blip_t = np.linspace(0, 0.05, blip_len, False)
        if timbre == 'sawtooth':
            blip = 2 * (blip_t * freq - np.floor(blip_t * freq + 0.5))
        else:
            blip = np.sin(2 * np.pi * freq * blip_t)
            
        end_idx = min(idx + blip_len, len(ticks))
        ticks[idx:end_idx] += blip[:end_idx - idx]
        
    if np.max(np.abs(ticks)) > 0:
        ticks = ticks / np.max(np.abs(ticks))
    
    pcm_data = np.int16(ticks * 32767).tobytes()
    return {
        "__audio_payload__": pcm_data,
        "status": f"Played {duration} seconds of {timbre} geiger audio at {volatility} volatility directly to user's speakers."
    }

live_api_tools = [
    get_ensemble_weights,
    get_seasonal_status,
    get_drift_alert,
    get_audio_params,
    get_full_state,
    simulate_geiger_audio
]
