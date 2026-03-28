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

live_api_tools = [
    get_ensemble_weights,
    get_seasonal_status,
    get_drift_alert,
    get_audio_params,
    get_full_state
]
