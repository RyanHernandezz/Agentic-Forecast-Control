from live_tools import (
    PIPELINE_STATE, get_ensemble_weights, get_seasonal_status, 
    get_drift_alert, get_audio_params, get_full_state
)

def test_live_tools():
    # Inject a mock but functionally representative state into the singleton mapping
    print("[*] Passing mock data state into singleton PIPELINE_STATE...")
    
    mock_state = {
        "regime": "transition",
        "drift_alert": "Drift Alert! Top anomalies: SPY (2.42z)",
        "is_drifting": True,
        "sonification_params": {"ch1_freq": 230.12},
        "structural_weights": {"ARIMA": 0.1, "RandomForest": 0.8},
        "calendar_month_factor": 1.05,
        "drift_score": 2.2,
        "seasonal_zscore": 0.85,
        "current_residual": 12.4
    }
    
    # Dictionary update is equivalent to a singleton binding natively.
    PIPELINE_STATE.update(mock_state)
    
    print("\n[*] Validating Tool Endpoints via Direct REPL execution:")
    
    res_weights = get_ensemble_weights()
    assert res_weights["dominant_model"] == "RandomForest"
    print(" - get_ensemble_weights() -> PASS")
    
    res_seas = get_seasonal_status()
    assert res_seas["month_factor"] == 1.05
    print(" - get_seasonal_status() -> PASS")

    res_drift = get_drift_alert()
    assert res_drift["is_drifting"] is True
    print(" - get_drift_alert() -> PASS")
    
    res_audio = get_audio_params()
    assert "ch1_freq" in res_audio
    print(" - get_audio_params() -> PASS")
    
    res_full = get_full_state()
    assert "drift" in res_full
    assert res_full["seasonality"]["expected_direction"] == "upward pull"
    print(" - get_full_state() -> PASS")

if __name__ == "__main__":
    test_live_tools()
