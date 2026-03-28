import logging
from feature_agent import FeatureAgent
from regime_agent import RegimeAgent
from drift_sensor import DriftSensor
from seasonality_agent import SeasonalityAgent
from sonification_agent import SonificationAgent
from agents import ModelerAgent, ChallengerAgent, EnsembleAgent

def fetch_and_run_pipeline(period: str = "1y") -> dict:
    """
    Executes the macro time-series pipeline sequentially applying strict train/test data leakage buffers.
    """
    logging.info("Step 1: Fetching core macros...")
    f_agent = FeatureAgent()
    features_df = f_agent.fetch_data(period=period)
    
    logging.info("Step 2: Classifying broad asset regime...")
    regime = RegimeAgent(features_df)
    
    logging.info("Step 3: Calculating statistical structural drift...")
    drift = DriftSensor(features_df)
    
    logging.info("Step 4: Mapping internal market seasonality...")
    seasonality = SeasonalityAgent(features_df["Close"])
    
    logging.info("Step 5: Training ensemble subset via non-leaking walk-forward window...")
    modeler = ModelerAgent()
    
    # Critical Fix: Establish strict boundaries isolating train data from evaluation data.
    test_size = 15
    train_df = features_df.iloc[:-test_size].copy()
    test_actuals = features_df["Close"].iloc[-test_size:].values
    
    # Note: ModelerAgent organically builds Lags 1-3 internally.
    predictions = modeler.train_and_forecast(train_df, test_size)
    
    logging.info("Step 6: Executing out-of-sample Alpha Challenger Scoring...")
    challenger = ChallengerAgent()
    scores = challenger.evaluate(test_actuals, predictions)
    
    logging.info("Step 7: Ensembling ML outcomes...")
    ensemble = EnsembleAgent()
    combined, weights = ensemble.mix(predictions, scores)
    
    logging.info("Step 8: Constructing Audio Arrays...")
    sonification = SonificationAgent(
        ensemble_weights=weights,
        regime_label=regime.regime_label(),
        seasonal_zscore=seasonality.seasonal_zscore(),
        drift_score=drift.drift_score()
    )
    
    # Note: Ensure floats and native mappings.
    state_payload = {
        "regime": regime.regime_label(),
        "drift_alert": drift.alert_message(),
        "is_drifting": drift.is_drifting(),
        "sonification_params": sonification.audio_params(),
        "structural_weights": {k: float(v) for k, v in weights.items()},
        "calendar_month_factor": sonification.calendar_month_factor(),
        "drift_score": drift.drift_score(),
        "seasonal_zscore": seasonality.seasonal_zscore(),
        "current_residual": seasonality.current_residual(),
        "actuals": test_actuals.tolist(),
        "ensemble_forecast": combined,
        "historical_close": features_df["Close"].tolist(),
        "historical_dates": [d.strftime("%Y-%m-%d") for d in features_df.index]
    }
    
    return state_payload
