import asyncio
import os
import json
import logging
import base64
import websockets
from dotenv import load_dotenv
load_dotenv()
from agents import DataAgent, ModelerAgent, ChallengerAgent, EnsembleAgent
from sonification import Sonifier

logging.basicConfig(level=logging.INFO)

async def orchestrate_run():
    logging.info("Starting Orchestrator Agent...")
    
    # Initialize Sub-Agents
    data_agent = DataAgent(ticker="CL=F")
    modeler = ModelerAgent()
    challenger = ChallengerAgent(alpha=0.3)
    ensemble = EnsembleAgent()
    sonifier = Sonifier(duration_per_point=0.1)

    # 1. Acquire Data
    logging.info("DataAgent fetching recent data for Crude Oil (CL=F)...")
    df = data_agent.fetch_data(period="2y")
    close_prices = df['Close'].values
    
    # 2. Simulate progressive forecasting pipeline over recent periods
    # We will step through the last 4 chunks of data (e.g. 15 days each)
    chunk_size = 15
    num_chunks = 4
    
    start_eval_idx = len(close_prices) - (chunk_size * num_chunks)
    
    metrics_history = []
    final_ensemble_predictions = []
    actuals_for_report = []

    for i in range(num_chunks):
        current_idx = start_eval_idx + (i * chunk_size)
        
        # Train on EVERYTHING before current_idx
        train_data = df.iloc[:current_idx]
        actual_test_data = df.iloc[current_idx:current_idx + chunk_size]['Close'].values
        
        logging.info(f"ModelerAgent training for period chunk {i+1}/{num_chunks}...")
        preds_dict = modeler.train_and_forecast(train_data, chunk_size)
        
        logging.info(f"ChallengerAgent evaluating new model predictions...")
        scores = challenger.evaluate(actual_test_data, preds_dict)
        
        logging.info(f"EnsembleAgent mixing models...")
        combined_preds, weights = ensemble.mix(preds_dict, scores)
        
        metrics_history.append({
            "chunk_index": i + 1,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "smoothed_rmse_scores": {k: round(v, 4) for k, v in scores.items()}
        })
        final_ensemble_predictions.extend(combined_preds)
        actuals_for_report.extend(actual_test_data)
        
    logging.info("Forecasting simulation complete. Saving generated data locally and preparing data sonification...")

    # Both Actuals and Predictions Sonified
    pcm_audio = sonifier.sonify(actuals_for_report + final_ensemble_predictions)
    
    # Save the internal data structures to the disk so they can be reviewed outside the LLM environment
    with open("metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
        
    with open("ensemble_forecast_output.csv", "w") as f:
        f.write("Actual_Price,Ensemble_Predicted_Price\n")
        max_len = max(len(actuals_for_report), len(final_ensemble_predictions))
        for i in range(max_len):
            actual = actuals_for_report[i] if i < len(actuals_for_report) else ""
            pred = final_ensemble_predictions[i] if i < len(final_ensemble_predictions) else ""
            f.write(f"{actual},{pred}\n")
    
    # Prepare text summary
    summary_text = (
        f"Multi-Agent Forecasting Run Summary.\n"
        f"Data: Crude Oil (CL=F), {len(df)} total historical points.\n"
        f"Metrics tracked across the last {num_chunks} evaluation periods (15 days each):\n"
        f"{json.dumps(metrics_history, indent=2)}\n"
        "The audio just transmitted represents the sonified sequence of actual past values followed by the ensemble forecasted values."
    )

    logging.info("Initiating Gemini Live SDK connection and sending data stream...")

    # Using GenAI Live SDK to generate the report
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY environment variable not set. Please set it to proceed.")
        return

    prompt = (
        f"{summary_text}\n\n"
        "As the AI Orchestrator supervising this pipeline, please listen to the sonified trend to detect any audible structural breaks "
        "or volatility changes in the time-series space. "
        "Combine this sensory insight with the historical metrics attached above to generate a "
        "'Model Governance and Regime Change Report'. Identify which models gained dominance, potential "
        "market regime shifts (based on your audio ingestion and model switching), and recommendations."
    )

    logging.info("Bypassing WebSockets. Initiating stable REST Streaming API...")
    
    try:
        # Re-initialize the official GenAI Client (pulls from .env automatically)
        from google import genai
        client = genai.Client()
        
        from google.genai import types
        # Package the raw PCM bytes into the strict native SDK Part type
        # Natively, google-genai expects strict typing for multimedia bytes.
        audio_part = types.Part.from_bytes(
            data=pcm_audio,
            mime_type="audio/pcm;rate=16000"
        )
        
        logging.info("Transmitting sonified metrics and analytical prompt via REST...")
        response_stream = await client.aio.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=[audio_part, prompt]
        )
        
        logging.info("Awaiting AI Orchestrator response...")
        print("\n======= MODEL GOVERNANCE & REGIME CHANGE REPORT =======")
        async for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n=======================================================\n")
        logging.info("REST Pipeline execution optimal.")
        
    except Exception as e:
        logging.error(f"Multimodal REST transmission failed: {e}")

if __name__ == "__main__":
    asyncio.run(orchestrate_run())
