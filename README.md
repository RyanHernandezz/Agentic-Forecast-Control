# 🛢️ Agentic Forecast Control
A real-time, multi-agent WTI Crude Oil forecasting and control dashboard equipped with an interactive Google Gemini Multimodal Live API voice layer and a data sonification engine.

This system takes live `yfinance` data and manages market expectations utilizing **three competing agent-driven models** (Linear Regression, Random Forest, ARIMA). The ensemble is recursively refit into a "Walk-Forward" state, weighting models based on whichever predicted most accurately in recent cycles to maintain prediction resilience against macroeconomic drifts.

## 🚀 Features
- **Walk-Forward Agentic Retraining:** A background pipeline rolling regression that retrains every single hour on the latest out-of-sample data, displaying its RMSE outperformance natively against static forecasting.
- **Seasonality & Drift Sentinels:** Agentic macro checks across ^VIX, Dollar Index, and Yield distributions designed to audibly ping analysts if underlying conditions drift off-course.
- **Geiger Counter Simulator:** Synthesizes audio tick density dynamically around micro-volume clusters.
- **Multimodal Voice Control Layer (`live_session.py`):** Enables hands-free natural language querying using `gemini-3.1-flash-live-preview`. The agent connects mathematically to local python tools on your machine.
- **Dynamic On-the-Fly Audio Synthesis:** Ask the agent what volatility sounds like! It builds aggressive, jagged sawtooth waveforms dynamically in Python and bypasses the Google voice layer to blast the audio straight to your speakers.

## 🛠️ Required Setup

Everything is built completely in Python utilizing Streamlit and Google GenAI.

1. **Install requirements:**
   Ensure you're using a modern Python 3.x stack and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment File Definition:**
   In the root of the project folder, create a `.env` file containing your Gemini key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

## 🧠 Running the System

You can run both interfaces seamlessly at the same time:

### The Control Desk GUI
The visual dashboard that handles the live tracking and Javascript sonification panels.
```bash
python -m streamlit run app.py
```
*Navigate to `http://localhost:8502/` to view the control desk.*

### The Voice Interrogator 
A terminal-based local daemon that connects your system microphone into the Multimodal Gemini pipeline.
```bash
python live_session.py
```
Wait for `Live session connected. Start speaking.` and then simply ask the AI:
- *"What model holds the most weight in the ensemble right now?"*
- *"Are any of our macro tracking inputs drifting from normal ranges?"*
- *"Play the audio for what high volatility would sound like using the geiger tool."*

## 📁 Core Infrastructure
- `app.py`: Streamlit visual dashboard routing.
- `live_session.py`: The live Google GenAI socket connection bridging real-time microphone to live internal pipeline variables.
- `live_tools.py`: Connective tool schema and custom wave synthesis.
- `sonification.py`: Mathematical waveform generator translating pure prices and volatility to raw 16-bit PCM arrays.
- `pipeline.py`: Root background logic managing the data pulling and drift assessments.