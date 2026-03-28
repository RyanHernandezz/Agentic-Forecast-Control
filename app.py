import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
import json
from pipeline import fetch_and_run_pipeline
from live_tools import PIPELINE_STATE
 
# ---- CACHED BACKEND ----
@st.cache_data(ttl=900)
def cached_pipeline_run():
    return fetch_and_run_pipeline(period="1y")
 
@st.cache_data(ttl=900)
def generate_true_future_forecast():
    from feature_agent import FeatureAgent
    from agents import ModelerAgent, EnsembleAgent
    f_agent = FeatureAgent()
    features_df = f_agent.fetch_data(period="1y")
    modeler = ModelerAgent()
    ensemble = EnsembleAgent()
 
    preds = modeler.train_and_forecast(features_df, test_data_length=15)
    pipeline_weights = PIPELINE_STATE.get("structural_weights", {"LinearRegression": 0.33, "RandomForest": 0.33, "ARIMA": 0.33})
 
    future = []
    for step in range(15):
        s_val = 0
        for name, p_arr in preds.items():
            s_val += pipeline_weights.get(name, 0) * p_arr[step]
        future.append(s_val)
 
    last_date = features_df.index[-1]
    b_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15, freq="B")
    return features_df.index[-60:].tolist(), features_df["Close"].iloc[-60:].tolist(), b_days.tolist(), future
 
 
st.set_page_config(page_title="Agentic Forecast Control", layout="wide")
st.title("🛢️ WTI Crude Oil — Agentic Forecast Control Desk")
st.caption("Five macro inputs. Three competing models. One ensemble that reweights itself every cycle based on who was right recently.")
 
try:
    pipeline_state = cached_pipeline_run()
    PIPELINE_STATE.update(pipeline_state)
except Exception as e:
    st.error(f"Pipeline failed: {e}")
    st.stop()
 
col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Regime", pipeline_state.get("regime", "Unknown").title())
col2.metric("Macro Drift Score", f'{pipeline_state.get("drift_score", 0.0):.2f}z')
weights = pipeline_state.get("structural_weights", {})
dom_model = max(weights, key=weights.get) if weights else "Unknown"
col3.metric("Dominant Model", dom_model)
col4.metric("Its Weight", f"{weights.get(dom_model, 0)*100:.1f}%" if weights else "")

# ---- LIVE API VOICE PANEL ----
st.markdown("#### 🎙️ Voice Interface — Ask the Pipeline")
st.info(
    "Run `python live_session.py` in a terminal to activate the voice layer. "
    "Then speak directly to the pipeline — ask about regime, drift, ensemble weights, "
    "or seasonal status. Gemini calls live tool functions and responds with actual model numbers."
)
with st.expander("Example questions to ask", expanded=False):
    st.markdown("""
- *"What regime are we in and which model is dominant?"*
- *"Is the ensemble drifting from seasonal expectations?"*
- *"Are any macro inputs anomalous right now?"*
- *"What do the sonification parameters look like?"*
- *"Give me a full system status."*
    """)


 
# ---- GOVERNANCE HEALTH ----
with st.expander("🩺 Model Governance — Three Live Checks", expanded=True):
    sz = pipeline_state.get('seasonal_zscore', 0.0)
    dr = pipeline_state.get('drift_score', 0.0)
 
    sz_status = f"✅ On-seasonal ({sz:.2f}z)" if abs(sz) < 1.5 else f"⚠️ Off-seasonal ({sz:.2f}z)"
    dr_status = f"✅ Stable ({dr:.2f}z)" if abs(dr) < 2.0 else f"🚨 Drift Detected ({dr:.2f}z)"
 
    st.markdown(f"""
**1. Is the forecast consistent with seasonal expectations?**
> **{sz_status}** — STL decomposition isolates the seasonal component. Above 1.5z means current price action is departing from 40 years of documented WTI seasonal structure (Mar/Apr bullish, Nov/Dec weak).
 
**2. Are the macro inputs the models trained on still valid?**
> **{dr_status}** — Rolling z-scores across VIX, Dollar Index, SPY, and 10yr yield. Above 2.0z means the input distribution has shifted outside its 60-day envelope. The models are operating out-of-sample.
 
**3. Is model performance decaying or is the market just noisy?**
> Watch the weight chart below. Weight drift = healthy adaptation. Collapse toward one model = the others have stopped working.
    """)
 
# ---- 15-DAY FORWARD FORECAST ----
st.subheader("15-Day Forward Forecast")
st.caption("Ensemble prediction beyond today. Weights reflect recent out-of-sample RMSE — whoever was right recently gets more vote.")
f_t_dates, f_t_prices, f_dates, f_preds = generate_true_future_forecast()
 
fig_ft = go.Figure()
f_extended_dates = f_t_dates[-1:] + [d.strftime("%Y-%m-%d") for d in f_dates]
f_extended_preds = f_t_prices[-1:] + f_preds
fig_ft.add_trace(go.Scatter(x=f_t_dates, y=f_t_prices, mode='lines', name='Historical', line=dict(color='white')))
fig_ft.add_trace(go.Scatter(x=f_extended_dates, y=f_extended_preds, mode='lines+markers', name='Ensemble Forecast', line=dict(color='#00ff9d', width=3, dash='dot')))
fig_ft.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20), xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
st.plotly_chart(fig_ft, use_container_width=True)
 
# ---- SONIFICATION EXPLAINER ----
st.markdown("#### 🎵 Why Sound?")
st.info("""
Markets are temporal processes. Charts are spatial. Sound is the right modality for slowly drifting data that has no single threshold to trigger an alert.
 
**Seasonal baseline (Channel 1):** A steady pitch derived from the expected WTI seasonal curve for this month — the note the market *should* be playing.  
**Ensemble forecast (Channel 2):** Tracks model output. Divergence from Channel 1 creates audible beating — the physical sensation of model disagreement with seasonal expectation.  
**Drift rhythm (Channel 3):** Tempo tracks the macro drift score. Faster = inputs are further from what the models trained on. Silence = familiar ground.
 
In the A/B backtest below, play Reality then play the Forecast. The dissonance between them is the forecast error, made audible.
""")
 
 
# ---- JAVASCRIPT AUDIO WRAPPERS ----
def build_dual_audio_wrapper(element_id: str, title: str, traces_json: str, seq_json: str, btn1_text: str, btn2_text: str, height=450):
    return f"""
    <!DOCTYPE html>
    <html>
    <head><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body style="margin: 0; font-family: sans-serif; background-color: transparent;">
        <div style="background-color: transparent; border-radius: 8px;">
            <div id="chart_{element_id}" style="width: 100%; height: {height}px;"></div>
            <div style="display: flex; align-items: center; justify-content: flex-start; gap: 10px; padding: 10px;">
                <button id="btn1_{element_id}" style="padding: 10px 16px; font-weight: bold; font-family: monospace; background-color: #00ff9d; color: black; border: none; border-radius: 4px; cursor: pointer; transition: 0.2s;">{btn1_text}</button>
                <button id="btn2_{element_id}" style="padding: 10px 16px; font-weight: bold; font-family: monospace; background-color: #00ff9d; color: black; border: none; border-radius: 4px; cursor: pointer; transition: 0.2s;">{btn2_text}</button>
                <div id="status_{element_id}" style="color: #aaa; font-size: 13px; font-family: monospace; margin-left:10px;">Click a button to play. <strong>Click the chart to seek.</strong></div>
            </div>
        </div>
        <script>
            const traces = {traces_json}; const sequence = {seq_json};
            const layout = {{ title: '{title}', margin: {{l: 40, r: 40, t: 40, b:40}}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{color: '#aaa', family: 'sans-serif'}}, hovermode: 'x unified', xaxis: {{showgrid: false}}, yaxis: {{showgrid: true, gridcolor: '#333'}}, shapes: [] }};
            Plotly.newPlot('chart_{element_id}', traces, layout, {{responsive: true}});
            let audioCtx = null; let activeInterval = null; let step = 0; let osc1, osc2, osc3, gain1, gain2, gain3; let targetMode = null;
            let btn1Text = "{btn1_text}"; let btn2Text = "{btn2_text}";
 
            const chartElem = document.getElementById('chart_{element_id}');
            chartElem.on('plotly_click', function(data) {{
                if (!data.points || data.points.length === 0) return;
                let clickedX = data.points[0].x;
                let newStep = sequence.findIndex(p => p.x === clickedX);
                if (newStep !== -1) {{
                    step = newStep;
                    Plotly.relayout('chart_{element_id}', {{ 'shapes': [{{ type: 'line', x0: clickedX, y0: 0, x1: clickedX, y1: 1, xref: 'x', yref: 'paper', line: {{color: '#ff0080', width: 2}} }}] }});
                }}
            }});
 
            function toggleAudio(mode) {{
                if (activeInterval && targetMode === mode) {{
                    clearInterval(activeInterval); activeInterval = null;
                    if (audioCtx) audioCtx.suspend();
                    document.getElementById('btn1_{element_id}').innerHTML = btn1Text; document.getElementById('btn2_{element_id}').innerHTML = btn2Text;
                    document.getElementById('btn1_{element_id}').style.backgroundColor = '#00ff9d'; document.getElementById('btn2_{element_id}').style.backgroundColor = '#00ff9d';
                    return;
                }}
                if (activeInterval && targetMode !== mode) {{
                    targetMode = mode;
                    if (mode === 'actual') {{
                        document.getElementById('btn1_{element_id}').style.backgroundColor = '#ff0080'; document.getElementById('btn1_{element_id}').innerHTML = "⏸ Playing Reality";
                        document.getElementById('btn2_{element_id}').style.backgroundColor = '#00ff9d'; document.getElementById('btn2_{element_id}').innerHTML = btn2Text;
                    }} else {{
                        document.getElementById('btn2_{element_id}').style.backgroundColor = '#ff0080'; document.getElementById('btn2_{element_id}').innerHTML = "⏸ Playing Forecast";
                        document.getElementById('btn1_{element_id}').style.backgroundColor = '#00ff9d'; document.getElementById('btn1_{element_id}').innerHTML = btn1Text;
                    }}
                    return;
                }}
                if (step >= sequence.length) {{ step = 0; if (audioCtx) {{ audioCtx.close(); audioCtx = null; }} }}
                targetMode = mode;
 
                if (!audioCtx) {{
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const masterGain = audioCtx.createGain(); masterGain.gain.value = 0.08; masterGain.connect(audioCtx.destination);
                    osc1 = audioCtx.createOscillator(); osc2 = audioCtx.createOscillator(); osc3 = audioCtx.createOscillator();
                    gain1 = audioCtx.createGain(); gain2 = audioCtx.createGain(); gain3 = audioCtx.createGain();
                    osc1.type = 'sine'; osc2.type = 'sine';
                    osc1.connect(gain1); gain1.connect(masterGain); osc2.connect(gain2); gain2.connect(masterGain); osc3.connect(gain3); gain3.connect(masterGain);
                    osc1.start(); osc2.start(); osc3.start();
                }} else if (audioCtx.state === 'suspended') {{ audioCtx.resume(); }}
 
                if (mode === 'actual') {{ document.getElementById('btn1_{element_id}').style.backgroundColor = '#ff0080'; document.getElementById('btn1_{element_id}').innerHTML = "⏸ Playing Reality"; }} else {{ document.getElementById('btn2_{element_id}').style.backgroundColor = '#ff0080'; document.getElementById('btn2_{element_id}').innerHTML = "⏸ Playing Forecast"; }}
 
                activeInterval = setInterval(() => {{
                    if (step >= sequence.length) {{
                        clearInterval(activeInterval); activeInterval = null;
                        if (audioCtx) {{ audioCtx.close(); audioCtx = null; }}
                        Plotly.relayout('chart_{element_id}', {{ shapes: [] }});
                        document.getElementById('btn1_{element_id}').innerHTML = btn1Text; document.getElementById('btn1_{element_id}').style.backgroundColor = '#00ff9d'; document.getElementById('btn2_{element_id}').innerHTML = btn2Text; document.getElementById('btn2_{element_id}').style.backgroundColor = '#00ff9d';
                        return;
                    }}
                    const p = sequence[step];
                    if (p.x !== undefined && p.x !== null) {{ Plotly.relayout('chart_{element_id}', {{ 'shapes': [{{ type: 'line', x0: p.x, y0: 0, x1: p.x, y1: 1, xref: 'x', yref: 'paper', line: {{color: '#ff0080', width: 2}} }}] }}); }}
                    const time = audioCtx.currentTime;
                    if (targetMode === 'actual') {{
                        osc1.frequency.setTargetAtTime(p.ch1_freq, time, 0.05); gain1.gain.setTargetAtTime(1.0, time, 0.05);
                        gain2.gain.setTargetAtTime(0.0, time, 0.05); gain3.gain.setTargetAtTime(0.0, time, 0.05);
                    }} else {{
                        osc2.frequency.setTargetAtTime(p.ch2_freq, time, 0.05); gain2.gain.setTargetAtTime(1.0, time, 0.05);
                        osc3.type = p.ch3_timbre || 'sine'; osc3.frequency.setTargetAtTime((p.ch3_tempo_bpm || 0) / 60, time, 0.05);
                        gain3.gain.setTargetAtTime(p.ch3_tempo_bpm > 0 ? 0.3 : 0.0, time, 0.05); gain1.gain.setTargetAtTime(0.0, time, 0.05);
                    }}
                    step++;
                }}, 250);
            }}
            document.getElementById('btn1_{element_id}').addEventListener('click', () => toggleAudio('actual')); document.getElementById('btn2_{element_id}').addEventListener('click', () => toggleAudio('forecast'));
        </script>
    </body>
    </html>
    """
 
def build_single_audio_wrapper(element_id: str, title: str, traces_json: str, seq_json: str, btn_text: str, desc: str, height=450):
    return build_dual_audio_wrapper(element_id, title, traces_json, seq_json, btn_text, "Bypass", height).replace(
        '<button id="btn2_', '<button style="display:none;" id="btn2_'
    )
 
 
# ---- OUT-OF-SAMPLE A/B BACKTEST ----
st.divider()
st.subheader("Out-of-Sample Backtest — Reality vs Ensemble")
st.caption("Model trained on everything except the last 15 days. Play Reality, then play the Forecast. The dissonance between them is the error.")
 
hist_prices = pipeline_state['historical_close']; hist_dates = pipeline_state['historical_dates']
actuals = pipeline_state['actuals']; preds = pipeline_state['ensemble_forecast']
context_size = 60
t_dates = hist_dates[-(context_size + 15):-15]; t_prices = hist_prices[-(context_size + 15):-15]
test_dates = hist_dates[-15:]
t_extended_dates = t_dates[-1:] + test_dates
t_extended_actuals = t_prices[-1:] + actuals
t_extended_preds = t_prices[-1:] + preds
 
traces = [
    {"x": t_dates, "y": t_prices, "mode": "lines", "name": "Training Context", "line": {"color": "#444"}},
    {"x": t_extended_dates, "y": t_extended_actuals, "mode": "lines", "name": "Actual (Hidden)", "line": {"color": "gray", "dash": "longdash"}},
    {"x": t_extended_dates, "y": t_extended_preds, "mode": "lines+markers", "name": "Ensemble Forecast", "line": {"color": "#00ff9d", "dash": "dot"}}
]
 
seq = []
anchor_price = hist_prices[-16]
for i in range(15):
    a_dev = actuals[i] - anchor_price
    p_dev = preds[i] - anchor_price
    error = abs(preds[i] - actuals[i])
    seq.append({
        "x": test_dates[i],
        "ch1_freq": 220.0 + (a_dev * 5.0),
        "ch2_freq": 220.0 + (p_dev * 5.0),
        "ch3_tempo_bpm": min(180, error * 15.0),
        "ch3_timbre": "sawtooth" if error > 2 else "sine"
    })
 
components.html(build_dual_audio_wrapper(
    "core",
    "15-Day A/B: Reality vs Forecast",
    json.dumps(traces), json.dumps(seq),
    "▶ Play Reality", "▶ Play Forecast"
), height=500)
 
 
# ---- AGENTIC VS STATIC ----
st.divider()
st.subheader("Agentic Retraining vs Static Forecast")
st.caption("Static model trains once, predicts 48 hours forward. Agentic model refits on every new hour before predicting the next. Same data, different discipline.")
 
@st.cache_data(persist="disk", ttl=86400)
def compute_agentic_precision_gains():
    from agents import ModelerAgent
    df_raw = yf.download("CL=F", period="1mo", interval="1h", progress=False)["Close"].ffill().dropna()
    if isinstance(df_raw, pd.Series):
        df = df_raw.to_frame("Close")
    else:
        df = df_raw.copy()
        df.columns = ["Close"]
 
    test_steps = 48
    train_base_idx = len(df) - test_steps
    if train_base_idx < 50:
        return None, None, None, None, None, None, None
 
    dates = df.index[-test_steps:].strftime("%m-%d %H:00").tolist()
    actuals = df["Close"].iloc[-test_steps:].values.tolist()
    modeler = ModelerAgent()
 
    train_static_df = df.iloc[:train_base_idx].copy()
    preds_static_dict = modeler.train_and_forecast(train_static_df, test_steps)
    static_preds = np.mean([p for p in preds_static_dict.values()], axis=0).tolist()
 
    agentic_preds = []
    for step in range(test_steps):
        train_agentic_df = df.iloc[:train_base_idx + step].copy()
        curr_preds = modeler.train_and_forecast(train_agentic_df, test_data_length=1)
        next_val = np.mean([cv[0] for cv in curr_preds.values()])
        agentic_preds.append(float(next_val))
 
    from sklearn.metrics import root_mean_squared_error
    static_rmse = root_mean_squared_error(actuals, static_preds)
    agent_rmse = root_mean_squared_error(actuals, agentic_preds)
    precision_gain = ((static_rmse - agent_rmse) / static_rmse) * 100 if static_rmse > 0 else 0
    return dates, actuals, static_preds, agentic_preds, precision_gain, static_rmse, agent_rmse
 
with st.spinner("Running 48 rolling retrains..."):
    ag_res = compute_agentic_precision_gains()
 
if ag_res[0]:
    ag_dates, ag_acts, ag_stat, ag_ag, ag_gain, ag_s_err, ag_a_err = ag_res
 
    ag_col1, ag_col2, ag_col3 = st.columns(3)
    ag_col1.metric("RMSE Reduction", f"+{ag_gain:.1f}%", help="Agentic vs static on held-out hourly data.")
    ag_col2.metric("Static RMSE", f"${ag_s_err:.3f}")
    ag_col3.metric("Agentic RMSE", f"${ag_a_err:.3f}")
 
    fig_ag = go.Figure()
    fig_ag.add_trace(go.Scatter(x=ag_dates, y=ag_acts, mode='lines', name='Actual', line=dict(color='white', width=1)))
    fig_ag.add_trace(go.Scatter(x=ag_dates, y=ag_stat, mode='lines', name='Static', line=dict(color='#ff0080', width=2, dash='dash')))
    fig_ag.add_trace(go.Scatter(x=ag_dates, y=ag_ag, mode='lines', name='Agentic', line=dict(color='#00ff9d', width=3)))
    fig_ag.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20), xaxis_title="Hour", yaxis_title="Price ($)", hovermode="x unified", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    st.plotly_chart(fig_ag, use_container_width=True)
 
 
# ---- ENSEMBLE WEIGHT EVOLUTION ----
st.divider()
st.subheader("Ensemble Weight Evolution — 30-Day Walk-Forward")
st.caption("Each point is a full retrain-and-score cycle. Watch which model the system trusts as conditions change. A regime shift should appear as a weight crossover.")
 
@st.cache_data(persist="disk", ttl=86400)
def simulate_walk_forward_metrics():
    from feature_agent import FeatureAgent
    from agents import ModelerAgent, ChallengerAgent, EnsembleAgent
    f_agent = FeatureAgent()
    features_df = f_agent.fetch_data("1y")
    modeler = ModelerAgent()
    challenger = ChallengerAgent(alpha=0.3)
    ensemble = EnsembleAgent()
 
    test_size = 15; sim_days = 30
    weights_history = {"LinearRegression": [], "RandomForest": [], "ARIMA": []}
    dates = []
    total_len = len(features_df)
    start_idx = total_len - sim_days - test_size
    if start_idx < 100:
        return [], {}
 
    for i in range(sim_days):
        current_idx = start_idx + i
        train_df = features_df.iloc[:current_idx].copy()
        test_actuals = features_df["Close"].iloc[current_idx: current_idx + test_size].values
        preds = modeler.train_and_forecast(train_df, test_size)
        smoothed_scores = challenger.evaluate(test_actuals, preds)
        ens_preds, step_weights = ensemble.mix(preds, smoothed_scores)
        for k in weights_history:
            weights_history[k].append(step_weights.get(k, 0.0))
        dates.append(features_df.index[current_idx].strftime("%Y-%m-%d"))
 
    return dates, weights_history
 
try:
    wf_dates, wf_weights = simulate_walk_forward_metrics()
    if wf_dates:
        fig_wf = go.Figure()
        colors = ['#ff0080', '#00ff9d', '#0088ff']
        for i, (model_name, w_array) in enumerate(wf_weights.items()):
            fig_wf.add_trace(go.Scatter(x=wf_dates, y=w_array, mode='lines', name=model_name, stackgroup='one', line=dict(width=0.5, color=colors[i]), fillcolor=colors[i]))
        fig_wf.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(range=[0, 1]), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_wf, use_container_width=True)
except Exception as e:
    st.warning(f"Walk-forward simulation failed: {e}")
 
 
# ---- ECONOMETRIC SEASONALITY ----
st.divider()
 
@st.cache_data(ttl=3600)
def get_econometric_seasonality():
    df_raw = yf.download("CL=F", period="max", progress=False)["Close"].dropna()
    df = df_raw.squeeze() if isinstance(df_raw, pd.DataFrame) else df_raw
    returns = np.log(df / df.shift(1)).dropna()
    t = np.arange(len(returns))
    X = pd.DataFrame()
    for k in range(1, 4):
        X[f's_{k}'] = np.sin(2 * np.pi * k * t / 252)
        X[f'c_{k}'] = np.cos(2 * np.pi * k * t / 252)
 
    model = LinearRegression(fit_intercept=False).fit(X, returns)
    t_yr = np.arange(252)
    X_yr = pd.DataFrame()
    for k in range(1, 4):
        X_yr[f's_{k}'] = np.sin(2 * np.pi * k * t_yr / 252)
        X_yr[f'c_{k}'] = np.cos(2 * np.pi * k * t_yr / 252)
 
    idx_curve = np.exp(np.cumsum(model.predict(X_yr))) * 100
    dates = pd.date_range(start='2025-01-01', periods=252).strftime("%b %d").tolist()
 
    seq = []
    for i, p in enumerate(idx_curve):
        dev = float(p - 100.0)
        seq.append({
            "x": dates[i],
            "ch1_freq": float(220.0 + (dev * 5.0)),
            "ch2_freq": float(220.0 + (dev * 5.0)),
            "ch2_amplitude": 0.8
        })
    traces = [{"x": dates, "y": idx_curve.tolist(), "mode": "lines", "name": "Fourier OLS Seasonal Curve", "line": {"color": "#ff0080"}}]
    return traces, seq
 
try:
    s_traces, s_seq = get_econometric_seasonality()
    components.html(build_single_audio_wrapper(
        "seasonality",
        "WTI Seasonal Baseline — Fourier OLS, Full History (Jan–Dec)",
        json.dumps(s_traces), json.dumps(s_seq),
        "▶ Play the Seasonal Curve", ""
    ), height=500)
except Exception as e:
    st.warning(f"Seasonality extraction failed: {e}")
 
st.markdown("<br>", unsafe_allow_html=True)
 
@st.cache_data(ttl=3600)
def get_recent_regime_change():
    df = yf.download(["CL=F", "^VIX"], start="2026-03-05", end="2026-03-11", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{c2}_{c1}" if c1 else c2 for c1, c2 in df.columns]
 
    cl = df["CL=F_Close"].ffill().dropna()
    vix = df["^VIX_Close"].ffill().dropna()
 
    seq = []; viz_d = []; viz_p = []
    for date in cl.index:
        d_str = date.strftime("%Y-%m-%d %H:%M")
        cp = float(cl.loc[date])
        vv = float(vix[vix.index <= date].iloc[-1]) if not vix[vix.index <= date].empty else 20.0
        viz_d.append(d_str); viz_p.append(cp)
        c2 = float(max(80.0, 220.0 + ((cp - 60.0) * 4.0)))
        seq.append({
            "x": d_str, "ch1_freq": c2, "ch2_freq": c2, "ch2_amplitude": 0.8,
            "ch3_tempo_bpm": float(vv * 300.0),
            "ch3_timbre": "sawtooth" if vv > 25 else "triangle"
        })
 
    traces = [{"x": viz_d, "y": viz_p, "mode": "lines", "name": "15m WTI", "line": {"color": "yellow"}}]
    return traces, seq
 
try:
    c_traces, c_seq = get_recent_regime_change()
    components.html(build_single_audio_wrapper(
        "shock",
        "March 5–11 2026: Regime Transition at 15-Minute Resolution",
        json.dumps(c_traces), json.dumps(c_seq),
        "▶ Play the Regime Shock", ""
    ), height=500)
except Exception as e:
    st.warning(f"Regime data unavailable: {e}")
 
 
# ---- GEIGER ORDER FLOW SIMULATOR ----
st.divider()
st.subheader("Order Flow Translator — Geiger Simulator")
st.info("""
Each 15-minute candle is subdivided into micro-ticks proportional to its volume. High-volume bars = dense pings. Low-volume bars = sparse.  
**Pitch** = price level. **Waveform** = volatility regime. Below VIX 25: sine (smooth). Above: your selected timbre (sawtooth = crisis, triangle = transition).  
Adjust the sliders to tune sensitivity, then hit play.
""")
 
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
custom_base_hz = m_col1.slider("Base Pitch (Hz)", 50, 800, 220, 10)
custom_tick_ms = m_col2.slider("Tick Duration (ms)", 10, 200, 50, 10)
custom_timbre  = m_col3.selectbox("High-Vol Waveform", ["sawtooth", "square", "triangle"], index=0)
custom_v_mult  = m_col4.slider("Volume Multiplier", 1, 10, 4)
 
def build_geiger_simulator_wrapper(element_id: str, title: str, traces_json: str, seq_json: str, btn_text: str, custom_tick_ms: int, height=450):
    return f"""
    <!DOCTYPE html>
    <html>
    <head><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body style="margin: 0; font-family: sans-serif; background-color: transparent;">
        <div style="background-color: transparent; border-radius: 8px;">
            <div id="chart_{element_id}" style="width: 100%; height: {height}px;"></div>
            <div style="padding: 10px;">
                <button id="startBtn_{element_id}" style="padding: 10px 16px; font-weight: bold; background-color: #00ff9d; color: black; border: none; border-radius: 4px; cursor: pointer; transition: 0.2s;">{btn_text}</button>
                <span style="color:#aaa; font-family:monospace; margin-left:15px;" id="lbl_{element_id}">{custom_tick_ms}ms per tick. <strong>Click chart to seek.</strong></span>
            </div>
        </div>
        <script>
            const traces = {traces_json}; const sequence = {seq_json};
            const layout = {{ title: '{title}', margin: {{l: 40, r: 40, t: 40, b: 40}}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{color: '#aaa', family: 'sans-serif'}}, hovermode: 'x unified', xaxis: {{showgrid: false}}, yaxis: {{showgrid: true, gridcolor: '#333'}}, shapes: [] }};
            Plotly.newPlot('chart_{element_id}', traces, layout, {{responsive: true}});
 
            let audioCtx = null; let activeInterval = null; let step = 0; let osc = null;
            let currentX = null;
 
            const chartElem = document.getElementById('chart_{element_id}');
            chartElem.on('plotly_click', function(data) {{
                if (!data.points || data.points.length === 0) return;
                let clickedX = data.points[0].x;
                let newStep = sequence.findIndex(p => p.x === clickedX);
                if (newStep !== -1) {{
                    step = newStep;
                    currentX = clickedX;
                    Plotly.relayout('chart_{element_id}', {{ 'shapes': [{{ type: 'line', x0: clickedX, y0: 0, x1: clickedX, y1: 1, xref: 'x', yref: 'paper', line: {{color: 'yellow', width: 2}} }}] }});
                }}
            }});
 
            document.getElementById('startBtn_{element_id}').addEventListener('click', function() {{
                if (activeInterval) {{
                    clearInterval(activeInterval); activeInterval = null;
                    if (audioCtx) audioCtx.suspend();
                    this.innerHTML = "▶ Resume";
                    this.style.backgroundColor = '#00ff9d';
                    return;
                }}
                if (step >= sequence.length) {{ step = 0; if (audioCtx) {{ audioCtx.close(); audioCtx = null; }} }}
                if (!audioCtx) {{
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const masterGain = audioCtx.createGain(); masterGain.gain.value = 0.05; masterGain.connect(audioCtx.destination);
                    osc = audioCtx.createOscillator();
                    osc.connect(masterGain); osc.start();
                }} else if (audioCtx.state === 'suspended') {{ audioCtx.resume(); }}
                this.innerHTML = "⏸ Pause"; this.style.backgroundColor = '#828282';
 
                activeInterval = setInterval(() => {{
                    if (step >= sequence.length) {{
                        clearInterval(activeInterval); activeInterval = null;
                        if (audioCtx) {{ audioCtx.close(); audioCtx = null; }}
                        Plotly.relayout('chart_{element_id}', {{ shapes: [] }});
                        document.getElementById('lbl_{element_id}').innerHTML = "Done.";
                        document.getElementById('startBtn_{element_id}').innerHTML = "↻ Replay"; document.getElementById('startBtn_{element_id}').style.backgroundColor = '#00ff9d';
                        return;
                    }}
 
                    const p = sequence[step];
                    if (p.x !== undefined && p.x !== null && p.x !== currentX) {{
                        currentX = p.x;
                        Plotly.relayout('chart_{element_id}', {{ 'shapes': [{{ type: 'line', x0: p.x, y0: 0, x1: p.x, y1: 1, xref: 'x', yref: 'paper', line: {{color: 'yellow', width: 2}} }}] }});
                    }}
 
                    document.getElementById('lbl_{element_id}').innerHTML = "$" + p.price.toFixed(2) + " | VIX " + p.vix.toFixed(1);
 
                    const time = audioCtx.currentTime;
                    osc.type = p.timbre;
                    osc.frequency.setValueAtTime(p.pitch, time);
 
                    step++;
                }}, parseInt('{custom_tick_ms}'));
            }});
        </script>
    </body>
    </html>
    """
 
@st.cache_data(ttl=3600)
def compute_geiger_simulator_data(base_hz, v_mult, vol_timbre):
    df = yf.download(["CL=F", "^VIX"], start="2026-03-05", end="2026-03-11", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{c2}_{c1}" if c1 else c2 for c1, c2 in df.columns]
 
    cl_close = df["CL=F_Close"].ffill().dropna()
    cl_high  = df["CL=F_High"].ffill().dropna()
    cl_low   = df["CL=F_Low"].ffill().dropna()
    cl_vol   = df["CL=F_Volume"].fillna(0)
    vix      = df["^VIX_Close"].ffill()
    max_vol  = cl_vol.max() or 1.0
 
    seq = []; viz_d = []; viz_p = []
    for date in cl_close.index:
        c = float(cl_close.loc[date]); h = float(cl_high.loc[date])
        l = float(cl_low.loc[date]);  v = float(cl_vol.loc[date])
        vx = float(vix[vix.index <= date].iloc[-1]) if not vix[vix.index <= date].empty else 20.0
 
        num_ticks = max(1, int((v / max_vol) * 50 * v_mult))
        viz_d.append(date.strftime("%m-%d %H:%M"))
        viz_p.append(c)
 
        ticks = np.random.uniform(l, h, num_ticks).tolist()
        for idx, t_p in enumerate(ticks):
            seq.append({
                "x": date.strftime("%m-%d %H:%M") if idx == 0 else None,
                "pitch": float(base_hz) + (t_p - 60.0) * 5.0,
                "vix": float(vx),
                "timbre": str(vol_timbre) if vx > 25 else "sine",
                "price": float(t_p)
            })
 
    traces = [{"x": viz_d, "y": viz_p, "mode": "lines", "name": "WTI 15m", "line": {"color": "cyan"}}]
    return traces, seq
 
try:
    g_traces, g_seq = compute_geiger_simulator_data(custom_base_hz, custom_v_mult, custom_timbre)
    components.html(build_geiger_simulator_wrapper(
        "geiger",
        "Order Flow — 15m Bars Subdivided by Volume",
        json.dumps(g_traces), json.dumps(g_seq),
        "▶ Start Geiger", custom_tick_ms, height=450
    ), height=500)
except Exception as e:
    st.warning(f"Geiger simulator failed: {e}")
