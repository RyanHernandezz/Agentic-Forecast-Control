# live_session.py
import asyncio
import os
import json
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pipeline import fetch_and_run_pipeline
from live_tools import (
    PIPELINE_STATE,
    get_full_state,
    get_ensemble_weights,
    get_drift_alert,
    get_seasonal_status,
    get_audio_params,
    simulate_geiger_audio
)
 
load_dotenv()
 
# ---- AUDIO CONFIG ----
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 4096
 
TOOL_MAP = {
    "get_full_state": get_full_state,
    "get_ensemble_weights": get_ensemble_weights,
    "get_drift_alert": get_drift_alert,
    "get_seasonal_status": get_seasonal_status,
    "get_audio_params": get_audio_params,
    "simulate_geiger_audio": simulate_geiger_audio,
}
 
SYSTEM_PROMPT = """You are the voice interface for a live WTI crude oil
multi-agent forecasting system. You have access to real-time pipeline state
via function calls. Rules:
- Always call a tool before answering any question about the market,
  models, regime, drift, or forecast.
- Respond in 2-3 sentences maximum using the actual numbers returned.
- Never guess or use prior knowledge about current market conditions.
- If asked what you can do, list the four tools available.
- Speak like a quant analyst, not a chatbot."""
 
TOOL_DECLARATIONS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_full_state",
        description="Returns current regime, ensemble weights, drift score, seasonal z-score, and audio parameters. Use for general status questions.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="get_ensemble_weights",
        description="Returns which model dominates the ensemble, all model weights, and current regime label.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="get_drift_alert",
        description="Returns whether macro inputs have drifted outside training distribution, the drift score, and which features are most anomalous.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="get_seasonal_status",
        description="Returns the seasonal z-score, STL residual, and whether current price action is consistent with historical WTI seasonality.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="get_audio_params",
        description="Returns the current sonification parameters including channel frequencies, tempo, timbre, and dissonance score.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="simulate_geiger_audio",
        description="Generates an actual local 3-second audio waveform that sounds like a Geiger counter representing specific 'high', 'medium', or 'low' market volatility. Call this when the user asks what the current anomalous volatility actually sounds like.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "volatility": {"type": "STRING", "description": "The volatility level: 'high', 'medium', or 'low'."}
            },
            required=["volatility"]
        )
    ),
])
 
 
async def run_live_session():
    print("Loading pipeline state...")
    state = fetch_and_run_pipeline()
    PIPELINE_STATE.update(state)
    print(f"Pipeline ready. Regime: {state.get('regime')} | Drift: {state.get('drift_score', 0):.2f}z")
 
    client = genai.Client()
 
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[TOOL_DECLARATIONS],
        system_instruction=SYSTEM_PROMPT,
    )
 
    async with client.aio.live.connect(
        model="gemini-3.1-flash-live-preview",
        config=config
    ) as session:
        print("Live session connected. Start speaking.")
 
        audio_queue = asyncio.Queue()
        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
 
        # ---- MIC INPUT ----
        async def capture_mic():
            def mic_callback(indata, frames, time, status):
                if status:
                    print(f"[Mic status: {status}]")
                audio_array = np.frombuffer(indata, dtype=np.int16)
                vol = np.abs(audio_array).mean()
                if vol > 200:
                    print(f"\r🎤 Heard something! (Vol: {int(vol)})", end="")
                audio_bytes = indata.tobytes()
                future = asyncio.run_coroutine_threadsafe(
                    session.send_realtime_input(
                        audio=types.Blob(
                            data=audio_bytes,
                            mime_type="audio/pcm;rate=16000"
                        )
                    ),
                    loop
                )
                def on_done(f):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[Error sending audio] {e}")
                future.add_done_callback(on_done)
 
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=CHUNK_SIZE,
                callback=mic_callback
            ):
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)
 
        # ---- SPEAKER OUTPUT ----
        async def play_audio():
            stream = sd.OutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=CHUNK_SIZE
            )
            stream.start()
            try:
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(
                            audio_queue.get(), timeout=0.1
                        )
                        if len(chunk) % 2 != 0:
                            chunk = chunk[:-1]  # Ensure even bytes for int16
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        await loop.run_in_executor(None, stream.write, audio_array)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"\n[Playback Error] {e}")
            finally:
                stream.stop()
                stream.close()
 
        # ---- RESPONSE HANDLER ----
        async def handle_responses():
            try:
                while not stop_event.is_set():
                    async for message in session.receive():
                        # Handle function calls
                        if message.tool_call:
                            responses = []
                            for fc in message.tool_call.function_calls:
                                fn = TOOL_MAP.get(fc.name)
                                if fn:
                                    # Support args if present
                                    args = dict(fc.args) if fc.args else {}
                                    result = fn(**args)
                                    
                                    # Handle raw python-generated audio bypass!
                                    if isinstance(result, dict) and "__audio_payload__" in result:
                                        raw_pcm = result.pop("__audio_payload__")
                                        await audio_queue.put(raw_pcm)

                                    print(f"[Tool called: {fc.name}] -> {json.dumps(result)[:120]}...")
                                else:
                                    result = {"error": f"Unknown tool: {fc.name}"}
                                responses.append(
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": json.dumps(result)}
                                    )
                                )
                            await session.send_tool_response(function_responses=responses)
        
                        # Queue audio chunks for playback
                        if message.server_content:
                            if message.server_content.model_turn:
                                for part in message.server_content.model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        await audio_queue.put(part.inline_data.data)
                                    if hasattr(part, 'text') and part.text:
                                        print(f"Agent: {part.text}")
        
                            if message.server_content.turn_complete:
                                print("[Turn complete — listening...]")
            except Exception as e:
                print(f"\n[Session Receive Error] {e}")
            finally:
                print("\n[Connection Closed]")
                stop_event.set()
 
        # Run all three concurrently
        try:
            await asyncio.gather(
                capture_mic(),
                play_audio(),
                handle_responses()
            )
        except KeyboardInterrupt:
            stop_event.set()
            print("\nSession ended.")
 
 
if __name__ == "__main__":
    asyncio.run(run_live_session())
