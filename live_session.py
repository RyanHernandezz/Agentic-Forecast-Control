# live_session.py
import asyncio
import os
import json
import pyaudio
import numpy as np
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
    get_audio_params
)

load_dotenv()

# ---- AUDIO CONFIG ----
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

TOOL_MAP = {
    "get_full_state": get_full_state,
    "get_ensemble_weights": get_ensemble_weights,
    "get_drift_alert": get_drift_alert,
    "get_seasonal_status": get_seasonal_status,
    "get_audio_params": get_audio_params,
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
])

async def run_live_session():
    print("Loading pipeline state...")
    state = fetch_and_run_pipeline()
    PIPELINE_STATE.update(state)
    print(f"Pipeline ready. Regime: {state.get('regime')} | Drift: {state.get('drift_score', 0):.2f}z")

    client = genai.Client()
    pya = pyaudio.PyAudio()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[TOOL_DECLARATIONS],
        system_instruction=SYSTEM_PROMPT,
    )

    async with client.aio.live.connect(
        model="gemini-live-2.5-flash-preview",
        config=config
    ) as session:
        print("Live session connected. Start speaking.")

        audio_queue = asyncio.Queue()
        stop_event = asyncio.Event()

        # ---- MIC INPUT ----
        async def capture_mic():
            mic_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            try:
                while not stop_event.is_set():
                    data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, False
                    )
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
            finally:
                mic_stream.stop_stream()
                mic_stream.close()

        # ---- SPEAKER OUTPUT ----
        async def play_audio():
            speaker = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            try:
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        await asyncio.to_thread(speaker.write, chunk)
                    except asyncio.TimeoutError:
                        continue
            finally:
                speaker.stop_stream()
                speaker.close()

        # ---- RESPONSE HANDLER ----
        async def handle_responses():
            async for message in session.receive():

                # Handle function calls
                if message.tool_call:
                    responses = []
                    for fc in message.tool_call.function_calls:
                        fn = TOOL_MAP.get(fc.name)
                        if fn:
                            result = fn()
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

        # Run mic, speaker, and response handler concurrently
        try:
            await asyncio.gather(
                capture_mic(),
                play_audio(),
                handle_responses()
            )
        except KeyboardInterrupt:
            stop_event.set()
            print("\nSession ended.")
        finally:
            pya.terminate()


if __name__ == "__main__":
    asyncio.run(run_live_session())
