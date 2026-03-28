import asyncio
import os
import numpy as np
import traceback
from google import genai

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing!")
        return

    client = genai.Client(http_options={'api_version': 'v1alpha'})
    model_id = 'gemini-3.1-flash-live-preview'
    
    print("-----------------------------------")
    print("Test 1: Simple Text Connection")
    print("-----------------------------------")
    try:
        async with client.aio.live.connect(model=model_id) as session:
            await session.send(input="Hello, can you see my messages?", end_of_turn=True)
            async for chunk in session.receive():
                if getattr(chunk, 'text', None):
                    print("Gemini>", chunk.text)
        print("Test 1 Passed.")
    except Exception as e:
        print("Test 1 Failed.")
        traceback.print_exc()


    print("\n-----------------------------------")
    print("Test 2: Multimodal (Text + Audio) Connection")
    print("-----------------------------------")
    # Generate 1 second of simple 16kHz sine wave audio (a dial tone)
    t = np.linspace(0, 1, 16000, False)
    pcm_audio = np.int16(np.sin(440 * t * 2 * np.pi) * 32767).tobytes()

    try:
        async with client.aio.live.connect(model=model_id) as session:
            await session.send(input=[
                {"mime_type": "audio/pcm;rate=16000", "data": pcm_audio},
                "What sound did you just hear?"
            ], end_of_turn=True)
            
            async for chunk in session.receive():
                if getattr(chunk, 'text', None):
                    print("Gemini>", chunk.text)
        print("Test 2 Passed.")
    except Exception as e:
        print("Test 2 Failed.")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
