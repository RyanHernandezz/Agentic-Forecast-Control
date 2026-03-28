import asyncio
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

async def test_barebones_live():
    # 1. Initialize the default Client exactly as Google officially documents
    # No forced v1alpha routing, no manual configs, no multimodal audio blobs.
    client = genai.Client()
    
    print("=== BAREBONES GEMINI LIVE API TESTER ===")
    
    # 2. Iterate through the primary models known to historically/currently support Live
    models_to_test = [
        "gemini-3.1-flash-live-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp"
    ]
    
    for model_name in models_to_test:
        print(f"\n[*] Testing model: {model_name}")
        try:
            # The most basic WebSockets initialization possible
            async with client.aio.live.connect(model=model_name) as session:
                print(" |--> Connection Established! Opening Bidi JSON Stream...")
                
                # Send a simple text string, no complex audio arrays
                print(" |--> Transmitting 'Hello' payload...")
                await session.send(input="Hello, are you receiving my websocket stream?", end_of_turn=True)
                
                print(" |--> Awaiting server response...")
                response_received = False
                
                async for chunk in session.receive():
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response_received = True
                
                if response_received:
                    print("\n[SUCCESS] The Live API WebSockets endpoint is fully operational.")
                    break
                else:
                    print("\n[WARNING] Connection survived, but server returned zero bytes.")
                    
        except Exception as e:
            # Print the exact abstracted exception
            print(f" [FAILED] Could not execute on {model_name}.")
            print(f"   Reason: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_barebones_live())
