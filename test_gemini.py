import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from live_tools import PIPELINE_STATE, live_api_tools

def populate_mock():
    print("[*] Pushing dummy state into singleton...")
    PIPELINE_STATE.update({
        "structural_weights": {"ARIMA": 0.1, "RandomForest": 0.9},
        "regime": "transition",
        "drift_score": 1.2
    })

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not set. Aborting.")
        return

    populate_mock()
    print("[*] Initiating GenAI Client...")
    
    client = genai.Client()
    
    print("[*] Prompting Gemini 2.5 Flash equipped with the python live_tools tracking array...")
    prompt = "I need to know the active state of my models. Call the get_ensemble_weights tool and tell me what the dominant model is."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=live_api_tools,
                temperature=0.0
            )
        )
        
        # Check if Gemini invoked the tool natively through the configuration payload
        if response.function_calls:
            print("\n[SUCCESS] The routing succeeded! Gemini dynamically selected a Pipeline Tool.")
            for fc in response.function_calls:
                print(f" -> Gemini Invoked: {fc.name}")
                if fc.name == "get_ensemble_weights":
                    # In a typical execution loop, the runner executes it and passes it back. 
                    # We confirm execution natively returns our mocked payload:
                    print(f" -> Execution Yielded: {live_api_tools[0]()}")
        else:
            print("\n[WARNING] Gemini bypassed the function call and responded directly:")
            print(f" -> Response: {response.text}")
            
    except Exception as e:
        print(f"\n[FAILED] Error invoking Gemini: {e}")

if __name__ == "__main__":
    main()
