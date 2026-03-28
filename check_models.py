import traceback
from google import genai

try:
    client = genai.Client()
    models = client.models.list()
    supported_bidi = []
    
    print("Searching for Gemini models that support Live API (bidiGenerateContent)...")
    for m in models:
        # Safe check for generation methods depending on your exact google-genai SDK version
        methods_raw = getattr(m, 'supported_generation_methods', [])
        methods = [str(x).lower() for x in methods_raw] if methods_raw else []
        
        # If bidiGenerateContent is directly listed
        if any('bidi' in method for method in methods):
            supported_bidi.append(m.name)

    if supported_bidi:
        print("\nThe following models explicitly support the Live API:")
        for name in supported_bidi:
            print(" -", name)
    else:
        print("\nCould not explicitly detect 'bidiGenerateContent' capability via the SDK attributes.")
        print("However, here are all the 'flash' models available to your account right now:")
        for m in models:
            if "flash" in m.name.lower():
                print(" -", m.name)

except Exception as e:
    print("Failed to fetch models.")
    traceback.print_exc()
