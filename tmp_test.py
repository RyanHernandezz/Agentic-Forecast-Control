from pipeline import fetch_and_run_pipeline
import traceback

def main():
    try:
        data = fetch_and_run_pipeline()
        print("[SUCCESS] Pipeline executed. Map Output:", data["regime"])
    except Exception as e:
        print(f"[FAIL] Pipeline threw an exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
