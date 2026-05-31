import os
import sys
from utils import run_agent_sync

# Force utf-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def print_progress(msg):
    print(f"[PROGRESS] {msg}")

if __name__ == "__main__":
    try:
        print("Starting test...")
        result = run_agent_sync(
            use_youtube=True,
            use_drive=False,
            use_notion=True,
            user_goal="I want to learn python basics in 2 days",
            progress_callback=print_progress,
            model_name="gemini-2.5-flash"
        )
        print("\n\n==== TEST RESULT ====\n\n")
        print(result)
    except Exception as e:
        print(f"FAILED: {e}")
