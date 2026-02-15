#!/usr/bin/env python3
import sys
import subprocess
import webbrowser
import os
import time

USAGE = """
Usage: cl <command>

Commands:
  rps       Launch CL-RPS (Rock Paper Scissors with Neural AI)
"""

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    command = sys.argv[1]

    if command == "rps":
        print("Starting CL-RPS server...")
        # Ensure we run uvicorn from the rps folder
        cwd = os.path.join(os.path.dirname(__file__), "rps")
        # Launch uvicorn in a subprocess
        subprocess.Popen(
            ["uvicorn", "main:app", "--reload"],
            cwd=cwd
        )
        # Wait a second to let server start
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")
    else:
        print(f"Unknown command: {command}")
        print(USAGE)
        sys.exit(1)

if __name__ == "__main__":
    main()