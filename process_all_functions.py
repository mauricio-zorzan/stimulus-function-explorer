#!/usr/bin/env python3
"""
Simple script to process all functions with AI descriptions.
This will update all function metadata with comprehensive AI-generated descriptions.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Process all functions with AI descriptions."""
    print("🚀 Starting AI description generation for all functions...")
    print("This will analyze each function and generate detailed descriptions.")
    print()

    # Check if OpenAI API key is set
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print(
            "Please set your OpenAI API key in the .env file or environment variables."
        )
        return

    print("✅ OpenAI API key found")
    print()

    # Run the AI description generator
    try:
        result = subprocess.run(
            [sys.executable, "generate_ai_descriptions.py"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ AI description generation completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Error during AI description generation:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ Error running AI description generator: {e}")


if __name__ == "__main__":
    main()
