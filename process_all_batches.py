#!/usr/bin/env python3
"""
Helper script to process all functions in batches.
This runs the AI description generation for all functions, processing them in batches of 10.
"""

import subprocess
import sys
from pathlib import Path


def count_total_functions():
    """Count total number of functions to process."""
    functions_dir = Path("data/functions")
    return len(list(functions_dir.glob("*.json")))


def main():
    batch_size = 10
    total_functions = count_total_functions()
    total_batches = (total_functions + batch_size - 1) // batch_size  # Ceiling division

    print("=" * 80)
    print("🚀 AI DESCRIPTION GENERATION - BATCH PROCESSOR")
    print("=" * 80)
    print(f"Total functions: {total_functions}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print("=" * 80)
    print()

    input("Press Enter to start processing all batches (Ctrl+C to cancel)...")
    print()

    successful_batches = 0
    failed_batches = []

    for batch_num in range(1, total_batches + 1):
        print("\n" + "=" * 80)
        print(f"📦 STARTING BATCH {batch_num} of {total_batches}")
        print("=" * 80)

        try:
            # Run the batch
            result = subprocess.run(
                [
                    sys.executable,
                    "generate_ai_descriptions.py",
                    "--batch",
                    str(batch_num),
                    "--batch-size",
                    str(batch_size),
                    "--force",
                ],
                check=True,
                capture_output=False,  # Show output in real-time
            )

            successful_batches += 1
            print(f"\n✅ Batch {batch_num} completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Batch {batch_num} failed with error code {e.returncode}")
            failed_batches.append(batch_num)

            # Ask if we should continue
            response = input("\nContinue with next batch? (y/n): ")
            if response.lower() != "y":
                print("\n⚠️  Stopping batch processing.")
                break
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Stopping batch processing.")
            break

    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"Total batches: {total_batches}")
    print(f"Successful: {successful_batches}")
    print(f"Failed: {len(failed_batches)}")

    if failed_batches:
        print(f"\nFailed batches: {', '.join(map(str, failed_batches))}")
        print("\nTo retry failed batches, run:")
        for batch_num in failed_batches:
            print(f"  python3 generate_ai_descriptions.py --batch {batch_num} --force")
    else:
        print("\n🎉 All batches completed successfully!")

    print("=" * 80)


if __name__ == "__main__":
    main()
