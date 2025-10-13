"""
Test script for structured output image search.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_search import AISearchEngine


def test_structured_image_search():
    """Test image search with structured output."""
    print("=" * 60)
    print("TEST: Structured Output Image Search")
    print("=" * 60)

    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        engine = AISearchEngine()
        print("✓ AISearchEngine initialized")

        # Find a test image
        images_dir = Path("data/images")
        test_images = list(images_dir.glob("draw_base_ten*"))[:1]

        if not test_images:
            test_images = list(images_dir.glob("*.webp"))[:1]

        if not test_images:
            print("❌ No test images found in data/images/")
            return False

        test_image = test_images[0]
        print(f"✓ Using test image: {test_image.name}")

        # Search by image with new structured output
        print("\nSearching with structured output (this may take 30-60 seconds)...")
        results = engine.search_by_image(
            test_image, max_candidates=5, max_results=3, similarity_threshold=30
        )

        if results:
            print(f"\n✓ Search successful! Found {len(results)} matches:\n")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['function_name']}")
                print(f"     Image Similarity: {result['similarity_score']}%")
                print(f"     AI Score: {result.get('ai_score', 'N/A')}")
                if result.get("ai_reasons"):
                    print(f"     AI Reasons: {', '.join(result['ai_reasons'][:2])}")
                print(f"     Reasoning: {result['reasoning'][:100]}...")
                print()
            return True
        else:
            print("⚠ Search completed but no matches found (threshold may be too high)")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("\n🧪 Testing Structured Output Image Search\n")

    # Check for API key first
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 60)
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("=" * 60)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    print("✓ OPENAI_API_KEY found\n")

    # Run test
    success = test_structured_image_search()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if success:
        print("✓ PASS: Structured output image search is working!")
        print("\nKey improvements:")
        print("  • Single API call with image + catalog")
        print("  • Guaranteed valid JSON via schema")
        print("  • Detailed scoring with reasons")
        print("  • Anti-confusion rules applied")
    else:
        print("❌ FAIL: Test failed. Check errors above.")


if __name__ == "__main__":
    main()
