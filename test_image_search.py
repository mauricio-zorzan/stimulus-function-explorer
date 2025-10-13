"""
Test script for image search functionality.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_search import AISearchEngine


def test_image_analysis():
    """Test basic image analysis."""
    print("=" * 60)
    print("TEST 1: Image Analysis")
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
        test_images = list(images_dir.glob("*.webp"))[:1]

        if not test_images:
            print("❌ No test images found in data/images/")
            return False

        test_image = test_images[0]
        print(f"✓ Using test image: {test_image.name}")

        # Analyze the image
        print("Analyzing image (this may take 10-15 seconds)...")
        visual_features = engine.analyze_image(test_image)
        
        if visual_features and visual_features.get("visual_signature"):
            print("✓ Image analysis successful!")
            print(f"Visual Signature: {visual_features['visual_signature']}")
            print(f"Families: {', '.join(visual_features.get('families', []))}")
            print(f"Primitives: {', '.join(visual_features.get('primitives', [])[:5])}...")
            print(f"Patterns: {', '.join(visual_features.get('patterns', []))}")
            counts = visual_features.get('counts', {})
            print(f"Counts: grid={counts.get('grid_rows', 0)}x{counts.get('grid_cols', 0)}, bars={counts.get('bars', 0)}, points={counts.get('points', 0)}")
            return True
        else:
            print("❌ Image analysis returned empty or invalid features")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_image_comparison():
    """Test image comparison."""
    print("\n" + "=" * 60)
    print("TEST 2: Image Comparison")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        engine = AISearchEngine()
        print("✓ AISearchEngine initialized")

        # Find two test images
        images_dir = Path("data/images")
        test_images = list(images_dir.glob("*.webp"))[:2]

        if len(test_images) < 2:
            print("❌ Need at least 2 test images")
            return False

        image1 = test_images[0]
        image2 = test_images[1]
        print(f"✓ Comparing {image1.name} with {image2.name}")

        # Compare images
        print("Comparing images (this may take 15-20 seconds)...")
        result = engine.compare_images(image1, f"images/{image2.name}")

        if result and "similarity_score" in result:
            print("✓ Image comparison successful!")
            print(f"Similarity Score: {result['similarity_score']}%")
            print(f"Reasoning: {result['reasoning']}")
            return True
        else:
            print("❌ Image comparison failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_search_by_image():
    """Test full image search workflow."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Image Search")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        engine = AISearchEngine()
        print("✓ AISearchEngine initialized")

        # Find a test image
        images_dir = Path("data/images")
        test_images = list(images_dir.glob("*.webp"))[:1]

        if not test_images:
            print("❌ No test images found")
            return False

        test_image = test_images[0]
        print(f"✓ Using test image: {test_image.name}")

        # Search by image
        print("Searching (this may take 30-60 seconds)...")
        results = engine.search_by_image(
            test_image, max_candidates=5, max_results=3, similarity_threshold=30
        )

        if results:
            print(f"✓ Search successful! Found {len(results)} matches:")
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result['function_name']}")
                print(f"     Similarity: {result['similarity_score']}%")
                print(f"     Reasoning: {result['reasoning'][:100]}...")
            return True
        else:
            print("⚠ Search completed but no matches found (this is okay)")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 Testing AI Image Search Functionality\n")

    # Check for API key first
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 60)
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("=" * 60)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr add it to a .env file in the project root.")
        return

    print("✓ OPENAI_API_KEY found\n")

    # Run tests
    results = []
    results.append(("Image Analysis", test_image_analysis()))
    results.append(("Image Comparison", test_image_comparison()))
    results.append(("Full Image Search", test_search_by_image()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All tests passed! Image search is working correctly.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
