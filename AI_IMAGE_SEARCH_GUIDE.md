# AI Image Search - User Guide

## Overview

The Stimulus Function Explorer now includes **AI-powered image search** functionality that allows users to upload an image and find similar stimulus functions in the database.

## How It Works

The AI image search uses a **two-stage search process**:

### Stage 1: Image Analysis & Description-Based Search

1. AI analyzes the uploaded image using OpenAI's Vision API (GPT-4o-mini)
2. Generates a detailed description of:
   - Type of visualization (graph, geometric shape, number line, etc.)
   - Mathematical concepts shown
   - Visual elements (colors, shapes, labels, numbers)
   - Educational purpose
3. Uses this description to search function descriptions and narrow down candidates

### Stage 2: Visual Comparison

1. Takes the top candidate functions from Stage 1
2. Compares the user's uploaded image with actual example images from each candidate function
3. Scores similarity on a 0-100 scale based on:
   - Type of visualization
   - Mathematical concepts
   - Visual structure and layout
   - Educational purpose
4. Returns only functions above the similarity threshold (default: 40%)

## How to Use

### 1. Start the App

```bash
streamlit run app_new.py
```

### 2. Select AI-Powered Search

- Click on the "🤖 AI-powered Search" radio button at the top

### 3. Choose Image Upload Mode

- Click on "🖼️ Image Upload" under the search mode options

### 4. Upload Your Image

- Click "Browse files" or drag-and-drop an image
- Supported formats: PNG, JPG, JPEG, WEBP
- The image will be displayed as a preview

### 5. Search

- Click the "🔍 Search by Image" button
- Wait 30-60 seconds for AI analysis (progress indicator shown)
- Results will appear below with similarity scores

### 6. View Results

- See similarity scores and reasoning for each match
- Click "View Similarity Scores" expander for detailed information
- Click on any function card to view full details

## What If No Matches Are Found?

If the AI returns no similar functions (or similarity scores below 40%), you'll see a message:

> "🤖 No similar functions found for your image."

This means:

- The image doesn't closely match any existing stimulus functions
- Try adjusting the image (better lighting, clearer view)
- Try a different type of educational stimulus image

## Tips for Best Results

✓ **DO:**

- Use clear, well-lit images
- Upload images of mathematical/educational content (graphs, diagrams, geometric shapes)
- Use images that show complete visualizations
- Try images similar to existing stimulus types (number lines, bar graphs, etc.)

✗ **DON'T:**

- Upload blurry or low-quality images
- Use images with text only (no visual elements)
- Upload non-educational content
- Use images that are too complex or contain multiple unrelated concepts

## Technical Requirements

### Dependencies

- OpenAI Python package (`openai >= 1.0.0`)
- PIL/Pillow for image processing
- Streamlit for the web interface

### Environment Setup

You must have an OpenAI API key set:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY='your-key-here'

# Option 2: .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
```

## API Cost Considerations

The image search uses:

- **GPT-4o-mini** for vision analysis (affordable)
- Multiple API calls per search:
  - 1 call for image analysis
  - 1 call for text-based function search
  - Up to 10 calls for image comparison (one per candidate)

Estimated cost per search: **$0.01 - $0.05** depending on:

- Image size and complexity
- Number of candidates compared
- Token usage in responses

## Customization

You can adjust search parameters in the code:

```python
image_search_results = ai_engine.search_by_image(
    pil_image,
    max_candidates=10,        # Stage 1: how many candidates to consider
    max_results=5,            # Final: how many results to return
    similarity_threshold=40   # Minimum score to include (0-100)
)
```

## Testing

Run the test suite to verify functionality:

```bash
python test_image_search.py
```

This will run three tests:

1. Image Analysis - Tests AI's ability to describe images
2. Image Comparison - Tests similarity scoring between images
3. Full Image Search - Tests the complete workflow

## Troubleshooting

### "AI search is not available"

- Check that OPENAI_API_KEY is set in your environment
- Verify the key is valid and has credits

### "No similar functions found"

- Try lowering the similarity_threshold (currently 40)
- Upload a different, clearer image
- Check that the image contains educational/mathematical content

### Slow Performance

- Image search takes 30-60 seconds by design (multiple AI calls)
- Check your internet connection
- Reduce max_candidates to speed up Stage 2

### API Errors

- Check OpenAI API status: https://status.openai.com/
- Verify API key permissions
- Check rate limits on your OpenAI account

## Code Structure

### Key Files

- `src/ai_search.py` - AISearchEngine class with image search methods
- `app_new.py` - Streamlit UI with image upload and results display
- `test_image_search.py` - Test suite for image search functionality

### Key Methods

- `analyze_image()` - Analyzes an image and returns description
- `compare_images()` - Compares two images and returns similarity score
- `search_by_image()` - Full two-stage search workflow

## Future Enhancements

Potential improvements:

- [ ] Compare with multiple images per function (not just the first)
- [ ] Cache image analysis results to avoid re-analyzing
- [ ] Add support for sketches/hand-drawn images
- [ ] Fine-tune similarity threshold based on image type
- [ ] Add image preprocessing (resize, normalize, enhance)
- [ ] Support batch image uploads
- [ ] Add visual diff viewer to show why images match/don't match

## Support

For issues or questions:

1. Check this guide first
2. Run the test suite to verify setup
3. Check OpenAI API status and credits
4. Review error messages in the Streamlit app

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Requires:** OpenAI API access with GPT-4o-mini vision capabilities
