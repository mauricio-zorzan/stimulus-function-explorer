"""
AI-powered search functionality for stimulus functions using OpenAI API.
"""

import os
import json
import base64
from typing import List, Dict, Optional, Union
import openai
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()


class AISearchEngine:
    """AI-powered search engine for finding stimulus functions based on descriptions."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the AI search engine.

        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly."
            )

        # Initialize OpenAI client (new API format)
        self.client = openai.OpenAI(api_key=self.api_key)

        # Cache for function descriptions to avoid repeated API calls
        self._descriptions_cache = None
        self._cache_file = Path("data/ai_search_cache.json")

    def _load_function_descriptions(self) -> List[Dict]:
        """Load function descriptions from the data directory."""
        if self._descriptions_cache is not None:
            return self._descriptions_cache

        functions_dir = Path("data/functions")
        if not functions_dir.exists():
            return []

        descriptions = []
        for json_file in functions_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    function_data = json.load(f)

                # Extract relevant information for AI search
                description_entry = {
                    "function_name": function_data.get("function_name", ""),
                    "description": function_data.get("description", ""),
                    "category": function_data.get("category", ""),
                    "tags": function_data.get("tags", []),
                    "parameters": function_data.get("parameters", []),
                    "example_usage": function_data.get("example_usage", ""),
                }
                descriptions.append(description_entry)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {json_file}: {e}")
                continue

        self._descriptions_cache = descriptions
        return descriptions

    def _create_search_prompt(
        self, query: str, function_descriptions: List[Dict]
    ) -> str:
        """Create a prompt for OpenAI to find relevant functions."""

        # Convert function descriptions to JSON format for the prompt
        functions_json = json.dumps(function_descriptions, indent=2)

        prompt = f"""You are an AI assistant that identifies the most relevant stimulus-drawing functions for a user-provided text query.
Your job is to analyze the query and match it to the function catalog.

**Inputs**

Text Query: "{query}"

Function Catalog (JSON list of objects): 
{functions_json}

Each object has at least: function_name, description, category, tags, and may include: parameters, example_usage, etc.

**Step 1 — Understand the query**

Parse the user's text query to identify:
- Mathematical concepts (fractions, place value, geometry, etc.)
- Visual elements (grids, bars, circles, number lines, graphs, etc.)
- Educational context (grade level, standards, etc.)
- Specific features (labels, measurements, colors, etc.)

**Step 2 — Normalize terms (synonym map)**

When comparing the query to functions, treat the following as equivalent:

- Base ten blocks ≈ place value blocks, Dienes blocks, base-10 blocks, flats/rods/units, hundreds/tens/ones
- Flat ≈ hundreds square, 10×10 grid
- Rod/Long ≈ tens bar, 1×10 bar
- Unit/Cube ≈ ones square, 1×1
- Fraction models ≈ fraction bars, area models (pie/circle), partitioned rectangles
- Number line ≈ line with ticks/intervals
- Graph ≈ chart, plot, visualization
- Geometric shapes ≈ polygons, figures, 2D shapes
- 3D shapes ≈ solids, prisms, polyhedra

(You may also use tags, category names, and description text as synonyms.)

**Step 3 — Scoring rules (additive; higher is better)**

Score each function using these weights:

**Exact/near-exact matches:**
- Function function_name or tags contain normalized key terms from query: +4
- description contains normalized key terms: +3
- category contains normalized key terms: +2

**Semantic relevance:**
- Query and function are in the same mathematical domain (e.g., both about fractions): +3
- Query mentions a visual type that matches function's output (e.g., "graph" matches graphing functions): +3

**Category relevance:**
- Matching high-level category (e.g., arithmetic, geometry, statistics): +2

**Anti-confusion penalties (apply when concepts conflict):**
- Query asks for fractions but function is about place value: −3
- Query asks for 2D shapes but function draws 3D: −2
- Query asks for graphs but function draws geometric shapes: −2

**Tie-breakers:**
- Function has more tags that match query concepts: +1
- Function example_usage is relevant to query: +1

**Step 4 — Select and order results**

Compute a score for each function; select the top 10 with score > 0, ordered by score (desc).

If nothing scores > 0, return the top 5 by semantic similarity only.

**Output (machine-readable, with backward compatibility)**

Return an object with both a ranked list and a simple list. Keep names exactly as in the catalog.

{{
  "ranked": [
    {{"name": "function_name_1", "score": 12, "reasons": ["tags match: base_ten_blocks", "description contains key terms"]}},
    {{"name": "function_name_2", "score": 8, "reasons": ["category: arithmetic", "semantic match"]}}
  ],
  "functions": ["function_name_1", "function_name_2"]
}}

**IMPORTANT:** Always return valid JSON in the exact format above. The "functions" array must contain only function names that exist in the catalog."""

        return prompt

    def search_functions(self, query: str, max_results: int = 10) -> List[str]:
        """Search for functions using AI based on the query.

        Args:
            query: The search query (e.g., "functions with rectangles")
            max_results: Maximum number of results to return

        Returns:
            List of function names ordered by relevance
        """
        if not query.strip():
            return []

        try:
            # Load function descriptions
            function_descriptions = self._load_function_descriptions()
            if not function_descriptions:
                return []

            # Create the search prompt
            prompt = self._create_search_prompt(query, function_descriptions)

            # Make API call to OpenAI (new API format)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that finds relevant stimulus functions based on user queries. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.1,
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                parsed_response = json.loads(response_text)

                # New format: {"ranked": [...], "functions": [...]}
                if isinstance(parsed_response, dict) and "functions" in parsed_response:
                    function_names = parsed_response["functions"]
                # Old format: ["function_name_1", "function_name_2"]
                elif isinstance(parsed_response, list):
                    function_names = parsed_response
                else:
                    function_names = []

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract function names from text
                function_names = self._extract_function_names_from_text(response_text)

            # Filter to only include valid function names and limit results
            valid_functions = [
                name for name in function_names if self._is_valid_function_name(name)
            ]
            return valid_functions[:max_results]

        except Exception as e:
            print(f"Error in AI search: {e}")
            return []

    def _extract_function_names_from_text(self, text: str) -> List[str]:
        """Extract function names from text response if JSON parsing fails."""
        # Simple extraction - look for function names in quotes or after colons
        import re

        # Pattern to match function names (typically start with draw_, create_, generate_)
        pattern = r"\b(draw_|create_|generate_)[a-zA-Z_]+"
        matches = re.findall(pattern, text)

        # Also look for quoted strings that might be function names
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)

        # Combine and deduplicate
        all_matches = list(set(matches + quoted_matches))
        return all_matches

    def _is_valid_function_name(self, name: str) -> bool:
        """Check if a function name is valid by checking if it exists in our data."""
        if not isinstance(name, str) or not name.strip():
            return False

        # Check if the function file exists
        function_file = Path("data/functions") / f"{name.strip()}.json"
        return function_file.exists()

    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on the query."""
        if len(query) < 2:
            return []

        # Load function descriptions
        function_descriptions = self._load_function_descriptions()

        # Simple suggestion based on partial matches
        suggestions = []
        query_lower = query.lower()

        for func in function_descriptions:
            # Check function name
            if query_lower in func["function_name"].lower():
                suggestions.append(func["function_name"])

            # Check description
            if query_lower in func["description"].lower():
                suggestions.append(func["function_name"])

            # Check tags
            for tag in func["tags"]:
                if query_lower in tag.lower():
                    suggestions.append(func["function_name"])

        # Remove duplicates and limit
        return list(set(suggestions))[:5]

    def _encode_image_to_base64(
        self, image_input: Union[str, Path, Image.Image, bytes]
    ) -> str:
        """Encode an image to base64 for API transmission.

        Args:
            image_input: Can be a file path, PIL Image, or bytes

        Returns:
            Base64 encoded string of the image
        """
        if isinstance(image_input, (str, Path)):
            # Read from file
            with open(image_input, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _create_image_search_prompt(
        self, visual_signature: str, function_descriptions: List[Dict]
    ) -> str:
        """Create a prompt for matching visual signature to functions."""

        # Convert function descriptions to JSON format
        functions_json = json.dumps(function_descriptions, indent=2)

        prompt = f"""You are an AI assistant that identifies the most relevant stimulus-drawing functions for a user-provided image visual signature.
Your job is to analyze the visual signature and match it to the function catalog.

**Inputs**

Visual Signature: "{visual_signature}"

Function Catalog (JSON list of objects): 
{functions_json}

Each object has at least: function_name, description, category, tags, and may include: parameters, example_usage, etc.

**Step 1 — Normalize terms (synonym map)**

When comparing the visual signature to functions, treat the following as equivalent:

- Base ten blocks ≈ place value blocks, Dienes blocks, base-10 blocks, flats/rods/units, hundreds/tens/ones
- Flat ≈ hundreds square, 10×10 grid
- Rod/Long ≈ tens bar, 1×10 bar
- Unit/Cube ≈ ones square, 1×1
- Fraction models ≈ fraction bars, area models (pie/circle), partitioned rectangles
- Number line ≈ line with ticks/intervals
- Graph ≈ chart, plot, visualization
- Coordinate plane ≈ xy-axis, Cartesian plane
- Geometric shapes ≈ polygons, figures, 2D shapes
- 3D shapes ≈ solids, prisms, polyhedra
- Grid ≈ lattice, array, matrix of squares

**Step 2 — Scoring rules (additive; higher is better)**

Score each function using these weights:

**Exact/near-exact matches:**
- Function function_name or tags contain normalized key terms from visual signature: +4
- description contains normalized key terms: +3
- category contains normalized key terms: +2

**Visual structure alignment (from visual signature):**
- Evidence of 10×10 grid + 1×10 bars + 1×1 units: +4
- Repeated discrete unit squares arranged in place-value groupings: +3
- Split-shape fraction cues (single shape partitioned with a part/whole emphasis): +3
- Coordinate plane with plotted elements (points, lines, graphs): +4
- Number line with ticks/intervals: +3
- Geometric shapes with measurements or labels: +3

**Anti-confusion penalties (apply when structures conflict):**
- Image shows flats/rods/units but function is a fraction model (pie/bar/area with partitions): −4
- Image shows fraction partitions but function is base-ten: −4
- Generic "grid" functions that lack place-value semantics when base-ten is evident: −2
- Image shows 2D shapes but function draws 3D objects: −2
- Image shows graphs/plots but function draws geometric shapes: −3

**Tie-breakers:**
- Prefer functions whose tags include specific visual elements mentioned: +2
- Prefer functions with category matching the visual domain: +1

**Step 3 — Select and order results**

Compute a score for each function; select the top 8 with score > 0, ordered by score (desc).

If nothing scores > 0, return the top 3 by semantic similarity only.

**Output (machine-readable, with backward compatibility)**

Return an object with both a ranked list and a simple list. Keep names exactly as in the catalog.

{{
  "ranked": [
    {{"name": "function_name_1", "score": 12, "reasons": ["10×10 grid + rods + units match", "tags: base_ten_blocks, place_value"]}},
    {{"name": "function_name_2", "score": 8, "reasons": ["category: arithmetic/place value", "grid + unit squares"]}}
  ],
  "functions": ["function_name_1", "function_name_2"]
}}

**IMPORTANT:** Always return valid JSON in the exact format above. The "functions" array must contain only function names that exist in the catalog."""

        return prompt

    def analyze_image(self, image_input: Union[str, Path, Image.Image, bytes]) -> Dict:
        """Analyze an image and extract structured visual features using OpenAI Vision API.

        Args:
            image_input: The image to analyze (file path, PIL Image, or bytes)

        Returns:
            Dictionary with visual_signature, families, primitives, patterns, counts, detected_text
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_input)

            # Define the visual parsing schema
            visual_schema = {
                "name": "VisualFeatureExtraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "visual_signature": {"type": "string"},
                        "families": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "AREA_MODEL",
                                    "BAR_CHART",
                                    "BOX_PLOT",
                                    "CIRCLE_GRAPH",
                                    "COORDINATE_PLANE",
                                    "DOT_PLOT",
                                    "FRACTION_MODEL",
                                    "FUNCTION_GRAPH",
                                    "GEOMETRY_2D",
                                    "GEOMETRY_3D",
                                    "GRID_GENERIC",
                                    "HISTOGRAM",
                                    "LINE_PLOT",
                                    "MEASUREMENT_DIAGRAM",
                                    "NUMBER_LINE",
                                    "PLACE_VALUE_MODEL",
                                    "SET_MODEL",
                                    "TABLE",
                                    "TREE_DIAGRAM",
                                    "TALLY_CHART",
                                ],
                            },
                        },
                        "primitives": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "POINT",
                                    "LINE_SEGMENT",
                                    "RAY",
                                    "ARROW",
                                    "AXIS_X",
                                    "AXIS_Y",
                                    "TICK_MARK",
                                    "RECTANGLE",
                                    "SQUARE",
                                    "CIRCLE",
                                    "ARC",
                                    "SECTOR",
                                    "BAR",
                                    "GRID",
                                    "BRACE",
                                    "POLYGON",
                                    "ANGLE_MARK",
                                    "HASH_MARK",
                                    "TEXT_LABEL",
                                    "CUBE",
                                    "PRISM",
                                    "CYLINDER",
                                ],
                            },
                        },
                        "patterns": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "GRID_MxN",
                                    "PARTITION_EQUAL",
                                    "PARTITION_UNEQUAL",
                                    "STACKED_BARS",
                                    "GROUPS_OF_N",
                                    "PARALLEL_LINES",
                                    "INTERSECTING_LINES",
                                    "SYMMETRY",
                                    "COORDINATE_TICKS",
                                    "NUMBER_LINE_ZERO_MARKED",
                                    "AXES_LABELED",
                                ],
                            },
                        },
                        "counts": {
                            "type": "object",
                            "properties": {
                                "bars": {"type": "integer"},
                                "sectors": {"type": "integer"},
                                "grid_rows": {"type": "integer"},
                                "grid_cols": {"type": "integer"},
                                "points": {"type": "integer"},
                                "lines": {"type": "integer"},
                            },
                            "required": [
                                "bars",
                                "sectors",
                                "grid_rows",
                                "grid_cols",
                                "points",
                                "lines",
                            ],
                            "additionalProperties": False,
                        },
                        "detected_text": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "visual_signature",
                        "families",
                        "primitives",
                        "patterns",
                        "counts",
                        "detected_text",
                    ],
                    "additionalProperties": False,
                },
            }

            # System message
            system_message = "You are a vision parser for educational math stimuli. Return ONLY JSON that matches the schema."

            # User instructions
            user_instructions = """Goal: extract general, model-agnostic visual features.

Instructions:
- Choose families broadly (e.g., NUMBER_LINE, COORDINATE_PLANE, GEOMETRY_2D, PLACE_VALUE_MODEL).
- List all visible primitives (RECTANGLE, SQUARE, GRID, AXIS_X, TEXT_LABEL, etc.).
- Use patterns when obvious (GRID_MxN, PARTITION_EQUAL, COORDINATE_TICKS, GROUPS_OF_N).
- If unsure, include GRID_GENERIC or TABLE rather than guessing pedagogy.
- Keep visual_signature ≤ 250 chars. No explanations beyond the signature.
- Count bars, sectors, grid dimensions, points, lines accurately. Use 0 if none present.
- Extract any visible text labels into detected_text array.

Return ONLY JSON matching the schema."""

            # Use OpenAI Vision API to analyze the image
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_schema", "json_schema": visual_schema},
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_instructions},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
            )

            # Parse the guaranteed valid JSON response
            response_content = response.choices[0].message.content
            parsed_result = json.loads(response_content)
            return parsed_result

        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                "visual_signature": "",
                "families": [],
                "primitives": [],
                "patterns": [],
                "counts": {
                    "bars": 0,
                    "sectors": 0,
                    "grid_rows": 0,
                    "grid_cols": 0,
                    "points": 0,
                    "lines": 0,
                },
                "detected_text": [],
            }

    def compare_images(
        self,
        user_image: Union[str, Path, Image.Image, bytes],
        candidate_image_path: str,
    ) -> Dict[str, any]:
        """Compare user image with a candidate function image.

        Args:
            user_image: The user's uploaded image
            candidate_image_path: Path to a candidate function image

        Returns:
            Dictionary with similarity score and reasoning
        """
        try:
            # Encode both images
            user_image_b64 = self._encode_image_to_base64(user_image)
            candidate_path = Path("data") / candidate_image_path

            if not candidate_path.exists():
                return {"similarity_score": 0, "reasoning": "Candidate image not found"}

            candidate_image_b64 = self._encode_image_to_base64(candidate_path)

            # Use OpenAI Vision to compare
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Compare these two educational stimulus images. 
First image is the user's query image, second is a candidate match.

Rate their similarity on a scale of 0-100 based on:
1. Type of visualization (e.g., both are bar graphs, number lines, etc.)
2. Mathematical concepts shown
3. Visual structure and layout
4. Educational purpose

Respond ONLY with a JSON object in this exact format:
{
  "similarity_score": <0-100>,
  "reasoning": "<brief explanation of why they match or don't match>"
}""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{user_image_b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{candidate_image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.1,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Try to extract score from text if JSON parsing fails
                import re

                score_match = re.search(r"(\d+)", response_text)
                if score_match:
                    return {
                        "similarity_score": int(score_match.group(1)),
                        "reasoning": response_text,
                    }
                return {"similarity_score": 0, "reasoning": response_text}

        except Exception as e:
            print(f"Error comparing images: {e}")
            return {"similarity_score": 0, "reasoning": f"Error: {str(e)}"}

    def search_by_image(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        max_candidates: int = 10,
        max_results: int = 5,
        similarity_threshold: int = 40,
        use_shortlist: bool = True,
        shortlist_size: int = 30,
    ) -> List[Dict]:
        """Search for functions by analyzing an uploaded image using structured output.

        This uses OpenAI's JSON schema feature for guaranteed valid responses.
        The process:
        1. Extract structured visual features from image (families, primitives, patterns, counts)
        2. Optionally shortlist functions using embedding similarity for efficiency
        3. Match visual features to function catalog using detailed scoring rules
        4. Optionally compare user image with top candidate images for visual verification

        Args:
            image_input: The user's uploaded image
            max_candidates: Maximum number of candidates to compare visually (Stage 2)
            max_results: Maximum number of final results to return
            similarity_threshold: Minimum similarity score (0-100) to include in results
            use_shortlist: Whether to use embedding-based shortlisting for efficiency
            shortlist_size: Number of functions to shortlist before detailed matching

        Returns:
            List of dictionaries with function_name, similarity_score, reasoning, ai_score, ai_reasons
        """
        try:
            # Get structured visual features from image
            print("=" * 60)
            print("STAGE 1: Extracting visual features from image...")
            print("=" * 60)
            visual_features = self.analyze_image(image_input)

            if not visual_features or not visual_features.get("visual_signature"):
                print("❌ Failed to extract visual features")
                return []

            # Log extracted features
            print(f"✓ Visual signature: {visual_features['visual_signature']}")
            print(
                f"✓ Families detected: {', '.join(visual_features.get('families', [])) or 'None'}"
            )
            print(
                f"✓ Primitives: {', '.join(visual_features.get('primitives', [])[:8])}..."
            )
            print(
                f"✓ Patterns: {', '.join(visual_features.get('patterns', [])) or 'None'}"
            )
            counts = visual_features.get("counts", {})
            print(
                f"✓ Counts: grid={counts.get('grid_rows', 0)}x{counts.get('grid_cols', 0)}, "
                + f"bars={counts.get('bars', 0)}, sectors={counts.get('sectors', 0)}, "
                + f"points={counts.get('points', 0)}, lines={counts.get('lines', 0)}"
            )

            # Load function descriptions
            all_function_descriptions = self._load_function_descriptions()
            if not all_function_descriptions:
                return []

            # Optional: Shortlist using semantic similarity for efficiency
            if use_shortlist and len(all_function_descriptions) > shortlist_size:
                print("\n" + "=" * 60)
                print(f"STAGE 2: Shortlisting top {shortlist_size} candidates...")
                print("=" * 60)

                # Create search query from visual features
                search_terms = []
                search_terms.append(visual_features["visual_signature"])
                search_terms.extend(visual_features.get("families", []))
                search_terms.extend(visual_features.get("primitives", [])[:10])
                search_terms.extend(visual_features.get("patterns", []))
                search_query = " ".join(search_terms)

                print(f"Search query: {search_query[:200]}...")

                # Use existing text search to shortlist
                shortlist_names = self.search_functions(
                    search_query, max_results=shortlist_size
                )

                # Filter to shortlisted functions
                function_descriptions = [
                    f
                    for f in all_function_descriptions
                    if f["function_name"] in shortlist_names
                ]
                print(
                    f"✓ Shortlisted {len(function_descriptions)} candidates from {len(all_function_descriptions)} total functions"
                )
                print(f"  Shortlist: {', '.join(shortlist_names[:5])}...")
            else:
                function_descriptions = all_function_descriptions
                print(
                    f"\n✓ Using all {len(function_descriptions)} functions (shortlist disabled)"
                )

            # Convert to JSON strings
            features_json = json.dumps(visual_features, indent=2)
            functions_json = json.dumps(function_descriptions, indent=2)
            # Truncate if too large (keep first 100k chars)
            if len(functions_json) > 100000:
                print(f"⚠ Truncating catalog from {len(functions_json)} to 100k chars")
                functions_json = functions_json[:100000]

            # Create enum of valid function names for strict validation
            enum_names = sorted([f["function_name"] for f in function_descriptions])
            print(f"✓ Created enum with {len(enum_names)} valid function names")

            # Define the JSON schema for structured output with enum validation
            schema = {
                "name": "StimulusFunctionMatches",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "score", "reasons"],
                                "properties": {
                                    "name": {"type": "string", "enum": enum_names},
                                    "score": {"type": "number"},
                                    "reasons": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "functions": {
                            "type": "array",
                            "items": {"type": "string", "enum": enum_names},
                        },
                    },
                    "required": ["ranked", "functions"],
                    "additionalProperties": False,
                },
            }

            # System message
            system_message = "You are a matcher. Return only JSON that conforms to the provided schema."

            # User instructions with improved matching prompt
            user_instructions = f"""Input:
- features (from vision): {features_json}
- function catalog (JSON list): {functions_json}

Goal: rank the most relevant functions using general visual evidence. Be conservative about families: prefer clear visual alignment over loose semantics.

Scoring (additive):
1) Name/Tags/Description
   +4 if function_name OR tags include any family/primitives/patterns terms from features
   +3 if description includes them
   +1 if category broadly matches (statistics/geometry/arithmetic/measurement)

2) Family alignment (choose all that apply)
   +5 if function's primary family equals a features.family (e.g., NUMBER_LINE, COORDINATE_PLANE, BAR_CHART, FRACTION_MODEL, GEOMETRY_2D/3D, TABLE, TREE_DIAGRAM, SET_MODEL, PLACE_VALUE_MODEL, AREA_MODEL, FUNCTION_GRAPH)
   +2 if function lists that family as secondary or in tags

3) Primitive & pattern overlap
   +1 for each overlapping primitive up to +5
   +1 for each overlapping pattern up to +4
   +1 if counts suggest similar structure (e.g., GRID_MxN, many points/lines)

4) Parameter-type hints (if available)
   +2 if function parameters mention elements that match features (e.g., ticks, axes, partitions, sectors, bars, polygons)

Mutual-exclusion penalties (apply all that fit):
   -5 if families are contradictory (e.g., FRACTION_MODEL vs NUMBER_LINE; BAR_CHART vs COORDINATE_PLANE unless chart-on-axes is explicit)
   -4 if function relies on 3D solids but features suggest strictly 2D geometry or charts
   -3 if function is a TABLE but features show a plotted/diagrammatic visualization
   -2 if function is a generic GRID but a more specific family is evident from features

Tie-breakers:
   +2 if example images/metadata (images, file_patterns) mention the same families/patterns
   +1 if educational_standards or tags directly reference the visualization type

Selection:
- Compute scores for all catalog items; return top 8 with score > 0, sorted desc.
- If none > 0, return top 3 by textual similarity to features.visual_signature and families.

Output: conform to the provided JSON schema. No prose outside JSON."""

            print("\n" + "=" * 60)
            print(
                f"STAGE 3: Matching features to {len(function_descriptions)} functions..."
            )
            print("=" * 60)
            print("Using detailed scoring with:")
            print("  • Name/Tags/Description matching (+1 to +4)")
            print("  • Family alignment (+2 to +5)")
            print("  • Primitive & pattern overlap (+1 each)")
            print("  • Mutual-exclusion penalties (-2 to -5)")

            # Make API call with structured output (text only, no image needed)
            # Using temperature=0 and seed for deterministic results
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                seed=42,  # Deterministic seed
                max_tokens=1000,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_instructions},
                ],
            )

            # Parse the guaranteed valid JSON response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)

            # Extract candidate function names from the response
            candidate_function_names = parsed_response.get("functions", [])
            ranked_results = parsed_response.get("ranked", [])

            # Log AI matching results
            print(
                f"\n✓ AI Matching complete! Found {len(ranked_results)} scored functions"
            )
            print("\nTop candidates from AI matching:")
            for i, result in enumerate(ranked_results[:5], 1):
                print(f"  {i}. {result['name']} (score: {result['score']})")
                if result.get("reasons"):
                    print(f"     Reasons: {'; '.join(result['reasons'][:2])}")

            # Filter to valid functions and limit to max_candidates
            valid_candidates = []
            for func_name in candidate_function_names:
                if self._is_valid_function_name(func_name):
                    valid_candidates.append(func_name)
                    if len(valid_candidates) >= max_candidates:
                        break

            if not valid_candidates:
                print("\n❌ No valid candidate functions found from matching")
                return []

            print(
                f"\n✓ Selected {len(valid_candidates)} candidates for visual comparison"
            )

            # Stage 4: Compare user image with candidate function images
            print("\n" + "=" * 60)
            print(
                f"STAGE 4: Visual comparison with top {len(valid_candidates)} candidates..."
            )
            print("=" * 60)
            results = []

            for i, func_name in enumerate(valid_candidates, 1):
                # Load function data to get image paths
                function_file = Path("data/functions") / f"{func_name}.json"
                if not function_file.exists():
                    continue

                with open(function_file, "r") as f:
                    function_data = json.load(f)

                # Get the first image for comparison
                images = function_data.get("images", [])
                if not images:
                    continue

                # Compare with the first image (could be enhanced to check multiple)
                first_image_path = images[0].get("path", "")
                if not first_image_path:
                    continue

                print(
                    f"[{i}/{len(valid_candidates)}] Comparing with {func_name}...",
                    end=" ",
                )
                comparison_result = self.compare_images(image_input, first_image_path)

                similarity_score = comparison_result.get("similarity_score", 0)
                reasoning = comparison_result.get("reasoning", "")

                # Find score and reasons from ranked results
                score_info = next(
                    (r for r in ranked_results if r.get("name") == func_name), {}
                )
                ai_score = score_info.get("score", 0)
                ai_reasons = score_info.get("reasons", [])

                print(f"similarity: {similarity_score}% (AI score: {ai_score})")

                if similarity_score >= similarity_threshold:
                    results.append(
                        {
                            "function_name": func_name,
                            "similarity_score": similarity_score,
                            "reasoning": reasoning,
                            "ai_score": ai_score,
                            "ai_reasons": ai_reasons,
                            "visual_features": visual_features,  # Keep features for debugging
                        }
                    )

            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Log final results
            print("\n" + "=" * 60)
            print(
                f"FINAL RESULTS: {len(results)} matches above {similarity_threshold}% threshold"
            )
            print("=" * 60)
            for i, result in enumerate(results[:max_results], 1):
                print(f"\n{i}. {result['function_name']}")
                print(f"   Visual Similarity: {result['similarity_score']}%")
                print(f"   AI Matching Score: {result['ai_score']}")
                if result.get("ai_reasons"):
                    print(f"   AI Reasons: {', '.join(result['ai_reasons'][:2])}")

            if not results:
                print("❌ No functions met the similarity threshold")

            print("=" * 60 + "\n")

            # Return top results
            return results[:max_results]

        except Exception as e:
            print(f"Error in image search: {e}")
            return []
