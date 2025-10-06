"""
Script to generate AI-enhanced descriptions for all stimulus functions.
This will analyze each function and generate detailed descriptions, categories, and stimulus type specifications.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AIDescriptionGenerator:
    """Generates AI-enhanced descriptions for stimulus functions."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.drawing_functions_dir = Path("drawing_functions")

    def analyze_function_code(self, function_name: str) -> Dict:
        """Analyze the function code to understand what it does."""
        # Try different possible function file names
        possible_names = [
            function_name,
            function_name.replace("create_", "")
            .replace("draw_", "")
            .replace("generate_", ""),
            function_name.replace("create_", "draw_").replace("generate_", "draw_"),
            function_name.replace("draw_", "create_").replace("generate_", "create_"),
            # Special cases for specific mappings
            "area_models" if "area_model" in function_name else None,
            "base_ten_blocks" if "base_ten" in function_name else None,
        ]
        possible_names = [name for name in possible_names if name]  # Remove None values

        function_file = None
        for name in possible_names:
            potential_file = self.drawing_functions_dir / f"{name}.py"
            if potential_file.exists():
                function_file = potential_file
                break

        if not function_file:
            return {
                "code": "",
                "imports": [],
                "error": f"Function file not found. Tried: {possible_names}",
            }

        try:
            with open(function_file, "r") as f:
                code = f.read()

            # Extract imports and basic structure
            lines = code.split("\n")
            imports = []
            for line in lines:
                if line.strip().startswith(("import ", "from ")):
                    imports.append(line.strip())

            return {"code": code, "imports": imports, "error": None}
        except Exception as e:
            return {"code": "", "imports": [], "error": str(e)}

    def generate_ai_description(self, function_name: str, existing_data: Dict) -> Dict:
        """Generate AI-enhanced description for a function."""

        # Analyze the function code
        code_analysis = self.analyze_function_code(function_name)

        if code_analysis["error"]:
            print(
                f"Warning: Could not analyze code for {function_name}: {code_analysis['error']}"
            )
            return existing_data

        # Create prompt for AI
        prompt = f"""You are an expert in educational technology and mathematical visualization. I need you to analyze a Python function that generates educational stimulus images and provide a comprehensive description.

Function Name: {function_name}

Function Code:
```python
{code_analysis["code"][:3000]}  # Truncated for token limits
```

Current Basic Description: {existing_data.get("description", "No description available")}

Please provide a comprehensive analysis in the following JSON format:

{{
    "description": "A detailed description of what this function draws and its educational purpose (2-3 sentences)",
    "category": "The most appropriate category (e.g., 'fractions', 'geometry', 'arithmetic', 'data_visualization', 'measurement', 'algebra')",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "educational_level": "Elementary/Middle/High School/College",
    "mathematical_concepts": ["concept1", "concept2", "concept3"],
    "stimulus_type_specification": {{
        "title": "Available Images: [Descriptive Title]",
        "description": "A comprehensive description of what the images show and their educational purpose (2-3 paragraphs)",
        "specifications": [
            {{
                "section": "Section Name",
                "details": [
                    "Detail 1",
                    "Detail 2",
                    "Detail 3"
                ]
            }}
        ],
        "educational_purpose": "Detailed explanation of how this helps students learn (1-2 paragraphs)"
    }}
}}

Focus on:
1. What visual elements are drawn
2. What mathematical concepts are represented
3. How it helps students learn
4. What inputs/parameters control the output
5. Educational applications and use cases

Be specific and educational in your descriptions. The stimulus_type_specification should be detailed and follow the format of the base 10 blocks example provided."""

        try:
            print(f"    📡 Sending request to OpenAI API...")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in educational technology and mathematical visualization. Provide detailed, accurate descriptions of educational stimulus functions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            print(f"    📥 Received response from OpenAI API")

            response_text = response.choices[0].message.content.strip()

            # Try to parse the JSON response
            try:
                # Clean the response - remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                ai_data = json.loads(response_text)

                # Merge with existing data, preserving important fields
                enhanced_data = existing_data.copy()
                enhanced_data.update(
                    {
                        "description": ai_data.get(
                            "description", existing_data.get("description", "")
                        ),
                        "category": ai_data.get(
                            "category", existing_data.get("category", "general")
                        ),
                        "tags": ai_data.get("tags", existing_data.get("tags", [])),
                        "educational_level": ai_data.get(
                            "educational_level", "Elementary/Middle/High School/College"
                        ),
                        "mathematical_concepts": ai_data.get(
                            "mathematical_concepts", []
                        ),
                        "stimulus_type_specification": ai_data.get(
                            "stimulus_type_specification", {}
                        ),
                        "ai_generated": True,
                        "last_updated": "2025-01-06T00:00:00.000000",  # Update timestamp
                    }
                )

                return enhanced_data

            except json.JSONDecodeError as e:
                print(f"Error parsing AI response for {function_name}: {e}")
                print(f"Response: {response_text[:500]}...")
                return existing_data

        except Exception as e:
            print(f"Error generating AI description for {function_name}: {e}")
            return existing_data

    def process_all_functions(self, dry_run: bool = False) -> Dict:
        """Process all functions and generate AI descriptions."""
        if not self.functions_dir.exists():
            print("Functions directory not found!")
            return {}

        results = {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}

        function_files = list(self.functions_dir.glob("*.json"))
        print(f"Found {len(function_files)} function files to process")
        print("=" * 50)

        for i, function_file in enumerate(function_files, 1):
            function_name = function_file.stem
            print(f"\n[{i}/{len(function_files)}] Processing: {function_name}")
            print(f"Progress: {i / len(function_files) * 100:.1f}%")

            try:
                # Load existing data
                with open(function_file, "r") as f:
                    existing_data = json.load(f)

                # Check if already has AI-generated data
                if existing_data.get("ai_generated", False) and not dry_run:
                    print(
                        f"  ⏭️  Skipping {function_name} - already has AI-generated data"
                    )
                    results["skipped"] += 1
                    continue

                print(f"  🤖 Generating AI description for {function_name}...")
                # Generate AI description
                enhanced_data = self.generate_ai_description(
                    function_name, existing_data
                )

                if not dry_run:
                    # Save enhanced data
                    with open(function_file, "w") as f:
                        json.dump(enhanced_data, f, indent=2)
                    print(f"  ✅ Successfully updated {function_name}")
                    results["updated"] += 1
                else:
                    print(f"  🔍 Would update {function_name}")
                    results["processed"] += 1

            except Exception as e:
                print(f"  ❌ Error processing {function_name}: {e}")
                results["errors"] += 1

        return results

    def process_single_function(self, function_name: str) -> bool:
        """Process a single function."""
        function_file = self.functions_dir / f"{function_name}.json"

        if not function_file.exists():
            print(f"Function file not found: {function_file}")
            return False

        try:
            # Load existing data
            with open(function_file, "r") as f:
                existing_data = json.load(f)

            # Generate AI description
            enhanced_data = self.generate_ai_description(function_name, existing_data)

            # Save enhanced data
            with open(function_file, "w") as f:
                json.dump(enhanced_data, f, indent=2)

            print(f"✅ Successfully updated {function_name}")
            return True

        except Exception as e:
            print(f"❌ Error processing {function_name}: {e}")
            return False


def main():
    """Main function to run the AI description generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate AI-enhanced descriptions for stimulus functions"
    )
    parser.add_argument("--function", "-f", help="Process a single function by name")
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force update even if already AI-generated"
    )

    args = parser.parse_args()

    try:
        generator = AIDescriptionGenerator()

        if args.function:
            # Process single function
            success = generator.process_single_function(args.function)
            if success:
                print(f"\n🎉 Successfully processed {args.function}")
            else:
                print(f"\n❌ Failed to process {args.function}")
        else:
            # Process all functions
            print("🚀 Starting AI description generation for all functions...")
            results = generator.process_all_functions(dry_run=args.dry_run)

            print(f"\n📊 Results:")
            print(f"  Processed: {results['processed']}")
            print(f"  Updated: {results['updated']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Errors: {results['errors']}")

            if results["updated"] > 0:
                print(f"\n🎉 Successfully updated {results['updated']} functions!")

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
