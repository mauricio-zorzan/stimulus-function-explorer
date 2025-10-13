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
        self.stimulus_descriptions_dir = Path("stimulus_descriptions")

    def get_stimulus_description(self, function_name: str) -> Dict:
        """Get the stimulus description class for a function."""
        # Convert function name to stimulus description file name
        # e.g., create_categorical_graph -> categorical_graph.py
        base_name = (
            function_name.replace("create_", "")
            .replace("draw_", "")
            .replace("generate_", "")
        )

        # Try different naming patterns
        possible_files = [
            f"{base_name}.py",
            f"{function_name}.py",
        ]

        stimulus_file = None
        for filename in possible_files:
            potential_file = self.stimulus_descriptions_dir / filename
            if potential_file.exists():
                stimulus_file = potential_file
                break

        if not stimulus_file:
            return {
                "code": "",
                "class_name": "",
                "error": f"Stimulus description file not found. Tried: {possible_files}",
            }

        try:
            with open(stimulus_file, "r") as f:
                code = f.read()

            # Extract main class name (usually ends with List)
            import re

            # Look for class definitions
            classes = re.findall(r"class (\w+)\(", code)
            main_class = next(
                (c for c in classes if c.endswith("List")),
                classes[0] if classes else "",
            )

            return {
                "code": code,
                "class_name": main_class,
                "file": stimulus_file.name,
                "error": None,
            }
        except Exception as e:
            return {"code": "", "class_name": "", "error": str(e)}

    def find_function_in_files(self, function_name: str) -> tuple:
        """Find which file contains the function and extract its code."""
        import ast

        # Search all Python files in drawing_functions
        for py_file in self.drawing_functions_dir.glob("*.py"):
            try:
                with open(py_file, "r") as f:
                    file_content = f.read()

                # Parse the file
                tree = ast.parse(file_content)

                # Look for the function definition
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        # Extract just this function's code
                        function_lines = file_content.split("\n")
                        start_line = node.lineno - 1

                        # Find the end of the function (next def or class at same indent level, or EOF)
                        end_line = len(function_lines)
                        func_indent = len(function_lines[start_line]) - len(
                            function_lines[start_line].lstrip()
                        )

                        for i in range(start_line + 1, len(function_lines)):
                            line = function_lines[i]
                            if line.strip() and not line.strip().startswith("#"):
                                line_indent = len(line) - len(line.lstrip())
                                if line_indent <= func_indent and (
                                    line.strip().startswith("def ")
                                    or line.strip().startswith("class ")
                                    or line.strip().startswith("@")
                                ):
                                    end_line = i
                                    break

                        function_code = "\n".join(function_lines[start_line:end_line])
                        return py_file, function_code

            except Exception as e:
                continue

        return None, None

    def analyze_function_code(self, function_name: str) -> Dict:
        """Analyze the function code to understand what it does."""

        # Find the function in any file
        function_file, function_code = self.find_function_in_files(function_name)

        if not function_file:
            return {
                "code": "",
                "imports": [],
                "file": "",
                "error": f"Function '{function_name}' not found in any drawing_functions file",
            }

        try:
            # Read the full file to get imports
            with open(function_file, "r") as f:
                full_code = f.read()

            # Extract imports
            lines = full_code.split("\n")
            imports = []
            for line in lines:
                if line.strip().startswith(("import ", "from ")):
                    imports.append(line.strip())

            return {
                "code": function_code,
                "imports": imports,
                "file": function_file.name,
                "error": None,
            }
        except Exception as e:
            return {"code": "", "imports": [], "file": "", "error": str(e)}

    def generate_ai_description(self, function_name: str, existing_data: Dict) -> Dict:
        """Generate AI-enhanced description for a function."""

        # Analyze the function code
        code_analysis = self.analyze_function_code(function_name)

        # Get stimulus description
        stimulus_desc = self.get_stimulus_description(function_name)

        if code_analysis["error"]:
            print(
                f"Warning: Could not analyze code for {function_name}: {code_analysis['error']}"
            )
            return existing_data

        # Build stimulus description section
        stimulus_section = ""
        if not stimulus_desc["error"]:
            stimulus_section = f"""

Stimulus Description Class ({stimulus_desc["class_name"]}):
```python
{stimulus_desc["code"][:2000]}  # Truncated for token limits
```

The function accepts input of type `{stimulus_desc["class_name"]}`. This class defines the structure and constraints of the input data."""
        else:
            print(
                f"Warning: Could not load stimulus description: {stimulus_desc['error']}"
            )

        # Create prompt for AI
        prompt = f"""You are an expert in educational technology and mathematical visualization. Analyze this Python function that generates educational stimulus images.

Function Name: {function_name}

Drawing Function Code:
```python
{code_analysis["code"][:2000]}  # Truncated for token limits
```
{stimulus_section}

Current Basic Description: {existing_data.get("description", "No description available")}

Based on the function code and stimulus description class, provide a comprehensive analysis in the following JSON format:

{{
    "description": "A detailed description of what this function draws, focusing on the visual output and its educational purpose (2-3 sentences)",
    "category": "The most appropriate category (e.g., 'fractions', 'geometry', 'arithmetic', 'data_visualization', 'measurement', 'algebra')",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "stimulus_type_specification": {{
        "title": "Available Images: [Descriptive Title]",
        "description": "A comprehensive description that includes:\n- What types of visualizations/images this function can draw\n- What inputs it accepts (based on the stimulus description class)\n- Key constraints and limits (e.g., value ranges, required fields, validation rules)\n- Educational purpose and use cases\n(2-3 paragraphs)",
        "specifications": [
            {{
                "section": "Input Parameters",
                "details": [
                    "List all key input fields from the stimulus description class",
                    "Describe what each input controls",
                    "Note any optional vs required fields"
                ]
            }},
            {{
                "section": "Constraints & Validation",
                "details": [
                    "List any value limits or ranges",
                    "Describe validation rules",
                    "Note any special conditions or requirements"
                ]
            }},
            {{
                "section": "Visual Output",
                "details": [
                    "Describe what visual elements are drawn",
                    "List variations or options available",
                    "Explain how inputs affect the output"
                ]
            }}
        ]
    }}
}}

IMPORTANT: Focus on:
1. **What it draws**: Describe the visual output clearly (e.g., "bar graphs, histograms, or picture graphs")
2. **Input parameters**: List all inputs from the stimulus description class and what they control
3. **Constraints**: Identify validation rules, limits, and requirements (e.g., "frequencies cannot exceed 35 for picture graphs")
4. **Variations**: If the function can draw different types, list them explicitly

Be specific and comprehensive. Extract constraints directly from the Pydantic validators and field descriptions in the stimulus description class."""

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

    def process_all_functions(
        self, dry_run: bool = False, batch_size: int = None, batch_number: int = None
    ) -> Dict:
        """Process all functions and generate AI descriptions."""
        if not self.functions_dir.exists():
            print("Functions directory not found!")
            return {}

        results = {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}

        function_files = sorted(list(self.functions_dir.glob("*.json")))
        total_functions = len(function_files)

        # Apply batch filtering if specified
        if batch_size and batch_number is not None:
            start_idx = (batch_number - 1) * batch_size
            end_idx = start_idx + batch_size
            function_files = function_files[start_idx:end_idx]
            print(
                f"📦 Processing batch {batch_number} ({len(function_files)} functions)"
            )
            print(
                f"   Range: {start_idx + 1} to {min(end_idx, total_functions)} of {total_functions} total"
            )
        else:
            print(f"Found {len(function_files)} function files to process")

        print("=" * 70)

        for i, function_file in enumerate(function_files, 1):
            function_name = function_file.stem
            actual_index = (
                (batch_number - 1) * batch_size + i
                if batch_size and batch_number
                else i
            )
            total_to_process = (
                total_functions if not batch_size else len(function_files)
            )

            print(f"\n[{i}/{len(function_files)}] Processing: {function_name}")
            if batch_size and batch_number:
                print(f"Overall progress: {actual_index}/{total_functions}")

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

    def process_single_function(self, function_name: str, force: bool = False) -> bool:
        """Process a single function."""
        function_file = self.functions_dir / f"{function_name}.json"

        if not function_file.exists():
            print(f"Function file not found: {function_file}")
            return False

        try:
            # Load existing data
            with open(function_file, "r") as f:
                existing_data = json.load(f)

            # Check if already has AI-generated data
            if existing_data.get("ai_generated", False) and not force:
                print(
                    f"⏭️  Skipping {function_name} - already has AI-generated data (use --force to override)"
                )
                return False

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

    def process_test_functions(self, force: bool = False) -> Dict:
        """Process a few test functions to validate the approach."""
        test_functions = [
            "create_categorical_graph",
            "draw_tree_diagram",
            "generate_area_stimulus",
        ]

        results = {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}

        print(
            f"🧪 Testing AI description generation on {len(test_functions)} functions..."
        )
        print("=" * 60)

        for i, function_name in enumerate(test_functions, 1):
            print(f"\n[{i}/{len(test_functions)}] Testing: {function_name}")
            print("-" * 60)

            try:
                success = self.process_single_function(function_name, force=force)
                if success:
                    results["updated"] += 1
                else:
                    results["skipped"] += 1
                results["processed"] += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                results["errors"] += 1

        return results


def main():
    """Main function to run the AI description generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate AI-enhanced descriptions for stimulus functions"
    )
    parser.add_argument("--function", "-f", help="Process a single function by name")
    parser.add_argument(
        "--test", "-t", action="store_true", help="Test on a few functions first"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Number of functions to process per batch (default: 10)",
    )
    parser.add_argument(
        "--batch", type=int, help="Process a specific batch number (1-indexed)"
    )
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
            success = generator.process_single_function(args.function, force=args.force)
            if success:
                print(f"\n🎉 Successfully processed {args.function}")
            else:
                print(f"\n❌ Failed to process {args.function}")
        elif args.test:
            # Process test functions
            results = generator.process_test_functions(force=args.force)

            print(f"\n📊 Test Results:")
            print(f"  Processed: {results['processed']}")
            print(f"  Updated: {results['updated']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Errors: {results['errors']}")

            if results["updated"] > 0:
                print(f"\n🎉 Successfully updated {results['updated']} test functions!")
                print(
                    "\n💡 Review the results and run without --test to process all functions"
                )
        else:
            # Process all functions (with optional batching)
            if args.batch:
                print(
                    f"🚀 Starting AI description generation for batch {args.batch}..."
                )
                results = generator.process_all_functions(
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                    batch_number=args.batch,
                )
            else:
                print("🚀 Starting AI description generation for all functions...")
                results = generator.process_all_functions(dry_run=args.dry_run)

            print(f"\n📊 Results:")
            print(f"  Processed: {results['processed']}")
            print(f"  Updated: {results['updated']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Errors: {results['errors']}")

            if results["updated"] > 0:
                print(f"\n🎉 Successfully updated {results['updated']} functions!")

            # Show next batch info if processing in batches
            if args.batch:
                next_batch = args.batch + 1
                print(f"\n💡 To process the next batch, run:")
                print(
                    f"   python3 generate_ai_descriptions.py --batch {next_batch} --force"
                )

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
