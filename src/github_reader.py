"""
Simple GitHub repository reader for stimulus functions.
"""

import ast
import json
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from github import Github

load_dotenv()


class StimulusFunctionReader:
    def __init__(self):
        self.github = Github(os.getenv("GITHUB_TOKEN"))
        self.repo_owner = os.getenv("GITHUB_REPO_OWNER")
        self.repo_name = os.getenv("GITHUB_REPO_NAME")
        self.repo = self.github.get_repo(f"{self.repo_owner}/{self.repo_name}")

    def extract_function_from_code(self, code: str, function_name: str) -> str:
        """Extract a specific function from Python code."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Find the function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the function's source code
                    lines = code.split("\n")
                    start_line = node.lineno - 1  # AST line numbers are 1-based

                    # Find the end of the function by looking for the next function or class at the same indentation level
                    function_indent = len(lines[start_line]) - len(
                        lines[start_line].lstrip()
                    )
                    end_line = len(lines)

                    for i in range(start_line + 1, len(lines)):
                        line = lines[i]
                        if line.strip():  # Skip empty lines
                            current_indent = len(line) - len(line.lstrip())
                            # If we find a line at the same or lower indentation that starts with def/class, we've found the end
                            if current_indent <= function_indent and (
                                line.strip().startswith("def ")
                                or line.strip().startswith("class ")
                            ):
                                end_line = i
                                break

                    # Extract the function
                    function_code = "\n".join(lines[start_line:end_line])
                    return function_code

            # Fallback: try regex if AST parsing fails
            return self.extract_function_with_regex(code, function_name)

        except Exception:
            # If AST parsing fails, fall back to regex
            return self.extract_function_with_regex(code, function_name)

    def extract_function_with_regex(self, code: str, function_name: str) -> str:
        """Fallback method to extract function using regex."""
        # Pattern to match the function definition and its body
        pattern = rf"^def {re.escape(function_name)}\(.*?\):"

        lines = code.split("\n")
        function_start = None

        # Find the function start
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                function_start = i
                break

        if function_start is None:
            return f"# Function '{function_name}' not found in this file"

        # Find function end by looking for the next function/class at same indentation
        function_indent = len(lines[function_start]) - len(
            lines[function_start].lstrip()
        )
        function_end = len(lines)

        for i in range(function_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= function_indent and (
                    line.strip().startswith("def ") or line.strip().startswith("class ")
                ):
                    function_end = i
                    break

        return "\n".join(lines[function_start:function_end])

    def get_stimulus_function_names(self) -> List[str]:
        """Extract stimulus function names from mcq_payload.json"""
        try:
            # Get the mcq_payload.json file
            file_path = "src/content_generators/question_generator/schemas/json_schemas/mcq_payload.json"
            content = self.repo.get_contents(file_path)
            json.loads(content.decoded_content)

            # Navigate to the stimulus function enum
            # This is a simplified extraction - you might need to adjust based on the exact JSON structure
            function_names = []

            # Search for the enum with stimulus function names
            content_str = content.decoded_content.decode("utf-8")

            # Find the enum section with function names (looking for the pattern you showed)
            enum_match = re.search(r'"enum":\s*\[(.*?)\]', content_str, re.DOTALL)
            if enum_match:
                enum_content = enum_match.group(1)
                # Extract quoted strings
                function_names = re.findall(r'"([^"]+)"', enum_content)

            return function_names

        except Exception as e:
            print(f"Error reading stimulus function names: {e}")
            return []

    def get_function_details(self, function_name: str) -> Dict:
        """Get details for a specific stimulus function"""
        try:
            details = {
                "name": function_name,
                "implementation": None,
                "schema": None,
                "tests": None,
                "error": None,
            }

            # Try to find the function implementation
            drawing_functions_path = "src/content_generators/additional_content/stimulus_image/drawing_functions"

            # Look for the function in various files
            try:
                contents = self.repo.get_contents(drawing_functions_path)
                for file in contents:
                    if file.name.endswith(".py") and file.name != "__init__.py":
                        file_content = file.decoded_content.decode("utf-8")
                        if f"def {function_name}" in file_content:
                            # Extract only the specific function
                            function_code = self.extract_function_from_code(
                                file_content, function_name
                            )
                            details["implementation"] = {
                                "file": file.name,
                                "content": function_code,
                                "url": file.html_url,
                                "full_file_url": file.html_url,  # Keep link to full file
                            }
                            break
            except Exception as e:
                details["error"] = f"Could not find implementation: {e}"

            # Try to find corresponding test file using smart test discovery
            try:
                # Use the test runner's smart discovery
                from src.test_runner import StimulusTestRunner

                test_runner = StimulusTestRunner()

                # Set up environment for test discovery
                test_runner.setup_test_environment()
                test_file_path = test_runner.get_function_test_file(function_name)

                if test_file_path:
                    # Get the relative path for GitHub API
                    import os

                    test_file_name = os.path.basename(test_file_path)
                    github_test_path = (
                        f"{drawing_functions_path}/tests/{test_file_name}"
                    )

                    try:
                        test_content = self.repo.get_contents(github_test_path)
                        details["tests"] = {
                            "file": test_content.name,
                            "content": test_content.decoded_content.decode("utf-8"),
                            "url": test_content.html_url,
                        }
                    except Exception as e:
                        # Fallback: if GitHub API fails, we still found the test file
                        details["tests"] = {
                            "file": test_file_name,
                            "content": "Test file found via smart discovery",
                            "note": f"Found test file: {test_file_name}",
                            "error": f"Could not fetch content via GitHub API: {e}",
                        }
                else:
                    # Smart discovery didn't find anything, try fallback patterns
                    fallback_patterns = [
                        f"test_{function_name.replace('draw_', '').replace('create_', '').replace('generate_', '')}.py",
                        f"test_{function_name}.py",
                    ]

                    found_fallback = False
                    for pattern in fallback_patterns:
                        try:
                            test_file_path = f"{drawing_functions_path}/tests/{pattern}"
                            test_content = self.repo.get_contents(test_file_path)
                            details["tests"] = {
                                "file": test_content.name,
                                "content": test_content.decoded_content.decode("utf-8"),
                                "url": test_content.html_url,
                            }
                            found_fallback = True
                            break
                        except Exception:
                            continue

                    if not found_fallback:
                        details["tests"] = {
                            "error": f"No test file found for {function_name}"
                        }

            except Exception as e:
                details["tests"] = {"error": f"Error during test discovery: {e}"}

            return details

        except Exception as e:
            return {"name": function_name, "error": str(e)}

    def list_all_functions(self) -> List[Dict]:
        """Get basic info for all stimulus functions"""
        function_names = self.get_stimulus_function_names()
        functions = []

        print(f"Found {len(function_names)} stimulus functions")

        # Return all functions (be mindful of API rate limits)
        for name in function_names:  # Show all functions
            functions.append({"name": name, "status": "available"})

        return functions


if __name__ == "__main__":
    # Test the reader
    reader = StimulusFunctionReader()
    functions = reader.list_all_functions()
    print(f"Found {len(functions)} functions")
    for func in functions:
        print(f"- {func['name']}")
