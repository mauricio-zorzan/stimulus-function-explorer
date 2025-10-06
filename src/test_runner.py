"""
Test runner for stimulus functions to generate images.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import json
import re
from datetime import datetime

from src.github_reader import StimulusFunctionReader
from src.repository_executor import RepositoryExecutor


class StimulusTestRunner:
    def __init__(self):
        self.github_reader = StimulusFunctionReader()
        self.temp_dir = None

    def setup_test_environment(self) -> str:
        """Set up a temporary directory with the repository for testing."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        self.temp_dir = tempfile.mkdtemp(prefix="stimulus_test_")
        print(f"Created temp directory: {self.temp_dir}")

        # Check environment variables
        repo_owner = os.getenv("GITHUB_REPO_OWNER")
        repo_name = os.getenv("GITHUB_REPO_NAME")

        if not repo_owner or not repo_name:
            raise Exception(
                f"Missing environment variables: GITHUB_REPO_OWNER={repo_owner}, GITHUB_REPO_NAME={repo_name}"
            )

        # Clone the repository to temp directory
        repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        print(f"Cloning repository from: {repo_url}")

        try:
            result = subprocess.run(
                ["git", "clone", repo_url, self.temp_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Git clone successful. Output: {result.stdout}")
            return self.temp_dir
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to clone repository: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            print(error_msg)
            raise Exception(error_msg)
        except FileNotFoundError:
            raise Exception(
                "Git command not found. Please ensure Git is installed and available in PATH."
            )

    def get_function_test_file(self, function_name: str) -> Optional[str]:
        """Find the test file for a specific function."""
        test_dir = os.path.join(
            self.temp_dir,
            "src/content_generators/additional_content/stimulus_image/drawing_functions/tests",
        )

        if not os.path.exists(test_dir):
            return None

        # Generate multiple possible test file patterns
        base_name = (
            function_name.replace("draw_", "")
            .replace("create_", "")
            .replace("generate_", "")
        )

        # Create comprehensive list of possible test file names
        test_patterns = [
            f"test_{function_name}.py",  # test_draw_mixed_fractional_models.py
            f"test_{base_name}.py",  # test_mixed_fractional_models.py
        ]

        # Add patterns for common word variations
        if "_" in base_name:
            parts = base_name.split("_")
            # Try combinations of the parts
            if len(parts) >= 2:
                test_patterns.extend(
                    [
                        f"test_{parts[0]}_{parts[1]}.py",  # test_mixed_fractional.py -> test_mixed_fraction.py
                        f"test_{'_'.join(parts[:2])}.py",  # first two parts
                        f"test_{parts[0]}.py",  # just first part
                    ]
                )

            # Handle common word variations and patterns
            variations = {
                "fractional": "fraction",
                "models": "model",
                "blocks": "block",
                "diagrams": "diagram",
            }

            for original, variant in variations.items():
                if original in base_name:
                    variant_name = base_name.replace(original, variant)
                    test_patterns.append(f"test_{variant_name}.py")

            # Special patterns for common function naming conventions
            if "mixed_fractional_models" in base_name:
                test_patterns.extend(
                    ["test_mixed_fraction.py", "test_mixed.py", "test_fractional.py"]
                )

            if "base_ten_blocks" in base_name:
                test_patterns.extend(["test_base_ten.py", "test_base_ten_block.py"])

            if "coordinate" in base_name:
                test_patterns.extend(["test_coordinate.py", "test_coordinates.py"])

        # First try exact pattern matches
        for pattern in test_patterns:
            test_file = os.path.join(test_dir, pattern)
            if os.path.exists(test_file):
                print(f"Found test file by pattern: {pattern}")
                return test_file

        # Fallback: search ALL test files for actual function calls and find the best match
        print(
            f"No pattern match found, searching all test files for function calls to: {function_name}"
        )

        import re

        # Pattern to match function calls
        patterns = [
            rf"file_name\s*=\s*{re.escape(function_name)}\s*\(",  # file_name = function_name( (most specific)
            rf"\w+\s*=\s*{re.escape(function_name)}\s*\(",  # variable = function_name(
            rf"{re.escape(function_name)}\s*\(",  # function_name( (direct call)
        ]

        candidates = []

        for test_file in os.listdir(test_dir):
            if test_file.startswith("test_") and test_file.endswith(".py"):
                test_path = os.path.join(test_dir, test_file)
                try:
                    with open(test_path, "r") as f:
                        content = f.read()

                    # Count matches for each pattern (higher priority patterns get more weight)
                    total_score = 0
                    pattern_matches = []

                    for i, pattern in enumerate(patterns):
                        matches = re.findall(pattern, content)
                        if matches:
                            # Weight: file_name= gets 3x, variable= gets 2x, direct call gets 1x
                            weight = 3 - i
                            score = len(matches) * weight
                            total_score += score
                            pattern_matches.append((pattern, matches, score))

                    if total_score > 0:
                        candidates.append(
                            {
                                "file": test_file,
                                "path": test_path,
                                "score": total_score,
                                "matches": pattern_matches,
                            }
                        )
                        print(
                            f"Found {len(sum([m[1] for m in pattern_matches], []))} function calls in {test_file} (score: {total_score})"
                        )

                except Exception as e:
                    print(f"Error reading test file {test_file}: {e}")
                    continue

        # Return the file with the highest score
        if candidates:
            best_candidate = max(candidates, key=lambda x: x["score"])
            print(
                f"Best match: {best_candidate['file']} with score {best_candidate['score']}"
            )
            return best_candidate["path"]

        # Final fallback: simple string search for backwards compatibility
        print(f"No regex matches found, falling back to simple string search...")
        for test_file in os.listdir(test_dir):
            if test_file.startswith("test_") and test_file.endswith(".py"):
                test_path = os.path.join(test_dir, test_file)
                try:
                    with open(test_path, "r") as f:
                        content = f.read()

                    if function_name in content:
                        print(
                            f"Found function {function_name} mentioned in test file: {test_file}"
                        )
                        return test_path

                except Exception as e:
                    continue

        print(f"No test file found for function: {function_name}")
        return None

    def run_function_test(self, function_name: str) -> Dict:
        """Run tests for a specific stimulus function and capture results."""
        try:
            print(f"Starting test execution for: {function_name}")

            # Try repository executor first to avoid types conflicts
            print(f"🚀 Attempting repository execution for: {function_name}")
            repo_executor = RepositoryExecutor()
            repo_result = repo_executor.run_repository_function_test(function_name)

            if repo_result["success"]:
                print(f"✅ Repository execution succeeded!")
                return repo_result

            print(f"⚠️ Repository execution failed, trying pytest...")

            # Setup test environment
            print("Setting up test environment...")
            repo_dir = self.setup_test_environment()
            print(f"Repository cloned to: {repo_dir}")

            # Find the test file
            print(f"Looking for test file for function: {function_name}")
            test_file = self.get_function_test_file(function_name)
            if not test_file:
                available_tests = []
                test_dir = os.path.join(
                    self.temp_dir,
                    "src/content_generators/additional_content/stimulus_image/drawing_functions/tests",
                )
                if os.path.exists(test_dir):
                    available_tests = [
                        f for f in os.listdir(test_dir) if f.endswith(".py")
                    ]

                return {
                    "success": False,
                    "error": f"No test file found for function: {function_name}. Available test files: {available_tests}",
                    "images": [],
                    "debug_info": {
                        "temp_dir": self.temp_dir,
                        "test_dir": test_dir,
                        "test_dir_exists": os.path.exists(test_dir)
                        if test_dir
                        else False,
                        "available_tests": available_tests,
                    },
                }

            print(f"Found test file: {test_file}")

            # Create output directory for images (matching the test expectation)
            output_dir = os.path.join(self.temp_dir, "content", "tests")
            os.makedirs(output_dir, exist_ok=True)

            # Set up environment variables for test execution
            test_env = os.environ.copy()

            # Build PYTHONPATH that avoids conflicts with built-in modules
            # Only add the specific content_generators path to avoid 'types' conflict
            pythonpath_parts = [
                os.path.join(repo_dir, "src", "content_generators"),
            ]

            # Add current working directory Python path if it exists
            if "PYTHONPATH" in test_env:
                pythonpath_parts.append(test_env["PYTHONPATH"])

            test_env.update(
                {
                    "MPLBACKEND": "Agg",  # Non-interactive backend
                    "FIGURE_DPI": "150",  # Good quality but fast rendering
                    "PYTHONPATH": os.pathsep.join(pythonpath_parts),
                    # Remove any existing tmp dir settings that might interfere
                    "TMPDIR": output_dir,
                    # Add some additional env vars that might help with imports
                    "CONTENT_GENERATORS_ROOT": repo_dir,
                    "PROJECT_ROOT": repo_dir,
                }
            )

            # Run all tests in the test file (since they test the function we want)
            cmd = [
                "python",
                "-m",
                "pytest",
                test_file,
                "-v",
                "--tb=short",
                "-m",
                "drawing_functions",  # Use the pytest marker instead
                "--no-header",
                "--quiet",
            ]

            # Execute the test
            result = subprocess.run(
                cmd,
                cwd=repo_dir,
                env=test_env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Collect generated images
            images = self.collect_generated_images(output_dir, function_name)

            # Check if test failed and provide helpful error information
            if result.returncode != 0:
                stderr_output = result.stderr
                stdout_output = result.stdout

                # Check for common import errors and provide helpful messages
                combined_output = stderr_output + "\n" + stdout_output

                if (
                    "ModuleNotFoundError" in combined_output
                    and "shapely" in combined_output
                ):
                    error_message = (
                        "✅ Function and tests found! However, this function requires additional geometric computing dependencies "
                        "like 'shapely' for polygon operations. The test exists but requires specialized packages to run locally. "
                        "This function would work in the full repository environment."
                    )
                    error_type = "missing_shapely"
                elif (
                    "circular import" in combined_output and "types" in combined_output
                ):
                    # Use repository executor to resolve types conflict
                    print(
                        f"⚡ Types conflict detected, repository executor should handle this..."
                    )
                    # Repository executor already tried above, continue to other fallbacks
                    error_message = (
                        "✅ Function and tests found! However, there's a naming conflict between the repository's 'types' module "
                        "and Python's built-in 'types' module. All execution methods failed."
                    )
                    error_type = "types_conflict"
                elif (
                    "ImportError" in combined_output
                    or "ModuleNotFoundError" in combined_output
                ):
                    error_message = (
                        "✅ Function and tests found! However, this function requires dependencies that aren't available "
                        "in this simplified test environment. The test exists in the repository but requires the full project "
                        "dependencies to run tests locally."
                    )
                    error_type = "missing_dependencies"
                else:
                    error_message = (
                        f"Test execution failed with return code {result.returncode}"
                    )
                    error_type = "execution_error"

                return {
                    "success": False,
                    "error": error_message,
                    "error_type": error_type,
                    "function_found": True,
                    "stdout": stdout_output,
                    "stderr": stderr_output,
                    "return_code": result.returncode,
                    "images": images,
                    "test_file": os.path.basename(test_file),
                    "execution_time": datetime.now().isoformat(),
                    "error_details": combined_output,
                }
            else:
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "images": images,
                    "test_file": os.path.basename(test_file),
                    "execution_time": datetime.now().isoformat(),
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out (5 minutes)",
                "images": [],
            }
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            error_str = str(e)

            # Check for common import errors and provide helpful messages
            if "ModuleNotFoundError" in error_details and "shapely" in error_details:
                helpful_message = (
                    "✅ Function and tests found! However, this function requires additional geometric computing dependencies "
                    "like 'shapely' for polygon operations. The test exists but requires specialized packages to run locally. "
                    "This function would work in the full repository environment."
                )
                error_type = "missing_shapely"
            elif (
                "ImportError" in error_details or "ModuleNotFoundError" in error_details
            ):
                helpful_message = (
                    "✅ Function and tests found! However, this function requires dependencies that aren't available "
                    "in this simplified test environment. The test exists in the repository but requires the full project "
                    "dependencies to run tests locally."
                )
                error_type = "missing_dependencies"
            else:
                helpful_message = f"Test execution failed: {error_str}"
                error_type = "execution_error"

            print(f"Test execution error: {error_details}")
            return {
                "success": False,
                "error": helpful_message,
                "error_details": error_details,
                "error_type": error_type,
                "function_found": True,  # We found the function and test file
                "images": [],
            }
        finally:
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass  # Best effort cleanup

    def collect_generated_images(
        self, output_dir: str, function_name: str
    ) -> List[Dict]:
        """Collect images generated during test execution."""
        images = []

        if not os.path.exists(output_dir):
            return images

        # Look for common image file extensions
        image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".svg"]

        for file in os.listdir(output_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(output_dir, file)

                try:
                    # Get file size
                    file_size = os.path.getsize(file_path)

                    # Read the image file as base64 for direct display
                    import base64

                    with open(file_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()

                    image_info = {
                        "filename": file,
                        "path": file_path,
                        "size_bytes": file_size,
                        "function_name": function_name,
                        "generated_at": datetime.now().isoformat(),
                        "image_data": img_data,  # Base64 encoded image for display
                        "file_extension": os.path.splitext(file)[1].lower(),
                    }

                    # Images are displayed locally using base64 encoding

                    images.append(image_info)

                except Exception as e:
                    print(f"Error processing image {file}: {e}")

        return images

    def run_multiple_tests(self, function_names: List[str]) -> Dict[str, Dict]:
        """Run tests for multiple functions."""
        results = {}

        for function_name in function_names:
            print(f"Running tests for: {function_name}")
            results[function_name] = self.run_function_test(function_name)

        return results

    def get_available_test_functions(self) -> List[str]:
        """Get list of functions that have tests available."""
        try:
            # Setup test environment
            repo_dir = self.setup_test_environment()

            test_dir = os.path.join(
                repo_dir,
                "src/content_generators/additional_content/stimulus_image/drawing_functions/tests",
            )

            if not os.path.exists(test_dir):
                return []

            functions_with_tests = []
            all_functions = self.github_reader.get_stimulus_function_names()

            for function_name in all_functions:
                test_file = self.get_function_test_file(function_name)
                if test_file:
                    functions_with_tests.append(function_name)

            return functions_with_tests

        except Exception as e:
            print(f"Error getting available test functions: {e}")
            return []
        finally:
            # Clean up
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    # Test the runner
    runner = StimulusTestRunner()

    # Example: run a single test
    result = runner.run_function_test("draw_base_ten_blocks")
    print(json.dumps(result, indent=2))
