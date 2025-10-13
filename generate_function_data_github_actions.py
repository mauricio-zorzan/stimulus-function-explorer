#!/usr/bin/env python3
"""
Generate function data using GitHub Actions artifacts from the working repository.
This approach leverages the successful workflow without environment setup issues.
"""

import json
import os
import base64
import requests
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil

from src.github_reader import StimulusFunctionReader


class GitHubActionsGenerator:
    def __init__(self, github_token: Optional[str] = None):
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.images_dir = self.data_dir / "images"
        self.index_file = self.data_dir / "index.json"
        self.progress_file = self.data_dir / "progress.json"

        # Ensure directories exist
        self.functions_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

        # Initialize components
        self.github_reader = StimulusFunctionReader()

        # GitHub API setup
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo_owner = "trilogy-group"
        self.repo_name = "coach-bot-external-content-generators"
        self.workflow_file = "drawing-tests.yml"

        # GitHub API headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.github_token}"
            if self.github_token
            else None,
        }
        if not self.github_token:
            self.headers.pop("Authorization")

        # Load or initialize progress
        self.progress = self.load_progress()

    def load_progress(self) -> Dict:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # If JSON is corrupted, start fresh
                pass
        return {
            "processed_functions": [],
            "failed_functions": [],
            "last_workflow_run": None,
            "last_artifact_download": None,
            "start_time": datetime.now().isoformat(),
        }

    def save_progress(self):
        """Save current progress to file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def find_latest_successful_run(self) -> Optional[str]:
        """Find the latest successful workflow run."""
        print("🔍 Looking for latest successful workflow run...")

        # Get workflow runs
        runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        response = requests.get(runs_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to get workflow runs: {response.status_code}")
            return None

        runs = response.json().get("workflow_runs", [])

        # Find the latest successful run of our workflow
        for run in runs:
            if (
                run["name"] == "Run Tests (Drawing Functions)"
                and run["status"] == "completed"
                and run["conclusion"] == "success"
            ):
                run_id = str(run["id"])
                print(f"✅ Found successful run: {run_id}")
                print(f"   Created: {run['created_at']}")
                print(f"   Updated: {run['updated_at']}")
                return run_id

        print("❌ No successful workflow runs found")
        return None

    def create_trigger_pr(self) -> bool:
        """Create a PR to trigger the workflow (if you have write access)."""
        print("🚀 Creating PR to trigger workflow...")

        # This would require more complex GitHub API calls to create a branch and PR
        # For now, just provide instructions
        print("💡 To trigger the workflow manually:")
        print(
            "1. Go to the repository: https://github.com/trilogy-group/coach-bot-external-content-generators"
        )
        print("2. Create a new branch")
        print("3. Make a small change (like adding a comment)")
        print("4. Create a pull request to main")
        print("5. The workflow will run automatically")
        print("6. Once it completes, run this script again")

        return False

    def get_latest_workflow_run(self) -> Optional[str]:
        """Get the latest workflow run ID."""
        runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        response = requests.get(runs_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to get workflow runs: {response.status_code}")
            return None

        runs = response.json().get("workflow_runs", [])
        if runs:
            latest_run = runs[0]
            run_id = str(latest_run["id"])
            print(f"📋 Latest workflow run: {run_id} (status: {latest_run['status']})")
            return run_id

        return None

    def wait_for_workflow_completion(self, run_id: str, timeout: int = 1800) -> bool:
        """Wait for workflow to complete and return success status."""
        print(f"⏳ Waiting for workflow {run_id} to complete...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            run_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}"
            response = requests.get(run_url, headers=self.headers)

            if response.status_code != 200:
                print(f"❌ Failed to get workflow status: {response.status_code}")
                return False

            run_data = response.json()
            status = run_data["status"]
            conclusion = run_data.get("conclusion")

            print(f"📊 Workflow status: {status}, conclusion: {conclusion}")

            if status == "completed":
                if conclusion == "success":
                    print("✅ Workflow completed successfully")
                    return True
                else:
                    print(f"❌ Workflow failed with conclusion: {conclusion}")
                    return False

            # Wait before checking again
            time.sleep(30)

        print(f"⏰ Workflow timed out after {timeout} seconds")
        return False

    def download_artifacts(self, run_id: str) -> Optional[str]:
        """Download artifacts from the workflow run."""
        print(f"📥 Downloading artifacts from run {run_id}...")

        # Get artifacts
        artifacts_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/artifacts"
        response = requests.get(artifacts_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to get artifacts: {response.status_code}")
            return None

        artifacts = response.json().get("artifacts", [])
        if not artifacts:
            print("❌ No artifacts found")
            return None

        # Find the test-images artifact
        test_images_artifact = None
        for artifact in artifacts:
            if artifact["name"] == "test-images":
                test_images_artifact = artifact
                break

        if not test_images_artifact:
            print("❌ test-images artifact not found")
            return None

        # Download the artifact
        download_url = test_images_artifact["archive_download_url"]
        response = requests.get(download_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to download artifact: {response.status_code}")
            return None

        # Save and extract the artifact
        temp_dir = tempfile.mkdtemp(prefix="github_artifacts_")
        zip_path = os.path.join(temp_dir, "artifacts.zip")

        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"✅ Artifacts downloaded and extracted to: {extract_dir}")
        return extract_dir

    def process_artifact_images(self, artifact_dir: str) -> List[Dict]:
        """Process images from the artifact directory."""
        print("🖼️ Processing images from artifacts...")

        processed_functions = []

        # Find all image files in the artifact directory
        image_files = []
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file.lower().endswith((".webp", ".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))

        print(f"Found {len(image_files)} images in artifacts")

        for image_file in image_files:
            try:
                # Extract function name from filename
                filename = os.path.basename(image_file)
                function_name = self.extract_function_name_from_filename(filename)

                if function_name:
                    # Copy image to our images directory
                    new_image_name = f"{function_name}_example{Path(image_file).suffix}"
                    new_image_path = self.images_dir / new_image_name

                    shutil.copy2(image_file, new_image_path)

                    # Create metadata
                    metadata = self.create_function_metadata(
                        function_name, str(new_image_path)
                    )

                    # Save metadata
                    self.save_function_metadata(metadata)

                    processed_functions.append(
                        {
                            "function_name": function_name,
                            "original_file": image_file,
                            "new_file": str(new_image_path),
                        }
                    )

                    print(f"✅ Processed: {function_name}")
                else:
                    print(f"⚠️ Could not extract function name from: {filename}")

            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
                continue

        return processed_functions

    def extract_function_name_from_filename(self, filename: str) -> Optional[str]:
        """Extract function name from image filename."""
        # Remove extension
        name_without_ext = Path(filename).stem

        # Get all available function names
        all_functions = self.github_reader.get_stimulus_function_names()

        # Remove timestamps first
        import re

        cleaned = re.sub(r"_\d+$", "", name_without_ext)  # Remove trailing timestamp
        cleaned = re.sub(r"^\d+_", "", cleaned)  # Remove leading timestamp

        # Load the complete mapping (includes both prefix_{time} and {time}_suffix patterns)
        try:
            import json

            with open("function_filename_mapping.json", "r") as f:
                complete_mappings = json.load(f)
        except FileNotFoundError:
            print("⚠️ function_filename_mapping.json not found, using fallback mappings")
            complete_mappings = {}

        # Try direct pattern match first
        if cleaned in complete_mappings:
            function_name = complete_mappings[cleaned]
            if function_name in all_functions:
                return function_name

        # Try fuzzy matching with all patterns
        for pattern, function_name in complete_mappings.items():
            # Check if pattern matches (handles both naming conventions)
            if pattern in cleaned or cleaned in pattern or pattern in name_without_ext:
                # Verify the function exists
                if function_name in all_functions:
                    return function_name

        # Fallback to original mappings for patterns not found in function code
        fallback_mappings = {
            "crosssec": "draw_cross_section",
            "symmetry_identification_flower": "generate_lines_of_symmetry",
            "compound_area_6th": "create_area_model",
            "histogram": "draw_histogram",
            "equation_tape": "create_equation_tape",
            "blank_coordinate_plane": "create_blank_coordinate_plane",
        }

        # Try fallback mappings
        for pattern, function_name in fallback_mappings.items():
            if pattern in cleaned.lower():
                if function_name in all_functions:
                    return function_name

        # Try to find function name in the filename (original logic)
        for function_name in all_functions:
            if function_name in name_without_ext:
                return function_name

        # Try cleaned version
        for function_name in all_functions:
            if function_name == cleaned or function_name in cleaned:
                return function_name

        # Last resort: try to match parts of the filename
        words = cleaned.split("_")
        for word in words:
            for function_name in all_functions:
                if word in function_name and len(word) > 3:  # Avoid short matches
                    return function_name

        return None

    def create_function_metadata(self, function_name: str, image_path: str) -> Dict:
        """Create metadata for a function."""
        # Basic description based on function name
        name_parts = function_name.replace("_", " ").split()

        if "draw" in name_parts:
            action = "Draws"
        elif "create" in name_parts:
            action = "Creates"
        elif "generate" in name_parts:
            action = "Generates"
        elif "plot" in name_parts:
            action = "Plots"
        else:
            action = "Creates"

        content_parts = [
            part
            for part in name_parts
            if part not in ["draw", "create", "generate", "plot"]
        ]
        content = " ".join(content_parts)
        description = f"{action} {content} for educational visualization."

        # Categorize
        if any(word in function_name for word in ["fraction", "fractional"]):
            category = "fractions"
        elif any(
            word in function_name
            for word in ["polygon", "triangle", "rectangle", "shape"]
        ):
            category = "geometry"
        elif any(word in function_name for word in ["table", "data_table"]):
            category = "tables"
        elif any(word in function_name for word in ["graph", "plot", "chart"]):
            category = "graphs"
        elif any(word in function_name for word in ["number_line", "numberline"]):
            category = "number_lines"
        elif any(word in function_name for word in ["angle", "angles"]):
            category = "angles"
        else:
            category = "general"

        # Get relative path for image
        relative_image_path = f"images/{Path(image_path).name}"

        return {
            "function_name": function_name,
            "description": description,
            "parameters": {"data": "Input data for the function"},
            "example_usage": f"# Example usage for {function_name}\nresult = {function_name}(data)",
            "image_path": relative_image_path,
            "category": category,
            "tags": [category],
            "status": "working",
            "last_updated": datetime.now().isoformat(),
            "test_data": "Generated from GitHub Actions workflow",
        }

    def save_function_metadata(self, metadata: Dict) -> None:
        """Save function metadata to JSON file."""
        function_file = self.functions_dir / f"{metadata['function_name']}.json"
        with open(function_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def update_index(self) -> None:
        """Update the master index file."""
        # Load existing index safely
        try:
            if self.index_file.exists():
                with open(self.index_file, "r") as f:
                    index_data = json.load(f)
            else:
                index_data = {
                    "metadata": {
                        "total_functions": 0,
                        "last_updated": "",
                        "version": "1.0.0",
                    },
                    "functions": [],
                }
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Error loading index.json: {e}. Creating new index.")
            index_data = {
                "metadata": {
                    "total_functions": 0,
                    "last_updated": "",
                    "version": "1.0.0",
                },
                "functions": [],
            }

        # Get all generated function files safely
        generated_functions = []
        for func_file in self.functions_dir.glob("*.json"):
            if func_file.name != "template.json":
                try:
                    with open(func_file, "r") as f:
                        func_data = json.load(f)
                        generated_functions.append(
                            {
                                "function_name": func_data["function_name"],
                                "category": func_data["category"],
                                "status": func_data["status"],
                                "image_path": func_data["image_path"],
                            }
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ Error reading {func_file.name}: {e}. Skipping.")

        # Update metadata
        index_data["metadata"]["total_functions"] = len(generated_functions)
        index_data["metadata"]["last_updated"] = datetime.now().isoformat()
        index_data["functions"] = generated_functions

        # Save updated index safely
        try:
            with open(self.index_file, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"📊 Updated index with {len(generated_functions)} functions")
        except Exception as e:
            print(f"❌ Error saving index.json: {e}")
            # Create backup
            import time

            backup_file = self.data_dir / f"index_backup_{int(time.time())}.json"
            try:
                with open(backup_file, "w") as f:
                    json.dump(index_data, f, indent=2)
                print(f"💾 Created backup: {backup_file}")
            except Exception as backup_e:
                print(f"❌ Failed to create backup: {backup_e}")

    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline: find successful run, download, process."""
        print("🚀 Starting GitHub Actions pipeline...")
        print("=" * 60)

        # Step 1: Find latest successful workflow run
        run_id = self.find_latest_successful_run()
        if not run_id:
            print("❌ No successful workflow runs found")
            print(
                "💡 The workflow needs to run successfully first (usually triggered by PRs)"
            )
            return False

        # Step 2: Download artifacts
        artifact_dir = self.download_artifacts(run_id)
        if not artifact_dir:
            print("❌ Failed to download artifacts")
            return False

        # Step 3: Process images
        processed_functions = self.process_artifact_images(artifact_dir)

        # Step 4: Update progress and index
        self.progress["last_workflow_run"] = run_id
        self.progress["last_artifact_download"] = datetime.now().isoformat()
        self.progress["processed_functions"].extend(
            [f["function_name"] for f in processed_functions]
        )
        self.save_progress()
        self.update_index()

        # Step 5: Clean up
        shutil.rmtree(artifact_dir, ignore_errors=True)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Functions processed: {len(processed_functions)}")
        print(f"Workflow run ID: {run_id}")

        if processed_functions:
            print("\nProcessed functions:")
            for func in processed_functions:
                print(f"  ✅ {func['function_name']}")

        return True


def main():
    """Main function to run the GitHub Actions pipeline."""
    print("🚀 GitHub Actions Function Data Generator")
    print("This will find successful workflow runs and download generated images")
    print("=" * 60)

    # Check for GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️ No GITHUB_TOKEN found in environment variables")
        print("You can:")
        print("1. Set GITHUB_TOKEN environment variable")
        print("2. Create a GitHub Personal Access Token with 'actions' scope")
        print("3. Continue without token (may have rate limits)")

        response = input("\nContinue without token? (y/n): ").lower().strip()
        if response not in ["y", "yes", ""]:
            print("❌ Exiting. Please set GITHUB_TOKEN and try again.")
            return

    # Initialize generator
    generator = GitHubActionsGenerator(github_token)

    # Run the pipeline
    success = generator.run_full_pipeline()

    if success:
        print("\n🎉 GitHub Actions pipeline completed successfully!")
        print("You can now run your Streamlit app to view the generated functions.")
    else:
        print("\n❌ GitHub Actions pipeline failed.")
        print("💡 This usually means:")
        print("   - No successful workflow runs found")
        print("   - The workflow needs to run first (triggered by PRs)")
        print("   - Check the repository for recent successful runs")

        response = (
            input("\nWould you like instructions to trigger the workflow? (y/n): ")
            .lower()
            .strip()
        )
        if response in ["y", "yes", ""]:
            generator.create_trigger_pr()


if __name__ == "__main__":
    main()
