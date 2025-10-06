#!/usr/bin/env python3
"""
Script to trigger the GitHub Actions workflow that runs ALL tests (not just drawing_functions).
This should generate images for all 134 functions.
"""

import requests
import time
import os
import tempfile
import zipfile
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, List

from src.github_reader import StimulusFunctionReader


class AllTestsWorkflowTrigger:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set")

        # Our repository details
        self.repo_owner = "mauriciozorzan"  # Your GitHub username
        self.repo_name = "stimulus-function-explorer"
        self.workflow_file = "run-all-tests.yml"

        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.github_reader = StimulusFunctionReader()

    def trigger_workflow(self) -> Optional[str]:
        """Trigger the workflow that runs all tests."""
        print("🚀 Triggering workflow to run ALL tests...")

        # Get workflow ID
        workflow_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows"
        response = requests.get(workflow_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to get workflows: {response.status_code}")
            return None

        workflows = response.json().get("workflows", [])
        workflow_id = None

        for workflow in workflows:
            if workflow["name"] == "Run All Tests (All Functions)":
                workflow_id = workflow["id"]
                break

        if not workflow_id:
            print("❌ Workflow 'Run All Tests (All Functions)' not found")
            return None

        print(f"✅ Found workflow ID: {workflow_id}")

        # Trigger the workflow
        trigger_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/dispatches"
        trigger_data = {
            "ref": "main",  # Branch to run on
            "inputs": {},  # No inputs needed
        }

        response = requests.post(trigger_url, headers=self.headers, json=trigger_data)

        if response.status_code == 204:
            print("✅ Workflow triggered successfully!")
            return workflow_id
        else:
            print(f"❌ Failed to trigger workflow: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    def wait_for_workflow_completion(
        self, workflow_id: str, timeout_minutes: int = 30
    ) -> Optional[str]:
        """Wait for workflow to complete and return the run ID."""
        print(
            f"⏳ Waiting for workflow to complete (timeout: {timeout_minutes} minutes)..."
        )

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while time.time() - start_time < timeout_seconds:
            # Get workflow runs
            runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/runs"
            response = requests.get(runs_url, headers=self.headers)

            if response.status_code != 200:
                print(f"❌ Failed to get workflow runs: {response.status_code}")
                return None

            runs = response.json().get("workflow_runs", [])

            if runs:
                latest_run = runs[0]
                status = latest_run["status"]
                conclusion = latest_run.get("conclusion")

                print(f"   Status: {status}, Conclusion: {conclusion}")

                if status == "completed":
                    if conclusion == "success":
                        print("✅ Workflow completed successfully!")
                        return str(latest_run["id"])
                    else:
                        print(f"❌ Workflow failed with conclusion: {conclusion}")
                        return None

            time.sleep(30)  # Check every 30 seconds

        print(f"⏰ Workflow timed out after {timeout_minutes} minutes")
        return None

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

        # Find the all-test-images artifact
        target_artifact = None
        for artifact in artifacts:
            if artifact["name"] == "all-test-images":
                target_artifact = artifact
                break

        if not target_artifact:
            print("❌ 'all-test-images' artifact not found")
            return None

        print(
            f"✅ Found artifact: {target_artifact['name']} ({target_artifact['size_in_bytes']} bytes)"
        )

        # Download artifact
        download_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/artifacts/{target_artifact['id']}/zip"
        response = requests.get(download_url, headers=self.headers)

        if response.status_code != 200:
            print(f"❌ Failed to download artifact: {response.status_code}")
            return None

        # Save and extract artifact
        temp_dir = tempfile.mkdtemp(prefix="all_tests_artifacts_")
        zip_path = os.path.join(temp_dir, "artifacts.zip")

        with open(zip_path, "wb") as f:
            f.write(response.content)

        print(f"📦 Extracting artifacts to: {temp_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        # Move extracted files
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".webp"):
                    src = os.path.join(root, file)
                    dst = os.path.join(extracted_dir, file)
                    os.rename(src, dst)

        print(f"✅ Artifacts extracted to: {extracted_dir}")
        return extracted_dir

    def process_images(self, images_dir: str) -> Dict[str, List[str]]:
        """Process the downloaded images and map them to functions."""
        print("🖼️ Processing images and mapping to functions...")

        # Use the same mapping logic from our GitHub Actions script
        from generate_function_data_github_actions import GitHubActionsGenerator

        generator = GitHubActionsGenerator()
        function_images = {}

        for filename in os.listdir(images_dir):
            if filename.endswith(".webp"):
                function_name = generator.extract_function_name_from_filename(filename)
                if function_name:
                    if function_name not in function_images:
                        function_images[function_name] = []
                    function_images[function_name].append(filename)
                    print(f"✅ Mapped: {filename} → {function_name}")
                else:
                    print(f"⚠️ Could not map: {filename}")

        return function_images

    def save_images_and_metadata(
        self, images_dir: str, function_images: Dict[str, List[str]]
    ) -> None:
        """Save images and create metadata for all functions."""
        print("💾 Saving images and creating metadata...")

        data_dir = Path("data")
        functions_dir = data_dir / "functions"
        images_dir_local = data_dir / "images"

        functions_dir.mkdir(exist_ok=True)
        images_dir_local.mkdir(exist_ok=True)

        processed_count = 0

        for function_name, image_files in function_images.items():
            if not image_files:
                continue

            # Use the first image as the example
            source_image = os.path.join(images_dir, image_files[0])
            target_image = images_dir_local / f"{function_name}_example.webp"

            # Copy image
            import shutil

            shutil.copy2(source_image, target_image)

            # Create metadata
            metadata = {
                "function_name": function_name,
                "description": f"Generated from all-tests workflow for {function_name}",
                "category": "general",
                "status": "working",
                "image_path": f"images/{function_name}_example.webp",
                "parameters": "Generated from test data",
                "example_usage": f"Call {function_name}() with appropriate parameters",
                "tags": ["generated", "all-tests"],
                "last_updated": datetime.now().isoformat(),
                "test_data": "Generated from all-tests workflow",
            }

            # Save metadata
            function_file = functions_dir / f"{function_name}.json"
            with open(function_file, "w") as f:
                json.dump(metadata, f, indent=2)

            processed_count += 1
            print(f"✅ Processed: {function_name}")

        print(f"📊 Processed {processed_count} functions")

    def update_index(self) -> None:
        """Update the master index with all processed functions."""
        print("📊 Updating master index...")

        from generate_function_data_github_actions import GitHubActionsGenerator

        generator = GitHubActionsGenerator()
        generator.update_index()

        # Check results
        with open("data/index.json", "r") as f:
            index_data = json.load(f)

        print(
            f"✅ Index updated with {index_data['metadata']['total_functions']} functions"
        )

    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline: trigger workflow, wait, download, process."""
        print("🚀 Starting All Tests Workflow Pipeline")
        print("=" * 60)

        try:
            # Step 1: Trigger workflow
            workflow_id = self.trigger_workflow()
            if not workflow_id:
                return False

            # Step 2: Wait for completion
            run_id = self.wait_for_workflow_completion(workflow_id)
            if not run_id:
                return False

            # Step 3: Download artifacts
            images_dir = self.download_artifacts(run_id)
            if not images_dir:
                return False

            # Step 4: Process images
            function_images = self.process_images(images_dir)
            print(f"📊 Mapped {len(function_images)} functions")

            # Step 5: Save images and metadata
            self.save_images_and_metadata(images_dir, function_images)

            # Step 6: Update index
            self.update_index()

            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Functions processed: {len(function_images)}")
            print("🚀 Your app should now show all functions with images!")

            return True

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return False


def main():
    """Main function to run the all-tests workflow pipeline."""
    print("🚀 All Tests Workflow Pipeline")
    print(
        "This will trigger a workflow to run ALL tests and generate images for all 134 functions"
    )
    print("=" * 60)

    try:
        trigger = AllTestsWorkflowTrigger()
        success = trigger.run_full_pipeline()

        if success:
            print("\n🎉 SUCCESS! All functions should now have images.")
            print("🚀 Restart your Streamlit app to see all 134 functions!")
        else:
            print("\n❌ Pipeline failed. Check the logs above for details.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
