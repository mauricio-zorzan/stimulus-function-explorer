#!/usr/bin/env python3
"""
Retrieve and organize images for functions using the updated mapping.
Supports multiple images per function.
"""

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import requests


class FunctionImageRetriever:
    def __init__(self, github_token: Optional[str] = None):
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.images_dir = self.data_dir / "images"
        self.index_file = self.data_dir / "index.json"

        # Ensure directories exist
        self.functions_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir.mkdir(exist_ok=True, parents=True)

        # Load the function filename mapping
        self.mapping = self.load_mapping()

        # Reverse mapping: function_name -> [file_patterns]
        self.reverse_mapping = self.create_reverse_mapping()

        # GitHub setup
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo_owner = "trilogy-group"
        self.repo_name = "coach-bot-external-content-generators"

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"🗺️  Loaded {len(self.mapping)} file pattern mappings")
        print(f"🔄 Reverse mapping covers {len(self.reverse_mapping)} functions")

    def load_mapping(self) -> Dict[str, str]:
        """Load the function filename mapping."""
        mapping_file = Path("function_filename_mapping.json")
        if mapping_file.exists():
            with open(mapping_file) as f:
                return json.load(f)
        return {}

    def create_reverse_mapping(self) -> Dict[str, List[str]]:
        """Create reverse mapping: function_name -> [file_patterns]."""
        reverse = {}
        for pattern, function_name in self.mapping.items():
            if function_name not in reverse:
                reverse[function_name] = []
            reverse[function_name].append(pattern)
        return reverse

    def match_filename_to_function(self, filename: str) -> Optional[str]:
        """
        Match a filename to a function using the mapping.
        Handles both patterns: prefix_{time} and {time}_suffix
        """
        # Remove extension
        name_without_ext = Path(filename).stem

        # Clean the filename by removing timestamps
        # Pattern 1: Remove trailing timestamp (prefix_1234567890)
        cleaned = re.sub(r"_\d{10,}$", "", name_without_ext)
        # Pattern 2: Remove leading timestamp (1234567890_suffix)
        cleaned = re.sub(r"^\d{10,}_", "", cleaned)

        # Try direct match first
        if cleaned in self.mapping:
            return self.mapping[cleaned]

        # Try to find any pattern that matches
        # Sort patterns by length (longest first) to match more specific patterns before shorter ones
        sorted_patterns = sorted(
            self.mapping.items(), key=lambda x: len(x[0]), reverse=True
        )

        for pattern, function_name in sorted_patterns:
            # For prefix matching, check if the filename starts with the pattern
            # and the next character is either underscore, digit, or end of string
            if name_without_ext.startswith(pattern):
                # Check if this is a proper prefix match (not a substring)
                remaining = name_without_ext[len(pattern) :]
                if not remaining or remaining[0] in (
                    "_",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ):
                    return function_name

            # For suffix matching (reversed pattern), check if pattern is in the cleaned name
            if pattern in cleaned:
                return function_name

        return None

    def scan_local_images(self) -> Dict[str, List[Path]]:
        """
        Scan local images directory and group by function.
        Returns: {function_name: [image_paths]}
        """
        print("\n🔍 Scanning local images directory...")
        function_images = {}

        if not self.images_dir.exists():
            print("⚠️  Images directory does not exist")
            return function_images

        # Get all image files
        image_extensions = {".webp", ".png", ".jpg", ".jpeg", ".gif"}
        image_files = [
            f
            for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        print(f"📸 Found {len(image_files)} total images")

        # Match each image to a function
        matched = 0
        unmatched = []

        for image_file in image_files:
            function_name = self.match_filename_to_function(image_file.name)

            if function_name:
                if function_name not in function_images:
                    function_images[function_name] = []
                function_images[function_name].append(image_file)
                matched += 1
            else:
                unmatched.append(image_file.name)

        print(f"✅ Matched {matched} images to functions")
        print(f"❌ Unmatched {len(unmatched)} images")

        if unmatched and len(unmatched) <= 10:
            print("\n⚠️  Unmatched images:")
            for name in unmatched:
                print(f"  - {name}")

        return function_images

    def find_latest_successful_run(self) -> Optional[str]:
        """Find the latest successful workflow run with artifacts."""
        print("\n🔍 Looking for latest successful workflow run with artifacts...")

        runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        params = {"per_page": 50}  # Get more runs to increase chances

        try:
            response = requests.get(runs_url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"❌ Failed to get workflow runs: {response.status_code}")
                return None

            runs = response.json().get("workflow_runs", [])

            # Find the latest successful run with artifacts
            for run in runs:
                if run["status"] == "completed" and run["conclusion"] == "success":
                    run_id = str(run["id"])

                    # Check if this run has artifacts
                    artifacts_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/artifacts"
                    artifacts_response = requests.get(
                        artifacts_url, headers=self.headers
                    )

                    if artifacts_response.status_code == 200:
                        artifacts = artifacts_response.json().get("artifacts", [])
                        if artifacts:
                            print(f"✅ Found successful run with artifacts: {run_id}")
                            print(f"   Name: {run['name']}")
                            print(f"   Created: {run['created_at']}")
                            print(f"   Artifacts: {len(artifacts)}")
                            return run_id

            print("❌ No successful workflow runs with artifacts found")
            return None

        except Exception as e:
            print(f"❌ Error finding workflow run: {e}")
            return None

    def download_artifacts(self, run_id: str) -> Optional[Path]:
        """Download artifacts from GitHub Actions run."""
        print(f"\n📥 Downloading artifacts from run {run_id}...")

        # Get artifacts list
        artifacts_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/artifacts"

        try:
            response = requests.get(artifacts_url, headers=self.headers)

            if response.status_code != 200:
                print(f"❌ Failed to get artifacts: {response.status_code}")
                return None

            artifacts = response.json().get("artifacts", [])
            if not artifacts:
                print("❌ No artifacts found")
                return None

            print(f"📦 Found {len(artifacts)} artifact(s)")

            # Find test-images artifact
            test_images_artifact = None
            for artifact in artifacts:
                print(f"  - {artifact['name']} ({artifact['size_in_bytes']} bytes)")
                if (
                    "image" in artifact["name"].lower()
                    or "test" in artifact["name"].lower()
                ):
                    test_images_artifact = artifact

            if not test_images_artifact:
                print("⚠️  No test-images artifact found, using first artifact")
                test_images_artifact = artifacts[0]

            # Download the artifact
            download_url = test_images_artifact["archive_download_url"]
            print(f"⬇️  Downloading {test_images_artifact['name']}...")

            response = requests.get(download_url, headers=self.headers, stream=True)

            if response.status_code != 200:
                print(f"❌ Failed to download: {response.status_code}")
                return None

            # Save to temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix="github_artifacts_"))
            zip_path = temp_dir / "artifacts.zip"

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"✅ Downloaded and extracted to {extract_dir}")
            return extract_dir

        except Exception as e:
            print(f"❌ Error downloading artifacts: {e}")
            return None

    def process_downloaded_images(self, artifact_dir: Path) -> Dict[str, List[Path]]:
        """Process images from downloaded artifacts."""
        print(f"\n🖼️  Processing images from {artifact_dir}...")

        function_images = {}

        # Find all images recursively
        image_extensions = {".webp", ".png", ".jpg", ".jpeg", ".gif"}
        image_files = []

        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)

        print(f"📸 Found {len(image_files)} images in artifacts")

        # Process each image
        matched = 0
        for image_file in image_files:
            function_name = self.match_filename_to_function(image_file.name)

            if function_name:
                # Copy to images directory with organized naming
                if function_name not in function_images:
                    function_images[function_name] = []

                # Create unique filename: function_name_index_timestamp.ext
                index = len(function_images[function_name]) + 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{function_name}_{index}_{timestamp}{image_file.suffix}"
                new_path = self.images_dir / new_name

                shutil.copy2(image_file, new_path)
                function_images[function_name].append(new_path)
                matched += 1

                print(f"✅ {function_name}: {image_file.name} -> {new_name}")

        print(f"\n✅ Processed {matched} images for {len(function_images)} functions")
        return function_images

    def create_function_metadata(
        self, function_name: str, image_paths: List[Path]
    ) -> Dict:
        """Create comprehensive metadata for a function with multiple images."""

        # Basic categorization
        category = self.categorize_function(function_name)

        # Create description
        description = self.generate_description(function_name)

        # Prepare image data
        images = []
        for i, img_path in enumerate(image_paths, 1):
            images.append(
                {
                    "path": f"images/{img_path.name}",
                    "filename": img_path.name,
                    "index": i,
                    "size": img_path.stat().st_size if img_path.exists() else 0,
                }
            )

        return {
            "function_name": function_name,
            "description": description,
            "category": category,
            "tags": self.generate_tags(function_name, category),
            "images": images,
            "image_count": len(images),
            "status": "active",
            "last_updated": datetime.now().isoformat(),
            "file_patterns": self.reverse_mapping.get(function_name, []),
        }

    def categorize_function(self, function_name: str) -> str:
        """Categorize function based on name."""
        name_lower = function_name.lower()

        categories = {
            "fractions": ["fraction", "frac", "division_model"],
            "geometry": [
                "polygon",
                "triangle",
                "rectangle",
                "circle",
                "shape",
                "geo",
                "quadrilateral",
                "trapezoid",
                "rhombus",
                "prism",
                "parallelogram",
            ],
            "graphs": [
                "graph",
                "plot",
                "scatter",
                "bar",
                "line_graph",
                "histogram",
                "categorical",
            ],
            "tables": ["table", "data"],
            "number_lines": ["number_line", "numberline"],
            "angles": ["angle", "transversal", "protractor"],
            "measurements": ["measurement", "ruler"],
            "coordinates": ["coordinate", "graphing"],
            "statistics": ["stats", "box_plot", "dot_plot"],
            "base_ten": ["base_ten", "blocks"],
            "other": [],
        }

        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                return category

        return "other"

    def generate_description(self, function_name: str) -> str:
        """Generate description from function name."""
        # Determine action verb
        if function_name.startswith("draw_"):
            action = "Draws"
            content = function_name[5:]
        elif function_name.startswith("create_"):
            action = "Creates"
            content = function_name[7:]
        elif function_name.startswith("generate_"):
            action = "Generates"
            content = function_name[9:]
        elif function_name.startswith("plot_"):
            action = "Plots"
            content = function_name[5:]
        else:
            action = "Generates"
            content = function_name

        # Format content
        content_readable = content.replace("_", " ").title()

        return f"{action} {content_readable.lower()} for educational visualization."

    def generate_tags(self, function_name: str, category: str) -> List[str]:
        """Generate tags for a function."""
        tags = [category]

        name_lower = function_name.lower()

        # Add specific tags
        if "3d" in name_lower:
            tags.append("3d")
        if "decimal" in name_lower:
            tags.append("decimals")
        if "equation" in name_lower:
            tags.append("equations")
        if "comparison" in name_lower:
            tags.append("comparison")
        if "multi" in name_lower:
            tags.append("multiple")

        return list(set(tags))

    def save_function_data(self, function_name: str, metadata: Dict):
        """Save function metadata to JSON file."""
        output_file = self.functions_dir / f"{function_name}.json"
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def update_index(self, all_functions: Dict[str, Dict]):
        """Update the master index file."""
        index_data = {
            "metadata": {
                "total_functions": len(all_functions),
                "total_images": sum(f["image_count"] for f in all_functions.values()),
                "last_updated": datetime.now().isoformat(),
                "version": "2.0.0",
            },
            "functions": [
                {
                    "function_name": name,
                    "category": data["category"],
                    "image_count": data["image_count"],
                    "status": data["status"],
                }
                for name, data in sorted(all_functions.items())
            ],
        }

        with open(self.index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        print(
            f"📊 Updated index: {len(all_functions)} functions, {index_data['metadata']['total_images']} images"
        )

    def run_local_scan(self):
        """Scan local images and organize them."""
        print("\n🚀 Starting local image scan...")
        print("=" * 60)

        # Scan local images
        function_images = self.scan_local_images()

        if not function_images:
            print("\n❌ No images found locally")
            return False

        # Create metadata for each function
        all_metadata = {}
        for function_name, image_paths in function_images.items():
            metadata = self.create_function_metadata(function_name, image_paths)
            self.save_function_data(function_name, metadata)
            all_metadata[function_name] = metadata
            print(f"✅ Saved metadata for {function_name} ({len(image_paths)} images)")

        # Update index
        self.update_index(all_metadata)

        print("\n" + "=" * 60)
        print("📊 LOCAL SCAN SUMMARY")
        print("=" * 60)
        print(f"Functions with images: {len(function_images)}")
        print(f"Total images: {sum(len(imgs) for imgs in function_images.values())}")

        return True

    def run_github_download(self):
        """Download images from GitHub Actions and organize them."""
        print("\n🚀 Starting GitHub Actions download...")
        print("=" * 60)

        # Find latest successful run
        run_id = self.find_latest_successful_run()
        if not run_id:
            return False

        # Download artifacts
        artifact_dir = self.download_artifacts(run_id)
        if not artifact_dir:
            return False

        try:
            # Process downloaded images
            function_images = self.process_downloaded_images(artifact_dir)

            # Merge with existing local images
            local_images = self.scan_local_images()
            for func, imgs in local_images.items():
                if func in function_images:
                    function_images[func].extend(imgs)
                else:
                    function_images[func] = imgs

            # Create metadata
            all_metadata = {}
            for function_name, image_paths in function_images.items():
                metadata = self.create_function_metadata(function_name, image_paths)
                self.save_function_data(function_name, metadata)
                all_metadata[function_name] = metadata

            # Update index
            self.update_index(all_metadata)

            print("\n" + "=" * 60)
            print("📊 DOWNLOAD SUMMARY")
            print("=" * 60)
            print(f"Functions with images: {len(function_images)}")
            print(
                f"Total images: {sum(len(imgs) for imgs in function_images.values())}"
            )

            return True

        finally:
            # Clean up temp directory
            if artifact_dir and artifact_dir.exists():
                shutil.rmtree(artifact_dir, ignore_errors=True)


def main():
    """Main function."""
    print("🖼️  Function Image Retriever")
    print("Organize and retrieve images for all functions")
    print("=" * 60)

    retriever = FunctionImageRetriever()

    print("\nOptions:")
    print("1. Scan local images directory")
    print("2. Download from GitHub Actions")
    print("3. Both (download then merge with local)")

    choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or "1"

    if choice == "1":
        retriever.run_local_scan()
    elif choice == "2":
        retriever.run_github_download()
    elif choice == "3":
        retriever.run_github_download()  # Already merges with local
    else:
        print("❌ Invalid choice")
        return

    print("\n🎉 Complete! Check the data/ directory for results.")


if __name__ == "__main__":
    main()
