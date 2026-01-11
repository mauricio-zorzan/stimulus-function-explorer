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
import argparse
import sys
import hashlib
import base64
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import GitHub reader for function discovery
try:
    from src.github_reader import StimulusFunctionReader
    GITHUB_READER_AVAILABLE = True
except ImportError:
    GITHUB_READER_AVAILABLE = False


class FunctionImageRetriever:
    def __init__(self, github_token: Optional[str] = None):
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.images_dir = self.data_dir / "images"
        self.index_file = self.data_dir / "index.json"
        self.cache_file = self.data_dir / ".test_images_cache.json"

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
            # Try both token formats (token and Bearer)
            token = self.github_token.strip().strip('"').strip("'")
            self.headers["Authorization"] = f"token {token}"
        
        # Initialize GitHub reader for function discovery
        self.github_reader = None
        if GITHUB_READER_AVAILABLE and self.github_token:
            try:
                # Set environment variable for GitHub reader
                os.environ["GITHUB_TOKEN"] = self.github_token
                os.environ["GITHUB_REPO_OWNER"] = self.repo_owner
                os.environ["GITHUB_REPO_NAME"] = self.repo_name
                self.github_reader = StimulusFunctionReader()
            except Exception as e:
                print(f"⚠️  Could not initialize GitHub reader: {e}")

        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"🗺️  Loaded {len(self.mapping)} file pattern mappings")
        print(f"🔄 Reverse mapping covers {len(self.reverse_mapping)} functions")

    def load_cache(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                    return cache
            except Exception as e:
                print(f"⚠️  Could not load cache: {e}")
                return {}
        return {}
    
    def save_cache(self, cache_data: Dict):
        """Save cache to file."""
        try:
            # Ensure data directory exists
            self.cache_file.parent.mkdir(exist_ok=True, parents=True)
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            print(f"💾 Saved cache to {self.cache_file.name}")
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")

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

        # Match each image to a function, avoiding duplicates by filename
        matched = 0
        unmatched = []
        seen_filenames = set()  # Track seen filenames to avoid duplicates

        for image_file in image_files:
            # Skip if we've already seen this filename
            if image_file.name in seen_filenames:
                continue
                
            function_name = self.match_filename_to_function(image_file.name)

            if function_name:
                if function_name not in function_images:
                    function_images[function_name] = []
                function_images[function_name].append(image_file)
                seen_filenames.add(image_file.name)
                matched += 1
            else:
                unmatched.append(image_file.name)

        print(f"✅ Matched {matched} unique images to functions")
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

            if response.status_code == 401:
                print(f"❌ Authentication failed (401): Invalid or expired GITHUB_TOKEN")
                print("💡 Falling back to local scan only")
                return None
            elif response.status_code == 403:
                print(f"❌ Access forbidden (403): Token may lack required permissions")
                print("💡 Falling back to local scan only")
                return None
            elif response.status_code != 200:
                print(f"❌ Failed to get workflow runs: {response.status_code}")
                print("💡 Falling back to local scan only")
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

    def get_image_hash(self, image_path: Path) -> str:
        """Compute MD5 hash of image file."""
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def process_downloaded_images(self, artifact_dir: Path) -> Dict[str, List[Path]]:
        """Process images from downloaded artifacts, avoiding duplicates by content hash."""
        print(f"\n🖼️  Processing images from {artifact_dir}...")

        function_images = {}

        # Build hash map of existing images to avoid duplicates
        existing_hashes = set()
        if self.images_dir.exists():
            for existing_img in self.images_dir.iterdir():
                if existing_img.is_file() and existing_img.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg", ".gif"}:
                    img_hash = self.get_image_hash(existing_img)
                    if img_hash:
                        existing_hashes.add(img_hash)

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
        skipped_duplicates = 0
        for image_file in image_files:
            function_name = self.match_filename_to_function(image_file.name)

            if function_name:
                # Check if this image content already exists
                img_hash = self.get_image_hash(image_file)
                if img_hash and img_hash in existing_hashes:
                    skipped_duplicates += 1
                    continue

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
                existing_hashes.add(img_hash)  # Track this new image
                matched += 1

                print(f"✅ {function_name}: {image_file.name} -> {new_name}")

        if skipped_duplicates > 0:
            print(f"\n⏭️  Skipped {skipped_duplicates} duplicate images (same content already exists)")
        print(f"\n✅ Processed {matched} new images for {len(function_images)} functions")
        return function_images

    def create_function_metadata(
        self, function_name: str, image_paths: List
    ) -> Dict:
        """Create comprehensive metadata for a function with multiple images.
        
        Args:
            function_name: Name of the function
            image_paths: List of image paths (can be Path objects or strings)
        """

        # Basic categorization
        category = self.categorize_function(function_name)

        # Create description
        description = self.generate_description(function_name)

        # Prepare image data
        images = []
        for i, img_path in enumerate(image_paths, 1):
            # Handle both Path objects and strings
            if isinstance(img_path, str):
                # If it's already a relative path string (from copy_test_images_to_local)
                if img_path.startswith("images/"):
                    path_str = img_path
                    filename = Path(img_path).name
                else:
                    # Convert string to Path for processing
                    img_path_obj = Path(img_path)
                    path_str = f"images/{img_path_obj.name}"
                    filename = img_path_obj.name
                # Get file size
                full_path = self.data_dir / path_str
                size = full_path.stat().st_size if full_path.exists() else 0
            else:
                # It's a Path object
                path_str = f"images/{img_path.name}"
                filename = img_path.name
                size = img_path.stat().st_size if img_path.exists() else 0
            
            images.append(
                {
                    "path": path_str,
                    "filename": filename,
                    "index": i,
                    "size": size,
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

    def discover_all_functions_from_repo(self) -> Set[str]:
        """Discover all functions from the GitHub repository."""
        all_functions = set()
        
        # Try GitHub API first if token is available
        if self.github_token:
            try:
                print("\n🔍 Discovering all functions from GitHub repository...")
                
                # Try using GitHub reader if available
                if self.github_reader:
                    try:
                        function_names = self.github_reader.get_stimulus_function_names()
                        all_functions = set(function_names)
                        print(f"✅ Found {len(all_functions)} functions via GitHub reader")
                        return all_functions
                    except Exception as e:
                        print(f"⚠️  GitHub reader failed: {e}, trying direct API...")
                
                # Fallback: Use GitHub API directly to get mcq_payload.json
                file_path = "src/content_generators/question_generator/schemas/json_schemas/mcq_payload.json"
                api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
                
                response = requests.get(api_url, headers=self.headers)
                if response.status_code == 200:
                    content_data = response.json()
                    file_content = base64.b64decode(content_data["content"]).decode("utf-8")
                    
                    # Try to parse as JSON first (more reliable)
                    try:
                        data = json.loads(file_content)
                        
                        # Recursively find all enum fields
                        def find_all_enums(obj):
                            enums = []
                            if isinstance(obj, dict):
                                for key, value in obj.items():
                                    if key == "enum" and isinstance(value, list):
                                        enums.append(value)
                                    else:
                                        enums.extend(find_all_enums(value))
                            elif isinstance(obj, list):
                                for item in obj:
                                    enums.extend(find_all_enums(item))
                            return enums
                        
                        all_enum_lists = find_all_enums(data)
                        function_names = set()
                        
                        # Extract function names from all enums
                        stimulus_patterns = ["draw_", "create_", "generate_", "plot_"]
                        for enum_list in all_enum_lists:
                            for val in enum_list:
                                if isinstance(val, str) and any(val.startswith(pattern) for pattern in stimulus_patterns):
                                    function_names.add(val)
                        
                        if function_names:
                            all_functions = function_names
                            print(f"✅ Found {len(all_functions)} functions via GitHub API (JSON parsing)")
                            return all_functions
                    except json.JSONDecodeError:
                        # Fall back to regex if JSON parsing fails
                        pass
                    
                    # Fallback: Extract function names from enum using regex
                    enum_match = re.search(r'"enum":\s*\[(.*?)\]', file_content, re.DOTALL)
                    if enum_match:
                        enum_content = enum_match.group(1)
                        function_names = re.findall(r'"([^"]+)"', enum_content)
                        all_functions = set(function_names)
                        print(f"✅ Found {len(all_functions)} functions via GitHub API (regex fallback)")
                        return all_functions
                    else:
                        print("⚠️  Could not find enum in mcq_payload.json")
                else:
                    print(f"⚠️  Failed to fetch mcq_payload.json: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️  Error discovering functions from repo: {e}")
        
        # Fallback: Discover from coach-bot-repo (local drawing_functions is deprecated)
        print("\n🔍 Discovering functions from coach-bot-repo...")
        repo_functions = self.discover_functions_from_local_folder()
        if repo_functions:
            all_functions.update(repo_functions)
            print(f"✅ Found {len(repo_functions)} functions from coach-bot-repo")
        
        return all_functions
    
    def discover_functions_from_local_folder(self) -> Set[str]:
        """Discover functions from coach-bot-repo (local drawing_functions is deprecated)."""
        import ast
        functions = set()
        
        # Use coach-bot-repo instead of local drawing_functions
        drawing_functions_dir = Path("coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions")
        
        if not drawing_functions_dir.exists():
            # Fallback: try local drawing_functions if repo doesn't exist (for backwards compatibility)
            drawing_functions_dir = Path("drawing_functions")
            if not drawing_functions_dir.exists():
                return functions
        
        # Pattern to match stimulus function definitions
        stimulus_patterns = ["draw_", "create_", "generate_", "plot_"]
        
        for py_file in drawing_functions_dir.glob("*.py"):
            if py_file.name == "__init__.py" or py_file.name == "main.py" or py_file.name == "constants.py":
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        # Check if it's a stimulus function
                        if any(func_name.startswith(pattern) for pattern in stimulus_patterns):
                            functions.add(func_name)
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        return functions
    
    def extract_filename_pattern_from_function(self, func_code: str, func_name: str) -> Optional[str]:
        """
        Extract filename pattern from a function's file_name assignment.
        Returns the pattern (e.g., 'spinner_shapes' or 'labeled_fraction_models').
        Checks both the main function and any helper functions it calls.
        Handles both time.time() and uuid-based patterns.
        """
        # Look for file_name assignments with f-strings
        # Pattern 1: prefix_{timestamp}.ext
        # Example: f".../spinner_shapes_{int(time.time() * 1000000)}.png"
        pattern1 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)_\{[^}]*time[^}]*\}\.[^"]*"'
        match = re.search(pattern1, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 2: {timestamp}_suffix.ext
        # Example: f".../{int(time.time())}_labeled_fraction_models.png"
        pattern2 = r'file_name\s*=\s*f"[^"]*?/\{[^}]*time[^}]*\}_([a-zA-Z0-9_]+)\.[^"]*"'
        match = re.search(pattern2, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 3: prefix{timestamp}.ext (no underscore before timestamp)
        # Example: f".../data_table{int(time.time())}.png"
        pattern3 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)\{[^}]*time[^}]*\}\.[^"]*"'
        match = re.search(pattern3, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 4: More flexible - find any string before time.time() in f-string
        # This catches patterns like: f".../some_pattern_{int(time.time())}.ext"
        pattern4 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)[_{]\s*int\s*\(\s*time\.time'
        match = re.search(pattern4, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 5: Simple pattern for suffix (works with multi-line f-strings)
        # Example: {int(time.time())}_labeled_fraction_models.
        # This pattern works even when file_name is split across multiple f-strings
        # But only match if it's within a file_name assignment context
        pattern5 = r'file_name\s*=[^=]*?\{[^}]*time[^}]*\}_([a-zA-Z0-9_]+)\.'
        match = re.search(pattern5, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 6: Simple pattern for prefix (works with multi-line f-strings)
        # Example: spinner_shapes_{int(time.time())}.
        # Only match if it's within a file_name assignment context
        pattern6 = r'file_name\s*=[^=]*?([a-zA-Z0-9_]+)_\{[^}]*time[^}]*\}\.'
        match = re.search(pattern6, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 7: UUID-based patterns (prefix_{uuid}.ext)
        # Example: f".../algebraic_tiles_{unique_suffix}.png" where unique_suffix = uuid.uuid4().hex[:8]
        # Look for pattern like: algebraic_tiles_{unique_suffix} or algebraic_tiles_{uuid_var}
        pattern7 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)_\{[^}]*uuid[^}]*\}\.[^"]*"'
        match = re.search(pattern7, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 8: UUID-based with variable (e.g., unique_suffix, suffix, etc.)
        # Look for: algebraic_tiles_{some_var} where some_var is assigned from uuid
        # First check if there's a uuid assignment, then look for the pattern
        uuid_pattern = r'([a-zA-Z0-9_]+)\s*=\s*uuid\.uuid4\(\)'
        uuid_match = re.search(uuid_pattern, func_code, re.MULTILINE | re.DOTALL)
        if uuid_match:
            uuid_var = uuid_match.group(1)
            # Now look for file_name with this variable
            pattern8 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)_\{[^}]*' + re.escape(uuid_var) + r'[^}]*\}\.[^"]*"'
            match = re.search(pattern8, func_code, re.MULTILINE | re.DOTALL)
            if match:
                return match.group(1)
        
        # Pattern 9: Generic pattern with variable suffix (catch-all for UUID patterns)
        # Look for: prefix_{variable}.ext where variable might be uuid-based
        pattern9 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)_\{[a-zA-Z0-9_]+\}\.[^"]*"'
        match = re.search(pattern9, func_code, re.MULTILINE | re.DOTALL)
        if match:
            # Only return if we haven't matched a time-based pattern above
            return match.group(1)
        
        # Pattern 10: Pattern with variable in middle, then time
        # Example: variant_net_{variant}_{int(time.time())}
        # Extract the base pattern before the first variable
        pattern10 = r'file_name\s*=\s*f"[^"]*?/([a-zA-Z0-9_]+)_\{[^}]*\}_\{[^}]*time[^}]*\}\.[^"]*"'
        match = re.search(pattern10, func_code, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1)
        
        return None
    
    def get_stimulus_functions_from_main(self) -> List[str]:
        """
        Parse main.py to extract the STIMULUS_DRAWER list as the source of truth.
        Returns a list of function names.
        """
        main_file = Path("coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions/main.py")
        
        if not main_file.exists():
            print("⚠️  main.py not found in coach-bot-repo")
            return []
        
        try:
            with open(main_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(main_file))
            
            # Find STIMULUS_DRAWER assignment - walk the tree to find it
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'STIMULUS_DRAWER':
                            # Get the list
                            if isinstance(node.value, ast.List):
                                functions = []
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Name):
                                        functions.append(elt.id)
                                    elif isinstance(elt, ast.Attribute):
                                        # Handle cases like module.function
                                        functions.append(elt.attr)
                                if functions:
                                    return functions
                            # If it's not a List, try to find it differently
                            break
            
            # Alternative: search by line number (STIMULUS_DRAWER is around line 403)
            lines = content.split('\n')
            in_stimulus_drawer = False
            functions = []
            for i, line in enumerate(lines):
                if 'STIMULUS_DRAWER' in line and '=' in line:
                    in_stimulus_drawer = True
                    continue
                if in_stimulus_drawer:
                    # Extract function names from lines like "    draw_something,"
                    stripped = line.strip()
                    if stripped.startswith(']'):
                        break
                    if stripped and not stripped.startswith('#'):
                        # Remove trailing comma and whitespace
                        func_name = stripped.rstrip(',').strip()
                        if func_name and not func_name.startswith('#'):
                            functions.append(func_name)
            
            if functions:
                return functions
                
        except Exception as e:
            print(f"⚠️  Error parsing main.py: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return []
    
    def find_function_definition(self, func_name: str, drawing_functions_dir: Path) -> Optional[tuple]:
        """
        Find the file and function definition for a given function name.
        Returns (file_path, func_code) or None.
        """
        for py_file in drawing_functions_dir.glob("*.py"):
            if py_file.name in ["__init__.py", "main.py", "constants.py"]:
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    file_content = f.read()
                
                tree = ast.parse(file_content, filename=str(py_file))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        func_start = node.lineno - 1
                        func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 500
                        func_code = '\n'.join(file_content.split('\n')[func_start:func_end])
                        
                        # If function doesn't have file_name directly, check helper functions it calls
                        if 'file_name' not in func_code and 'savefig' not in func_code:
                            # Find which helper functions this function calls
                            called_helpers = []
                            for call_node in ast.walk(node):
                                if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                                    called_name = call_node.func.id
                                    # Include all called functions (both underscore-prefixed and regular helpers)
                                    # Examples: _draw_pie_models, render_prism_net_variant, etc.
                                    called_helpers.append(called_name)
                            
                            # Look for those specific helper functions in the same file
                            for helper_node in ast.walk(tree):
                                if (isinstance(helper_node, ast.FunctionDef) and 
                                    helper_node.name in called_helpers):
                                    helper_start = helper_node.lineno - 1
                                    helper_end = helper_node.end_lineno if hasattr(helper_node, 'end_lineno') else helper_start + 300
                                    helper_code = '\n'.join(file_content.split('\n')[helper_start:helper_end])
                                    if 'file_name' in helper_code or 'savefig' in helper_code:
                                        func_code += '\n' + helper_code
                                        break
                        
                        return (py_file, func_code)
            except Exception as e:
                continue
        
        return None
    
    def auto_generate_mapping(self) -> Dict[str, str]:
        """
        Auto-generate the function filename mapping by parsing function definitions
        and matching them to actual image files.
        Uses STIMULUS_DRAWER list from main.py as the source of truth.
        """
        print("\n🔍 Auto-generating function filename mapping...")
        print("=" * 60)
        
        new_mapping = {}
        
        # Coach-bot repository (primary source)
        coach_bot_dir = Path("coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions")
        if not coach_bot_dir.exists():
            print("⚠️  coach-bot-repo not found!")
            print("   Run ./setup_repo_without_token.sh to clone the repository")
            return new_mapping
        
        print(f"📁 Using coach-bot-repo as source (local drawing_functions/ is deprecated)")
        print(f"   Location: {coach_bot_dir}")
        
        # Get the source of truth: STIMULUS_DRAWER list from main.py
        print("\n📋 Parsing STIMULUS_DRAWER list from main.py...")
        stimulus_functions = self.get_stimulus_functions_from_main()
        
        if not stimulus_functions:
            print("⚠️  Could not parse STIMULUS_DRAWER list from main.py")
            print("   Falling back to scanning all Python files...")
            # Fallback to old method
            stimulus_functions = []
            for py_file in coach_bot_dir.glob("*.py"):
                if py_file.name in ["__init__.py", "main.py", "constants.py"]:
                    continue
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            if any(func_name.startswith(p) for p in ["draw_", "create_", "generate_", "plot_"]):
                                stimulus_functions.append(func_name)
                except:
                    continue
        
        print(f"✅ Found {len(stimulus_functions)} functions in STIMULUS_DRAWER")
        
        # Get all image files to match against
        image_files = []
        if self.images_dir.exists():
            image_files = list(self.images_dir.glob("*.*"))
            for subdir in self.images_dir.iterdir():
                if subdir.is_dir():
                    image_files.extend(subdir.glob("*.*"))
        
        print(f"📸 Found {len(image_files)} image files to match")
        
        # Process each function from STIMULUS_DRAWER
        functions_processed = 0
        functions_with_patterns = 0
        
        print(f"\n📂 Processing functions from STIMULUS_DRAWER...")
        
        for func_name in stimulus_functions:
            functions_processed += 1
            
            # Find the function definition
            result = self.find_function_definition(func_name, coach_bot_dir)
            if not result:
                print(f"⚠️  Could not find definition for {func_name}")
                continue
            
            py_file, func_code = result
            
            # Extract filename pattern
            pattern = self.extract_filename_pattern_from_function(func_code, func_name)
            
            if pattern:
                functions_with_patterns += 1
                # Find matching images
                matching_images = []
                for img_file in image_files:
                    img_stem = img_file.stem
                    # Remove timestamp from image name
                    cleaned_img = re.sub(r"_\d{10,}$", "", img_stem)
                    cleaned_img = re.sub(r"^\d{10,}_", "", cleaned_img)
                    
                    # Check if pattern matches
                    if (pattern in cleaned_img or 
                        cleaned_img.startswith(pattern) or
                        cleaned_img.endswith(pattern) or
                        pattern in img_stem):
                        matching_images.append(img_file.name)
                
                if matching_images:
                    # Use the pattern as the key (only if not already mapped)
                    if pattern not in new_mapping:
                        new_mapping[pattern] = func_name
                        print(f"✅ {func_name} -> pattern '{pattern}' ({len(matching_images)} matching images)")
                    else:
                        print(f"⚠️  Pattern '{pattern}' already mapped to {new_mapping[pattern]}, skipping {func_name}")
                else:
                    # Still add it even without images (might be a new function)
                    if pattern not in new_mapping:
                        new_mapping[pattern] = func_name
                        print(f"⚠️  {func_name} -> pattern '{pattern}' (no matching images found)")
            else:
                print(f"⚠️  Could not extract pattern from {func_name}")
        
        print(f"\n📊 Processed {functions_processed} functions from STIMULUS_DRAWER")
        print(f"📊 Found patterns for {functions_with_patterns} functions")
        print(f"📊 Generated {len(new_mapping)} mappings")
        
        # Also check GitHub repository for functions not in local folder
        if self.github_token:
            try:
                print("\n🔍 Checking GitHub repository for additional functions...")
                
                # First verify token can access the repository
                repo_test_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
                repo_test = requests.get(repo_test_url, headers=self.headers)
                
                if repo_test.status_code == 401:
                    print("⚠️  GitHub token authentication failed (401)")
                    print("   Token may be invalid, expired, or missing 'repo' scope for private repositories")
                    print("   Skipping GitHub function discovery - using local functions only")
                elif repo_test.status_code == 404:
                    print("⚠️  Repository not found or no access (404)")
                    print("   You may not have access to this private repository")
                    print("   Skipping GitHub function discovery - using local functions only")
                elif repo_test.status_code == 200:
                    # Token is valid and can access repo
                    repo_data = repo_test.json()
                    is_private = repo_data.get("private", False)
                    if is_private:
                        print(f"✅ Accessing private repository: {self.repo_owner}/{self.repo_name}")
                    else:
                        print(f"✅ Accessing public repository: {self.repo_owner}/{self.repo_name}")
                    
                    # Get all functions from GitHub (using the same discovery method)
                    github_functions = self.discover_all_functions_from_repo()
                    local_function_names = {v for v in new_mapping.values()}
                    
                    # Find functions in GitHub that aren't in local
                    github_only_functions = [f for f in github_functions if f not in local_function_names]
                    
                    if github_only_functions:
                        print(f"📋 Found {len(github_only_functions)} functions in GitHub not in local folder")
                        
                        github_processed = 0
                        github_with_patterns = 0
                        
                        # Get function code from GitHub API
                        drawing_functions_path = "src/content_generators/additional_content/stimulus_image/drawing_functions"
                        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents/{drawing_functions_path}"
                        
                        response = requests.get(api_url, headers=self.headers)
                        if response.status_code == 200:
                            files = response.json()
                        
                        for func_name in github_only_functions:
                            try:
                                # Search for the function in GitHub files
                                func_code = None
                                for file_info in files:
                                    if file_info["type"] == "file" and file_info["name"].endswith(".py"):
                                        file_url = file_info["url"]
                                        file_response = requests.get(file_url, headers=self.headers)
                                        if file_response.status_code == 200:
                                            file_content = base64.b64decode(file_response.json()["content"]).decode("utf-8")
                                            if f"def {func_name}" in file_content:
                                                # Extract function code using AST
                                                try:
                                                    tree = ast.parse(file_content)
                                                    for node in ast.walk(tree):
                                                        if isinstance(node, ast.FunctionDef) and node.name == func_name:
                                                            func_start = node.lineno - 1
                                                            func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 200
                                                            func_code = '\n'.join(file_content.split('\n')[func_start:func_end])
                                                            break
                                                except:
                                                    # Fallback: find function in file content
                                                    lines = file_content.split('\n')
                                                    in_function = False
                                                    func_lines = []
                                                    indent_level = None
                                                    for i, line in enumerate(lines):
                                                        if f"def {func_name}" in line:
                                                            in_function = True
                                                            indent_level = len(line) - len(line.lstrip())
                                                            func_lines.append(line)
                                                        elif in_function:
                                                            if line.strip() and not line.startswith(' ' * (indent_level + 1)) and not line.startswith('\t'):
                                                                break
                                                            func_lines.append(line)
                                                    if func_lines:
                                                        func_code = '\n'.join(func_lines)
                                                break
                                
                                if func_code:
                                    github_processed += 1
                                    
                                    # Extract filename pattern
                                    pattern = self.extract_filename_pattern_from_function(func_code, func_name)
                                    
                                    if pattern:
                                        github_with_patterns += 1
                                        # Find matching images
                                        matching_images = []
                                        for img_file in image_files:
                                            img_stem = img_file.stem
                                            # Remove timestamp from image name
                                            cleaned_img = re.sub(r"_\d{10,}$", "", img_stem)
                                            cleaned_img = re.sub(r"^\d{10,}_", "", cleaned_img)
                                            
                                            # Check if pattern matches
                                            if (pattern in cleaned_img or 
                                                cleaned_img.startswith(pattern) or
                                                cleaned_img.endswith(pattern) or
                                                pattern in img_stem):
                                                matching_images.append(img_file.name)
                                        
                                        if matching_images:
                                            new_mapping[pattern] = func_name
                                            print(f"✅ {func_name} -> pattern '{pattern}' ({len(matching_images)} matching images) [GitHub]")
                                        else:
                                            # Still add it even without images (might be a new function)
                                            new_mapping[pattern] = func_name
                                            print(f"⚠️  {func_name} -> pattern '{pattern}' (no matching images found) [GitHub]")
                                    else:
                                        print(f"⚠️  Could not extract pattern from {func_name} [GitHub]")
                                else:
                                    print(f"⚠️  Could not find code for {func_name} in GitHub")
                            except Exception as e:
                                print(f"⚠️  Error processing {func_name} from GitHub: {e}")
                            continue
                        
                            print(f"\n📊 Processed {github_processed} GitHub functions")
                            print(f"📊 Found patterns for {github_with_patterns} GitHub functions")
                        else:
                            print(f"⚠️  Could not access GitHub repository contents: {response.status_code}")
                            if response.status_code == 401:
                                print("   Token may need 'repo' scope for private repository access")
                    else:
                        print("✅ All functions found locally - no additional functions in GitHub")
                else:
                    print(f"⚠️  Unexpected error accessing repository: {repo_test.status_code}")
                    print(f"   {repo_test.text[:200]}")
                
            except Exception as e:
                print(f"⚠️  Error checking GitHub repository: {e}")
        
        print(f"\n📊 Total mappings generated: {len(new_mapping)}")
        
        return new_mapping
    
    def update_mapping_file(self, new_mapping: Dict[str, str], merge: bool = True):
        """
        Update the function_filename_mapping.json file with new mappings.
        If merge=True, merge with existing mappings (new mappings take precedence).
        """
        mapping_file = Path("function_filename_mapping.json")
        
        # Load existing mapping if merging
        existing_mapping = {}
        if merge and mapping_file.exists():
            with open(mapping_file) as f:
                existing_mapping = json.load(f)
        
        # Merge mappings (new takes precedence)
        if merge:
            existing_mapping.update(new_mapping)
            final_mapping = existing_mapping
        else:
            final_mapping = new_mapping
        
        # Sort by key for readability
        final_mapping = dict(sorted(final_mapping.items()))
        
        # Save to file
        with open(mapping_file, "w") as f:
            json.dump(final_mapping, f, indent=2)
        
        print(f"✅ Updated {mapping_file} with {len(final_mapping)} mappings")
        print(f"   ({len(new_mapping)} new/updated, {len(existing_mapping) - len(new_mapping)} existing)")

    def run_local_scan(self):
        """Scan local images and organize them."""
        print("\n🚀 Starting local image scan...")
        print("=" * 60)

        # ONLY include functions that are in the mapping (have test image patterns)
        # This strictly filters out helper functions that don't have test images
        functions_with_mapping = set(self.reverse_mapping.keys())
        print(f"📋 Functions with test image mappings: {len(functions_with_mapping)}")

        # Scan local images
        function_images = self.scan_local_images()

        # Filter images to only those for functions in the mapping
        filtered_function_images = {
            func: imgs for func, imgs in function_images.items()
            if func in functions_with_mapping
        }

        # Load existing function data (ONLY functions in mapping)
        existing_functions = {}
        if self.functions_dir.exists():
            for func_file in self.functions_dir.glob("*.json"):
                try:
                    with open(func_file, "r") as f:
                        func_data = json.load(f)
                        func_name = func_data.get("function_name", func_file.stem)
                        # ONLY keep functions that are in mapping
                        if func_name in functions_with_mapping:
                            existing_functions[func_name] = func_data
                        else:
                            # Remove helper functions that aren't in mapping
                            print(f"🗑️  Removing helper function: {func_name} (not in mapping)")
                            func_file.unlink()  # Delete the file
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Only add new functions that are in mapping
        new_functions_count = 0
        for func_name in functions_with_mapping:
            if func_name not in existing_functions:
                # Create metadata for function (may have images from scan)
                image_paths = filtered_function_images.get(func_name, [])
                metadata = self.create_function_metadata(func_name, image_paths)
                existing_functions[func_name] = metadata
                self.save_function_data(func_name, metadata)
                new_functions_count += 1
        
        if new_functions_count > 0:
            print(f"✅ Created metadata for {new_functions_count} new functions with test images")

        if not filtered_function_images and not existing_functions:
            print("\n⚠️  No images found locally and no existing functions with test images")
            return False

        # Update metadata for functions with images (only those in mapping)
        all_metadata = existing_functions.copy()  # Start with existing functions
        updated_count = 0
        for function_name, image_paths in filtered_function_images.items():
            metadata = self.create_function_metadata(function_name, image_paths)
            self.save_function_data(function_name, metadata)
            all_metadata[function_name] = metadata
            updated_count += 1
            print(f"✅ Updated metadata for {function_name} ({len(image_paths)} images)")

        # Update index with ONLY functions that are in mapping (have test images)
        self.update_index(all_metadata)

        print("\n" + "=" * 60)
        print("📊 LOCAL SCAN SUMMARY")
        print("=" * 60)
        print(f"Total functions with test images: {len(all_metadata)}")
        print(f"Functions with images: {len(filtered_function_images)}")
        print(f"Functions updated: {updated_count}")
        print(f"Total images: {sum(len(imgs) for imgs in filtered_function_images.values())}")
        
        # Report filtered functions
        filtered_out = len(function_images) - len(filtered_function_images)
        if filtered_out > 0:
            print(f"⏭️  Filtered out {filtered_out} helper functions without test image mappings")

        return True

    def find_test_image_directory(self) -> Optional[Path]:
        """
        Find the directory where test images are stored.
        Based on conftest.py, images are saved to content/tests relative to repo root.
        """
        # Try multiple possible locations
        possible_paths = [
            Path("coach-bot-repo/content/tests"),
            Path("coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions/tests/../content/tests"),
            Path("coach-bot-repo/../content/tests"),  # If repo is nested
        ]
        
        for path in possible_paths:
            abs_path = path.resolve()
            if abs_path.exists() and abs_path.is_dir():
                # Check if it has image files
                image_files = list(abs_path.glob("*.webp")) + list(abs_path.glob("*.png"))
                if image_files:
                    return abs_path
        
        return None
    
    def parse_test_files_for_functions(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """
        Parse all test files to find which functions are being tested.
        Uses cache to only re-parse changed files.
        Returns: {function_name: [test_file_paths]}
        """
        test_dir = Path("coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions/tests")
        
        if not test_dir.exists():
            print(f"⚠️  Test directory not found: {test_dir}")
            return {}
        
        test_files = list(test_dir.glob("test_*.py"))
        
        # Load cache
        cache = self.load_cache() if use_cache else {}
        cached_test_files = cache.get("test_files", {})
        cached_function_to_tests = cache.get("function_to_tests", {})
        
        # Check which files have changed
        changed_files = []
        current_file_times = {}
        
        for test_file in test_files:
            try:
                mtime = test_file.stat().st_mtime
                current_file_times[str(test_file)] = mtime
                cached_mtime = cached_test_files.get(str(test_file))
                # Allow 1 second tolerance for filesystem precision differences
                if cached_mtime is None or abs(cached_mtime - mtime) > 1.0:
                    changed_files.append(test_file)
            except OSError:
                # File might have been deleted
                changed_files.append(test_file)
        
        if not changed_files and cached_function_to_tests:
            print(f"\n📋 Using cached test file parsing ({len(test_files)} files, all up-to-date)")
            print(f"   ⚡ Cache hit! Skipping parsing of {len(test_files)} files")
            return cached_function_to_tests
        
        if changed_files:
            print(f"\n📋 Parsing {len(changed_files)} changed test files (of {len(test_files)} total)...")
            print(f"   📝 {len(test_files) - len(changed_files)} files using cache")
        else:
            print(f"\n📋 Parsing {len(test_files)} test files (no cache found)...")
        
        # Start with cached results
        function_to_tests = cached_function_to_tests.copy()
        
        # Re-parse only changed files
        for test_file in changed_files:
            try:
                # Remove old entries for this file
                for func_name in list(function_to_tests.keys()):
                    if str(test_file) in function_to_tests[func_name]:
                        function_to_tests[func_name].remove(str(test_file))
                        if not function_to_tests[func_name]:
                            del function_to_tests[func_name]
                
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Parse AST to find function calls
                tree = ast.parse(content, filename=str(test_file))
                
                # Find all function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Check if it's a direct function call (not a method call)
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                            # Check if it's a stimulus function (starts with draw_, create_, generate_, plot_)
                            if any(func_name.startswith(prefix) for prefix in ["draw_", "create_", "generate_", "plot_"]):
                                if func_name not in function_to_tests:
                                    function_to_tests[func_name] = []
                                if str(test_file) not in function_to_tests[func_name]:
                                    function_to_tests[func_name].append(str(test_file))
                        # Also check for attribute calls like module.function()
                        elif isinstance(node.func, ast.Attribute):
                            func_name = node.func.attr
                            if any(func_name.startswith(prefix) for prefix in ["draw_", "create_", "generate_", "plot_"]):
                                if func_name not in function_to_tests:
                                    function_to_tests[func_name] = []
                                if str(test_file) not in function_to_tests[func_name]:
                                    function_to_tests[func_name].append(str(test_file))
            except Exception as e:
                print(f"⚠️  Error parsing {test_file.name}: {e}")
                continue
        
        # Update cache
        if use_cache:
            cache["test_files"] = current_file_times
            cache["function_to_tests"] = function_to_tests
            self.save_cache(cache)
        
        print(f"✅ Found {len(function_to_tests)} functions in tests")
        return function_to_tests
    
    def scan_test_images(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """
        Scan test image directory and match images to functions using filename patterns.
        Uses cache to only re-scan if images changed.
        Returns: {function_name: [image_paths]}
        """
        print("\n🔍 Scanning test images...")
        print("=" * 60)
        
        # Find test image directory
        test_image_dir = self.find_test_image_directory()
        if not test_image_dir:
            print("⚠️  Test image directory not found")
            print("   Expected location: coach-bot-repo/content/tests")
            print("   Run the tests first to generate images")
            return {}
        
        print(f"📁 Found test image directory: {test_image_dir}")
        
        # Get all image files with modification times
        image_files = list(test_image_dir.glob("*.webp")) + list(test_image_dir.glob("*.png"))
        print(f"📸 Found {len(image_files)} test images")
        
        if not image_files:
            print("⚠️  No test images found")
            return {}
        
        # Load cache
        cache = self.load_cache() if use_cache else {}
        cached_images = cache.get("test_images", {})  # {image_path: mtime}
        cached_function_images = cache.get("function_images", {})  # {function_name: [image_paths]}
        
        # Check which images are new or changed
        current_image_times = {}
        for img in image_files:
            try:
                current_image_times[str(img)] = img.stat().st_mtime
            except OSError:
                # File might have been deleted, skip it
                continue
        
        new_or_changed = []
        for img in image_files:
            img_str = str(img)
            if img_str not in cached_images:
                new_or_changed.append(img)
            else:
                cached_mtime = cached_images[img_str]
                current_mtime = current_image_times.get(img_str)
                if current_mtime is None or abs(cached_mtime - current_mtime) > 1.0:  # Allow 1 second tolerance
                    new_or_changed.append(img)
        
        if not new_or_changed and cached_function_images:
            print(f"✅ Using cached image matching (all {len(image_files)} images up-to-date)")
            print(f"   ⚡ Cache hit! Skipping matching of {len(image_files)} images")
            return cached_function_images
        
        if new_or_changed:
            print(f"🔄 Re-matching {len(new_or_changed)} new/changed images...")
            print(f"   📝 {len(image_files) - len(new_or_changed)} images using cache")
        else:
            print(f"🔄 Matching {len(image_files)} images (no cache found)...")
        
        # Start with cached results, but remove entries for changed images
        function_images = {}
        for func_name, img_paths in cached_function_images.items():
            # Only keep images that haven't changed
            unchanged_paths = [
                path for path in img_paths
                if path not in [str(img) for img in new_or_changed]
            ]
            if unchanged_paths:
                function_images[func_name] = unchanged_paths
        
        # Match new/changed images to functions using filename patterns
        for pattern, func_name in self.mapping.items():
            matching_images = []
            for img_file in new_or_changed:
                img_stem = img_file.stem
                # Remove timestamp/UUID from image name
                cleaned_img = re.sub(r"_\d{10,}$", "", img_stem)
                cleaned_img = re.sub(r"^\d{10,}_", "", cleaned_img)
                cleaned_img = re.sub(r"_[a-f0-9]{8}$", "", cleaned_img)  # Remove UUID suffix
                cleaned_img = re.sub(r"^[a-f0-9]{8}_", "", cleaned_img)  # Remove UUID prefix
                
                # Check if pattern matches
                if (pattern in cleaned_img or 
                    cleaned_img.startswith(pattern) or
                    cleaned_img.endswith(pattern) or
                    pattern in img_stem):
                    matching_images.append(str(img_file))
            
            if matching_images:
                if func_name not in function_images:
                    function_images[func_name] = []
                function_images[func_name].extend(matching_images)
        
        # Update cache
        if use_cache:
            cache["test_images"] = current_image_times
            cache["function_images"] = function_images
            self.save_cache(cache)
        
        print(f"✅ Matched {len(function_images)} functions to test images")
        total_images = sum(len(imgs) for imgs in function_images.values())
        print(f"📊 Total images matched: {total_images}")
        
        return function_images
    
    def copy_test_images_to_local(self, function_images: Dict[str, List[str]], use_cache: bool = True) -> Dict[str, List[str]]:
        """
        Copy test images to local data/images/ directory and return new paths.
        Uses cache to only copy new/changed images.
        Returns: {function_name: [local_image_paths]}
        """
        print("\n📦 Copying test images to local directory...")
        
        # Load cache
        cache = self.load_cache() if use_cache else {}
        cached_copied_images = cache.get("copied_images", {})  # {source_path: dest_path}
        
        local_image_paths = {}
        copied_count = 0
        skipped_count = 0
        
        for func_name, image_paths in function_images.items():
            local_paths = []
            for img_path in image_paths:
                img_file = Path(img_path)
                if not img_file.exists():
                    continue
                
                # Create destination path
                dest_path = self.images_dir / img_file.name
                
                # Check cache first
                if use_cache and img_path in cached_copied_images:
                    cached_dest = cached_copied_images[img_path]
                    if Path(cached_dest).exists():
                        # Verify it's still valid (same size)
                        if dest_path.exists() and dest_path.stat().st_size == img_file.stat().st_size:
                            skipped_count += 1
                            local_paths.append(str(dest_path.relative_to(self.data_dir)))
                            continue
                
                # Check if already exists (by content hash to avoid duplicates)
                if dest_path.exists():
                    # Compare file sizes and modification times
                    if (dest_path.stat().st_size == img_file.stat().st_size and
                        dest_path.stat().st_mtime >= img_file.stat().st_mtime):
                        skipped_count += 1
                        local_paths.append(str(dest_path.relative_to(self.data_dir)))
                        # Update cache
                        if use_cache:
                            cached_copied_images[img_path] = str(dest_path.relative_to(self.data_dir))
                        continue
                
                # Copy the file
                try:
                    shutil.copy2(img_file, dest_path)
                    local_paths.append(str(dest_path.relative_to(self.data_dir)))
                    copied_count += 1
                    # Update cache
                    if use_cache:
                        cached_copied_images[img_path] = str(dest_path.relative_to(self.data_dir))
                except Exception as e:
                    print(f"⚠️  Error copying {img_file.name}: {e}")
                    continue
            
            if local_paths:
                local_image_paths[func_name] = local_paths
        
        # Save cache
        if use_cache:
            cache["copied_images"] = cached_copied_images
            self.save_cache(cache)
        
        if copied_count == 0 and skipped_count > 0:
            print(f"✅ All {skipped_count} images already copied (cache hit!)")
        else:
            print(f"✅ Copied {copied_count} images, skipped {skipped_count} (cached/duplicates)")
        return local_image_paths
    
    def run_test_images_scan(self):
        """
        Scan test files, find test images, and link them to functions.
        This replaces the GitHub Actions and local scan methods.
        """
        print("\n🚀 Starting test images scan...")
        print("=" * 60)
        
        # Parse test files to find which functions are tested (with caching)
        function_to_tests = self.parse_test_files_for_functions(use_cache=True)
        
        if not function_to_tests:
            print("⚠️  No functions found in test files")
            return False
        
        # Scan test images and match to functions (with caching)
        function_images = self.scan_test_images(use_cache=True)
        
        if not function_images:
            print("⚠️  No test images found or matched")
            print("   Make sure tests have been run to generate images")
            return False
        
        # Copy images to local directory (with caching)
        local_image_paths = self.copy_test_images_to_local(function_images, use_cache=True)
        
        if not local_image_paths:
            print("⚠️  No images were copied")
            return False
        
        # ONLY include functions that are in the mapping (have test image patterns)
        functions_with_mapping = set(self.reverse_mapping.keys())
        
        # Load existing function data
        existing_functions = {}
        if self.functions_dir.exists():
            for func_file in self.functions_dir.glob("*.json"):
                try:
                    with open(func_file, "r") as f:
                        func_data = json.load(f)
                        func_name = func_data.get("function_name", func_file.stem)
                        if func_name in functions_with_mapping:
                            existing_functions[func_name] = func_data
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Update or create metadata for functions with test images
        updated_count = 0
        new_count = 0
        
        for func_name, image_paths in local_image_paths.items():
            if func_name not in functions_with_mapping:
                continue  # Skip helper functions
            
            metadata = self.create_function_metadata(func_name, image_paths)
            self.save_function_data(func_name, metadata)
            
            if func_name in existing_functions:
                updated_count += 1
            else:
                new_count += 1
        
        # Update index
        self.update_index(all_metadata)
        
        print("\n" + "=" * 60)
        print("📊 TEST IMAGES SCAN SUMMARY")
        print("=" * 60)
        print(f"Functions found in tests: {len(function_to_tests)}")
        print(f"Functions with test images: {len(local_image_paths)}")
        print(f"New functions: {new_count}")
        print(f"Updated functions: {updated_count}")
        print(f"Total images: {sum(len(imgs) for imgs in local_image_paths.values())}")

        return True

    def run_github_download(self):
        """Download images from GitHub Actions and organize them."""
        print("\n🚀 Starting GitHub Actions download...")
        print("=" * 60)

        # Check if we have a token - if not, try to proceed anyway (might work for public repos)
        if not self.github_token:
            print("⚠️  No GITHUB_TOKEN found")
            print("   Attempting to access GitHub Actions (may fail for private repos)")
            print("   For private repos, you'll need a token with 'repo' scope")
            print("   Falling back to local scan...")
            return self.run_local_scan()

        # Only include functions that are in the mapping (have test image patterns)
        # This filters out helper functions that don't have test images
        functions_with_mapping = set(self.reverse_mapping.keys())
        print(f"📋 Functions with test image mappings: {len(functions_with_mapping)}")

        # Load existing function data (ONLY functions in mapping)
        existing_functions = {}
        if self.functions_dir.exists():
            for func_file in self.functions_dir.glob("*.json"):
                try:
                    with open(func_file, "r") as f:
                        func_data = json.load(f)
                        func_name = func_data.get("function_name", func_file.stem)
                        # ONLY keep functions that are in mapping
                        if func_name in functions_with_mapping:
                            existing_functions[func_name] = func_data
                        else:
                            # Remove helper functions that aren't in mapping
                            print(f"🗑️  Removing helper function: {func_name} (not in mapping)")
                            func_file.unlink()  # Delete the file
                except (json.JSONDecodeError, KeyError):
                    pass

        # Find latest successful run
        run_id = self.find_latest_successful_run()
        if not run_id:
            print("⚠️  No successful workflow run found. Falling back to local scan.")
            # Fall back to local scan but preserve existing functions
            return self.run_local_scan()

        # Download artifacts
        artifact_dir = self.download_artifacts(run_id)
        if not artifact_dir:
            return False

        try:
            # Process downloaded images
            function_images = self.process_downloaded_images(artifact_dir)

            # Merge with existing local images, avoiding duplicates
            local_images = self.scan_local_images()
            for func, imgs in local_images.items():
                if func in function_images:
                    # Get existing image filenames to avoid duplicates
                    existing_filenames = {img.name for img in function_images[func]}
                    # Only add images that don't already exist
                    for img in imgs:
                        if img.name not in existing_filenames:
                            function_images[func].append(img)
                            existing_filenames.add(img.name)
                else:
                    function_images[func] = imgs

            # Only include functions that have images AND are in the mapping
            # Filter out any helper functions that don't have test images
            filtered_function_images = {
                func: imgs for func, imgs in function_images.items()
                if func in functions_with_mapping
            }
            
            # Start with existing functions (already filtered to only those with mappings)
            all_metadata = existing_functions.copy()
            
            # Update metadata for functions with images (only if in mapping)
            for function_name, image_paths in filtered_function_images.items():
                metadata = self.create_function_metadata(function_name, image_paths)
                self.save_function_data(function_name, metadata)
                all_metadata[function_name] = metadata

            # Update index with only functions that have test images
            self.update_index(all_metadata)

            print("\n" + "=" * 60)
            print("📊 DOWNLOAD SUMMARY")
            print("=" * 60)
            print(f"Total functions with test images: {len(all_metadata)}")
            print(f"Functions with images: {len(filtered_function_images)}")
            print(
                f"Total images: {sum(len(imgs) for imgs in filtered_function_images.values())}"
            )
            
            # Report if any functions were filtered out
            filtered_out = len(function_images) - len(filtered_function_images)
            if filtered_out > 0:
                print(f"⏭️  Filtered out {filtered_out} helper functions without test images")

            return True

        finally:
            # Clean up temp directory
            if artifact_dir and artifact_dir.exists():
                shutil.rmtree(artifact_dir, ignore_errors=True)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Retrieve and organize images for functions from GitHub Actions"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "github", "both", "update-mapping", "test-images"],
        default="test-images",
        help="Retrieval mode: 'local' (scan local only), 'github' (download from GitHub), 'both' (download then merge with local), 'update-mapping' (auto-generate mapping), 'test-images' (scan test files and images - recommended). Default: test-images",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub token for API access (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--no-auto-update-mapping",
        action="store_true",
        help="Skip auto-updating mapping before sync (default: auto-update enabled)",
    )

    args = parser.parse_args()

    print("🖼️  Function Image Retriever")
    print("Organize and retrieve images for all functions")
    print("=" * 60)

    retriever = FunctionImageRetriever(github_token=args.github_token)

    # If mode is update-mapping, just update the mapping
    if args.mode == "update-mapping":
        new_mapping = retriever.auto_generate_mapping()
        retriever.update_mapping_file(new_mapping, merge=True)
        print("\n🎉 Mapping updated! You can now run sync to use the new mappings.")
        sys.exit(0)

    # Auto-update mapping before sync operations (unless explicitly disabled)
    if not args.no_auto_update_mapping:
        print("\n🔄 Auto-updating function filename mapping...")
        new_mapping = retriever.auto_generate_mapping()
        if new_mapping:
            retriever.update_mapping_file(new_mapping, merge=True)
            # Reload mapping after update
            retriever.mapping = retriever.load_mapping()
            retriever.reverse_mapping = retriever.create_reverse_mapping()
            print(f"✅ Mapping updated with {len(new_mapping)} patterns")
        else:
            print("⚠️  No new mappings found, using existing mapping")

    # If mode is specified, run non-interactively
    if args.mode:
        choice = args.mode
    else:
        # Interactive mode
        print("\nOptions:")
        print("1. Scan local images directory (deprecated)")
        print("2. Download from GitHub Actions (deprecated)")
        print("3. Both (download then merge with local) (deprecated)")
        print("4. Update mapping (auto-generate function filename mapping)")
        print("5. Scan test files and images (recommended)")

        choice = input("\nSelect option (1/2/3/4/5) [default: 5]: ").strip() or "5"

    # Map choice to action
    if choice == "1" or choice == "local":
        success = retriever.run_local_scan()
    elif choice == "2" or choice == "github":
        success = retriever.run_github_download()
    elif choice == "3" or choice == "both":
        success = retriever.run_github_download()  # Already merges with local
    elif choice == "4" or choice == "update-mapping":
        new_mapping = retriever.auto_generate_mapping()
        retriever.update_mapping_file(new_mapping, merge=True)
        success = True
    elif choice == "5" or choice == "test-images":
        success = retriever.run_test_images_scan()
    else:
        print("❌ Invalid choice")
        sys.exit(1)

    if success:
        print("\n🎉 Complete! Check the data/ directory for results.")
        sys.exit(0)
    else:
        print("\n❌ Retrieval failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
