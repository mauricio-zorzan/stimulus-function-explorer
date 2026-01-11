"""
New Streamlit app using file-based function data for instant loading.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Increase PIL decompression bomb limit for large educational images
# Default is ~89MP, setting to 200MP to handle stimulus images
Image.MAX_IMAGE_PIXELS = 200_000_000


# Configure Streamlit page FIRST (must be before any other Streamlit commands)
st.set_page_config(
    page_title="Stimulus Function Explorer", page_icon="🎯", layout="wide"
)


# Helper function to display images with version compatibility
def display_image_compat(image_path: str, use_column_width: bool = True, **kwargs):
    """Display image with automatic version compatibility handling."""
    if use_column_width:
        try:
            # Try new width parameter first (Streamlit >= 1.40.0)
            st.image(image_path, width="content", **kwargs)
        except TypeError:
            try:
                st.image(image_path, width="stretch", **kwargs)
            except TypeError:
                st.image(image_path, **kwargs)
    else:
        # For gallery/thumbnails - use default sizing
        st.image(image_path, **kwargs)


# Import AI search functionality (after page config)
try:
    from src.ai_search import AISearchEngine

    AI_SEARCH_AVAILABLE = True
except ImportError:
    AI_SEARCH_AVAILABLE = False

# Note: Standards are now loaded from function data files, not live database queries


class FunctionDataManager:
    """Manages loading and accessing function data from files."""

    def __init__(self):
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.images_dir = self.data_dir / "images"
        self.index_file = self.data_dir / "index.json"

    @st.cache_data(ttl=30)  # Cache for 30 seconds (reduced for faster updates after sync)
    def load_index(_self) -> Dict:
        """Load the master index file with caching."""
        if not _self.index_file.exists():
            return {"metadata": {"total_functions": 0}, "functions": []}

        try:
            with open(_self.index_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error loading index.json: {e}")
            st.info("Creating a fresh index...")
            return {"metadata": {"total_functions": 0}, "functions": []}

    @st.cache_data(ttl=30)  # Cache for 30 seconds (reduced for faster updates)
    def _load_function_data_cached(_self, function_name: str) -> Optional[Dict]:
        """Cached version of load_function_data for internal use."""
        function_file = _self.functions_dir / f"{function_name}.json"
        
        if not function_file.exists():
            return None
        
        try:
            with open(function_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return None
    
    def load_function_data(
        self, function_name: str, show_errors: bool = True
    ) -> Optional[Dict]:
        """Load data for a specific function."""
        # Use cached version for better performance
        data = self._load_function_data_cached(function_name)
        
        if data is None:
            if show_errors:
                function_file = self.functions_dir / f"{function_name}.json"
                if not function_file.exists():
                    st.error(f"Function file not found: {function_file}")
                else:
                    st.error(f"Error loading function data for {function_name}")
            return None
        
        return data

    def search_functions(self, query: str) -> List[Dict]:
        """Search functions by name, category, tags, description, or educational standards."""
        if not query.strip():
            return []

        index_data = self.load_index()
        all_functions = index_data.get("functions", [])
        query_lower = query.lower()

        results = []
        seen_functions = set()  # To avoid duplicates

        for func in all_functions:
            function_name = func.get("function_name", "")
            if function_name in seen_functions:
                continue

            # Search in function name
            if query_lower in func.get("function_name", "").lower():
                results.append(func)
                seen_functions.add(function_name)
                continue

            # Search in category
            if query_lower in func.get("category", "").lower():
                results.append(func)
                seen_functions.add(function_name)
                continue

            # Search in tags
            tags = func.get("tags", [])
            if any(query_lower in tag.lower() for tag in tags):
                results.append(func)
                seen_functions.add(function_name)
                continue

            # Search in description
            description = func.get("description", "").lower()
            if query_lower in description:
                results.append(func)
                seen_functions.add(function_name)
                continue

            # Search in educational standards
            if function_name:
                function_data = self.load_function_data(
                    function_name, show_errors=False
                )
                if function_data:
                    standards = function_data.get("educational_standards", [])
                    for standard in standards:
                        # Search in external_id
                        external_id = standard.get("external_id") or ""
                        if external_id and query_lower in external_id.lower():
                            results.append(func)
                            seen_functions.add(function_name)
                            break

                        # Search in display_name
                        display_name = standard.get("display_name") or ""
                        if display_name and query_lower in display_name.lower():
                            results.append(func)
                            seen_functions.add(function_name)
                            break

                        # Search in description
                        standard_description = standard.get("description") or ""
                        if (
                            standard_description
                            and query_lower in standard_description.lower()
                        ):
                            results.append(func)
                            seen_functions.add(function_name)
                            break

        return results


def display_function_image(image_path: str) -> bool:
    """Display function image and return True if successful."""
    try:
        full_path = Path("data") / image_path
        if full_path.exists():
            display_image_compat(str(full_path))
            return True
        else:
            st.warning(f"Image not found: {image_path}")
            return False
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return False


def display_image_carousel(images: List[Dict], function_name: str):
    """
    Display multiple images with navigation arrows.

    Args:
        images: List of image dicts with 'path', 'filename', 'index' keys
        function_name: Function name for unique session state key
    """
    if not images:
        st.warning("No images available")
        return

    # Initialize session state for this function's carousel
    carousel_key = f"carousel_{function_name}"
    if carousel_key not in st.session_state:
        st.session_state[carousel_key] = 0

    num_images = len(images)
    current_index = st.session_state[carousel_key]

    # Ensure index is within bounds
    current_index = max(0, min(current_index, num_images - 1))
    st.session_state[carousel_key] = current_index

    # Display current image
    current_image = images[current_index]
    image_path = current_image.get("path", "")

    # Create carousel with side-by-side layout
    if num_images > 1:
        col1, col2, col3 = st.columns([1, 10, 1])

        with col1:
            st.write("")  # Spacer for vertical centering
            st.write("")
            st.write("")
            st.write("")
            if st.button(
                "◀",
                key=f"prev_{function_name}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[carousel_key] = (current_index - 1) % num_images
                st.rerun()

        with col2:
            if image_path:
                full_path = Path("data") / image_path
                if full_path.exists():
                    display_image_compat(str(full_path))
                else:
                    st.warning(f"Image not found: {image_path}")

        with col3:
            st.write("")  # Spacer for vertical centering
            st.write("")
            st.write("")
            st.write("")
            if st.button(
                "▶",
                key=f"next_{function_name}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[carousel_key] = (current_index + 1) % num_images
                st.rerun()

        # Image counter and navigation dots below
        st.markdown(
            f"<div style='text-align: center; margin-top: 10px; color: #666; font-size: 0.9em;'>"
            f"Image {current_index + 1} of {num_images}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Clickable dot indicators with better styling
        # Show dots for reasonable number of images, or dropdown for many images
        if num_images > 1:
            st.write("")

            if num_images <= 40:  # Show dots for up to 40 images
                # Create clickable dot buttons - all use primary type for consistent circle styling
                dot_cols = st.columns(num_images)
                for idx, col in enumerate(dot_cols):
                    with col:
                        # Use button for all dots - filled (●) when active, hollow (○) when inactive
                        button_label = "●" if idx == current_index else "○"

                        # Active dot won't trigger action, inactive dots navigate
                        if idx != current_index:
                            if st.button(
                                button_label,
                                key=f"dot_{function_name}_{idx}",
                                type="primary",
                                help=f"Go to image {idx + 1}",
                            ):
                                st.session_state[carousel_key] = idx
                                st.rerun()
                        else:
                            # Show active dot as disabled button (same size, no action)
                            st.button(
                                button_label,
                                key=f"dot_{function_name}_{idx}",
                                type="primary",
                                disabled=True,
                                help=f"Current image {idx + 1}",
                            )
            else:
                # For many images, show a dropdown selector
                selected_index = st.selectbox(
                    "Jump to image:",
                    options=list(range(num_images)),
                    index=current_index,
                    format_func=lambda x: f"Image {x + 1}",
                    key=f"select_{function_name}",
                )
                if selected_index != current_index:
                    st.session_state[carousel_key] = selected_index
                    st.rerun()
    else:
        # Single image - no navigation needed
        if image_path:
            full_path = Path("data") / image_path
            if full_path.exists():
                display_image_compat(str(full_path))
            else:
                st.warning(f"Image not found: {image_path}")


def display_search_results(functions: List[Dict], search_type: str):
    """Display search results in a grid format."""
    # Display search results in a grid with images
    num_cols = 3
    cols = st.columns(num_cols)

    for i, func in enumerate(functions):
        with cols[i % num_cols]:
            # Create function card with custom styling
            display_name = (
                func["function_name"]
                .replace("draw_", "")
                .replace("create_", "")
                .replace("generate_", "")
                .replace("_", " ")
                .title()
            )

            # Try new format first (images array)
            images = func.get("images", [])
            image_path = ""
            
            # Extract image path from images array
            if images and len(images) > 0:
                # Get the first image
                first_image = images[0]
                if isinstance(first_image, dict):
                    image_path = first_image.get("path", "")
                elif isinstance(first_image, str):
                    # If images is a list of strings (old format)
                    image_path = first_image
            else:
                # Fallback to old format
                image_path = func.get("image_path", "")

            # Create a unique key using hash of function name and search type
            import hashlib

            key_hash = hashlib.md5(
                f"{search_type}_{func['function_name']}_{i}".encode()
            ).hexdigest()[:8]

            # Create a container that wraps both the card and button
            with st.container():
                # Show function name with badge
                if images and len(images) > 1:
                    st.markdown(
                        f'<div style="text-align: center; font-weight: bold; color: #2c3e50; margin-bottom: 8px; font-size: 1.1em;">'
                        f'{display_name} <span style="background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.7em;">{len(images)}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="text-align: center; font-weight: bold; color: #2c3e50; margin-bottom: 8px; font-size: 1.1em;">{display_name}</div>',
                        unsafe_allow_html=True,
                    )

                # Display image using Streamlit's native image component
                if image_path:
                    full_path = Path("data") / image_path
                    if full_path.exists():
                        # Use the compatibility function for consistent rendering
                        display_image_compat(str(full_path), use_column_width=True)
                    else:
                        st.caption("⚠️ Image not found")
                else:
                    st.caption("📷 No image available")

                # Clickable button inside the same container
                if st.button(
                    f"View {display_name}",
                    key=f"btn_{key_hash}",
                    use_container_width=True,
                ):
                    # Force rerun by adding timestamp to ensure state change
                    import time

                    st.session_state["selected_function"] = func["function_name"]
                    st.session_state["function_selection_time"] = time.time()
                    st.rerun()


def display_educational_standards(function_data: Dict):
    """Display educational standards for a function in a collapsible format"""
    standards = function_data.get("educational_standards", [])
    standards_last_updated = function_data.get("standards_last_updated", "")

    if standards:
        # Show last updated info
        if standards_last_updated:
            try:
                from datetime import datetime

                last_updated = datetime.fromisoformat(
                    standards_last_updated.replace("Z", "+00:00")
                )
                st.caption(
                    f"Standards last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except:
                st.caption(f"Standards last updated: {standards_last_updated}")

        # Create collapsible section for standards
        with st.expander(
            f"📚 Educational Standards ({len(standards)} standards)", expanded=False
        ):
            for i, standard in enumerate(standards):
                external_id = standard["external_id"]
                display_name = standard["display_name"]
                description = standard["description"]
                specifications = standard.get("stimulus_type_specifications", [])

                # Create a clean display for each standard
                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.code(external_id, language=None)

                    with col2:
                        st.write(f"**{display_name}**")
                        if description and description.strip():
                            st.caption(description)
                    
                    # Display stimulus type specifications if available
                    if specifications:
                        st.write("**Stimulus Type Specifications:**")
                        for j, spec in enumerate(specifications, 1):
                            with st.expander(f"📋 Specification {j}", expanded=False):
                                # Display as markdown
                                st.markdown(spec)

                    if i < len(standards) - 1:  # Add separator between standards
                        st.divider()

    else:
        st.info("📚 No educational standards found for this function.")
        st.info(
            "Run the update script to populate standards: `python update_function_standards.py`"
        )


def generate_plain_text_specification(stimulus_spec: Dict) -> str:
    """Generate a plain text version of the stimulus type specification."""
    lines = []

    title = stimulus_spec.get("title", "")
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    spec_description = stimulus_spec.get("description", "")
    if spec_description:
        lines.append("Description:")
        lines.append(spec_description)
        lines.append("")

    specifications = stimulus_spec.get("specifications", [])
    if specifications:
        lines.append("Specifications:")
        lines.append("")
        for spec in specifications:
            section = spec.get("section", "")
            details = spec.get("details", [])
            if section:
                lines.append(f"{section}:")
                for detail in details:
                    lines.append(f"  - {detail}")
                lines.append("")

    educational_purpose = stimulus_spec.get("educational_purpose", "")
    if educational_purpose:
        lines.append("Educational Purpose:")
        lines.append(educational_purpose)

    return "\n".join(lines)


def sync_repo_and_functions():
    """Sync repository, functions, and images from GitHub repository."""
    import subprocess
    
    st.info("🔄 Syncing repository, functions, and images...")
    
    # Create a status container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Sync the repository first
        status_text.text("Syncing coach-bot repository...")
        progress_bar.progress(10)
        
        sync_script = Path("sync_coach_bot.sh")
        if not sync_script.exists():
            st.error("❌ sync_coach_bot.sh not found!")
            st.info("Please run ./setup_repo_without_token.sh first to set up the repository")
            return False
        
        # Run the sync script
        try:
            result = subprocess.run(
                ["bash", str(sync_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                st.success("✅ Repository synced successfully")
                if result.stdout:
                    st.code(result.stdout, language="text")
            else:
                st.warning(f"⚠️ Repository sync had issues: {result.stderr}")
                if result.stdout:
                    st.code(result.stdout, language="text")
        except subprocess.TimeoutExpired:
            st.warning("⚠️ Repository sync timed out")
        except Exception as e:
            st.warning(f"⚠️ Could not sync repository: {e}")
            st.info("Continuing with existing repository state...")
        
        # Step 2: Auto-update mapping from repository
        status_text.text("Extracting filename patterns from repository functions...")
        progress_bar.progress(30)
        
        # Import the retriever class
        from retrieve_function_images import FunctionImageRetriever
        
        github_token = os.getenv("GITHUB_TOKEN")
        retriever = FunctionImageRetriever(github_token=github_token)
        
        # Auto-update mapping before syncing
        new_mapping = retriever.auto_generate_mapping()
        if new_mapping:
            retriever.update_mapping_file(new_mapping, merge=True)
            # Reload mapping after update
            retriever.mapping = retriever.load_mapping()
            retriever.reverse_mapping = retriever.create_reverse_mapping()
            st.success(f"✅ Updated mapping with {len(new_mapping)} patterns from repository")
        
        # Step 3: Sync images from test files
        status_text.text("Scanning test files and images...")
        progress_bar.progress(60)
        success = retriever.run_test_images_scan()
        if not success:
            st.warning("⚠️ Test images scan failed. Make sure tests have been run to generate images.")
            st.info("💡 Run the tests in coach-bot-repo to generate test images")
        
        progress_bar.progress(95)
        
        if success:
            # Clear all caches to refresh data
            FunctionDataManager.load_index.clear()
            # Create a temporary instance to clear the function data cache
            temp_data_manager = FunctionDataManager()
            temp_data_manager._load_function_data_cached.clear()
            
            progress_bar.progress(100)
            status_text.text("✅ Sync completed successfully!")
            
            st.success("✅ Repository, functions, and images synced successfully!")
            st.info("🔄 Refreshing page to show updated data...")
            
            # Rerun to show updated data
            st.rerun()
        else:
            progress_bar.progress(100)
            status_text.text("❌ Sync completed with errors")
            st.error("❌ Sync completed with errors. Check the logs for details.")
        
        return success
        
    except Exception as e:
        st.error(f"❌ Error during sync: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return False


def sync_standards():
    """Sync educational standards from database."""
    st.info("🔄 Syncing educational standards from database...")
    
    # Create a status container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update educational standards from database
        status_text.text("Updating educational standards from database...")
        progress_bar.progress(10)
        
        try:
            from update_function_standards import update_all_standards
            
            def progress_cb(progress):
                progress_bar.progress(progress)
            
            def status_cb(message):
                status_text.text(message)
            
            standards_success = update_all_standards(
                progress_callback=progress_cb,
                status_callback=status_cb
            )
            
            if standards_success:
                progress_bar.progress(100)
                status_text.text("✅ Standards updated successfully!")
                st.success("✅ Standards updated successfully!")
                
                # Clear caches to refresh data
                FunctionDataManager.load_index.clear()
                temp_data_manager = FunctionDataManager()
                temp_data_manager._load_function_data_cached.clear()
                
                st.info("🔄 Refreshing page to show updated data...")
                st.rerun()
            else:
                progress_bar.progress(100)
                status_text.text("⚠️ Standards update skipped or had issues")
                st.warning("⚠️ Standards update skipped or had issues")
                return False
        except Exception as e:
            st.warning(f"⚠️ Could not update standards: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return False
        
        return standards_success
        
    except Exception as e:
        st.error(f"❌ Error during standards sync: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return False
            
    except ImportError:
        # Fallback: run as subprocess if import fails
        status_text.text("Running sync script...")
        progress_bar.progress(50)
        
        try:
            result = subprocess.run(
                [sys.executable, "retrieve_function_images.py", "--mode", mode],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            progress_bar.progress(90)
            
            if result.returncode == 0:
                # Clear cache
                FunctionDataManager.load_index.clear()
                
                progress_bar.progress(100)
                status_text.text("✅ Sync completed successfully!")
                
                st.success("✅ Functions and images synced successfully!")
                st.info("🔄 Refreshing page to show updated data...")
                
                # Show output if available
                if result.stdout:
                    with st.expander("📋 Sync Details", expanded=False):
                        st.code(result.stdout)
                
                st.rerun()
            else:
                progress_bar.progress(100)
                status_text.text("❌ Sync failed")
                st.error("❌ Sync failed. Check the error details below.")
                
                if result.stderr:
                    st.error(f"Error: {result.stderr}")
                if result.stdout:
                    with st.expander("📋 Sync Output", expanded=True):
                        st.code(result.stdout)
                        
        except subprocess.TimeoutExpired:
            progress_bar.progress(100)
            status_text.text("⏱️ Sync timed out")
            st.error("⏱️ Sync timed out after 5 minutes. The process may still be running.")
        except Exception as e:
            progress_bar.progress(100)
            status_text.text(f"❌ Error: {str(e)}")
            st.error(f"❌ Error running sync: {str(e)}")
    except Exception as e:
        progress_bar.progress(100)
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"❌ Error during sync: {str(e)}")
        st.exception(e)


def display_function_details(function_data: Dict):
    """Display detailed information about a function."""
    st.subheader(f"📋 {function_data['function_name']}")

    # Description
    st.write("**Description:**")
    st.write(function_data.get("description", "No description available"))

    # Educational Standards
    st.write("---")
    display_educational_standards(function_data)

    # Category and tags
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Category:** {function_data.get('category', 'Unknown')}")
    with col4:
        tags = function_data.get("tags", [])
        if tags:
            st.write(f"**Tags:** {', '.join(tags)}")

    # Parameters
    parameters = function_data.get("parameters", {})
    if parameters:
        st.write("**Parameters:**")
        if isinstance(parameters, dict):
            for param, description in parameters.items():
                st.write(f"- **{param}**: {description}")
        elif isinstance(parameters, list):
            for param in parameters:
                st.write(f"- {param}")

    # Stimulus Type Specification
    stimulus_spec = function_data.get("stimulus_type_specification", {})
    if stimulus_spec:
        st.write("---")

        # Create collapsible section for stimulus type specification
        with st.expander("📋 Stimulus Type Specification", expanded=False):
            # Generate the complete specification text
            spec_text = generate_plain_text_specification(stimulus_spec)

            # Display the formatted specification
            title = stimulus_spec.get("title", "")
            if title:
                st.write(f"**{title}**")

            spec_description = stimulus_spec.get("description", "")
            if spec_description:
                st.write("**Description:**")
                st.write(spec_description)

            specifications = stimulus_spec.get("specifications", [])
            if specifications:
                st.write("**Specifications:**")
                for spec in specifications:
                    section = spec.get("section", "")
                    details = spec.get("details", [])
                    if section:
                        st.write(f"**{section}:**")
                        for detail in details:
                            st.write(f"- {detail}")

    # Images - support both new (multiple images) and old (single image) formats
    st.write("---")

    images = function_data.get("images", [])
    if images:
        # New format: multiple images with carousel
        display_image_carousel(images, function_data["function_name"])
    else:
        # Fallback: old single image format
        image_path = function_data.get("image_path", "")
        if image_path:
            display_function_image(image_path)
        else:
            st.warning("No example images available")


def main():
    st.title("🎯 Stimulus Function Explorer")

    # Add custom CSS for function cards
    st.markdown(
        """
    <style>
    /* Container that wraps both card and button */
    .stContainer > div {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ffffff;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stContainer > div:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border-color: #4CAF50;
    }
    
    .function-card {
        border: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        background-color: transparent;
        box-shadow: none;
        text-align: center;
    }
    
    
    .function-name {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    
    .function-category {
        color: #7f8c8d;
        font-size: 0.9em;
        margin-bottom: 15px;
    }
    
    .function-image {
        border-radius: 8px;
        margin: 0 auto;
        display: block;
        max-width: 200px;
        width: 100%;
    }
    
    .function-card img {
        border-radius: 8px;
        margin: 0 auto;
        display: block;
        max-width: 200px;
        width: 100%;
    }
    
    /* Image carousel navigation buttons - arrow buttons only */
    /* Target arrow buttons by their content (they contain arrows) */
    div[data-testid="column"] > div > div > button[data-testid="baseButton-secondary"] {
        min-height: 80px !important;
        font-size: 32px !important;
        font-weight: bold !important;
        background-color: rgba(240, 242, 246, 0.9) !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        width: auto !important;
        height: auto !important;
    }
    
    div[data-testid="column"] > div > div > button[data-testid="baseButton-secondary"]:hover {
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
        transform: scale(1.15) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Dot navigation buttons - all primary type, styled as circles */
    /* Active dot (disabled) - filled green circle */
    button[data-testid="baseButton-primary"]:disabled {
        width: 50px !important;
        height: 50px !important;
        min-width: 50px !important;
        min-height: 50px !important;
        max-width: 50px !important;
        max-height: 50px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        margin: 0 auto !important;
        font-size: 32px !important;
        line-height: 50px !important;
        background-color: #4CAF50 !important;
        border: 3px solid #4CAF50 !important;
        color: white !important;
        opacity: 1 !important;
        cursor: default !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Inactive dots (enabled) - hollow gray circles */
    button[data-testid="baseButton-primary"]:not(:disabled) {
        width: 50px !important;
        height: 50px !important;
        min-width: 50px !important;
        min-height: 50px !important;
        max-width: 50px !important;
        max-height: 50px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        margin: 0 auto !important;
        font-size: 32px !important;
        line-height: 50px !important;
        background-color: white !important;
        border: 3px solid #ccc !important;
        color: #ccc !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
    }
    
    button[data-testid="baseButton-primary"]:not(:disabled):hover {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
        transform: scale(1.15) !important;
    }
    
    /* Style the buttons to be more integrated */
    .stButton > button {
        width: 100% !important;
        margin-top: 10px !important;
        margin-bottom: 0 !important;
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #45a049 !important;
    }
    
    /* Standards styling removed - now using Streamlit native components */
    
    /* Removed old standard styling - now using Streamlit native components */
    </style>
    
    """,
        unsafe_allow_html=True,
    )

    # Initialize data manager
    data_manager = FunctionDataManager()

    # Load index data
    index_data = data_manager.load_index()
    total_functions = index_data.get("metadata", {}).get("total_functions", 0)
    last_updated = index_data.get("metadata", {}).get("last_updated", "Unknown")

    # Sidebar with stats and update button
    st.sidebar.header("📊 Statistics")
    st.sidebar.metric("Total Functions", total_functions)
    st.sidebar.caption(f"Last updated: {last_updated}")
    
    st.sidebar.divider()
    st.sidebar.header("🔄 Sync Options")
    
    # Sync Repository & Functions button
    if st.sidebar.button("📦 Sync Repository & Functions", use_container_width=True, type="primary"):
        st.session_state["sync_repo_triggered"] = True
    
    # Sync Standards button
    if st.sidebar.button("📚 Sync Standards", use_container_width=True, type="secondary"):
        st.session_state["sync_standards_triggered"] = True
    
    # Show sync UI in main area if triggered
    if st.session_state.get("sync_repo_triggered", False):
        st.session_state["sync_repo_triggered"] = False  # Reset flag
        sync_repo_and_functions()
    
    if st.session_state.get("sync_standards_triggered", False):
        st.session_state["sync_standards_triggered"] = False  # Reset flag
        sync_standards()

    # Check if data exists
    if total_functions == 0:
        st.error("⚠️ No function data found!")
        st.info("""
        To generate function data:
        1. Run `python generate_function_data.py` to create examples
        2. This will generate metadata and images for all working functions
        3. Refresh this page to see the results
        """)
        return

    # Search type toggle
    search_type = st.radio(
        "Choose search type:",
        ["🔍 Name-based Search", "🤖 AI-powered Search"],
        horizontal=True,
        key="search_type_toggle",
    )

    # Search functionality based on toggle
    if search_type == "🔍 Name-based Search":
        st.subheader("🔍 Search Functions by Name")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search by function name, category, or tags:",
                placeholder="e.g., draw_base_ten_blocks, fractional, coordinate...",
                key="search_input",
            )

        with col2:
            st.button("🔍 Search", key="search_btn")

        # Show name-based search results
        if search_query:
            functions = data_manager.search_functions(search_query)

            if functions:
                # Load full function data for each result to get images
                functions_with_images = []
                for func in functions:
                    func_name = func.get("function_name")
                    if func_name:
                        # Load full function data directly from file (bypass cache)
                        function_file = data_manager.functions_dir / f"{func_name}.json"
                        if function_file.exists():
                            try:
                                with open(function_file, "r") as f:
                                    file_data = json.load(f)
                                    # Ensure images array exists
                                    if "images" not in file_data:
                                        file_data["images"] = []
                                    functions_with_images.append(file_data)
                            except Exception:
                                # If file read fails, use the search result data
                                functions_with_images.append(func)
                        else:
                            # File doesn't exist, use search result data
                            functions_with_images.append(func)
                
                st.write(f"💡 **Found {len(functions_with_images)} matching functions:**")
                display_search_results(functions_with_images, "name_search")
            else:
                st.info(f"No functions found matching: '{search_query}'")

    else:  # AI-powered search
        st.subheader("🤖 AI-Powered Search")

        col3, col4 = st.columns([3, 1])
        with col3:
            ai_search_query = st.text_input(
                "Describe what you're looking for:",
                placeholder="e.g., functions that draw rectangles, mathematical graphs, geometric shapes...",
                key="ai_search_input",
            )

        with col4:
            ai_search_clicked = st.button("🤖 AI Search", key="ai_search_btn")

        # Initialize session state for AI search
        if "ai_search_results" not in st.session_state:
            st.session_state.ai_search_results = None
        if "ai_search_last_query" not in st.session_state:
            st.session_state.ai_search_last_query = ""

        # Run AI search only when button is clicked or query changes
        should_search = ai_search_clicked or (
            ai_search_query and ai_search_query != st.session_state.ai_search_last_query
        )

        if should_search and ai_search_query:
            if AI_SEARCH_AVAILABLE:
                try:
                    ai_engine = AISearchEngine()
                    ai_function_names = ai_engine.search_functions(ai_search_query)

                    # Convert function names to full function data
                    functions = []
                    for func_name in ai_function_names:
                        # Load full function data directly from file (bypass cache)
                        function_file = data_manager.functions_dir / f"{func_name}.json"
                        if function_file.exists():
                            try:
                                with open(function_file, "r") as f:
                                    file_data = json.load(f)
                                    # Ensure images array exists
                                    if "images" not in file_data:
                                        file_data["images"] = []
                                    functions.append(file_data)
                            except Exception:
                                # If file read fails, try cached version
                                func_data = data_manager.load_function_data(func_name, show_errors=False)
                                if func_data:
                                    functions.append(func_data)
                        else:
                            # File doesn't exist, try cached version
                            func_data = data_manager.load_function_data(func_name, show_errors=False)
                            if func_data:
                                functions.append(func_data)

                    # Store results in session state
                    st.session_state.ai_search_results = functions
                    st.session_state.ai_search_last_query = ai_search_query

                except Exception as e:
                    st.error(f"AI search error: {e}")
                    st.info("Please check your OpenAI API key and try again.")
                    st.session_state.ai_search_results = None
            else:
                st.error(
                    "AI search is not available. Please set OPENAI_API_KEY environment variable."
                )
                st.session_state.ai_search_results = None

        # Display cached AI search results
        if ai_search_query and st.session_state.ai_search_results is not None:
            functions = st.session_state.ai_search_results
            if functions:
                st.success(
                    f"🤖 AI found {len(functions)} relevant functions for: '{st.session_state.ai_search_last_query}'"
                )
                display_search_results(functions, "ai_search")
            else:
                st.info(
                    f"🤖 AI didn't find any functions matching: '{st.session_state.ai_search_last_query}'"
                )
        elif ai_search_query and not AI_SEARCH_AVAILABLE:
            st.error(
                "AI search is not available. Please set OPENAI_API_KEY environment variable."
            )

    # Display selected function details (only if a function is selected)
    if "selected_function" in st.session_state:
        st.divider()

        st.subheader("📋 Function Details")

        selected_function_name = st.session_state["selected_function"]
        function_data = data_manager.load_function_data(selected_function_name)

        if function_data:
            display_function_details(function_data)
        else:
            st.error(f"Could not load data for function: {selected_function_name}")

    # Function Gallery
    st.divider()
    st.subheader("🎨 Function Gallery")
    st.write("Browse all available functions:")

    # Load full function data for gallery (index only has basic info)
    all_functions_index = index_data.get("functions", [])
    
    if all_functions_index:
        # Load full function data for each function to get images
        # Always load directly from file to bypass cache and get fresh data
        all_functions_with_images = []
        for func_index in all_functions_index:
            func_name = func_index.get("function_name")
            if func_name:
                # Always load directly from file (bypass cache) to ensure we get latest data with images
                function_file = data_manager.functions_dir / f"{func_name}.json"
                if function_file.exists():
                    try:
                        with open(function_file, "r") as f:
                            file_data = json.load(f)
                            # Ensure images array exists (even if empty)
                            if "images" not in file_data:
                                file_data["images"] = []
                            # Always add the file data (has images if they exist)
                            all_functions_with_images.append(file_data)
                    except Exception as e:
                        # If file read fails, try cached version as fallback
                        full_func_data = data_manager.load_function_data(func_name, show_errors=False)
                        if full_func_data:
                            all_functions_with_images.append(full_func_data)
                        else:
                            # Last resort: use index data
                            all_functions_with_images.append(func_index)
                else:
                    # File doesn't exist, try cached version
                    full_func_data = data_manager.load_function_data(func_name, show_errors=False)
                    if full_func_data:
                        all_functions_with_images.append(full_func_data)
                    else:
                        # Last resort: use index data
                        all_functions_with_images.append(func_index)
        
        if all_functions_with_images:
            # Verify we have images for debugging
            functions_with_images_count = sum(1 for f in all_functions_with_images if f.get("images"))
            if functions_with_images_count < len(all_functions_with_images):
                # Some functions don't have images - this is expected for some functions
                pass
            display_search_results(all_functions_with_images, "gallery")
        else:
            st.info("No functions available in the gallery.")
    else:
        st.info("No functions available in the gallery.")

    # Show AI search availability message at bottom if not available
    if not AI_SEARCH_AVAILABLE:
        st.info(
            "💡 AI-powered search is not available. To enable it, install OpenAI package and set OPENAI_API_KEY environment variable."
        )


if __name__ == "__main__":
    main()
