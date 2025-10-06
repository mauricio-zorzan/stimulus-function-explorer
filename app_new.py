"""
New Streamlit app using file-based function data for instant loading.
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import AI search functionality
try:
    from src.ai_search import AISearchEngine

    AI_SEARCH_AVAILABLE = True
except ImportError:
    AI_SEARCH_AVAILABLE = False
    st.warning(
        "AI search not available. Install OpenAI package and set OPENAI_API_KEY to enable AI search."
    )

# Note: Standards are now loaded from function data files, not live database queries

# Configure Streamlit page
st.set_page_config(
    page_title="Stimulus Function Explorer", page_icon="🎯", layout="wide"
)


class FunctionDataManager:
    """Manages loading and accessing function data from files."""

    def __init__(self):
        self.data_dir = Path("data")
        self.functions_dir = self.data_dir / "functions"
        self.images_dir = self.data_dir / "images"
        self.index_file = self.data_dir / "index.json"

    def load_index(self) -> Dict:
        """Load the master index file."""
        if not self.index_file.exists():
            return {"metadata": {"total_functions": 0}, "functions": []}

        try:
            with open(self.index_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error loading index.json: {e}")
            st.info("Creating a fresh index...")
            return {"metadata": {"total_functions": 0}, "functions": []}

    def load_function_data(self, function_name: str) -> Optional[Dict]:
        """Load data for a specific function."""
        function_file = self.functions_dir / f"{function_name}.json"

        if not function_file.exists():
            st.error(f"Function file not found: {function_file}")
            return None

        try:
            with open(function_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON for {function_name}: {e}")
            return None
        except Exception:
            st.error(f"Error loading function data for {function_name}")
            return None

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
                function_data = self.load_function_data(function_name)
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
            st.image(str(full_path), use_column_width=True)
            return True
        else:
            st.warning(f"Image not found: {image_path}")
            return False
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return False


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

            # Function image (standardized size)
            image_html = ""
            image_path = func.get("image_path", "")
            if image_path:
                try:
                    full_path = Path("data") / image_path
                    if full_path.exists():
                        # Convert image to base64 for embedding in HTML
                        with open(full_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            image_html = f'<img src="data:image/webp;base64,{img_data}" class="function-image" alt="{display_name}">'
                    else:
                        image_html = '<div style="color: #999; text-align: center; padding: 20px;">Image not available</div>'
                except Exception:
                    image_html = '<div style="color: #999; text-align: center; padding: 20px;">Image preview unavailable</div>'
            else:
                image_html = '<div style="color: #999; text-align: center; padding: 20px;">No image available</div>'

            # Create clickable card with everything inside
            st.markdown(
                f"""
            <div class="function-card">
                <div class="function-name">{display_name}</div>
                <div class="function-category">{func.get("category", "Unknown category")}</div>
                {image_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Clickable button that covers the card
            # Create a unique key using hash of function name and search type
            import hashlib

            key_hash = hashlib.md5(
                f"{search_type}_{func['function_name']}_{i}".encode()
            ).hexdigest()[:8]
            if st.button(f"View {display_name}", key=f"btn_{key_hash}"):
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

                # Create a clean display for each standard
                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.code(external_id, language=None)

                    with col2:
                        st.write(f"**{display_name}**")
                        if description and description.strip():
                            st.caption(description)

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

            educational_purpose = stimulus_spec.get("educational_purpose", "")
            if educational_purpose:
                st.write("**Educational Purpose:**")
                st.write(educational_purpose)

            # Copy toggle at the bottom
            if spec_text.strip():
                # Use a toggle to show/hide copy content
                copy_key = f"show_copy_{function_data['function_name']}"
                show_copy = st.toggle(
                    "📋 Copy Complete Specification", key=copy_key, value=False
                )

                # Show copy content if toggled
                if show_copy:
                    st.code(spec_text, language="text")

    # Image
    image_path = function_data.get("image_path", "")
    if image_path:
        st.write("---")
        st.write("**Example Output:**")
        display_function_image(image_path)
    else:
        st.warning("No example image available")

    # AI Generated indicator
    if function_data.get("ai_generated", False):
        st.caption("🤖 This description was generated by AI")


def main():
    st.title("🎯 Stimulus Function Explorer")
    st.caption("Explore stimulus functions with instant loading")

    # Add custom CSS for function cards
    st.markdown(
        """
    <style>
    .function-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ffffff;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        text-align: center;
    }
    
    .function-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border-color: #4CAF50;
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
    
    /* Style the buttons to be more integrated */
    .stButton > button {
        width: 100% !important;
        margin-top: 10px !important;
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

    # Sidebar with stats
    st.sidebar.header("📊 Statistics")
    st.sidebar.metric("Total Functions", total_functions)
    st.sidebar.caption(f"Last updated: {last_updated}")

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
                st.write(f"💡 **Found {len(functions)} matching functions:**")
                display_search_results(functions, "name_search")
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
            st.button("🤖 AI Search", key="ai_search_btn")

        # Show AI search results
        if ai_search_query:
            if AI_SEARCH_AVAILABLE:
                try:
                    ai_engine = AISearchEngine()
                    ai_function_names = ai_engine.search_functions(ai_search_query)

                    # Convert function names to full function data
                    functions = []
                    for func_name in ai_function_names:
                        func_data = data_manager.load_function_data(func_name)
                        if func_data:
                            functions.append(func_data)

                    if functions:
                        st.success(
                            f"🤖 AI found {len(functions)} relevant functions for: '{ai_search_query}'"
                        )
                        display_search_results(functions, "ai_search")
                    else:
                        st.info(
                            f"🤖 AI didn't find any functions matching: '{ai_search_query}'"
                        )

                except Exception as e:
                    st.error(f"AI search error: {e}")
                    st.info("Please check your OpenAI API key and try again.")
            else:
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

    # Get all functions for the gallery
    index_data = data_manager.load_index()
    all_functions = index_data.get("functions", [])

    if all_functions:
        display_search_results(all_functions, "gallery")
    else:
        st.info("No functions available in the gallery.")


if __name__ == "__main__":
    main()
