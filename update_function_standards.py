#!/usr/bin/env python3
"""
Script to update function data with educational standards from the database.
This script should be run periodically to keep the standards data up to date.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.database import connect_to_db, get_standards_for_function


def load_function_data(function_name: str) -> Optional[Dict]:
    """Load function data from JSON file"""
    function_file = Path("data/functions") / f"{function_name}.json"

    if not function_file.exists():
        print(f"Function file not found: {function_file}")
        return None

    try:
        with open(function_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {function_name}: {e}")
        return None
    except Exception as e:
        print(f"Error loading function data for {function_name}: {e}")
        return None


def save_function_data(function_name: str, function_data: Dict):
    """Save function data to JSON file (thread-safe for different files)"""
    function_file = Path("data/functions") / f"{function_name}.json"

    try:
        # Use atomic write: write to temp file first, then rename
        temp_file = function_file.with_suffix('.json.tmp')
        with open(temp_file, "w") as f:
            json.dump(function_data, f, indent=2)
        temp_file.replace(function_file)
        # Only print in verbose mode to avoid cluttering output in concurrent mode
        # print(f"✅ Updated {function_name}.json")
    except Exception as e:
        print(f"❌ Error saving {function_name}: {e}")


def get_standards_for_function_from_db(
    connection, function_name: str
) -> List[Dict[str, str]]:
    """Get standards for a function from the database using the working function"""
    return get_standards_for_function(connection, function_name)


def update_function_with_standards(connection, function_name: str, verbose: bool = True):
    """Update a single function with standards data"""
    if verbose:
        print(f"\n🔍 Processing function: {function_name}")

    # Load existing function data
    function_data = load_function_data(function_name)
    if not function_data:
        if verbose:
            print(f"❌ Could not load data for {function_name}")
        return False

    # Get standards from database
    standards = get_standards_for_function_from_db(connection, function_name)

    if standards:
        # Add standards to function data
        function_data["educational_standards"] = standards
        function_data["standards_last_updated"] = datetime.now().isoformat()

        # Save updated function data
        save_function_data(function_name, function_data)
        return True
    else:
        if verbose:
            print(f"⚠️  No standards found for {function_name}")
        return False


def update_single_function_worker(function_name: str, verbose: bool = False):
    """Worker function for concurrent processing - creates its own database connection"""
    try:
        # Each thread gets its own connection
        connection = connect_to_db()
        try:
            result = update_function_with_standards(connection, function_name, verbose=verbose)
            return function_name, result, None
        finally:
            connection.close()
    except Exception as e:
        return function_name, False, str(e)


def update_all_standards(progress_callback=None, status_callback=None, max_workers: int = 10):
    """Update standards for all functions using concurrent processing. Can be called from sync process."""
    # Check if database credentials are available
    if not os.environ.get("DB_USERNAME") or not os.environ.get("DB_PASSWORD"):
        if status_callback:
            status_callback("⚠️ Database credentials not found - skipping standards update")
        return False

    # Test database connection
    try:
        test_connection = connect_to_db()
        test_connection.close()
        if status_callback:
            status_callback("✅ Database connection test successful")
    except Exception as e:
        if status_callback:
            status_callback(f"⚠️ Could not connect to database: {e}")
        return False

    try:
        # Get all function names
        function_names = get_all_function_names()
        total_functions = len(function_names)
        
        if status_callback:
            status_callback(f"📋 Updating standards for {total_functions} functions using {max_workers} workers...")

        # Process functions concurrently
        updated_count = 0
        error_count = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_function = {
                executor.submit(update_single_function_worker, func_name, verbose=False): func_name
                for func_name in function_names
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_function):
                completed += 1
                function_name, success, error = future.result()
                
                # Update progress
                if progress_callback:
                    progress = int(85 + (completed / total_functions) * 10)  # 85-95% range
                    progress_callback(progress)
                
                with lock:
                    if success:
                        updated_count += 1
                    if error:
                        error_count += 1
                        if status_callback:
                            status_callback(f"⚠️ Error processing {function_name}: {error}")

                # Status update every 10 functions
                if completed % 10 == 0 and status_callback:
                    status_callback(f"📊 Progress: {completed}/{total_functions} functions processed ({updated_count} updated, {error_count} errors)")

        if status_callback:
            status_callback(f"✅ Updated {updated_count} functions with standards ({error_count} errors)")
        
        return True

    except Exception as e:
        if status_callback:
            status_callback(f"❌ Fatal error during standards update: {e}")
        return False


def get_all_function_names() -> List[str]:
    """Get all function names from the functions directory"""
    functions_dir = Path("data/functions")
    if not functions_dir.exists():
        print("❌ Functions directory not found")
        return []

    function_names = []
    for json_file in functions_dir.glob("*.json"):
        if json_file.name != "template.json":  # Skip template
            function_name = json_file.stem
            function_names.append(function_name)

    return sorted(function_names)


def debug_single_function(function_name: str = "create_scatterplot"):
    """Debug a single function to understand the data flow"""
    print(f"🔍 Debugging function: {function_name}")

    try:
        connection = connect_to_db()
        print("✅ Connected to database")

        # Test the new query first
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT
            st.external_id,
            st.value,
            st.properties
        FROM content_gen_extended_attributes st
        JOIN content_gen_extended_attribute_types st_type
            ON st_type.id = st.type_id
        WHERE st_type.type = 'StimulusType'
          AND JSON_EXTRACT(st.properties, '$.stimulusFunction') = %s
        ORDER BY st.external_id
        LIMIT 5
        """
        cursor.execute(query, (function_name,))

        print(f"New query results for {function_name}:")
        for row in cursor:
            print(f"  - external_id: {row['external_id']}")
            print(f"    value: {row.get('value', 'N/A')}")

        cursor.close()

        # Test the full function
        from src.database import get_standards_for_function, parse_standard_from_external_id

        standards = get_standards_for_function(connection, function_name)

        print(f"\nFound {len(standards)} standards for {function_name}")
        for standard in standards:
            print(f"  ✓ {standard['external_id']}: {standard['display_name']}")
            if 'stimulus_type_specifications' in standard:
                specs = standard['stimulus_type_specifications']
                print(f"    Stimulus Type Specs ({len(specs)}): {', '.join(specs)}")
            elif 'stimulus_type_specification' in standard:
                # Legacy format (single specification)
                print(f"    Stimulus Type Spec: {standard['stimulus_type_specification']}")

        connection.close()

    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to update all functions with standards"""
    print("🚀 Starting function standards update...")

    # Check if database credentials are available
    if not os.environ.get("DB_USERNAME") or not os.environ.get("DB_PASSWORD"):
        print("❌ Database credentials not found!")
        print("Please set DB_USERNAME and DB_PASSWORD environment variables")
        return

    # Connect to database
    try:
        connection = connect_to_db()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    try:
        # Get all function names
        function_names = get_all_function_names()
        total_functions = len(function_names)
        print(f"📋 Found {total_functions} functions to process")
        print(f"🚀 Using concurrent processing with 10 workers...")

        # Process functions concurrently
        updated_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_function = {
                executor.submit(update_single_function_worker, func_name, verbose=False): func_name
                for func_name in function_names
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_function):
                completed += 1
                function_name, success, error = future.result()
                
                if success:
                    updated_count += 1
                if error:
                    error_count += 1
                    print(f"❌ Error processing {function_name}: {error}")

                # Progress update every 10 functions
                if completed % 10 == 0 or completed == total_functions:
                    print(f"📊 Progress: {completed}/{total_functions} ({((completed / total_functions) * 100):.1f}%) - {updated_count} updated, {error_count} errors")

        print(f"\n🎉 Update complete!")
        print(f"✅ Updated {updated_count} functions with standards")
        print(f"⚠️  {len(function_names) - updated_count} functions had no standards")
        if error_count > 0:
            print(f"❌ {error_count} functions had errors")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        function_name = sys.argv[2] if len(sys.argv) > 2 else "generate_table"
        debug_single_function(function_name)
    else:
        main()
