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
    """Save function data to JSON file"""
    function_file = Path("data/functions") / f"{function_name}.json"

    try:
        with open(function_file, "w") as f:
            json.dump(function_data, f, indent=2)
        print(f"✅ Updated {function_name}.json")
    except Exception as e:
        print(f"❌ Error saving {function_name}: {e}")


def get_standards_for_function_from_db(
    connection, function_name: str
) -> List[Dict[str, str]]:
    """Get standards for a function from the database using the working function"""
    return get_standards_for_function(connection, function_name)


def update_function_with_standards(connection, function_name: str):
    """Update a single function with standards data"""
    print(f"\n🔍 Processing function: {function_name}")

    # Load existing function data
    function_data = load_function_data(function_name)
    if not function_data:
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
        print(f"⚠️  No standards found for {function_name}")
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


def debug_single_function(function_name: str = "generate_table"):
    """Debug a single function to understand the data flow"""
    print(f"🔍 Debugging function: {function_name}")

    try:
        connection = connect_to_db()
        print("✅ Connected to database")

        # Test the basic query first
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT external_id FROM content_gen_extended_attributes 
        WHERE type_id = 'e87ba42e-89ed-11ef-ae50-0eb28d3c3f3f'
        AND properties LIKE %s
        LIMIT 5
        """
        cursor.execute(query, (f'%"stimulusFunction": "{function_name}"%',))

        print(f"Basic query results for {function_name}:")
        for row in cursor:
            print(f"  - {row['external_id']}")

        cursor.close()

        # Test the full function
        from src.database import get_standards_for_function

        standards = get_standards_for_function(connection, function_name)

        print(f"Found {len(standards)} standards for {function_name}")
        for standard in standards:
            print(f"  ✓ {standard['external_id']}: {standard['display_name']}")

        connection.close()

    except Exception as e:
        print(f"❌ Debug error: {e}")


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
        print(f"📋 Found {len(function_names)} functions to process")

        # Process each function with connection management
        updated_count = 0
        total_functions = len(function_names)

        for i, function_name in enumerate(function_names):
            try:
                print(
                    f"\n📊 Progress: {i + 1}/{total_functions} ({((i + 1) / total_functions) * 100:.1f}%)"
                )

                if update_function_with_standards(connection, function_name):
                    updated_count += 1

                # Reconnect every 10 functions to avoid connection issues
                if (i + 1) % 10 == 0:
                    print(f"🔄 Reconnecting after {i + 1} functions...")
                    connection.close()
                    connection = connect_to_db()
                    print("✅ Reconnected to database")

            except Exception as e:
                print(f"❌ Error processing {function_name}: {e}")
                # Try to reconnect on error
                try:
                    connection.close()
                    connection = connect_to_db()
                    print("✅ Reconnected after error")
                except:
                    print("❌ Failed to reconnect, stopping")
                    break

        print(f"\n🎉 Update complete!")
        print(f"✅ Updated {updated_count} functions with standards")
        print(f"⚠️  {len(function_names) - updated_count} functions had no standards")

    finally:
        if connection:
            try:
                connection.close()
                print("🔌 Database connection closed")
            except:
                pass  # Ignore errors when closing connection


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        function_name = sys.argv[2] if len(sys.argv) > 2 else "generate_table"
        debug_single_function(function_name)
    else:
        main()
