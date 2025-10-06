"""
Database connection module for retrieving educational standards.
"""

import os
from typing import Dict, List, Optional, Tuple
import mysql.connector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def connect_to_db():
    """Connect to the database using credentials from environment variables"""
    db_config = {
        "host": "coachbot-prod-db.rp.devfactory.com",
        "user": os.environ.get("DB_USERNAME"),
        "password": os.environ.get("DB_PASSWORD"),
        "database": "alphacoachbot",
    }

    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        raise


def get_content_ids_for_function(connection, function_name: str) -> List[str]:
    """Query the database to get content IDs for a specific function"""
    cursor = connection.cursor(dictionary=True)

    try:
        # Query for content IDs that contain the specific function name in properties
        query = """
        SELECT external_id, properties FROM content_gen_extended_attributes 
        WHERE type_id = 'e87ba42e-89ed-11ef-ae50-0eb28d3c3f3f'
        AND properties LIKE %s
        """
        cursor.execute(query, (f"%{function_name}%",))

        # Extract external_ids and clean them up
        external_ids = []
        for row in cursor:
            external_id = row["external_id"]
            properties = row["properties"]

            # Debug: print what we found
            print(f"Found external_id: {external_id}, properties: {properties}")

            # Remove the -StimulusType-2 part
            if "-StimulusType-" in external_id:
                clean_id = external_id.split("-StimulusType-")[0]
                external_ids.append(clean_id)
            else:
                external_ids.append(external_id)

        return external_ids

    except mysql.connector.Error as err:
        print(f"Error querying content IDs for function {function_name}: {err}")
        return []
    finally:
        cursor.close()


def get_all_content_ids_to_validate(connection, function_name: str) -> List[str]:
    """Query the database to get all content IDs that need validation (for debugging)"""
    cursor = connection.cursor(dictionary=True)

    try:
        query = """
        SELECT external_id, properties FROM content_gen_extended_attributes 
        WHERE type_id = 'e87ba42e-89ed-11ef-ae50-0eb28d3c3f3f'
        AND properties LIKE %s
        LIMIT 10
        """
        cursor.execute(query, (f"%{function_name}%",))

        # Extract external_ids and clean them up
        external_ids = []
        for row in cursor:
            external_id = row["external_id"]
            properties = row["properties"]

            # Debug: print what we found
            print(f"Sample external_id: {external_id}, properties: {properties}")

            # Remove the -StimulusType-2 part
            if "-StimulusType-" in external_id:
                clean_id = external_id.split("-StimulusType-")[0]
                external_ids.append(clean_id)
            else:
                external_ids.append(external_id)

        return external_ids

    except mysql.connector.Error as err:
        print(f"Error querying all content IDs: {err}")
        return []
    finally:
        cursor.close()


def get_standards_details(connection, external_id: str) -> Optional[Dict[str, str]]:
    """Query the standards table for display_name and description"""
    cursor = connection.cursor(dictionary=True)

    try:
        query = """
        SELECT display_name, description FROM standards 
        WHERE external_id = %s
        """
        cursor.execute(query, (external_id,))
        result = cursor.fetchone()

        if result:
            print(
                f"  ✓ Found standard details for {external_id}: {result['display_name']}"
            )
            return {
                "display_name": result["display_name"],
                "description": result["description"],
            }
        else:
            print(f"  ⚠️  No standard found for external_id: {external_id}")
            return None

    except mysql.connector.Error as err:
        print(f"Error querying standards for {external_id}: {err}")
        return None
    finally:
        cursor.close()


def get_standards_for_function(connection, function_name: str) -> List[Dict[str, str]]:
    """Get all standards associated with a function using simple queries"""
    standards = []

    try:
        # Step 1: Get external_ids from content_gen_extended_attributes
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT external_id FROM content_gen_extended_attributes 
        WHERE type_id = 'e87ba42e-89ed-11ef-ae50-0eb28d3c3f3f'
        AND properties LIKE %s
        """
        # Use exact function name match in JSON
        cursor.execute(query, (f'%"stimulusFunction": "{function_name}"%',))

        external_ids = []
        for row in cursor:
            external_id = row["external_id"]
            # Clean up the external_id
            if "-StimulusType-" in external_id:
                clean_id = external_id.split("-StimulusType-")[0]
                external_ids.append(clean_id)
            else:
                external_ids.append(external_id)

        cursor.close()

        print(f"Found {len(external_ids)} external_ids for function {function_name}")

        # Step 2: Get standards for each external_id (only if we have IDs)
        if external_ids:
            cursor = connection.cursor(dictionary=True)
            placeholders = ",".join(["%s"] * len(external_ids))
            query = f"""
            SELECT external_id, display_name, description FROM standards 
            WHERE external_id IN ({placeholders})
            """
            cursor.execute(query, external_ids)

            for row in cursor:
                standards.append(
                    {
                        "external_id": row["external_id"],
                        "display_name": row["display_name"],
                        "description": row["description"],
                    }
                )
                print(f"  ✓ Found standard: {row['display_name']}")

            cursor.close()

        return standards

    except Exception as err:
        print(f"Error getting standards for function {function_name}: {err}")
        return []


def debug_database_structure(connection):
    """Debug function to understand the database structure"""
    try:
        print("=== DEBUGGING DATABASE STRUCTURE ===")

        # Get sample data from content_gen_extended_attributes
        sample_content_ids = get_all_content_ids_to_validate(
            connection, "generate_table"
        )
        print(f"Found {len(sample_content_ids)} sample content IDs")

        # Try to get standards for a sample external_id
        if sample_content_ids:
            sample_id = sample_content_ids[0]
            print(f"Testing with sample external_id: {sample_id}")
            standard_details = get_standards_details(connection, sample_id)
            if standard_details:
                print(f"Sample standard: {standard_details}")
            else:
                print("No standard found for sample ID")

        print("=== END DEBUG ===")

    except Exception as err:
        print(f"Error in debug function: {err}")


def get_standards_for_function_cached(function_name: str) -> List[Dict[str, str]]:
    """Get standards for a function with connection management"""
    connection = None
    try:
        connection = connect_to_db()
        return get_standards_for_function(connection, function_name)
    except Exception as err:
        print(f"Error connecting to database for function {function_name}: {err}")
        return []
    finally:
        if connection and connection.is_connected():
            connection.close()
