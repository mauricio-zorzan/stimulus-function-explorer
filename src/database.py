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


def parse_standard_from_external_id(external_id: str) -> str:
    """Parse the standard from external_id by removing -LearningObjective-X-StimulusType or -StimulusType-X suffix"""
    # Remove -LearningObjective-X-StimulusType pattern
    # Example: CCSS.MATH.CONTENT.6.EE.C.9+8-LearningObjective-4-StimulusType -> CCSS.MATH.CONTENT.6.EE.C.9+8
    # Also handle -StimulusType-X pattern (without LearningObjective)
    # Example: CCSS.MATH.CONTENT.7.SP.C.8.B+2-StimulusType-1 -> CCSS.MATH.CONTENT.7.SP.C.8.B+2
    import re
    # Match pattern like -LearningObjective-4-StimulusType or -LearningObjective-5-StimulusType
    pattern1 = r'-LearningObjective-\d+-StimulusType.*$'
    clean_id = re.sub(pattern1, '', external_id)
    # Also handle -StimulusType-X pattern (without LearningObjective)
    pattern2 = r'-StimulusType-\d+.*$'
    clean_id = re.sub(pattern2, '', clean_id)
    return clean_id


def get_stimulus_type_specifications(connection, learning_objective_external_id: str, verbose: bool = False) -> List[str]:
    """Get all stimulus type specifications for a learning objective"""
    cursor = connection.cursor(dictionary=True)
    
    try:
        query = """
        SELECT 
               child.value
        FROM content_gen_extended_attribute_mappings cam
        JOIN content_gen_extended_attributes child ON child.id = cam.to_attribute_id
        JOIN content_gen_extended_attribute_types child_type ON child_type.id = child.type_id
        WHERE cam.relationship LIKE 'LearningObjective-%'
          AND child_type.type IN ('StimulusTypeSpecification')
          AND child.external_id LIKE %s
        ORDER BY child.external_id
        """
        # Use LIKE with pattern matching
        pattern = f"{learning_objective_external_id}%"
        if verbose:
            print(f"    Querying for specifications with pattern: {pattern}")
        cursor.execute(query, (pattern,))
        results = cursor.fetchall()
        
        if verbose:
            print(f"    Found {len(results)} specification rows")
        
        # Extract all unique values
        specifications = []
        seen_values = set()
        for row in results:
            value = row["value"]
            if value and value not in seen_values:
                specifications.append(value)
                seen_values.add(value)
                if verbose:
                    print(f"      - {value}")
        
        return specifications
        
    except Exception as err:
        print(f"Error getting stimulus type specifications: {err}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        cursor.close()


def get_standards_for_function(connection, function_name: str) -> List[Dict[str, str]]:
    """Get all standards associated with a function using the new query structure"""
    standards = []

    try:
        # Step 1: Get external_ids from content_gen_extended_attributes using new query
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
        """
        cursor.execute(query, (function_name,))

        # Parse external_ids to extract standards
        # We need to track both the full learning objective ID and the parsed standard ID
        standard_external_ids = set()  # Will contain both base and +N versions
        learning_objective_ids = []  # Full IDs like "CCSS.MATH.CONTENT.6.EE.C.9+8-LearningObjective-4-StimulusType"
        lo_id_to_standard_id = {}  # Map learning objective ID to standard ID
        
        for row in cursor:
            external_id = row["external_id"]
            # Parse the standard from external_id (remove -LearningObjective-X-StimulusType or -StimulusType-X)
            standard_id = parse_standard_from_external_id(external_id)
            # Add the parsed standard (with +N if present) to the set
            standard_external_ids.add(standard_id)
            # Also add the base version (without +N) for database lookup
            standard_id_base = standard_id.split('+')[0] if '+' in standard_id else standard_id
            if standard_id_base != standard_id:
                standard_external_ids.add(standard_id_base)
            # Keep the full external_id for getting stimulus type specification
            learning_objective_ids.append(external_id)
            lo_id_to_standard_id[external_id] = standard_id

        cursor.close()

        print(
            f"Found {len(standard_external_ids)} unique standards for function {function_name}"
        )

        # Step 2: Get standards details for each standard external_id
        # Try both with and without +N suffix
        all_standard_ids_to_try = set(standard_external_ids)
        # Also add the versions with +N suffixes if we have them
        for std_id in list(standard_external_ids):
            # Check if we have any learning objectives with this base
            for lo_id in learning_objective_ids:
                parsed = parse_standard_from_external_id(lo_id)
                base = parsed.split('+')[0] if '+' in parsed else parsed
                if base == std_id and '+' in parsed:
                    # Add the version with +N
                    all_standard_ids_to_try.add(parsed)
        
        if all_standard_ids_to_try:
            cursor2 = connection.cursor(dictionary=True)
            placeholders = ",".join(["%s"] * len(all_standard_ids_to_try))
            query = f"""
            SELECT external_id, display_name, description FROM standards 
            WHERE external_id IN ({placeholders})
            """
            cursor2.execute(query, list(all_standard_ids_to_try))
            # Fetch all results immediately
            standards_rows = cursor2.fetchall()
            cursor2.close()

            # Create a mapping from standard_id (from database) to learning_objective_ids for stimulus spec lookup
            # We need to match database results (which might have +N) to learning objectives
            standard_to_lo_ids = {}
            for lo_id in learning_objective_ids:
                # Use the mapping we created earlier
                standard_id_base = lo_id_to_standard_id.get(lo_id)
                parsed_full = parse_standard_from_external_id(lo_id)
                
                # Map both the base (without +N) and the full version (with +N) to learning objectives
                # This allows matching regardless of whether the database has +N or not
                if standard_id_base:
                    if standard_id_base not in standard_to_lo_ids:
                        standard_to_lo_ids[standard_id_base] = []
                    standard_to_lo_ids[standard_id_base].append(lo_id)
                
                # Also map the full version if different (with +N)
                if parsed_full != standard_id_base:
                    if parsed_full not in standard_to_lo_ids:
                        standard_to_lo_ids[parsed_full] = []
                    standard_to_lo_ids[parsed_full].append(lo_id)
                
                # Also map the base without +N for matching
                parsed_base = parsed_full.split('+')[0] if '+' in parsed_full else parsed_full
                if parsed_base != standard_id_base and parsed_base != parsed_full:
                    if parsed_base not in standard_to_lo_ids:
                        standard_to_lo_ids[parsed_base] = []
                    standard_to_lo_ids[parsed_base].append(lo_id)

            # Filter: Only keep standards with +N suffix if both base and +N versions exist
            # First, identify which standards have +N versions
            standards_with_suffix = set()
            standards_base_only = set()
            
            for row in standards_rows:
                external_id = row["external_id"]
                if '+' in external_id:
                    # This is a +N version, extract the base
                    base = external_id.split('+')[0]
                    standards_with_suffix.add(base)
                else:
                    # This is a base version
                    standards_base_only.add(external_id)
            
            # Filter: Remove base versions if they have a +N version
            filtered_standards_rows = []
            for row in standards_rows:
                external_id = row["external_id"]
                if '+' in external_id:
                    # Always include +N versions
                    filtered_standards_rows.append(row)
                else:
                    # Only include base version if there's no +N version
                    if external_id not in standards_with_suffix:
                        filtered_standards_rows.append(row)
            
            # Use a dict to deduplicate by external_id and get stimulus type specifications
            standards_dict = {}
            for row in filtered_standards_rows:
                external_id = row["external_id"]
                # Only add if not already seen (prevents duplicates)
                if external_id not in standards_dict:
                    # Find matching learning objectives to get all stimulus type specifications
                    # Try matching with the exact external_id, base version, and with +N
                    all_specifications = []
                    matching_keys = []
                    
                    # Try exact match first
                    if external_id in standard_to_lo_ids:
                        matching_keys.append(external_id)
                    
                    # Try base version (without +N)
                    external_id_base = external_id.split('+')[0] if '+' in external_id else external_id
                    if external_id_base != external_id and external_id_base in standard_to_lo_ids:
                        matching_keys.append(external_id_base)
                    
                    # Try with +N if database has base
                    if '+' not in external_id:
                        # Check if we have any learning objectives with this base +N
                        for key in standard_to_lo_ids.keys():
                            if key.startswith(external_id + '+') and key not in matching_keys:
                                matching_keys.append(key)
                    
                    if matching_keys:
                        # Try each matching key to get all specifications
                        for key in matching_keys:
                            # Try each learning objective ID for this standard
                            for lo_id in standard_to_lo_ids[key]:
                                specs = get_stimulus_type_specifications(connection, lo_id, verbose=False)
                                all_specifications.extend(specs)
                    else:
                        print(f"  ⚠️  No learning objectives found for standard: {external_id}")
                        print(f"    Available keys in standard_to_lo_ids: {list(standard_to_lo_ids.keys())[:5]}...")
                    
                    # Remove duplicates while preserving order
                    unique_specifications = []
                    seen = set()
                    for spec in all_specifications:
                        if spec not in seen:
                            unique_specifications.append(spec)
                            seen.add(spec)
                    
                    standards_dict[external_id] = {
                        "external_id": external_id,
                        "display_name": row["display_name"],
                        "description": row["description"],
                    }
                    # Add all stimulus type specifications if found
                    if unique_specifications:
                        standards_dict[external_id]["stimulus_type_specifications"] = unique_specifications
                        print(f"  ✓ Found standard: {row['display_name']} ({len(unique_specifications)} specifications)")
                    else:
                        print(f"  ✓ Found standard: {row['display_name']} (no specifications)")
                    
                    # Debug: verify specifications were added
                    if external_id in standards_dict and "stimulus_type_specifications" in standards_dict[external_id]:
                        specs_count = len(standards_dict[external_id]["stimulus_type_specifications"])
                        print(f"    ✅ Verified: {specs_count} specifications stored for {external_id}")

            cursor.close()

            # Convert dict back to list
            standards = list(standards_dict.values())

        return standards

    except Exception as err:
        print(f"Error getting standards for function {function_name}: {err}")
        import traceback
        traceback.print_exc()
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
