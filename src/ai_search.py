"""
AI-powered search functionality for stimulus functions using OpenAI API.
"""

import os
import json
from typing import List, Dict, Optional
import openai
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AISearchEngine:
    """AI-powered search engine for finding stimulus functions based on descriptions."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the AI search engine.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Initialize OpenAI client (new API format)
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Cache for function descriptions to avoid repeated API calls
        self._descriptions_cache = None
        self._cache_file = Path("data/ai_search_cache.json")
    
    def _load_function_descriptions(self) -> List[Dict]:
        """Load function descriptions from the data directory."""
        if self._descriptions_cache is not None:
            return self._descriptions_cache
        
        functions_dir = Path("data/functions")
        if not functions_dir.exists():
            return []
        
        descriptions = []
        for json_file in functions_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    function_data = json.load(f)
                    
                # Extract relevant information for AI search
                description_entry = {
                    "function_name": function_data.get("function_name", ""),
                    "description": function_data.get("description", ""),
                    "category": function_data.get("category", ""),
                    "tags": function_data.get("tags", []),
                    "parameters": function_data.get("parameters", []),
                    "example_usage": function_data.get("example_usage", "")
                }
                descriptions.append(description_entry)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        self._descriptions_cache = descriptions
        return descriptions
    
    def _create_search_prompt(self, query: str, function_descriptions: List[Dict]) -> str:
        """Create a prompt for OpenAI to find relevant functions."""
        
        # Format function descriptions for the prompt
        functions_text = ""
        for i, func in enumerate(function_descriptions, 1):
            # Format parameters properly
            if isinstance(func['parameters'], dict):
                params_str = ', '.join([f"{k}: {v}" for k, v in func['parameters'].items()])
            elif isinstance(func['parameters'], list):
                params_str = ', '.join([str(p) for p in func['parameters']])
            else:
                params_str = str(func['parameters']) if func['parameters'] else 'None'
            
            functions_text += f"""
Function {i}: {func['function_name']}
Description: {func['description']}
Category: {func['category']}
Tags: {', '.join(func['tags']) if func['tags'] else 'None'}
Parameters: {params_str}
Example: {func['example_usage']}

"""
        
        prompt = f"""You are an AI assistant helping to find stimulus functions based on user queries. 

User Query: "{query}"

Here are all available stimulus functions with their descriptions:

{functions_text}

Please analyze the user query and return the most relevant functions. Be generous in your matching - include functions that:
1. Have direct matches in function names, descriptions, or tags
2. Are semantically related (e.g., "rectangles" matches functions that draw rectangular shapes, "fractions" matches any fractional functions)
3. Are in relevant categories
4. Have related parameter types

IMPORTANT: Be inclusive rather than exclusive. If the query could reasonably relate to a function, include it.

Return your response as a JSON array of function names, ordered by relevance (most relevant first). 
Include functions that are reasonably related to the query.

Example response format:
["function_name_1", "function_name_2", "function_name_3"]

Response:"""
        
        return prompt
    
    def search_functions(self, query: str, max_results: int = 10) -> List[str]:
        """Search for functions using AI based on the query.
        
        Args:
            query: The search query (e.g., "functions with rectangles")
            max_results: Maximum number of results to return
            
        Returns:
            List of function names ordered by relevance
        """
        if not query.strip():
            return []
        
        try:
            # Load function descriptions
            function_descriptions = self._load_function_descriptions()
            if not function_descriptions:
                return []
            
            # Create the search prompt
            prompt = self._create_search_prompt(query, function_descriptions)
            
            # Make API call to OpenAI (new API format)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that finds relevant stimulus functions based on user queries. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                function_names = json.loads(response_text)
                if not isinstance(function_names, list):
                    function_names = []
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract function names from text
                function_names = self._extract_function_names_from_text(response_text)
            
            # Filter to only include valid function names and limit results
            valid_functions = [name for name in function_names if self._is_valid_function_name(name)]
            return valid_functions[:max_results]
            
        except Exception as e:
            print(f"Error in AI search: {e}")
            return []
    
    def _extract_function_names_from_text(self, text: str) -> List[str]:
        """Extract function names from text response if JSON parsing fails."""
        # Simple extraction - look for function names in quotes or after colons
        import re
        
        # Pattern to match function names (typically start with draw_, create_, generate_)
        pattern = r'\b(draw_|create_|generate_)[a-zA-Z_]+'
        matches = re.findall(pattern, text)
        
        # Also look for quoted strings that might be function names
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)
        
        # Combine and deduplicate
        all_matches = list(set(matches + quoted_matches))
        return all_matches
    
    def _is_valid_function_name(self, name: str) -> bool:
        """Check if a function name is valid by checking if it exists in our data."""
        if not isinstance(name, str) or not name.strip():
            return False
        
        # Check if the function file exists
        function_file = Path("data/functions") / f"{name.strip()}.json"
        return function_file.exists()
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on the query."""
        if len(query) < 2:
            return []
        
        # Load function descriptions
        function_descriptions = self._load_function_descriptions()
        
        # Simple suggestion based on partial matches
        suggestions = []
        query_lower = query.lower()
        
        for func in function_descriptions:
            # Check function name
            if query_lower in func['function_name'].lower():
                suggestions.append(func['function_name'])
            
            # Check description
            if query_lower in func['description'].lower():
                suggestions.append(func['function_name'])
            
            # Check tags
            for tag in func['tags']:
                if query_lower in tag.lower():
                    suggestions.append(func['function_name'])
        
        # Remove duplicates and limit
        return list(set(suggestions))[:5]
