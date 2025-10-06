import json
from typing import Any

import jsonref
import pyperclip
from pydantic import BaseModel


def remove_title_from_schema(schema, parent_name=None):
    if isinstance(schema, dict):
        if parent_name != "example" and isinstance(schema.get("title"), str):
            schema.pop("title", None)
        for key, value in schema.items():
            remove_title_from_schema(value, key)
    elif isinstance(schema, list):
        for item in schema:
            remove_title_from_schema(item, parent_name)


def get_class_docstring(cls):
    doc_string = cls.__doc__
    if doc_string is None and hasattr(cls, "__bases__"):
        doc_string = cls.__bases__[0].__doc__
    if doc_string:
        return doc_string.strip()
    return ""


def generate_assistant_function_schema(
    model: type[BaseModel],
) -> dict:
    """
    Generate a JSON schema for an Assistant Function.

    Args:
        model: The model of the stimulus of the Assistant Function.

    Returns:
        A dictionary representing the JSON schema.
    """
    schema: Any = jsonref.JsonRef.replace_refs(
        model.model_json_schema(mode="serialization")
    )

    # Remove mandatory 'title' field from all objects in the schema
    # This leaves objects so that user-defined titles attributes remain
    remove_title_from_schema(schema)

    output = {
        "name": schema["properties"]["name"]["default"],
        "description": schema["properties"]["description"]["default"],
        "parameters": {
            "description": schema["properties"].get("description"),
            "properties": schema["properties"].get("parameters", {}).get("properties"),
            "required": schema["properties"].get("parameters", {}).get("required"),
            "additionalProperties": False,
        },
    }

    out = json.dumps(output, indent=2, ensure_ascii=False)
    pyperclip.copy(out)
    print(out)
    return output
