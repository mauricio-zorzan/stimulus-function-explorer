import functools
import json
from typing import Callable, Union

import matplotlib
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusFunctionProtocol,
)
from pydantic import BaseModel


def stimulus_function(func: Union[StimulusFunctionProtocol, Callable]):
    annotation = list(func.__annotations__.values())[0]
    # Expose stimulus type to the function
    func.stimulus_type = annotation
    matplotlib.use("Agg")

    # Wrapper to parse the input data into a stimulus model
    @functools.wraps(func, assigned=["stimulus_type"])
    def wrapper(input_data):
        if isinstance(input_data, str):
            try:
                parsed_input = json.loads(input_data)  # Parsing json strings
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON input")
        elif isinstance(
            input_data, (list, dict)
        ):  # Support for dictionary and list loading
            parsed_input = input_data
        elif isinstance(input_data, annotation):  # Support for direct model loading
            return func(input_data)
        elif isinstance(
            input_data, BaseModel
        ):  # Support for inheritance architectures on drawing functions
            return func(annotation(**input_data.model_dump()))
        else:
            raise ValueError("Unsupported input type")

        if isinstance(parsed_input, list):
            parsed_input = annotation(root=parsed_input)
        else:
            parsed_input = annotation(**parsed_input)
        return func(parsed_input)

    return wrapper
