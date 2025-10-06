from enum import Enum
from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class EAbbreviatedMeasurementUnit(str, Enum):
    """Abbreviated measurement units."""

    CENTIMETERS = "cm"
    METERS = "m"
    FEET = "ft"
    INCHES = "in"
    MILLIMETERS = "mm"
    KILOMETERS = "km"
    UNITS = "Units"


class PolygonStringSides(StimulusDescription):
    """
    Defines a polygon with side lengths specified as strings and optional labels.

    The polygon can have:
    - side_lengths: list of side lengths as strings (e.g., ["3", "4", "5"])
    - side_labels: optional list of custom labels for each side (e.g., ["a", "b", "c"])
    - unit: optional measurement unit (e.g., "cm", "units")
    """

    side_lengths: List[str] = Field(
        min_length=3,
        max_length=10,
        description="List of side lengths as strings. Can be numeric strings like '3' or variable names like 'x', 'y', 'z'",
    )
    side_labels: Optional[List[str]] = Field(
        default=None,
        description="Optional custom labels for each side. If not provided, uses side_lengths as labels",
    )
    unit: Optional[str] = Field(
        default="",
        description="Optional unit string to append to side lengths (e.g., 'cm', 'units')",
    )

    class Config:
        json_encoders = {str: lambda s: str(s)}
