from enum import Enum
from fractions import Fraction
from typing import Optional, Union

from pydantic import BaseModel, Field


class EAbbreviatedMeasurementUnit(str, Enum):
    """Abbreviated measurement units."""
    INCHES = "in"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    FEET = "ft"
    METERS = "m"
    UNITS = "Units"
    KILOMETERS = "km"


class ETrapezoidType(str, Enum):
    """Types of trapezoids that can be drawn."""
    REGULAR_TRAPEZOID = "regular_trapezoid"
    ISOSCELES_TRAPEZOID = "isosceles_trapezoid"
    RIGHT_TRAPEZOID = "right_trapezoid"
    LEFT_TRAPEZOID = "left_trapezoid"


class TrapezoidGrid(BaseModel):
    """
    Defines a trapezoid with labeled dimensions for area calculation problems.
    
    The trapezoid has:
    - base: the length of the bottom parallel side
    - top_length: the length of the top parallel side  
    - height: the perpendicular distance between the parallel sides
    - unit: the measurement unit for all dimensions
    """
    
    base: Union[int, float, Fraction] = Field(
        description="Length of the bottom parallel side (base) of the trapezoid"
    )
    top_length: Union[int, float, Fraction] = Field(
        description="Length of the top parallel side of the trapezoid"
    )
    height: Union[int, float, Fraction] = Field(
        description="Height of the trapezoid (perpendicular distance between parallel sides)"
    )
    unit: EAbbreviatedMeasurementUnit = Field(
        description="Unit of measurement for all dimensions"
    )
    trapezoid_type: ETrapezoidType = Field(
        default=ETrapezoidType.REGULAR_TRAPEZOID,
        description="Type of trapezoid to draw"
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional label for the trapezoid (e.g., 'Figure 1')"
    )
    show_variable_height: bool = Field(
        default=False,
        description="Whether to show height as a variable (h) instead of numeric value"
    )
    
    class Config:
        json_encoders = {
            Fraction: lambda f: float(f)
        } 