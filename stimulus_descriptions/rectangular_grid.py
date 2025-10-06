from enum import Enum
from typing import List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator, model_validator


class EAbbreviatedMeasurementUnit(str, Enum):
    INCHES = "in"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    FEET = "ft"
    METERS = "m"
    UNITS = "Units"
    KILOMETERS = "km"


class RectangularGrid(StimulusDescription):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit_case(cls, v):
        """Convert unit names to match enum values."""
        if isinstance(v, str):
            # Handle the special case where "units" should become "Units"
            if v.lower() == "units":
                return "Units"
            # For other values, try lowercase first
            return v.lower()
        return v

    length: int = Field(
        ...,
        ge=1,
        le=10,
        description="The length of the grid, must be a natural number less than 11.",
    )
    width: int = Field(
        ...,
        ge=1,
        le=10,
        description="The width of the grid, must be a natural number less than 11.",
    )
    label: Optional[str] = Field(
        default=None,
        description=(
            "Optional per-grid label such as 'Figure A' or 'Shape 1'. "
            "If all labels are omitted, defaults are 'Figure A', 'Figure B', ..."
        ),
    )
    irregular: bool = Field(
        default=False,
        description=(
            "If true, the grid will be drawn irregularly (with skewing) instead of as a regular rectangle."
        ),
    )
    extra_unit_squares: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional number of extra unit squares to append to the ends of rows/columns when irregular=true. "
            "Must be between 1 and max(length, width). Ignored when irregular=false."
        ),
    )

    @model_validator(mode="after")
    def validate_irregular_extras(self):
        # If irregular is false, extra_unit_squares must not be provided
        if not self.irregular and self.extra_unit_squares not in (None, 0):
            raise ValueError(
                "extra_unit_squares may only be set when irregular is true"
            )

        # If extras provided, ensure within bounds
        if self.extra_unit_squares is not None:
            max_allowed = max(self.length, self.width)
            if not (1 <= self.extra_unit_squares <= max_allowed):
                raise ValueError(
                    f"extra_unit_squares must be between 1 and {max_allowed}, got {self.extra_unit_squares}"
                )
        return self


class MultipleGrids(StimulusDescription):
    """
    Stimulus description for multiple unit square grids in one image.
    Supports up to 5 grids for comparison purposes.
    Useful for teaching area comparison, perimeter comparison, and unit understanding.
    """

    grids: List[RectangularGrid] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of 1 to 5 grids to display for comparison.",
    )
    title: Optional[str] = Field(
        None, description="Optional title for the multiple grids image."
    )
    irregularity: Literal["all_regular", "all_irregular", "mixed"] = Field(
        "all_irregular",
        description="Control grid irregularity: 'all_regular' (perfect rectangles), 'all_irregular' (missing squares), or 'mixed' (random selection)",
    )
    target_units: Optional[List[int]] = Field(
        None,
        description="Optional list of target unit counts for each grid. If provided, algorithm will attempt to achieve these exact counts while maintaining connectivity. Length must match the number of grids.",
    )

    @model_validator(mode="after")
    def validate_grids(self):
        """Validate that all grids have the same unit and set default labels if omitted."""
        if len(self.grids) > 1:
            first_unit = self.grids[0].unit
            for grid in self.grids[1:]:
                if grid.unit != first_unit:
                    raise ValueError(
                        "All grids must have the same unit of measurement for comparison."
                    )

        # Validate target_units if provided
        if self.target_units is not None:
            if len(self.target_units) != len(self.grids):
                raise ValueError(
                    f"Number of target units ({len(self.target_units)}) must match number of grids ({len(self.grids)})"
                )

            # Validate that all targets are positive
            for i, target in enumerate(self.target_units):
                if target <= 0:
                    raise ValueError(
                        f"Target unit count at index {i} must be positive, got {target}"
                    )

        # If every grid has empty/None label, set defaults "Figure A.."
        if self.grids and all((g.label is None or g.label == "") for g in self.grids):
            for idx, g in enumerate(self.grids, start=1):
                letter = chr(ord("A") + (idx - 1))
                g.label = f"Figure {letter}"

        return self


if __name__ == "__main__":
    RectangularGrid.generate_assistant_function_schema("mcq4")
