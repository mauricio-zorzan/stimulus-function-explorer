from typing import Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, field_validator


class UnitSquare(BaseModel):
    length: int = Field(..., description="The length of the unit square.", ge=1, le=10)
    width: int = Field(..., description="The width of the unit square.", ge=1, le=10)


class UnitSquares(StimulusDescriptionList[UnitSquare]):
    root: list[UnitSquare] = Field(
        ...,
        description="List of unit squares with specified lengths and widths.",
        min_length=2,
        max_length=4,
    )


class UnitSquareDecomposition(StimulusDescription):
    """
    A unit square decomposition showing a grid with some squares filled in.

    Creates a grid divided into unit squares (or rectangles) with a certain number of squares
    filled in to form a perfect rectangle (no leftover squares). The rectangle
    dimensions are randomly chosen from all possible factorizations of the filled_count
    that fit within the available space. The color of the filled squares is also
    randomly selected. Always leaves 2 rows at the bottom and 2 columns at the right empty.
    """

    size: int = Field(
        ...,
        description="Width of the grid (e.g., 6 creates a 6-unit wide grid). If rectangle_tiles is False, this also determines the height.",
        ge=1,
        le=10,
    )
    filled_count: int = Field(
        ..., description="Number of squares to fill in the grid", ge=0
    )
    rectangle_tiles: bool = Field(
        False,
        description="Whether to use rectangular tiles instead of square tiles. If True, requires height parameter. Tiles will have a 1.5:1 aspect ratio.",
    )
    height: Optional[int] = Field(
        None,
        description="Height of the grid when using rectangular tiles. Required if rectangle_tiles is True.",
        ge=1,
        le=10,
    )

    @field_validator("height")
    @classmethod
    def validate_height(cls, height, info):
        """Validate that height is provided when rectangle_tiles is True."""
        if "rectangle_tiles" in info.data and info.data["rectangle_tiles"]:
            if height is None:
                raise ValueError(
                    "height parameter is required when rectangle_tiles is True"
                )
        elif height is not None:
            raise ValueError(
                "height parameter should only be provided when rectangle_tiles is True"
            )
        return height

    @field_validator("filled_count")
    @classmethod
    def validate_filled_count(cls, filled_count, info):
        """Validate that filled_count doesn't exceed fillable area (leaving 2 rows/columns empty)."""
        if "size" in info.data:
            width = info.data["size"]

            # Determine grid height
            if (
                info.data.get("rectangle_tiles", False)
                and "height" in info.data
                and info.data["height"] is not None
            ):
                height = info.data["height"]
            else:
                height = width  # Square grid

            max_fillable_width = width - 2
            max_fillable_height = height - 2
            max_fillable_area = max_fillable_width * max_fillable_height

            if filled_count > max_fillable_area:
                raise ValueError(
                    f"filled_count ({filled_count}) cannot exceed fillable area ({max_fillable_area}) "
                    f"for grid size {width}x{height} (must leave 2 rows/columns empty)"
                )
        return filled_count
