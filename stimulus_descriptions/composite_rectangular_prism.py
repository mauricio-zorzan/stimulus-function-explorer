from typing import Annotated, Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BeforeValidator, Field


class CompositeRectangularPrism(StimulusDescription):
    figures: list[list[float]] = Field(
        ..., description="List of figures with dimensions.", min_length=2
    )
    units: Annotated[
        Literal["in", "m", "cm", "ft", "units"],
        BeforeValidator(
            lambda v: {
                "inch": "in",
                "inches": "in",
                "meter": "m",
                "meters": "m",
                "centimeter": "cm",
                "centimeters": "cm",
                "foot": "ft",
                "feet": "ft",
            }.get(v.lower(), v.lower())
        ),
    ] = Field(..., description="Unit of measurement.")
    hide_measurements: list[int] = Field(
        default_factory=list,
        description="List of prism indexes (0-based) for which measurements should be hidden. If empty, shows measurements for all prisms.",
    )


class CompositeRectangularPrism2(StimulusDescription):
    """
    figures: list[[h, w, l]]   (height=z, width=y, length=x) for each prism
    """

    figures: list[list[float]] = Field(..., min_length=2)
    units: Annotated[
        Literal["in", "m", "cm", "ft", "units"],
        BeforeValidator(
            lambda v: {
                "inch": "in",
                "inches": "in",
                "meter": "m",
                "meters": "m",
                "centimeter": "cm",
                "centimeters": "cm",
                "foot": "ft",
                "feet": "ft",
            }.get(v.lower(), v.lower())
        ),
    ]
    hide_measurements: list[int] = Field(
        default_factory=list,
        description="List of prism indexes (0-based) for which measurements should be hidden. If empty, shows measurements for all prisms.",
    )
    show_labels: list[bool] = Field(
        default_factory=list,
        description="List of boolean values for each prism indicating whether to show measurement labels. If empty or shorter than figures list, defaults to True for missing entries.",
    )

    # ---- layout knobs ----
    layout: Literal["auto", "stack", "side", "L"] = "auto"
    gap: float = 0.0  # spacing between prisms
    positions: list[list[float]] | None = Field(
        default=None,
        description="Explicit (x,y,z) positions for each prism as [[x,y,z], [x,y,z], ...]. If provided, must have same length as figures list.",
    )
    seed: int | None = None  # used for auto layout tie-breaks AND camera randomness
