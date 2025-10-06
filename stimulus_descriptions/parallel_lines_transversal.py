from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import AliasChoices, Field


class ParallelLinesTransversal(StimulusDescription):
    """
    This schema represents the description of a parallel lines traversal of two parallel lines and a transversal line intersecting at two points of the same angle within the range of 30 to 150 degrees.
    """

    angle: float = Field(
        ...,
        ge=30,
        le=150,
        description="The measure of the top-left-most angle in degrees by which the transversal line cuts the parallel lines.",
    )
    top_line_label: str = Field(
        default="A",
        min_length=1,
        max_length=1,
        description="The letter label override for the top parallel line.",
    )
    bottom_line_label: str = Field(
        default="B",
        min_length=1,
        max_length=1,
        description="The letter label override for the bottom parallel line.",
    )
    transversal_line_label: str = Field(
        default="C",
        min_length=1,
        max_length=1,
        description="The letter label override for the transversal line.",
    )
    top_line_top_left_angle_label: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "top_angle_label", "top_line_top_left_angle_label"
        ),
        description="The label for the angle that the transversal line makes with the top parallel line on the top left of the transversal line. Either an equation or a degree value.",
    )
    bottom_line_top_left_angle_label: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "bottom_angle_label", "bottom_line_top_left_angle_label"
        ),
        description="The label for the angle that the transversal line makes with the bottom parallel line on the top left of the transversal line. Either an equation or a degree value.",
    )
    top_line_top_right_angle_label: str | None = Field(
        default=None,
        description="The label for the angle that the transversal line makes with the top parallel line on the top right of the transversal line. Either an equation or a degree value.",
    )
    bottom_line_top_right_angle_label: str | None = Field(
        default=None,
        description="The label for the angle that the transversal line makes with the bottom parallel line on the top right of the transversal line. Either an equation or a degree value.",
    )


if __name__ == "__main__":
    ParallelLinesTransversal.generate_assistant_function_schema("mcq4")
