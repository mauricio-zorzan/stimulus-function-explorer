from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class TransversalAngleParams(StimulusDescription):
    given_angle: int = Field(
        ..., ge=0, le=180, description="The given angle in degrees."
    )
    x_angle_position: int = Field(
        ...,
        ge=1,
        le=8,
        description=(
            "Position number for the variable angle x (1 to 8). "
            "These positions refer to the 8 angles created by a transversal line "
            "crossing two parallel lines, numbered starting from the angle on the "
            "top line and going counter-clockwise, then continuing on the bottom line."
        ),
    )
    given_angle_position: int = Field(
        ...,
        ge=1,
        le=8,
        description=(
            "Position number for the given angle (1 to 8). "
            "These positions refer to the 8 angles created by a transversal line "
            "crossing two parallel lines, numbered starting from the angle on the "
            "top line and going counter-clockwise, then continuing on the bottom line."
        ),
    )

    def validate_positions(self):
        if self.x_angle_position == self.given_angle_position:
            raise ValueError(
                "The given angle and variable x angle positions must be different."
            )


if __name__ == "__main__":
    TransversalAngleParams.generate_assistant_function_schema("sat-math")
