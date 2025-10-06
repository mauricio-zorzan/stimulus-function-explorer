from pydantic import Field, field_validator

from .stimulus_description import StimulusDescription


class DivideIntoEqualGroups(StimulusDescription):
    """
    Stimulus description for creating a visual representation of dots divided into equal groups.
    Shows big circles (groups) with small circles (dots) inside each group.
    """
    
    number_of_dots_per_group: int = Field(
        ..., 
        description="The number of dots in each group",
        ge=1,
        le=10
    )
    
    number_of_groups: int = Field(
        ...,
        description="The number of groups to divide the dots into",
        ge=1,
        le=10
    )
    
    @field_validator("number_of_dots_per_group", "number_of_groups")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Must be a positive integer")
        return v


if __name__ == "__main__":
    DivideIntoEqualGroups.generate_assistant_function_schema("mcq4") 