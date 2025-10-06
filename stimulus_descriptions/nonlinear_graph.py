from typing import Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class NonlinearEquationParameters(BaseModel):
    coef1: Optional[float] = Field(
        default=None, description="The first coefficient of the equation"
    )
    coef2: Optional[float] = Field(
        default=None, description="The second coefficient of the equation"
    )
    coef3: Optional[float] = Field(
        default=None, description="The third coefficient of the equation"
    )
    coef4: Optional[float] = Field(
        default=None, description="The fourth coefficient of the equation"
    )
    coef5: Optional[float] = Field(
        default=None, description="The fifth coefficient of the equation"
    )
    coef6: Optional[float] = Field(
        default=None, description="The sixth coefficient of the equation"
    )


class NonlinearGraph(StimulusDescription):
    equation_type: Literal["quadratic", "cubic", "quartic", "quintic"] = Field(
        ..., description="The type of nonlinear equation"
    )
    parameters: NonlinearEquationParameters = Field(
        ..., description="Parameters of the nonlinear equation based on its type"
    )
