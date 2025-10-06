from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, RootModel

TStimulusDescription = TypeVar("TStimulusDescription", bound=BaseModel)

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class StimulusDescriptionProtocol:
    """
    The base class for all stimulus descriptions.
    """

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """
        Override this method to implement additional pipeline validation here
        This is called before the assistant validators

        :param validation_context: An instance of PipelineContext containing validation data.
        """
        if pipeline_context is None:
            raise ValueError("Validation context is None")

    def generate_image(self):
        """
        Override this method to implement custom image generation logic.
        """
        pass

    @classmethod
    def generate_assistant_function_schema(
        cls, type: Literal["mcq3", "mcq4", "sat-math"]
    ):
        from content_generators.additional_content.stimulus_image.legacy.generate_mcq3_choice import (
            GenerateMCQ3ChoiceSB,
        )
        from content_generators.additional_content.stimulus_image.legacy.generate_mcq4_choice import (
            GenerateMCQ4ChoiceSB,
        )
        from content_generators.additional_content.stimulus_image.legacy.generate_sat_mcq import (
            GenerateSATMathMCQSB,
        )
        from content_generators.additional_content.stimulus_image.stimulus_descriptions.common.assistant_function_schema import (
            generate_assistant_function_schema,
        )

        model_map = {
            "mcq3": GenerateMCQ3ChoiceSB,
            "mcq4": GenerateMCQ4ChoiceSB,
            "sat-math": GenerateSATMathMCQSB,
        }

        generate_assistant_function_schema(model=model_map[type][cls])


class StimulusDescription(BaseModel, StimulusDescriptionProtocol):
    """
    Stimulus Description base class.
    For use when you do not require a list of stimulus descriptions.
    Inherit your model from this in that case.
    """

    class Config:
        json_encoders = {Enum: lambda v: v.value}


class StimulusDescriptionList(
    StimulusDescriptionProtocol,
    RootModel[TStimulusDescription],
):
    """
    When using the StimulusDescriptionList, the root is a list of BaseModel objects.
    Do not use the StimulusDescription class as the base of your model that you want to inject into this list.
    """

    root: list[TStimulusDescription]

    class Config:
        json_encoders = {Enum: lambda v: v.value}

    def __init__(self, stimuli: list[dict | TStimulusDescription] = [], root=None):
        if root is not None:
            super().__init__(root)
        else:
            super().__init__(stimuli)  # type: ignore

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item) -> TStimulusDescription:
        return self.root[item]

    def __len__(self):
        return len(self.root)


file_name_str = str


class StimulusFunctionProtocol:
    stimulus_type: StimulusDescriptionProtocol | None = None

    def __call__(self, definition: dict | list[dict]) -> file_name_str:
        raise NotImplementedError()
