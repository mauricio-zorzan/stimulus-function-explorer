import re
from typing import TYPE_CHECKING, List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from content_generators.utils import String
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class DataPoint(BaseModel):
    """
    A data point for a line plot, with a maximum frequency of 8.
    """

    class Config:
        description = "A data point for a line plot."

    value: str = Field(
        description="The x-axis value for the data point as a whole number, mixed number or category.",
        json_schema_extra={"examples": ["1", "1 1/2", "category"]},
    )
    value_parsed: float | str = Field(
        description="The parsed x-axis value as a float, or string for easier sorting.",
        exclude=True,
    )
    value_label: str = Field(
        description="The label for the x-axis value, if it is a string.", exclude=True
    )
    frequency: int = Field(
        description="The frequency or count of occurrences at this value.", le=8, ge=0
    )

    def is_numerical_value(self):
        if isinstance(self.value_parsed, str):
            pattern = r"^(\d+\s+\d+\/\d+|\d+\/\d+)$"
            if re.match(pattern, self.value_parsed):
                return True
            else:
                return False
        else:
            return True

    # Add hidden fields that have data parsed as required
    @model_validator(mode="before")
    def validate_and_create_hidden_fields(cls, data_point: dict):
        value = String.convert_improper_fraction_to_proper(data_point["value"])
        data_point["value_parsed"] = String.convert_optional_fraction_string_to_float(
            value
        )
        data_point["value_label"] = (
            String.convert_string_with_optional_fraction_to_latex(value)
            if isinstance(data_point["value_parsed"], float)
            else value
        )
        return data_point


class LinePlot(BaseModel):
    """
    Line plot stimulus description
    """

    class Config:
        description = "Line plot stimulus description"

    title: str = Field(
        description="The title of the line plot with correct context for question."
    )
    x_axis_label: str = Field(
        description="The label for the x-axis of the line plot with correct context for question."
    )
    data_points: List[DataPoint] = Field(
        description="List of data points and their frequencies for the line plot.",
        json_schema_extra={"minItems": 1, "maxItems": 8},
    )

    @model_validator(mode="after")
    def validate_data_points(self):
        if self.data_points[0].is_numerical_value():
            self.data_points.sort(
                key=lambda x: x.value_parsed
            )  # Sort data points by value_parsed
        return self


class SingleLinePlot(StimulusDescription):
    """
    Single line plot stimulus description (for single-plot standards)
    """

    class Config:
        description = "Single line plot stimulus description"

    title: str = Field(
        description="The title of the line plot with correct context for question."
    )
    x_axis_label: str = Field(
        description="The label for the x-axis of the line plot with correct context for question."
    )
    data_points: List[DataPoint] = Field(
        description="List of data points and their frequencies for the line plot.",
        json_schema_extra={"minItems": 1, "maxItems": 8},
    )

    @model_validator(mode="after")
    def validate_data_points(self):
        if self.data_points and self.data_points[0].is_numerical_value():
            self.data_points.sort(key=lambda x: x.value_parsed)
        return self

    # Add any single-plot specific validation here if needed


class LinePlotList(StimulusDescriptionList[LinePlot]):
    def assert_correct_frequencies_is_unique_for_select(
        self, pipeline_context: "QuestionGeneratorContext"
    ):
        """
        Ensure that the correct set of frequencies is unique for each line plot in the list.
        """
        if pipeline_context.question is None:
            raise ValueError("Question is None")
        if not hasattr(pipeline_context.question, "correct_answer"):
            return
        correct_answer_text = pipeline_context.question.correct_answer.strip()  # type: ignore
        figure = next(
            (plot for plot in self.root if plot.title.strip() == correct_answer_text),
            None,
        )
        if figure is None:
            raise ValueError(
                f"No figure found with title matching the correct answer: {correct_answer_text}"
            )

        frequencies = [[dp.frequency for dp in plot.data_points] for plot in self.root]
        correct_frequencies = [dp.frequency for dp in figure.data_points]
        if frequencies.count(correct_frequencies) > 1:
            raise ValueError(
                f"The set of frequencies {correct_frequencies} is not unique among the line plots."
            )

    def assert_correctly_configured_select_question(
        self, pipeline_context: "QuestionGeneratorContext"
    ):
        assert len(self.root) == 4, "Select/Make sub standard that requires 4 plots."
        self.assert_correct_frequencies_is_unique_for_select(
            pipeline_context=pipeline_context
        )

    def assert_correct_answer_not_all_ones(
        self, pipeline_context: "QuestionGeneratorContext"
    ):
        """
        Assert that the correct answer does not have a frequency of 1 for all data points.
        """
        if pipeline_context.question is None:
            raise ValueError("Question is None")
        if not hasattr(pipeline_context.question, "correct_answer"):
            return
        correct_answer_text = pipeline_context.question.correct_answer.strip()  # type: ignore
        figure = next(
            (plot for plot in self.root if plot.title.strip() == correct_answer_text),
            None,
        )
        if figure is None:
            raise ValueError(
                f"No figure found with title matching the correct answer: {correct_answer_text}"
            )

        if all(dp.frequency == 1 for dp in figure.data_points):
            raise ValueError(
                "The correct answer has a frequency of 1 for all data points, which is not allowed."
            )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        match pipeline_context.standard_id:
            # The following 2 sub standards are select/make questions
            case "CCSS.MATH.CONTENT.6.SP.B.4+1":
                self.assert_correctly_configured_select_question(pipeline_context)
                self.assert_correct_answer_not_all_ones(pipeline_context)
            case "CCSS.MATH.CONTENT.6.SP.B.5.A":
                assert all(
                    isinstance(dp.value_parsed, float)
                    for plot in self.root
                    for dp in plot.data_points
                ), "All data points must have a parsed value that is a float (no categories allowed for CCSS.MATH.CONTENT.6.SP.B.5.A)."


if __name__ == "__main__":
    LinePlotList.generate_assistant_function_schema("mcq4")
    SingleLinePlot.generate_assistant_function_schema("single_plot")
