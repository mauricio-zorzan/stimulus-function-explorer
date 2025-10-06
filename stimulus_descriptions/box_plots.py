from typing import TYPE_CHECKING

from content_generators.types import UniqueList
from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class BoxPlotData(BaseModel):
    class_name: str | None = Field(default=None)
    min_value: float
    q1: float
    median: float
    q3: float
    max_value: float

    def __hash__(self):
        return hash(self.model_dump_json())

    @model_validator(mode="after")
    def validate_model(self):
        # Ensure that each successive value is bigger than the last
        values = [self.min_value, self.q1, self.median, self.q3, self.max_value]
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                raise ValueError(
                    f"Value {values[i]} should be less than {values[i + 1]}"
                )
        return self


class BoxPlotDescription(StimulusDescription):
    title: str | None = Field(default=None)
    data: UniqueList[BoxPlotData] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_model(self):
        if len(self.data) > 1:
            if self.title is None:
                raise ValueError("Title is required for multiple box plots.")
            for data in self.data:
                if data.class_name is None:
                    raise ValueError("Class name is required for multiple box plots.")
        else:
            # For single box plots, title is optional but allowed
            # Class name should be None for single box plots
            if self.data[0].class_name is not None:
                raise ValueError(
                    "Class name should not be provided for single box plots."
                )
        return self

    def get_display_title(self) -> str | None:
        """Get the title for display. Returns None if no title is provided."""
        return self.title

    @staticmethod
    def extract_distribution(text: str) -> list[float]:
        import re

        matches = re.findall(r"[\d.]+", text)
        return [float(match) for match in matches]

    def calculate_distribution_statistics(self, distribution: list[float]) -> dict:
        if not distribution:
            raise ValueError("Distribution list is empty.")

        sorted_distribution = sorted(distribution)
        n = len(sorted_distribution)

        def median(data: list[float]) -> float:
            mid = len(data) // 2
            if len(data) % 2 == 0:
                return (data[mid - 1] + data[mid]) / 2.0
            else:
                return data[mid]

        min_value = sorted_distribution[0]
        max_value = sorted_distribution[-1]
        q1 = median(sorted_distribution[: n // 2])
        q3 = median(sorted_distribution[(n + 1) // 2 :])
        med = median(sorted_distribution)

        return {
            "min_value": min_value,
            "q1": q1,
            "median": med,
            "q3": q3,
            "max_value": max_value,
        }

    def assert_distribution_is_correct(
        self, validation_context: "QuestionGeneratorContext"
    ):
        if not validation_context.question or not hasattr(
            validation_context.question, "correct_answer"
        ):
            return
        correct_answer = validation_context.question.correct_answer  # type: ignore
        distribution = self.extract_distribution(correct_answer)

        # Skip validation if only one number - likely not a full dataset
        if len(distribution) <= 1:
            return

        stats = self.calculate_distribution_statistics(distribution)

        for data in self.data:
            if (
                data.min_value != stats["min_value"]
                or data.q1 != stats["q1"]
                or data.median != stats["median"]
                or data.q3 != stats["q3"]
                or data.max_value != stats["max_value"]
            ):
                raise ValueError(
                    "Distribution statistics do not match the expected values."
                )

    def assert_answer_list_len_le(
        self, pipeline_context: "QuestionGeneratorContext", max_len: int
    ):
        """
        Enforce that any comma-separated list of answers is not longer than max_len,
        for question types that actually carry answers as strings.

        Supports:
        - MCQ-like: q.answer_options -> [AnswerOption(answer=str), ...]
        - Text-Entry-like: q.correct_answer (str or list[str]) or q.accepted_answers (list[str])

        Skips silently for other types (e.g., DragAndDrop), which don't expose those attributes.
        This is the new block
        """
        q = getattr(pipeline_context, "question", None)
        if q is None:
            return

        # Collect strings to validate
        answer_strings: list[str] = []

        # MCQ-style
        if hasattr(q, "answer_options") and q.answer_options:
            for opt in q.answer_options:
                s = getattr(opt, "answer", None)
                if isinstance(s, str):
                    answer_strings.append(s)

        # Text Entry (common shapes)
        if hasattr(q, "correct_answer"):
            ca = getattr(q, "correct_answer")
            if isinstance(ca, str):
                answer_strings.append(ca)
            elif isinstance(ca, (list, tuple)):
                answer_strings.extend([x for x in ca if isinstance(x, str)])

        if hasattr(q, "accepted_answers"):
            aa = getattr(q, "accepted_answers")
            if isinstance(aa, (list, tuple)):
                answer_strings.extend([x for x in aa if isinstance(x, str)])
            elif isinstance(aa, str):
                answer_strings.append(aa)

        # Nothing relevant to validate (e.g., DragAndDrop) -> exit quietly
        if not answer_strings:
            return

        # Apply the length rule on comma-separated tokens (ignore empty tokens/spaces)
        for s in answer_strings:
            tokens = [t for t in (piece.strip() for piece in s.split(",")) if t]
            if len(tokens) > max_len:
                raise ValueError(
                    f"Answer list length is greater than {max_len}. Got {len(tokens)} tokens in: {s!r}"
                )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.6.SP.B.4+3":
                self.assert_distribution_is_correct(pipeline_context)
                q = getattr(pipeline_context, "question", None)
                if q and (
                    hasattr(q, "answer_options")
                    or hasattr(q, "correct_answer")
                    or hasattr(q, "accepted_answers")
                ):
                    self.assert_answer_list_len_le(pipeline_context, 15)


if __name__ == "__main__":
    BoxPlotDescription.generate_assistant_function_schema(type="mcq4")
