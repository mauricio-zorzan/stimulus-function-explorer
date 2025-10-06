from collections import Counter
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class DualStatsLinePlot(StimulusDescription):
    top_title: str = Field(description="The title of the top data set")
    top_data: List[int] = Field(description="The list of data points for the top plot")
    bottom_title: str = Field(description="The title of the bottom data set")
    bottom_data: List[int] = Field(
        description="The list of data points for the bottom plot"
    )

    @model_validator(mode="after")
    def validate_data(self):
        # Validate top data
        top_data = self.top_data
        if len(top_data) < 10:
            raise ValueError("Top data must have at least 10 data points")

        top_unique_numbers = sorted(set(top_data))
        if len(top_unique_numbers) < 5 or len(top_unique_numbers) > 8:
            raise ValueError(
                "Top data must have at least 5 different numbers used, but no greater than 8"
            )

        # Check if the numbers are consecutive
        for i in range(len(top_unique_numbers) - 1):
            if top_unique_numbers[i + 1] - top_unique_numbers[i] != 1:
                raise ValueError("The unique numbers in top data must be consecutive")

        # Check if at least 50% of unique numbers are present at least twice
        top_counter = Counter(top_data)
        top_numbers_present_twice = sum(
            1 for count in top_counter.values() if count >= 2
        )
        if top_numbers_present_twice / len(top_unique_numbers) < 0.50:
            raise ValueError(
                "At least 50% of unique numbers in top data must be present at least twice"
            )

        # Validate bottom data
        bottom_data = self.bottom_data
        if len(bottom_data) < 10:
            raise ValueError("Bottom data must have at least 10 data points")

        bottom_unique_numbers = sorted(set(bottom_data))
        if len(bottom_unique_numbers) < 5 or len(bottom_unique_numbers) > 8:
            raise ValueError(
                "Bottom data must have at least 5 different numbers used, but no greater than 8"
            )

        # Check if the numbers are consecutive
        for i in range(len(bottom_unique_numbers) - 1):
            if bottom_unique_numbers[i + 1] - bottom_unique_numbers[i] != 1:
                raise ValueError(
                    "The unique numbers in bottom data must be consecutive"
                )

        # Check if at least 50% of unique numbers are present at least twice
        bottom_counter = Counter(bottom_data)
        bottom_numbers_present_twice = sum(
            1 for count in bottom_counter.values() if count >= 2
        )
        if bottom_numbers_present_twice / len(bottom_unique_numbers) < 0.50:
            raise ValueError(
                "At least 50% of unique numbers in bottom data must be present at least twice"
            )

        return self


if __name__ == "__main__":
    DualStatsLinePlot.generate_assistant_function_schema("sat-math")
