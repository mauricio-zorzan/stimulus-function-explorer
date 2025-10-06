from collections import Counter
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class StatsLinePlot(StimulusDescription):
    title: str = Field(description="The title of the data set")
    data: List[int] = Field(description="The list of data points")

    @model_validator(mode="after")
    def validate_data(self):
        data = self.data

        # Check total length first
        if len(data) < 10:
            raise ValueError("There must be at least 10 data points")

        unique_numbers = sorted(set(data))

        # Check if there are at least 5 and at most 8 unique numbers
        if len(unique_numbers) < 5 or len(unique_numbers) > 8:
            raise ValueError(
                "There must be at least 5 different numbers used, but no greater than 8"
            )

        # Check if the numbers are consecutive
        for i in range(len(unique_numbers) - 1):
            if unique_numbers[i + 1] - unique_numbers[i] != 1:
                raise ValueError("The unique numbers must be consecutive")

        # Check if at least 50% of unique numbers are present at least twice
        counter = Counter(data)
        numbers_present_twice = sum(1 for count in counter.values() if count >= 2)
        if numbers_present_twice / len(unique_numbers) < 0.50:
            raise ValueError(
                "At least 50% of unique numbers must be present at least twice"
            )

        return self


if __name__ == "__main__":
    StatsLinePlot.generate_assistant_function_schema("sat-math")
