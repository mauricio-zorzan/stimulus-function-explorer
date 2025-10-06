from typing import List

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription


class Line(BaseModel):
    slope: float = Field(..., description="The slope of the line")
    y_intercept: float = Field(..., description="The y-intercept of the line")
    label: str = Field(..., description="The label of the line")
    best_fit: bool = Field(
        ...,
        description="Indicates whether or not the line is the true line of best fit",
    )


class LinesOfBestFit(StimulusDescription):
    lines: List[Line] = Field(
        ..., description="List of lines of best fit", min_length=4, max_length=4
    )

    @model_validator(mode="after")
    def validate_best_fit(self):
        best_fit_count = sum(line.best_fit for line in self.lines)
        if best_fit_count != 1:
            raise ValueError("Exactly one line must have best_fit set to True")
        return self

    @model_validator(mode="after")
    def validate_lines_cross_10x10_space(self):
        for line in self.lines:
            if not line_crosses_10x10_space(line):
                raise ValueError(
                    f"Line {line.label} does not cross through the 10x10 space in the positive quadrant"
                )
        return self


def line_crosses_10x10_space(line: Line) -> bool:
    # Check if the line crosses the left edge (x = 0)
    y_at_x0 = line.y_intercept
    if 0 <= y_at_x0 <= 10:
        return True

    # Check if the line crosses the bottom edge (y = 0)
    x_at_y0 = -line.y_intercept / line.slope if line.slope != 0 else float("inf")
    if 0 <= x_at_y0 <= 10:
        return True

    # Check if the line crosses the right edge (x = 10)
    y_at_x10 = line.slope * 10 + line.y_intercept
    if 0 <= y_at_x10 <= 10:
        return True

    # Check if the line crosses the top edge (y = 10)
    x_at_y10 = (10 - line.y_intercept) / line.slope if line.slope != 0 else float("inf")
    if 0 <= x_at_y10 <= 10:
        return True

    return False


if __name__ == "__main__":
    LinesOfBestFit.generate_assistant_function_schema(type="mcq4")
