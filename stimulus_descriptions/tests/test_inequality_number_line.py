import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.number_lines import (
    create_inequality_number_line,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.inequality_number_line import (
    InequalityNumberLine,
    Line,
    Point,
    Range,
)
from pydantic import ValidationError


def test_invalid_range():
    with pytest.raises(ValidationError):
        InequalityNumberLine(
            range=Range(min=0, max=30),
            points=[Point(fill=True, value=3)],
            line=Line(min=3, max=7),
        )


def test_invalid_line_length():
    with pytest.raises(
        ValueError, match="Invalid line: max - min must be greater than 0."
    ):
        stim_desc = InequalityNumberLine(
            range=Range(min=0, max=10),
            points=[Point(fill=True, value=5)],
            line=Line(min=5, max=5),
        )
        create_inequality_number_line(stim_desc)


def test_line_point_mismatch():
    with pytest.raises(ValidationError):
        InequalityNumberLine(
            range=Range(min=0, max=10),
            points=[Point(fill=True, value=3)],
            line=Line(min=2, max=8),
        )
