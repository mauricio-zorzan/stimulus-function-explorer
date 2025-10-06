import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.number_lines import (
    create_inequality_number_line,
    create_multi_inequality_number_line,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.inequality_number_line import (
    InequalityNumberLine,
    Line,
    Point,
    Range,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.multi_inequality_number_line import (
    MultiInequalityNumberLine,
)


@pytest.mark.drawing_functions
def test_valid_inequality_number_line():
    stim_desc = InequalityNumberLine(
        range=Range(min=89, max=99),
        points=[Point(fill=True, value=93), Point(fill=False, value=95)],
        line=Line(min=93, max=95),
    )
    result = create_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_valid_inequality_number_line_long():
    stim_desc = InequalityNumberLine(
        range=Range(min=-10, max=10),
        points=[Point(fill=True, value=3), Point(fill=False, value=5)],
        line=Line(min=3, max=5),
    )
    result = create_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_valid_inequality_number_line_long_2_digit_labels():
    stim_desc = InequalityNumberLine(
        range=Range(min=70, max=90),
        points=[Point(fill=True, value=75), Point(fill=False, value=80)],
        line=Line(min=75, max=80),
    )
    result = create_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_open_ended_line():
    stim_desc = InequalityNumberLine(
        range=Range(min=0, max=10),
        points=[Point(fill=True, value=5)],
        line=Line(min=5, max=None),
    )
    result = create_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_open_ended_line_min():
    stim_desc = InequalityNumberLine(
        range=Range(min=0, max=10),
        points=[Point(fill=True, value=5)],
        line=Line(min=None, max=5),
    )
    result = create_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_valid_multi_inequality_number_line():
    """Test creating a 2x2 grid of inequality number lines."""
    # Create 4 different inequality number lines
    number_lines = [
        InequalityNumberLine(
            range=Range(min=0, max=10),
            points=[Point(fill=True, value=3), Point(fill=False, value=7)],
            line=Line(min=3, max=7),
        ),
        InequalityNumberLine(
            range=Range(min=-5, max=5),
            points=[Point(fill=True, value=-2), Point(fill=False, value=2)],
            line=Line(min=-2, max=2),
        ),
        InequalityNumberLine(
            range=Range(min=10, max=20),
            points=[Point(fill=True, value=15)],
            line=Line(min=15, max=None),
        ),
        InequalityNumberLine(
            range=Range(min=1, max=8),
            points=[Point(fill=True, value=5)],
            line=Line(min=None, max=5),
        ),
    ]

    stim_desc = MultiInequalityNumberLine(number_lines=number_lines)
    result = create_multi_inequality_number_line(stim_desc)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_multi_inequality_number_line_with_different_ranges():
    """Test creating a multi-inequality number line with different ranges."""
    number_lines = [
        InequalityNumberLine(
            range=Range(min=0, max=5),
            points=[Point(fill=True, value=2), Point(fill=False, value=4)],
            line=Line(min=2, max=4),
        ),
        InequalityNumberLine(
            range=Range(min=0, max=15),
            points=[Point(fill=True, value=5), Point(fill=False, value=10)],
            line=Line(min=5, max=10),
        ),
        InequalityNumberLine(
            range=Range(min=-10, max=0),
            points=[Point(fill=True, value=-5), Point(fill=False, value=-2)],
            line=Line(min=-5, max=-2),
        ),
        InequalityNumberLine(
            range=Range(min=20, max=30),
            points=[Point(fill=True, value=25)],
            line=Line(min=25, max=None),
        ),
    ]

    stim_desc = MultiInequalityNumberLine(number_lines=number_lines)
    result = create_multi_inequality_number_line(stim_desc)
    assert os.path.exists(result)
