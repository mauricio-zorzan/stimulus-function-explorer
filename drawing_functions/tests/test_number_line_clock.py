import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.number_lines_clock import (
    create_clock_number_line,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.number_line_clock_model import (
    NumberLineClockStimulus,
    Range,
    TimePoint,
)


@pytest.mark.drawing_functions
def test_basic_clock_number_line():
    stimulus = NumberLineClockStimulus(
        range=Range(min=7, max=10),
        points=[
            TimePoint(label="Start of call", hour=8, minute=45),
        ],
    )
    file_name = create_clock_number_line(stimulus.model_dump())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_basic_clock_number_line_2():
    stimulus = NumberLineClockStimulus(
        range=Range(min=3, max=6),
        points=[
            TimePoint(label="Start: 3:15 pm", hour=3, minute=15),
            TimePoint(label="End: 5:45 pm", hour=5, minute=45),
        ],
    )
    file_name = create_clock_number_line(stimulus.model_dump())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_clock_number_line_with_multiple_points():
    stimulus = NumberLineClockStimulus(
        range=Range(min=11, max=1),
        points=[
            TimePoint(label="Breakfast", hour=11, minute=50),
            TimePoint(label="Dinner", hour=12, minute=50),
        ],
    )
    file_name = create_clock_number_line(stimulus.model_dump())
    assert os.path.exists(file_name)
