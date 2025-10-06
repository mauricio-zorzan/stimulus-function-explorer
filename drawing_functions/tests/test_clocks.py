import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.clocks import (
    create_clock,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.clock import (
    Clock,
)


@pytest.fixture
def sample_analog_clock():
    return Clock(hour=3, minute=15, type="analog")


@pytest.fixture
def sample_digital_clock():
    return Clock(hour=3, minute=15, type="digital")


@pytest.mark.drawing_functions
def test_create_analog_clock(sample_analog_clock):
    file_name = create_clock(sample_analog_clock)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_digital_clock(sample_digital_clock):
    file_name = create_clock(sample_digital_clock)
    assert os.path.exists(file_name)
