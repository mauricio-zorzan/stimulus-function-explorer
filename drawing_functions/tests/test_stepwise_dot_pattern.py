import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.stepwise_dot_pattern import (
    draw_stepwise_shape_pattern,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stepwise_dot_pattern import (
    StepwiseShapePattern,
    StepwiseShapePatternStep,
)
from matplotlib import pyplot as plt


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_simple_sequence():
    """Test a simple 3-step sequence with increasing columns."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=5, columns=1, color="#e78be7", shape="circle", label="step 1"
            ),
            StepwiseShapePatternStep(
                rows=5, columns=2, color="#8ecae6", shape="circle", label="step 2"
            ),
            StepwiseShapePatternStep(
                rows=5, columns=3, color="#e78be7", shape="circle", label="step 3"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_triangle_rotation():
    """Test triangle rotation variations."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=1,
                columns=1,
                color="#ffb703",
                shape="triangle",
                rotation=0,
                label="0 deg",
            ),
            StepwiseShapePatternStep(
                rows=1,
                columns=1,
                color="#ffb703",
                shape="triangle",
                rotation=90,
                label="90 deg",
            ),
            StepwiseShapePatternStep(
                rows=1,
                columns=1,
                color="#ffb703",
                shape="triangle",
                rotation=180,
                label="180 deg",
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_mixed_shapes():
    """Test a sequence with different shapes and varying dimensions."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=2, columns=2, color="#ff6b6b", shape="square", label="2x2 square"
            ),
            StepwiseShapePatternStep(
                rows=3, columns=1, color="#4ecdc4", shape="circle", label="3x1 circle"
            ),
            StepwiseShapePatternStep(
                rows=2,
                columns=3,
                color="#ffd93d",
                shape="triangle",
                label="2x3 triangle",
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_large_sequence():
    """Test a large sequence with 6 steps that should wrap to multiple rows."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=1, columns=4, color="#6c5ce7", shape="circle", label="Step 1"
            ),
            StepwiseShapePatternStep(
                rows=2, columns=3, color="#a8e6cf", shape="square", label="Step 2"
            ),
            StepwiseShapePatternStep(
                rows=3, columns=2, color="#ff8b94", shape="triangle", label="Step 3"
            ),
            StepwiseShapePatternStep(
                rows=2, columns=4, color="#6c5ce7", shape="circle", label="Step 4"
            ),
            StepwiseShapePatternStep(
                rows=1, columns=5, color="#a8e6cf", shape="square", label="Step 5"
            ),
            StepwiseShapePatternStep(
                rows=3, columns=3, color="#ff8b94", shape="triangle", label="Step 6"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_single_column():
    """Test a sequence with single-column patterns of different heights."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=1, columns=1, color="#ff9a8b", shape="circle", label="1 row"
            ),
            StepwiseShapePatternStep(
                rows=3, columns=1, color="#ff9a8b", shape="circle", label="3 rows"
            ),
            StepwiseShapePatternStep(
                rows=5, columns=1, color="#ff9a8b", shape="circle", label="5 rows"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_complex_patterns():
    """Test complex patterns with varying dimensions and shapes."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=4, columns=4, color="#ff6b6b", shape="square", label="4x4 square"
            ),
            StepwiseShapePatternStep(
                rows=2, columns=6, color="#4ecdc4", shape="circle", label="2x6 circle"
            ),
            StepwiseShapePatternStep(
                rows=3,
                columns=5,
                color="#ffd93d",
                shape="triangle",
                label="3x5 triangle",
            ),
            StepwiseShapePatternStep(
                rows=5, columns=3, color="#ff6b6b", shape="square", label="5x3 square"
            ),
            StepwiseShapePatternStep(
                rows=4, columns=4, color="#4ecdc4", shape="circle", label="4x4 circle"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_with_missing_step():
    """Test pattern with a missing step (zero dimensions) but with a label."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=2, columns=2, color="#ff6b6b", shape="square", label="Step 1"
            ),
            StepwiseShapePatternStep(
                rows=2, columns=2, color="#ffd93d", shape="triangle", label="Step 2"
            ),
            StepwiseShapePatternStep(
                rows=0, columns=0, color="#4ecdc4", shape="circle", label="Missing Step"
            ),
            StepwiseShapePatternStep(
                rows=2, columns=2, color="#6c5ce7", shape="circle", label="Step 4"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_from_thing2_json():
    """Test the draw_stepwise_shape_pattern function with the object from thing2.json."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=3,
                columns=4,
                color="blue",
                shape="square",
                rotation=0.0,
                label="Step 1",
            ),
            StepwiseShapePatternStep(
                rows=6,
                columns=4,
                color="blue",
                shape="square",
                rotation=0.0,
                label="Step 2",
            ),
            StepwiseShapePatternStep(
                rows=0,
                columns=0,
                color="blue",
                shape="square",
                rotation=0.0,
                label="Step 3",
            ),
            StepwiseShapePatternStep(
                rows=12,
                columns=4,
                color="blue",
                shape="square",
                rotation=0.0,
                label="Step 4",
            ),
        ],
        shape_size=1.0,
        spacing=0.5,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_stepwise_shape_pattern_seven_steps_missing_six():
    """Test 7 steps, 3 columns each, 2 more rows per step, step 6 missing."""
    stimulus = StepwiseShapePattern(
        steps=[
            StepwiseShapePatternStep(
                rows=2, columns=3, color="#e78be7", shape="square", label="Step 1"
            ),
            StepwiseShapePatternStep(
                rows=4, columns=3, color="#8ecae6", shape="square", label="Step 2"
            ),
            StepwiseShapePatternStep(
                rows=6, columns=3, color="#ffb703", shape="square", label="Step 3"
            ),
            StepwiseShapePatternStep(
                rows=8, columns=3, color="#4ecdc4", shape="square", label="Step 4"
            ),
            StepwiseShapePatternStep(
                rows=10, columns=3, color="#ff6b6b", shape="square", label="Step 5"
            ),
            StepwiseShapePatternStep(
                rows=0, columns=0, color="#ffd93d", shape="square", label="Step 6"
            ),
            StepwiseShapePatternStep(
                rows=14, columns=3, color="#6c5ce7", shape="square", label="Step 7"
            ),
        ],
        shape_size=1.0,
        spacing=1.0,
    )
    file_name = draw_stepwise_shape_pattern(stimulus)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None
