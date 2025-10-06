import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.prism_nets import (
    draw_dual_nets,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.prism_net import (
    DualNetsShapeType,
    DualPrismNets,
    Position,
)


@pytest.mark.drawing_functions
def test_cube_nets_correct_left():
    """Test drawing cube nets with correct net on the left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.LEFT,  # Show correct net in Figure 1 (left)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_nets_correct_right():
    """Test drawing cube nets with correct net on the right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.RIGHT,  # Show correct net in Figure 2 (right)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_nets_correct_left():
    """Test drawing rectangular prism nets with correct net on the left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.LEFT,  # Show correct net in Figure 1 (left)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_nets_correct_right():
    """Test drawing rectangular prism nets with correct net on the right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.RIGHT,  # Show correct net in Figure 2 (right)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_triangular_prism_nets_correct_left():
    """Test drawing triangular prism nets with correct net on the left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.LEFT,  # Show correct net in Figure 1 (left)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_triangular_prism_nets_correct_right():
    """Test drawing triangular prism nets with correct net on the right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,  # The shape for incorrect option
        correct_shape_position=Position.RIGHT,  # Show correct net in Figure 2 (right)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_square_pyramid_nets_correct_left():
    """Test drawing square pyramid nets with correct net on the left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.SQUARE_PYRAMID,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,  # The shape for incorrect option
        correct_shape_position=Position.LEFT,  # Show correct net in Figure 1 (left)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_square_pyramid_nets_correct_right():
    """Test drawing square pyramid nets with correct net on the right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.SQUARE_PYRAMID,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,  # The shape for incorrect option
        correct_shape_position=Position.RIGHT,  # Show correct net in Figure 2 (right)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_pyramid_nets_correct_left():
    """Test drawing rectangular pyramid nets with correct net on the left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,  # The shape for incorrect option
        correct_shape_position=Position.LEFT,  # Show correct net in Figure 1 (left)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_pyramid_nets_correct_right():
    """Test drawing rectangular pyramid nets with correct net on the right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,  # The shape we're asking about
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,  # The shape for incorrect option
        correct_shape_position=Position.RIGHT,  # Show correct net in Figure 2 (right)
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


# Additional combinations with Cube
@pytest.mark.drawing_functions
def test_cube_vs_triangular_prism_left():
    """Test cube vs triangular prism with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_vs_triangular_prism_right():
    """Test cube vs triangular prism with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_vs_square_pyramid_left():
    """Test cube vs square pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_vs_square_pyramid_right():
    """Test cube vs square pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_vs_rectangular_pyramid_left():
    """Test cube vs rectangular pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_cube_vs_rectangular_pyramid_right():
    """Test cube vs rectangular pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.CUBE,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


# Additional combinations with Rectangular Prism
@pytest.mark.drawing_functions
def test_rectangular_prism_vs_square_pyramid_left():
    """Test rectangular prism vs square pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_vs_square_pyramid_right():
    """Test rectangular prism vs square pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_vs_rectangular_pyramid_left():
    """Test rectangular prism vs rectangular pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_vs_rectangular_pyramid_right():
    """Test rectangular prism vs rectangular pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.RECTANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


# Additional combinations with Triangular Prism
@pytest.mark.drawing_functions
def test_triangular_prism_vs_square_pyramid_left():
    """Test triangular prism vs square pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_triangular_prism_vs_square_pyramid_right():
    """Test triangular prism vs square pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.SQUARE_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_triangular_prism_vs_rectangular_pyramid_left():
    """Test triangular prism vs rectangular pyramid with correct net on left."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.LEFT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_triangular_prism_vs_rectangular_pyramid_right():
    """Test triangular prism vs rectangular pyramid with correct net on right."""
    stimulus = DualPrismNets(
        correct_shape_type=DualNetsShapeType.TRIANGULAR_PRISM,
        incorrect_shape_type=DualNetsShapeType.RECTANGULAR_PYRAMID,
        correct_shape_position=Position.RIGHT,
    )
    filename = draw_dual_nets(stimulus)
    assert os.path.exists(filename)
