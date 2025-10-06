import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing_function import (
    draw_graphing_function,
    draw_graphing_function_quadrant_one,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.graphing_function_model import (
    GraphingFunction,
    GraphingFunctionQuadrantOne,
)

# ========== ORIGINAL TESTS (Backward Compatibility) ==========


@pytest.mark.drawing_functions
def test_draw_graphing_function_linear():
    graphing_function = GraphingFunction(function_type="linear", a=1, b=-3, c=None)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadratic():
    graphing_function = GraphingFunction(function_type="quadratic", a=1, b=-3, c=4)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_exponential():
    graphing_function = GraphingFunction(function_type="exponential", a=1, b=3, c=None)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_linear():
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="linear", a=1, b=2, c=None
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_quadratic():
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="quadratic", a=0.5, b=1, c=2
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_exponential():
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="exponential", a=1, b=0.5, c=None
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_large_linear():
    """Test linear function with larger coefficients to show dynamic scaling"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="linear", a=3, b=8, c=None
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_large_quadratic():
    """Test quadratic function with larger coefficients to show dynamic scaling"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="quadratic", a=2, b=3, c=5
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_large_exponential():
    """Test exponential function with larger coefficients to show dynamic scaling"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="exponential", a=2, b=0.8, c=None
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_small_linear():
    """Test linear function with small coefficients to show minimum scaling"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="linear", a=0.3, b=0.5, c=None
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_steep_quadratic():
    """Test steep quadratic function to show maximum scaling"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="quadratic", a=8, b=1, c=1
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


# ========== NEW RELATION TYPE TESTS ==========


@pytest.mark.drawing_functions
def test_draw_graphing_function_cubic_basic():
    """Test basic cubic function: y = x³"""
    graphing_function = GraphingFunction(function_type="cubic", a=1, b=0, c=0, d=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_cubic_complex():
    """Test complex cubic function: y = 0.25x³ - 5x"""
    graphing_function = GraphingFunction(function_type="cubic", a=0.25, b=0, c=-5, d=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_square_root_basic():
    """Test basic square root function: y = √x"""
    graphing_function = GraphingFunction(function_type="square_root", a=1, b=0, c=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_square_root_shifted():
    """Test shifted square root function: y = √(x + 5)"""
    graphing_function = GraphingFunction(function_type="square_root", a=1, b=5, c=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_rational_basic():
    """Test basic rational function: y = 2/x"""
    graphing_function = GraphingFunction(function_type="rational", a=2, b=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_rational_shifted():
    """Test shifted rational function: y = 10/x + 2 (vertical shift)"""
    graphing_function = GraphingFunction(function_type="rational", a=10, b=2)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_circle_small():
    """Test small circle: x² + y² = 25"""
    graphing_function = GraphingFunction(function_type="circle", a=1, radius=5)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_circle_large():
    """Test large circle: x² + y² = 100"""
    graphing_function = GraphingFunction(function_type="circle", a=1, radius=10)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_sideways_parabola():
    """Test sideways parabola: x = y²"""
    graphing_function = GraphingFunction(
        function_type="sideways_parabola", a=1, b=0, c=0
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_hyperbola_standard():
    """Test standard hyperbola: x²/9 - y²/4 = 1"""
    graphing_function = GraphingFunction(
        function_type="hyperbola", a=1, x_radius=3, y_radius=2
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_hyperbola_elongated():
    """Test elongated hyperbola with different radii"""
    graphing_function = GraphingFunction(
        function_type="hyperbola", a=1, x_radius=5, y_radius=3
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_ellipse_standard():
    """Test standard ellipse: x²/4 + y²/4 = 1"""
    graphing_function = GraphingFunction(
        function_type="ellipse", a=1, x_radius=2, y_radius=2
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_ellipse_stretched():
    """Test stretched ellipse with different radii"""
    graphing_function = GraphingFunction(
        function_type="ellipse", a=1, x_radius=6, y_radius=3
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


# ========== QUADRANT ONE EXTENDED TESTS ==========


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_cubic():
    """Test cubic function in quadrant I"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="cubic", a=0.1, b=0, c=0, d=1
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_square_root():
    """Test square root function in quadrant I"""
    graphing_function = GraphingFunctionQuadrantOne(
        function_type="square_root", a=2, b=0, c=1
    )
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_graphing_function_quadrant_one_rational():
    """Test rational function in quadrant I"""
    graphing_function = GraphingFunctionQuadrantOne(function_type="rational", a=5, b=1)
    file_name = draw_graphing_function_quadrant_one(graphing_function)
    assert os.path.exists(file_name)


# ========== VALIDATION ERROR TESTS ==========


def test_quadratic_missing_coefficient_c():
    """Test that quadratic relations require coefficient c"""
    with pytest.raises(
        ValueError, match="Coefficient 'c' is required for quadratic relations"
    ):
        GraphingFunction(function_type="quadratic", a=1, b=2)


def test_cubic_missing_coefficient_d():
    """Test that cubic relations require coefficient d"""
    with pytest.raises(
        ValueError, match="Coefficient 'd' is required for cubic relations"
    ):
        GraphingFunction(function_type="cubic", a=1, b=2, c=3)


def test_circle_missing_radius():
    """Test that circular relations require radius parameter"""
    with pytest.raises(
        ValueError, match="Radius parameter is required for circular relations"
    ):
        GraphingFunction(function_type="circle", a=1)


def test_hyperbola_missing_radii():
    """Test that hyperbolic relations require both radii"""
    with pytest.raises(
        ValueError,
        match="Both x_radius and y_radius are required for hyperbola relations",
    ):
        GraphingFunction(function_type="hyperbola", a=1, x_radius=3)


def test_ellipse_missing_radii():
    """Test that elliptical relations require both radii"""
    with pytest.raises(
        ValueError,
        match="Both x_radius and y_radius are required for ellipse relations",
    ):
        GraphingFunction(function_type="ellipse", a=1, y_radius=4)


def test_linear_unused_coefficient():
    """Test that linear relations reject unused coefficients"""
    with pytest.raises(
        ValueError, match="Coefficient 'c' should not be provided for linear relations"
    ):
        GraphingFunction(function_type="linear", a=1, b=2, c=3)


def test_rational_zero_coefficient():
    """Test that rational relations reject zero primary coefficient"""
    with pytest.raises(
        ValueError, match="Coefficient 'a' cannot be zero for rational relations"
    ):
        GraphingFunction(function_type="rational", a=0, b=2)


def test_coefficient_value_too_large():
    """Test that coefficients cannot exceed maximum values"""
    with pytest.raises(ValueError, match="must be less than 20 in absolute value"):
        GraphingFunction(function_type="linear", a=25, b=2)


def test_negative_radius():
    """Test that radius parameters must be positive"""
    with pytest.raises(ValueError, match="Radius must be positive"):
        GraphingFunction(function_type="circle", a=1, radius=-5)


def test_quadrant_one_negative_output():
    """Test that quadrant I functions reject configurations producing negative values"""
    with pytest.raises(ValueError, match="negative y values in quadrant I"):
        GraphingFunctionQuadrantOne(function_type="linear", a=-2, b=1)


def test_square_root_invalid_domain_quadrant_one():
    """Test that square root functions validate domain in quadrant I"""
    with pytest.raises(
        ValueError,
        match="Coefficient 'b' must be less than 15 in absolute value for quadrant I",
    ):
        GraphingFunctionQuadrantOne(function_type="square_root", a=1, b=-15, c=0)


# ========== EDGE CASE VISUAL TESTS ==========


@pytest.mark.drawing_functions
def test_very_small_coefficients():
    """Test handling of very small coefficient values"""
    graphing_function = GraphingFunction(function_type="linear", a=0.01, b=0.001)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_steep_rational_function():
    """Test rational function with steep asymptotic behavior"""
    graphing_function = GraphingFunction(function_type="rational", a=0.1, b=0)
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_narrow_ellipse():
    """Test very narrow ellipse"""
    graphing_function = GraphingFunction(
        function_type="ellipse", a=1, x_radius=8, y_radius=1
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_wide_hyperbola():
    """Test wide hyperbola with large x_radius"""
    graphing_function = GraphingFunction(
        function_type="hyperbola", a=1, x_radius=7, y_radius=2
    )
    file_name = draw_graphing_function(graphing_function)
    assert os.path.exists(file_name)
