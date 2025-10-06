import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.prism_nets import (
    draw_custom_triangular_prism_net,
    draw_prism_net,
    draw_pyramid_net,
    draw_rectangular_prism_net,
    draw_triangular_prism_net,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.prism_net import (
    CustomTriangularPrismNet,
    EPrismType,
    PyramidPrismNet,
    RectangularPrismNet,
    TriangularPrismNet,
)
from content_generators.settings import settings
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_draw_rectangular_prism_net_basic():
    """Test drawing a basic rectangular prism net."""
    net = RectangularPrismNet(
        height=5,
        width=8,
        length=8,
        unit_label="in",
    )

    file_path = draw_rectangular_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )
    assert net.surface_area == 2 * (5 * 8 + 5 * 8 + 8 * 8)  # 2(lw + lh + wh)


@pytest.mark.drawing_functions
def test_draw_custom_triangular_prism_net_basic():
    """Test basic case for custom triangular prism net with all sides labeled."""
    h, w, l, side_w = 4, 6, 8, 5  # side_w as integer
    # Calculate surface area
    triangle_area = h * w  # Two triangular faces
    rect_area = (w * l) + (2 * side_w * l)  # front + two sides
    total_area = triangle_area + rect_area

    net = CustomTriangularPrismNet(
        height=h,
        width=w,
        length=l,
        side_w=side_w,  # Integer
        expected_surface_area=total_area,
        unit_label="in",
        blank_net=False,
    )
    file_path = draw_custom_triangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_rectangular_prism_net_square():
    """Test drawing a rectangular prism net with square faces (cube)."""
    net = RectangularPrismNet(
        height=5,
        width=5,
        length=5,
        unit_label="in",
    )

    file_path = draw_rectangular_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )
    assert net.surface_area == 2 * (5 * 5 + 5 * 5 + 5 * 5)  # 2(lw + lh + wh)


@pytest.mark.drawing_functions
def test_draw_triangular_prism_net_basic():
    """Test drawing a basic triangular prism net."""
    net = TriangularPrismNet(
        height=5,
        width=8,
        length=8,
        unit_label="in",
    )

    file_path = draw_triangular_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )

    # Calculate expected surface area
    side_w = (5**2 + (8 / 2) ** 2) ** 0.5
    expected_area = 8 * 5 + 8 * 8 + 2 * side_w * 8
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_triangular_prism_net_equilateral():
    """Test drawing a triangular prism net with equilateral triangle faces."""
    net = TriangularPrismNet(
        height=8,
        width=8,
        length=8,
        unit_label="in",
    )

    file_path = draw_triangular_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )

    # Calculate expected surface area for equilateral triangle
    side_w = (8**2 + (8 / 2) ** 2) ** 0.5
    expected_area = 8 * 8 + 8 * 8 + 2 * side_w * 8
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_triangular_prism_net__llm():
    """Test drawing a triangular prism net with equilateral triangle faces."""
    net = TriangularPrismNet(
        height=6,
        width=5,
        length=8,
        unit_label="in",
    )

    file_path = draw_triangular_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )

    assert net.surface_area == 174


@pytest.mark.drawing_functions
def test_draw_pyramid_net_basic():
    """Test drawing a basic pyramid net with rectangular base."""
    net = PyramidPrismNet(
        height=8,
        width=6,
        length=10,  # Rectangular base (not square)
        unit_label="in",
    )

    file_path = draw_pyramid_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )
    # Surface area calculation: base_area + triangular faces
    expected_area = 6 * 10 + 2 * (0.5 * 6 * 8) + 2 * (0.5 * 10 * 8)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_pyramid_net_square_base():
    """Test drawing a pyramid net with square base."""
    net = PyramidPrismNet(
        height=10,
        width=8,
        length=8,  # Square base
        unit_label="in",
    )

    file_path = draw_pyramid_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )
    # Surface area calculation: base_area + triangular faces
    expected_area = 8 * 8 + 2 * (0.5 * 8 * 10) + 2 * (0.5 * 8 * 10)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_pyramid_net_dimensions_valid():
    """Test that valid pyramid net dimensions are accepted (both square and rectangular bases)."""
    # Test rectangular pyramid
    valid_net = PyramidPrismNet(
        height=8,
        width=6,
        length=10,
        unit_label="in",
    )
    assert valid_net.height == 8
    assert valid_net.width == 6
    assert valid_net.length == 10


@pytest.mark.drawing_functions
def test_pyramid_net_dimensions_boundary():
    """Test that pyramid net dimensions are accepted without hardcoded limits."""
    valid_net = PyramidPrismNet(
        height=12,
        width=10,
        length=8,
        unit_label="in",
    )
    assert valid_net.height == 12
    assert valid_net.width == 10
    assert valid_net.length == 8


@pytest.mark.drawing_functions
def test_pyramid_net_constraint_height_too_large():
    """Test that pyramid nets reject when height is too large relative to base."""
    with pytest.raises(ValueError, match="height too large"):
        PyramidPrismNet(
            height=12,
            width=2,
            length=2,  # Height (12) > max_base (2) * 3 = 6
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_negative_dimensions_rectangular():
    """Test that negative dimensions are rejected for rectangular prism net."""
    with pytest.raises(ValueError):
        RectangularPrismNet(
            height=-5,
            width=8,
            length=8,
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_negative_dimensions_triangular():
    """Test that negative dimensions are rejected for triangular prism net."""
    with pytest.raises(ValueError):
        TriangularPrismNet(
            height=5,
            width=-8,
            length=8,
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_negative_dimensions_pyramid():
    """Test that negative dimensions are rejected for pyramid net."""
    with pytest.raises(ValueError):
        PyramidPrismNet(
            height=8,
            width=6,
            length=-8,
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_zero_dimensions_rectangular():
    """Test that zero dimensions are rejected for rectangular prism net."""
    with pytest.raises(ValueError):
        RectangularPrismNet(
            height=0,
            width=8,
            length=8,
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_zero_dimensions_triangular():
    """Test that zero dimensions are rejected for triangular prism net."""
    with pytest.raises(ValueError):
        TriangularPrismNet(
            height=5,
            width=0,
            length=8,
            unit_label="in",
        )


@pytest.mark.drawing_functions
def test_zero_dimensions_pyramid():
    """Test that zero dimensions are rejected for pyramid net."""
    with pytest.raises(ValueError):
        PyramidPrismNet(
            height=8,
            width=6,
            length=0,
            unit_label="in",
        )


# DIFFICULTY LEVEL COVERAGE TESTS
@pytest.mark.drawing_functions
def test_easy_level_cube():
    """Test EASY level: Simple cube net."""
    net = RectangularPrismNet(
        height=4,
        width=4,
        length=4,
        unit_label="cm",
    )
    file_path = draw_rectangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_medium_level_triangular_prism():
    """Test MEDIUM level: Triangular prism requiring face distinction."""
    net = TriangularPrismNet(
        height=6,
        width=8,
        length=10,
        unit_label="in",
    )
    file_path = draw_triangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_hard_level_rectangular_pyramid():
    """Test HARD level: Rectangular pyramid net with complex dimensional relationships."""
    net = PyramidPrismNet(
        height=12,
        width=8,
        length=10,  # Rectangular base
        unit_label="m",
    )
    file_path = draw_pyramid_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_hard_level_square_pyramid():
    """Test HARD level: Square pyramid net with dimensional relationships."""
    net = PyramidPrismNet(
        height=12,
        width=8,
        length=8,  # Square base
        unit_label="m",
    )
    file_path = draw_pyramid_net(net)
    assert os.path.exists(file_path)


# LABEL ALL SIDES FUNCTIONALITY TESTS
@pytest.mark.drawing_functions
def test_rectangular_prism_net_with_all_sides_labeled():
    """Test rectangular prism net with all sides labeled."""
    net = RectangularPrismNet(
        height=5,
        width=6,
        length=8,
        unit_label="mm",
        label_all_sides=True,
    )
    file_path = draw_rectangular_prism_net(net)
    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )


@pytest.mark.drawing_functions
def test_rectangular_prism_net_without_all_sides_labeled():
    """Test rectangular prism net without all sides labeled (default behavior)."""
    net = RectangularPrismNet(
        height=5,
        width=6,
        length=8,
        unit_label="mm",
        label_all_sides=False,
    )
    file_path = draw_rectangular_prism_net(net)
    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )


@pytest.mark.drawing_functions
def test_triangular_prism_net_with_all_sides_labeled():
    """Test triangular prism net with all sides labeled."""
    net = TriangularPrismNet(
        height=4,
        width=6,
        length=8,
        unit_label="in",
        label_all_sides=True,
    )
    file_path = draw_triangular_prism_net(net)
    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )


@pytest.mark.drawing_functions
def test_pyramid_net_with_all_sides_labeled():
    """Test pyramid net with all sides labeled."""
    net = PyramidPrismNet(
        height=5,
        width=6,
        length=6,
        unit_label="cm",
        label_all_sides=True,
    )
    file_path = draw_pyramid_net(net)
    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )


# COMPREHENSIVE LABEL ALL SIDES FUNCTIONALITY TESTS
@pytest.mark.drawing_functions
def test_rectangular_prism_net_label_all_sides_components():
    """Test rectangular prism net with label_all_sides=True - test all components."""
    # Test different dimension combinations
    test_cases = [
        # Small dimensions
        RectangularPrismNet(
            height=3, width=4, length=5, unit_label="cm", label_all_sides=True
        ),
        # Square faces (cube-like)
        RectangularPrismNet(
            height=6, width=6, length=6, unit_label="mm", label_all_sides=True
        ),
        # Large dimensions
        RectangularPrismNet(
            height=10, width=12, length=8, unit_label="in", label_all_sides=True
        ),
        # Different units
        RectangularPrismNet(
            height=4, width=7, length=9, unit_label="ft", label_all_sides=True
        ),
    ]

    for net in test_cases:
        file_path = draw_rectangular_prism_net(net)
        assert os.path.exists(file_path)
        assert file_path.endswith(
            f".{settings.additional_content_settings.stimulus_image_format}"
        )
        # Verify the net has the expected properties
        assert net.height > 0
        assert net.width > 0
        assert net.length > 0
        assert net.label_all_sides is True


@pytest.mark.drawing_functions
def test_triangular_prism_net_label_all_sides_components():
    """Test triangular prism net with label_all_sides=True - test all components."""
    # Test different dimension combinations
    test_cases = [
        # Equilateral triangle faces
        TriangularPrismNet(
            height=6, width=6, length=8, unit_label="cm", label_all_sides=True
        ),
        # Right triangle faces
        TriangularPrismNet(
            height=4, width=8, length=10, unit_label="mm", label_all_sides=True
        ),
        # Large dimensions
        TriangularPrismNet(
            height=12, width=10, length=15, unit_label="in", label_all_sides=True
        ),
        # Different units
        TriangularPrismNet(
            height=5, width=7, length=9, unit_label="ft", label_all_sides=True
        ),
    ]

    for net in test_cases:
        file_path = draw_triangular_prism_net(net)
        assert os.path.exists(file_path)
        assert file_path.endswith(
            f".{settings.additional_content_settings.stimulus_image_format}"
        )
        # Verify the net has the expected properties
        assert net.height > 0
        assert net.width > 0
        assert net.length > 0
        assert net.label_all_sides is True


@pytest.mark.drawing_functions
def test_pyramid_net_label_all_sides_components():
    """Test pyramid net with label_all_sides=True - test all components."""
    # Test different dimension combinations
    test_cases = [
        # Square base pyramid
        PyramidPrismNet(
            height=8, width=6, length=6, unit_label="cm", label_all_sides=True
        ),
        # Rectangular base pyramid
        PyramidPrismNet(
            height=10, width=8, length=12, unit_label="mm", label_all_sides=True
        ),
        # Large dimensions
        PyramidPrismNet(
            height=15, width=10, length=14, unit_label="in", label_all_sides=True
        ),
        # Different units
        PyramidPrismNet(
            height=6, width=5, length=7, unit_label="ft", label_all_sides=True
        ),
    ]

    for net in test_cases:
        file_path = draw_pyramid_net(net)
        assert os.path.exists(file_path)
        assert file_path.endswith(
            f".{settings.additional_content_settings.stimulus_image_format}"
        )
        # Verify the net has the expected properties
        assert net.height > 0
        assert net.width > 0
        assert net.length > 0
        assert net.label_all_sides is True


@pytest.mark.drawing_functions
def test_label_all_sides_default_behavior():
    """Test that label_all_sides defaults to False and doesn't break existing functionality."""
    # Test rectangular prism with default label_all_sides (False)
    net_default = RectangularPrismNet(height=5, width=6, length=8, unit_label="mm")
    assert net_default.label_all_sides is False

    file_path_default = draw_rectangular_prism_net(net_default)
    assert os.path.exists(file_path_default)

    # Test triangular prism with default label_all_sides (False)
    net_tri_default = TriangularPrismNet(height=4, width=6, length=8, unit_label="in")
    assert net_tri_default.label_all_sides is False

    file_path_tri_default = draw_triangular_prism_net(net_tri_default)
    assert os.path.exists(file_path_tri_default)

    # Test pyramid with default label_all_sides (False)
    net_pyr_default = PyramidPrismNet(height=5, width=6, length=6, unit_label="cm")
    assert net_pyr_default.label_all_sides is False

    file_path_pyr_default = draw_pyramid_net(net_pyr_default)
    assert os.path.exists(file_path_pyr_default)


@pytest.mark.drawing_functions
def test_label_all_sides_edge_cases():
    """Test label_all_sides functionality with edge cases."""
    # Test with minimum valid dimensions
    net_min = RectangularPrismNet(
        height=1, width=1, length=1, unit_label="mm", label_all_sides=True
    )
    file_path_min = draw_rectangular_prism_net(net_min)
    assert os.path.exists(file_path_min)

    # Test with larger dimensions (reduced from 20,25,30 to avoid font/overlap issues)
    net_large = RectangularPrismNet(
        height=8, width=10, length=12, unit_label="in", label_all_sides=True
    )
    file_path_large = draw_rectangular_prism_net(net_large)
    assert os.path.exists(file_path_large)

    # Test with different unit labels
    net_units = RectangularPrismNet(
        height=5, width=6, length=8, unit_label="units", label_all_sides=True
    )
    file_path_units = draw_rectangular_prism_net(net_units)
    assert os.path.exists(file_path_units)


@pytest.mark.drawing_functions
def test_label_all_sides_surface_area_calculation():
    """Test that label_all_sides doesn't affect surface area calculations."""
    # Test rectangular prism
    net_rect = RectangularPrismNet(
        height=5, width=6, length=8, unit_label="mm", label_all_sides=True
    )
    expected_area_rect = 2 * (5 * 6 + 5 * 8 + 6 * 8)  # 2(lw + lh + wh)
    assert net_rect.surface_area == expected_area_rect

    # Test triangular prism
    net_tri = TriangularPrismNet(
        height=4, width=6, length=8, unit_label="in", label_all_sides=True
    )
    side_w = (4**2 + (6 / 2) ** 2) ** 0.5
    expected_area_tri = 6 * 4 + 6 * 8 + 2 * side_w * 8
    assert net_tri.surface_area == expected_area_tri

    # Test pyramid
    net_pyr = PyramidPrismNet(
        height=5, width=6, length=6, unit_label="cm", label_all_sides=True
    )
    expected_area_pyr = 6 * 6 + 2 * (0.5 * 6 * 5) + 2 * (0.5 * 6 * 5)
    assert net_pyr.surface_area == expected_area_pyr


@pytest.mark.drawing_functions
def test_draw_prism_net_generic():
    """Test the generic draw_prism_net function with different net types."""
    test_cases = [
        (
            RectangularPrismNet(height=5, width=8, length=8, unit_label="in"),
            EPrismType.RECTANGULAR,
        ),
        (
            TriangularPrismNet(height=5, width=8, length=8, unit_label="in"),
            EPrismType.TRIANGULAR,
        ),
        (
            PyramidPrismNet(
                height=8, width=6, length=10, unit_label="in"
            ),  # Rectangular pyramid
            EPrismType.PYRAMIDAL,
        ),
    ]

    for net, expected_type in test_cases:
        file_path = draw_prism_net(net)
        assert os.path.exists(file_path)
        assert file_path.endswith(
            f".{settings.additional_content_settings.stimulus_image_format}"
        )
        assert net.net_type == expected_type


@pytest.mark.drawing_functions
def test_generic_prism_net_with_all_sides_labeled():
    """Test the generic draw_prism_net function with label_all_sides=True for all types."""
    test_cases = [
        (
            RectangularPrismNet(
                height=5, width=8, length=8, unit_label="in", label_all_sides=True
            ),
            EPrismType.RECTANGULAR,
        ),
        (
            TriangularPrismNet(
                height=5, width=8, length=8, unit_label="in", label_all_sides=True
            ),
            EPrismType.TRIANGULAR,
        ),
        (
            PyramidPrismNet(
                height=8, width=6, length=10, unit_label="in", label_all_sides=True
            ),
            EPrismType.PYRAMIDAL,
        ),
    ]

    for net, expected_type in test_cases:
        file_path = draw_prism_net(net)
        assert os.path.exists(file_path)
        assert file_path.endswith(
            f".{settings.additional_content_settings.stimulus_image_format}"
        )
        assert net.net_type == expected_type
        assert net.label_all_sides is True


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_with_specified_dimensions():
    """Test draw_prism_net function with pyramidal net using specific dimensions."""
    net = PyramidPrismNet(
        height=19,
        width=7,
        length=19,
        unit_label="cm",
        label_all_sides=True,
    )

    file_path = draw_prism_net(net)

    assert os.path.exists(file_path)
    assert file_path.endswith(
        f".{settings.additional_content_settings.stimulus_image_format}"
    )
    assert net.net_type == EPrismType.PYRAMIDAL
    assert net.label_all_sides is True
    assert net.height == 19
    assert net.width == 7
    assert net.length == 19
    assert net.unit_label == "cm"


# =====================================================================================
# COMPREHENSIVE DRAW_PRISM_NET TEST CASES - ALL SHAPES AND SIZES
# =====================================================================================


# RECTANGULAR PRISM TESTS - SMALL SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_small_cube():
    """Test draw_prism_net with small cube (all dimensions equal)."""
    net = RectangularPrismNet(
        height=2,
        width=2,
        length=2,
        unit_label="cm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (2 * 2 + 2 * 2 + 2 * 2)  # 2(lw + lh + wh) = 24


@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_small_elongated():
    """Test draw_prism_net with small elongated rectangular prism."""
    net = RectangularPrismNet(
        height=1,
        width=3,
        length=5,
        unit_label="mm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (1 * 3 + 1 * 5 + 3 * 5)  # 2(lw + lh + wh) = 62


@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_small_tall():
    """Test draw_prism_net with small tall rectangular prism."""
    net = RectangularPrismNet(
        height=6,
        width=2,
        length=3,
        unit_label="in",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (6 * 2 + 6 * 3 + 2 * 3)  # 2(lw + lh + wh) = 72


# RECTANGULAR PRISM TESTS - LARGE SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_large_cube():
    """Test draw_prism_net with large cube (all dimensions equal)."""
    net = RectangularPrismNet(
        height=12,
        width=12,
        length=12,
        unit_label="ft",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (
        12 * 12 + 12 * 12 + 12 * 12
    )  # 2(lw + lh + wh) = 864


@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_large_elongated():
    """Test draw_prism_net with large elongated rectangular prism."""
    net = RectangularPrismNet(
        height=8,
        width=15,
        length=20,
        unit_label="m",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (8 * 15 + 8 * 20 + 15 * 20)  # 2(lw + lh + wh) = 1160


@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_large_flat():
    """Test draw_prism_net with large flat rectangular prism."""
    net = RectangularPrismNet(
        height=2,
        width=18,
        length=25,
        unit_label="units",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (2 * 18 + 2 * 25 + 18 * 25)  # 2(lw + lh + wh) = 1076


# TRIANGULAR PRISM TESTS - SMALL SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_small_equilateral():
    """Test draw_prism_net with small equilateral triangular prism."""
    net = TriangularPrismNet(
        height=3,
        width=3,
        length=4,
        unit_label="cm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    # Calculate expected surface area for triangular prism
    side_w = (3**2 + (3 / 2) ** 2) ** 0.5
    expected_area = 3 * 3 + 3 * 4 + 2 * side_w * 4
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_small_right_triangle():
    """Test draw_prism_net with small right triangular prism."""
    net = TriangularPrismNet(
        height=4,
        width=5,
        length=6,
        unit_label="mm",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    side_w = (4**2 + (5 / 2) ** 2) ** 0.5
    expected_area = 4 * 5 + 5 * 6 + 2 * side_w * 6
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_small_isosceles():
    """Test draw_prism_net with small isosceles triangular prism."""
    net = TriangularPrismNet(
        height=6,
        width=4,
        length=3,
        unit_label="in",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    side_w = (6**2 + (4 / 2) ** 2) ** 0.5
    expected_area = 6 * 4 + 4 * 3 + 2 * side_w * 3
    assert net.surface_area == expected_area


# TRIANGULAR PRISM TESTS - LARGE SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_large_equilateral():
    """Test draw_prism_net with large equilateral triangular prism."""
    net = TriangularPrismNet(
        height=15,
        width=15,
        length=20,
        unit_label="ft",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    side_w = (15**2 + (15 / 2) ** 2) ** 0.5
    expected_area = 15 * 15 + 15 * 20 + 2 * side_w * 20
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_large_scalene():
    """Test draw_prism_net with large scalene triangular prism."""
    net = TriangularPrismNet(
        height=12,
        width=18,
        length=25,
        unit_label="m",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    side_w = (12**2 + (18 / 2) ** 2) ** 0.5
    expected_area = 12 * 18 + 18 * 25 + 2 * side_w * 25
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_large_tall():
    """Test draw_prism_net with large tall triangular prism."""
    net = TriangularPrismNet(
        height=20,
        width=8,
        length=12,
        unit_label="units",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    side_w = (20**2 + (8 / 2) ** 2) ** 0.5
    expected_area = 20 * 8 + 8 * 12 + 2 * side_w * 12
    assert net.surface_area == expected_area


# PYRAMIDAL PRISM TESTS - SMALL SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_small_square_base():
    """Test draw_prism_net with small square-based pyramid."""
    net = PyramidPrismNet(
        height=4,
        width=3,
        length=3,
        unit_label="cm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 3 * 3 + 2 * (0.5 * 3 * 4) + 2 * (0.5 * 3 * 4)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_small_rectangular_base():
    """Test draw_prism_net with small rectangular-based pyramid."""
    net = PyramidPrismNet(
        height=5,
        width=3,
        length=6,
        unit_label="mm",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 3 * 6 + 2 * (0.5 * 3 * 5) + 2 * (0.5 * 6 * 5)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_small_low():
    """Test draw_prism_net with small low pyramid."""
    net = PyramidPrismNet(
        height=2,
        width=4,
        length=5,
        unit_label="in",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 4 * 5 + 2 * (0.5 * 4 * 2) + 2 * (0.5 * 5 * 2)
    assert net.surface_area == expected_area


# PYRAMIDAL PRISM TESTS - LARGE SIZES
@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_large_square_base():
    """Test draw_prism_net with large square-based pyramid."""
    net = PyramidPrismNet(
        height=18,
        width=12,
        length=12,
        unit_label="ft",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 12 * 12 + 2 * (0.5 * 12 * 18) + 2 * (0.5 * 12 * 18)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_large_rectangular_base():
    """Test draw_prism_net with large rectangular-based pyramid."""
    net = PyramidPrismNet(
        height=15,
        width=10,
        length=20,
        unit_label="m",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 10 * 20 + 2 * (0.5 * 10 * 15) + 2 * (0.5 * 20 * 15)
    assert net.surface_area == expected_area


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_large_tall():
    """Test draw_prism_net with large tall pyramid."""
    net = PyramidPrismNet(
        height=24,
        width=8,
        length=10,
        unit_label="units",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    expected_area = 8 * 10 + 2 * (0.5 * 8 * 24) + 2 * (0.5 * 10 * 24)
    assert net.surface_area == expected_area


# EDGE CASE TESTS - MINIMUM AND MAXIMUM VALID DIMENSIONS
@pytest.mark.drawing_functions
def test_draw_prism_net_rectangular_minimum_dimensions():
    """Test draw_prism_net with minimum valid dimensions for rectangular prism."""
    net = RectangularPrismNet(
        height=1,
        width=1,
        length=1,
        unit_label="cm",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)
    assert net.surface_area == 2 * (1 * 1 + 1 * 1 + 1 * 1)  # 6


@pytest.mark.drawing_functions
def test_draw_prism_net_triangular_minimum_dimensions():
    """Test draw_prism_net with minimum valid dimensions for triangular prism."""
    net = TriangularPrismNet(
        height=1,
        width=1,
        length=1,
        unit_label="mm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_prism_net_pyramidal_minimum_dimensions():
    """Test draw_prism_net with minimum valid dimensions for pyramid."""
    net = PyramidPrismNet(
        height=1,
        width=1,
        length=1,
        unit_label="in",
        label_all_sides=True,
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)


# DIFFERENT UNIT LABEL TESTS
@pytest.mark.drawing_functions
def test_draw_prism_net_various_units():
    """Test draw_prism_net with different unit labels."""
    units = ["cm", "mm", "m", "in", "ft", "units", "yd", "km"]

    for i, unit in enumerate(units):
        net = RectangularPrismNet(
            height=3 + i,
            width=4 + i,
            length=5 + i,
            unit_label=unit,
            label_all_sides=(i % 2 == 0),  # Alternate label_all_sides
        )
        file_path = draw_prism_net(net)
        assert os.path.exists(file_path)
        assert unit in net.unit_label


# ASPECT RATIO TESTS
@pytest.mark.drawing_functions
def test_draw_prism_net_extreme_aspect_ratios():
    """Test draw_prism_net with extreme but valid aspect ratios."""

    # Very wide rectangular prism
    net_wide = RectangularPrismNet(
        height=1,
        width=20,
        length=2,
        unit_label="cm",
    )
    file_path_wide = draw_prism_net(net_wide)
    assert os.path.exists(file_path_wide)

    # Very tall rectangular prism
    net_tall = RectangularPrismNet(
        height=20,
        width=2,
        length=2,
        unit_label="m",
        label_all_sides=True,
    )
    file_path_tall = draw_prism_net(net_tall)
    assert os.path.exists(file_path_tall)

    # Very long triangular prism
    net_long = TriangularPrismNet(
        height=3,
        width=4,
        length=25,
        unit_label="ft",
    )
    file_path_long = draw_prism_net(net_long)
    assert os.path.exists(file_path_long)


# COMPREHENSIVE LABEL_ALL_SIDES TESTS
@pytest.mark.drawing_functions
def test_draw_prism_net_all_shapes_with_labels():
    """Test draw_prism_net with label_all_sides=True for all shapes."""

    # Rectangular with labels
    net_rect = RectangularPrismNet(
        height=7,
        width=9,
        length=11,
        unit_label="cm",
        label_all_sides=True,
    )
    file_path_rect = draw_prism_net(net_rect)
    assert os.path.exists(file_path_rect)
    assert net_rect.label_all_sides is True

    # Triangular with labels
    net_tri = TriangularPrismNet(
        height=8,
        width=10,
        length=12,
        unit_label="m",
        label_all_sides=True,
    )
    file_path_tri = draw_prism_net(net_tri)
    assert os.path.exists(file_path_tri)
    assert net_tri.label_all_sides is True

    # Pyramidal with labels
    net_pyr = PyramidPrismNet(
        height=9,
        width=6,
        length=8,
        unit_label="in",
        label_all_sides=True,
    )
    file_path_pyr = draw_prism_net(net_pyr)
    assert os.path.exists(file_path_pyr)
    assert net_pyr.label_all_sides is True


@pytest.mark.drawing_functions
def test_draw_prism_net_all_shapes_without_labels():
    """Test draw_prism_net with label_all_sides=False for all shapes."""

    # Rectangular without labels
    net_rect = RectangularPrismNet(
        height=6,
        width=8,
        length=10,
        unit_label="mm",
        label_all_sides=False,
    )
    file_path_rect = draw_prism_net(net_rect)
    assert os.path.exists(file_path_rect)
    assert net_rect.label_all_sides is False

    # Triangular without labels
    net_tri = TriangularPrismNet(
        height=7,
        width=9,
        length=11,
        unit_label="ft",
        label_all_sides=False,
    )
    file_path_tri = draw_prism_net(net_tri)
    assert os.path.exists(file_path_tri)
    assert net_tri.label_all_sides is False

    # Pyramidal without labels
    net_pyr = PyramidPrismNet(
        height=8,
        width=5,
        length=7,
        unit_label="units",
        label_all_sides=False,
    )
    file_path_pyr = draw_prism_net(net_pyr)
    assert os.path.exists(file_path_pyr)
    assert net_pyr.label_all_sides is False


# BOUNDARY CONDITION TESTS
@pytest.mark.drawing_functions
def test_draw_prism_net_pyramid_height_boundary():
    """Test draw_prism_net with pyramid at maximum allowed height ratio."""
    # Height = 3 * max(width, length) should be valid
    net = PyramidPrismNet(
        height=15,  # 3 * 5 = 15
        width=4,
        length=5,
        unit_label="cm",
    )
    file_path = draw_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_prism_net_performance_large_dimensions():
    """Test draw_prism_net performance with large valid dimensions."""

    # Large rectangular prism
    net_large = RectangularPrismNet(
        height=25,
        width=30,
        length=35,
        unit_label="m",
        label_all_sides=True,
    )
    file_path_large = draw_prism_net(net_large)
    assert os.path.exists(file_path_large)

    # Large triangular prism
    net_tri_large = TriangularPrismNet(
        height=22,
        width=28,
        length=35,
        unit_label="ft",
    )
    file_path_tri_large = draw_prism_net(net_tri_large)
    assert os.path.exists(file_path_tri_large)

    # Large pyramid (respecting height constraint)
    net_pyr_large = PyramidPrismNet(
        height=30,  # 3 * 10 = 30
        width=8,
        length=10,
        unit_label="units",
        label_all_sides=True,
    )
    file_path_pyr_large = draw_prism_net(net_pyr_large)
    assert os.path.exists(file_path_pyr_large)


@pytest.mark.drawing_functions
def test_custom_triangular_prism_net_boundary_min():
    """Test minimum valid dimensions for custom triangular prism net."""
    h, w, l, side_w = 2, 3, 3, 2  # side_w as integer
    expected_area = (h * w) + (w * l) + (2 * side_w * l)
    net = CustomTriangularPrismNet(
        height=h,
        width=w,
        length=l,
        side_w=side_w,  # Integer
        expected_surface_area=expected_area,
        unit_label="cm",
        blank_net=False,
    )
    file_path = draw_custom_triangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_custom_triangular_prism_net_boundary_max():
    """Test maximum valid dimensions within assessment boundary (≤ 40)."""
    h, w, l, side_w = 35, 40, 38, 37  # side_w as integer
    expected_area = (h * w) + (w * l) + (2 * side_w * l)
    net = CustomTriangularPrismNet(
        height=h,
        width=w,
        length=l,
        side_w=side_w,  # Integer
        expected_surface_area=expected_area,
        unit_label="cm",
        blank_net=False,
    )
    file_path = draw_custom_triangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_custom_triangular_prism_net_invalid_positive_areas():
    """Test invalid but positive surface areas."""
    h, w, l, side_w = 4, 6, 8, 5
    correct_area = (
        h * w  # Two triangular faces
        + w * l  # Front rectangular face
        + side_w * l * 2  # Two side rectangular faces
    )

    # Test cases with invalid but positive areas
    invalid_areas = [
        correct_area - 1,  # Just under correct area
        correct_area + 1,  # Just over correct area
        correct_area // 2,  # Much smaller
        correct_area * 2,  # Much larger
    ]

    for invalid_area in invalid_areas:
        with pytest.raises(
            ValueError, match="Expected surface area .* does not match calculated area"
        ):
            CustomTriangularPrismNet(
                height=h,
                width=w,
                length=l,
                side_w=side_w,
                expected_surface_area=invalid_area,
                unit_label="cm",
                blank_net=False,
            )


@pytest.mark.drawing_functions
def test_custom_triangular_prism_net_non_positive_areas():
    """Test that non-positive surface areas are rejected by Pydantic validation."""
    h, w, l, side_w = 4, 6, 8, 5

    # Test cases with non-positive areas
    invalid_areas = [0, -1]

    for invalid_area in invalid_areas:
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            CustomTriangularPrismNet(
                height=h,
                width=w,
                length=l,
                side_w=side_w,
                expected_surface_area=invalid_area,
                unit_label="cm",
                blank_net=False,
            )


@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "h,w,l,side_w",
    [
        (6, 16, 24, 10),  # very short h vs width
        (8, 20, 24, 12),  # short h
        (12, 32, 24, 20),  # your screenshot case
        (16, 24, 24, 20),  # h≈w
        (20, 20, 24, 20),  # equilateral-ish rectangle heights
        (24, 32, 24, 20),  # tall h
    ],
)
def test_custom_triangular_prism_net_label_scaling_matrix(h, w, l, side_w):
    expected_area = (h * w) + (w * l) + (2 * side_w * l)
    net = CustomTriangularPrismNet(
        height=h,
        width=w,
        length=l,
        side_w=side_w,
        expected_surface_area=expected_area,
        unit_label="cm",
        blank_net=False,
        label_all_sides=True,
    )
    file_path = draw_custom_triangular_prism_net(net)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "h,w,l,side_w",
    [
        (24, 16, 40, 18),  # tall l relative to w/side_w
        (30, 12, 36, 20),  # very tall center rectangles
        (18, 14, 32, 15),  # moderate tall
    ],
)
def test_custom_triangular_prism_net_labels_ok_when_rectangles_tall(h, w, l, side_w):
    expected_area = (h * w) + (w * l) + (2 * side_w * l)
    net = CustomTriangularPrismNet(
        height=h,
        width=w,
        length=l,
        side_w=side_w,
        expected_surface_area=expected_area,
        unit_label="cm",
        blank_net=False,
        label_all_sides=True,
    )
    file_path = draw_custom_triangular_prism_net(net)
    assert os.path.exists(file_path)
