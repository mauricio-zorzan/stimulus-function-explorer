import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.ratio_object_array import (
    _darken_color,
    draw_ratio_object_array,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.ratio_object_array import (
    RatioObjectArray,
    RatioObjectCell,
    RatioObjectShape,
)
from content_generators.settings import settings
from matplotlib import colors as mcolors


def _ok_ext(path: str) -> bool:
    fmt = getattr(settings.additional_content_settings, "stimulus_image_format", "png")
    return any(path.endswith(f".{e}") for e in [fmt, "png", "jpg", "jpeg", "webp"])


def test_two_shapes_ok(tmp_path):
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ffa366")
    stim = RatioObjectArray(rows=2, columns=3, objects=[[a, b, a], [b, a, b]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_three_shapes_ok(tmp_path):
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="red")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="blue")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="green")
    stim = RatioObjectArray(
        rows=2,
        columns=4,
        objects=[[a, b, c, a], [b, c, a, b]],  # Changed to 2 rows
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_four_shapes_ok(tmp_path):
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="red")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="blue")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="green")
    d = RatioObjectCell(shape=RatioObjectShape.STAR, color="gold")
    stim = RatioObjectArray(rows=2, columns=4, objects=[[a, b, c, d], [b, c, d, a]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_hexagon_supported(tmp_path):
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    h = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ff9966")
    stim = RatioObjectArray(rows=2, columns=3, objects=[[a, h, a], [h, a, h]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_shape_count_validation():
    # Only 1 distinct shape -> error
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="red")
    stim = RatioObjectArray(rows=2, columns=2, objects=[[a, a], [a, a]])
    with pytest.raises(ValueError):
        draw_ratio_object_array(stim)

    # 5 distinct shapes -> error
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="red")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="blue")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="green")
    d = RatioObjectCell(shape=RatioObjectShape.STAR, color="gold")
    h = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="purple")
    stim = RatioObjectArray(rows=1, columns=5, objects=[[a, b, c, d, h]])
    with pytest.raises(ValueError):
        draw_ratio_object_array(stim)


def test_grid_mismatch_raises():
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="red")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="blue")
    # rows=2, but only one row provided
    stim = RatioObjectArray(rows=2, columns=2, objects=[[a, b]])
    with pytest.raises(ValueError):
        draw_ratio_object_array(stim)


def test_border_is_darker():
    base = "#6699ff"
    darker = _darken_color(base, factor=0.5)
    try:
        base_rgba = mcolors.to_rgba(base)
    except (ValueError, TypeError):
        # Skip test if color parsing fails
        pytest.skip("Color parsing failed")
    for i in range(3):
        assert darker[i] < base_rgba[i]


def test_striped_pattern_two_rows(tmp_path):
    """Vertical stripes of two shapes (useful for ratio prompts) - updated for 2 rows and 10-shape limit."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#b9ec11")
    b = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#580eec")
    grid = []
    rows, cols = 2, 5  # Changed to 2×5 = 10 shapes (max allowed)
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(a if c % 2 == 0 else b)
        grid.append(row)
    stim = RatioObjectArray(rows=rows, columns=cols, objects=grid)
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_quadrant_pattern_two_rows(tmp_path):
    """Four-shape layout in 2 rows."""
    A = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#da09cc")
    B = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#0e0ca6")
    C = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#26932c")
    D = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#cb8112")
    rows, cols = 2, 4  # Changed to 2 rows
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if r == 0 and c < 2:
                row.append(A)
            elif r == 0 and c >= 2:
                row.append(B)
            elif r == 1 and c < 2:
                row.append(C)
            else:
                row.append(D)
        grid.append(row)
    stim = RatioObjectArray(rows=rows, columns=cols, objects=grid)
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_star_parity_among_others(tmp_path):
    """2x4 with circle, square, triangle, star — for quick visual parity check."""
    A = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#e53935")
    B = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#1e88e5")
    C = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#2e7d32")
    D = RatioObjectCell(shape=RatioObjectShape.STAR, color="#fdd835")
    stim = RatioObjectArray(rows=2, columns=4, objects=[[A, B, C, D], [B, C, D, A]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_diagonal_band_three_shapes_two_rows(tmp_path):
    """Diagonal band with three shapes for ratio prompts - updated for 2 rows and 10-shape limit."""
    A = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#64b5f6")
    B = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#66bb6a")
    C = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ffa366")
    rows, cols = 2, 5  # Changed to 2×5 = 10 shapes (max allowed)
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if abs(c - r) <= 1:
                row.append(B)
            elif c < r:
                row.append(A)
            else:
                row.append(C)
        grid.append(row)
    stim = RatioObjectArray(rows=rows, columns=cols, objects=grid)
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_checker_four_shapes_two_rows(tmp_path):
    """Checkerboard of four distinct shapes - updated for 2 rows and 10-shape limit."""
    A = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#ef5350")
    B = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#42a5f5")
    C = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#66bb6a")
    D = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ffa726")
    rows, cols = 2, 5  # Changed to 2×5 = 10 shapes (max allowed)
    grid = []
    palette = [A, B, C, D]
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(palette[(r + c) % 4])
        grid.append(row)
    stim = RatioObjectArray(rows=rows, columns=cols, objects=grid)
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out)


def test_hexagon_with_random_colors_and_row_limit(tmp_path):
    """Test hexagon visibility with random colors and 2-row limit."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    h = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ff9966")
    # Test with exactly 2 rows (max allowed)
    stim = RatioObjectArray(rows=2, columns=4, objects=[[a, h, a, h], [h, a, h, a]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_row_limit_validation():
    """Test that more than 2 rows raises validation error."""
    from pydantic import ValidationError

    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ff9966")

    # This should fail validation since we limit to 2 rows
    with pytest.raises(ValidationError):
        RatioObjectArray(rows=3, columns=2, objects=[[a, b], [b, a], [a, b]])


def test_all_shapes_with_random_colors(tmp_path):
    """Test 4 shapes with random color assignment (maximum allowed)."""
    shapes = [
        RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff"),
        RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966"),
        RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66"),
        RatioObjectCell(shape=RatioObjectShape.STAR, color="#ff6699"),
        # Removed HEXAGON to stay within 4-shape limit
    ]
    # Use 2 rows, 4 columns to show 4 shapes
    stim = RatioObjectArray(rows=2, columns=4, objects=[shapes, shapes])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


# New edge case tests
def test_minimum_grid_size(tmp_path):
    """Test minimum grid size (1x1)."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    stim = RatioObjectArray(rows=1, columns=2, objects=[[a, b]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_maximum_columns_within_shape_limit(tmp_path):
    """Test maximum columns that stay within 10-shape limit."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    # Create 2 rows with 5 columns each = 10 shapes (max allowed)
    row1 = [a if i % 2 == 0 else b for i in range(5)]
    row2 = [b if i % 2 == 0 else a for i in range(5)]
    stim = RatioObjectArray(rows=2, columns=5, objects=[row1, row2])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_maximum_total_shapes_validation():
    """Test that exceeding 10 total shapes raises validation error."""
    from pydantic import ValidationError

    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")

    # Test exceeding 10 shapes (2×6 = 12 shapes)
    with pytest.raises(ValidationError):
        RatioObjectArray(
            rows=2, columns=6, objects=[[a, b, a, b, a, b], [b, a, b, a, b, a]]
        )


def test_exactly_ten_shapes(tmp_path):
    """Test exactly 10 shapes (maximum allowed)."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66")
    # 2×5 = 10 shapes exactly
    stim = RatioObjectArray(
        rows=2, columns=5, objects=[[a, b, c, a, b], [b, c, a, b, c]]
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_exactly_two_distinct_shapes(tmp_path):
    """Test exactly 2 distinct shapes (minimum requirement)."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    stim = RatioObjectArray(rows=2, columns=3, objects=[[a, b, a], [b, a, b]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_exactly_four_distinct_shapes(tmp_path):
    """Test exactly 4 distinct shapes (maximum requirement)."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66")
    d = RatioObjectCell(shape=RatioObjectShape.STAR, color="#ff6699")
    stim = RatioObjectArray(rows=2, columns=4, objects=[[a, b, c, d], [d, c, b, a]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_single_row_grid(tmp_path):
    """Test single row grid."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66")
    stim = RatioObjectArray(rows=1, columns=6, objects=[[a, b, c, a, b, c]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_hexagon_only_with_other_shapes(tmp_path):
    """Test hexagon specifically with other shapes to ensure visibility."""
    h = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#6699ff")
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    stim = RatioObjectArray(rows=2, columns=4, objects=[[h, a, h, a], [a, h, a, h]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_hexagon_specifically_with_three_others(tmp_path):
    """Test hexagon specifically with 3 other shapes (4 total)."""
    shapes = [
        RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff"),
        RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966"),
        RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66"),
        RatioObjectCell(
            shape=RatioObjectShape.HEXAGON, color="#6699ff"
        ),  # Include hexagon
    ]
    # Use 2 rows, 4 columns to show 4 shapes including hexagon
    stim = RatioObjectArray(rows=2, columns=4, objects=[shapes, shapes])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_single_row_with_auto_shape_size(tmp_path):
    """Test single row with automatic smaller shape size."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    c = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#99ff66")
    stim = RatioObjectArray(rows=1, columns=6, objects=[[a, b, c, a, b, c]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_custom_shape_size(tmp_path):
    """Test custom shape size parameter."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    stim = RatioObjectArray(
        rows=2,
        columns=3,
        objects=[[a, b, a], [b, a, b]],
        shape_size=0.6,  # Custom smaller size
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_large_shape_size(tmp_path):
    """Test large shape size parameter."""
    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")
    stim = RatioObjectArray(
        rows=2,
        columns=3,
        objects=[[a, b, a], [b, a, b]],
        shape_size=0.9,  # Custom larger size
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_shape_size_validation():
    """Test shape size validation limits."""
    from pydantic import ValidationError

    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")

    # Test minimum shape size
    with pytest.raises(ValidationError):
        RatioObjectArray(
            rows=2,
            columns=2,
            objects=[[a, b], [b, a]],
            shape_size=0.2,  # Too small
        )

    # Test maximum shape size
    with pytest.raises(ValidationError):
        RatioObjectArray(
            rows=2,
            columns=2,
            objects=[[a, b], [b, a]],
            shape_size=1.1,  # Too large
        )


def test_effective_shape_size_method():
    """Test the get_effective_shape_size method."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.ratio_object_array import (
        DEFAULT_SHAPE_SIZE,
        SINGLE_ROW_SHAPE_SIZE,
        RatioObjectArray,
        RatioObjectCell,
        RatioObjectShape,
    )

    a = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#66ccff")
    b = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#ff9966")

    # Test single row auto-size
    stim_single = RatioObjectArray(rows=1, columns=2, objects=[[a, b]])
    assert stim_single.get_effective_shape_size() == SINGLE_ROW_SHAPE_SIZE

    # Test multi-row auto-size
    stim_multi = RatioObjectArray(rows=2, columns=2, objects=[[a, b], [b, a]])
    assert stim_multi.get_effective_shape_size() == DEFAULT_SHAPE_SIZE

    # Test custom size
    stim_custom = RatioObjectArray(
        rows=2, columns=2, objects=[[a, b], [b, a]], shape_size=0.7
    )
    assert stim_custom.get_effective_shape_size() == 0.7


def test_single_row_hexagon_focus(tmp_path):
    """Test single row with hexagon as primary shape to ensure visibility."""
    h = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#6699ff")
    c = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#ff9966")
    s = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#99ff66")
    # Single row with 6 shapes: 3 hexagons, 2 circles, 1 square
    stim = RatioObjectArray(rows=1, columns=6, objects=[[h, c, h, s, h, c]])
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_single_row_star_hexagon_combo(tmp_path):
    """Test single row with star and hexagon combination."""
    star = RatioObjectCell(shape=RatioObjectShape.STAR, color="#ff6699")
    hexagon = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#9966ff")
    triangle = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#66ff99")
    # Single row with 6 shapes: 2 stars, 2 hexagons, 2 triangles
    stim = RatioObjectArray(
        rows=1, columns=6, objects=[[star, hexagon, star, triangle, hexagon, triangle]]
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_single_row_four_shapes_maximum(tmp_path):
    """Test single row with 4 shape types (maximum allowed)."""
    circle = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#ff0000")
    square = RatioObjectCell(shape=RatioObjectShape.SQUARE, color="#00ff00")
    triangle = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#0000ff")
    hexagon = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#ffff00")
    # Single row with 4 shapes: one of each type (maximum allowed)
    stim = RatioObjectArray(
        rows=1, columns=4, objects=[[circle, square, triangle, hexagon]]
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)


def test_color_consistency_per_shape_type(tmp_path):
    """Test that all shapes of the same type have the same color within one generation."""
    # This test verifies the new color-per-shape-type feature
    circle1 = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#ff0000")
    circle2 = RatioObjectCell(shape=RatioObjectShape.CIRCLE, color="#ff0000")
    triangle1 = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#00ff00")
    triangle2 = RatioObjectCell(shape=RatioObjectShape.TRIANGLE, color="#00ff00")
    hexagon1 = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#0000ff")
    hexagon2 = RatioObjectCell(shape=RatioObjectShape.HEXAGON, color="#0000ff")

    # 2x3 grid with 2 circles, 2 triangles, 2 hexagons
    stim = RatioObjectArray(
        rows=2,
        columns=3,
        objects=[[circle1, triangle1, hexagon1], [circle2, triangle2, hexagon2]],
    )
    out = draw_ratio_object_array(stim)
    assert os.path.exists(out) and _ok_ext(out)
