import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.trapezoids_decomposition import (
    TrapezoidDecomposition,
    create_trapezoid_decomposition_decimal_only,
)
from matplotlib import pyplot as plt


@pytest.mark.drawing_functions
def test_trapezoid_basic():
    data = TrapezoidDecomposition(
        units="cm",
        vertices=[[2.0, 5.0], [6.0, 5.0], [0.0, 0.0], [8.0, 0.0]],
        shaded=False,
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_narrow_top():
    data = TrapezoidDecomposition(
        units="in",
        vertices=[[3.5, 4.0], [5.0, 4.0], [2.0, 0.0], [8.0, 0.0]],
        shaded=True,
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_altitude_left_case():
    v0, v1, v2, v3 = (0.0, 5.0), (2.0, 5.0), (0.0, 0.0), (4.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=False
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_altitude_right_case_equal():
    v0, v1, v2, v3 = (1.0, 5.0), (3.0, 5.0), (0.0, 0.0), (3.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=True
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_altitude_right_case_inside():
    v0, v1, v2, v3 = (1.0, 5.0), (3.0, 5.0), (0.0, 0.0), (4.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=False
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_wide_bottom_tiny_top():
    v0, v1, v2, v3 = (2.0, 6.0), (2.5, 6.0), (0.0, 0.0), (8.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=True
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_isosceles_symmetric():
    v0, v1, v2, v3 = (2.0, 5.0), (6.0, 5.0), (0.0, 0.0), (8.0, 0.0)
    data = TrapezoidDecomposition(
        units="m", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=False
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_off_center_top():
    v0, v1, v2, v3 = (1.5, 5.0), (4.5, 5.0), (0.0, 0.0), (7.0, 0.0)
    data = TrapezoidDecomposition(
        units="in", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=True
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_left_vertical_side_variant():
    v0, v1, v2, v3 = (0.0, 6.0), (3.0, 6.0), (0.0, 0.0), (7.0, 0.0)
    data = TrapezoidDecomposition(
        units="ft", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=False
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_equal_right_top_to_bottom():
    v0, v1, v2, v3 = (2.0, 5.0), (4.0, 5.0), (0.0, 0.0), (4.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm", vertices=[list(v0), list(v1), list(v2), list(v3)], shaded=True
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_mode_compute_height():
    v0, v1, v2, v3 = (1.0, 5.0), (3.0, 5.0), (0.0, 0.0), (4.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm",
        vertices=[list(v0), list(v1), list(v2), list(v3)],
        shaded=False,
        mode="compute_height",
        show_missing_placeholder=True,
        placeholder_text="?",
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_mode_compute_base():
    v0, v1, v2, v3 = (1.0, 6.0), (4.0, 6.0), (0.0, 0.0), (7.0, 0.0)
    data = TrapezoidDecomposition(
        units="in",
        vertices=[list(v0), list(v1), list(v2), list(v3)],
        shaded=True,
        mode="compute_base",
        show_missing_placeholder=True,
        placeholder_text="b?",
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_mode_compute_area():
    v0, v1, v2, v3 = (2.0, 6.0), (6.0, 6.0), (0.0, 0.0), (9.0, 0.0)
    data = TrapezoidDecomposition(
        units="m",
        vertices=[list(v0), list(v1), list(v2), list(v3)],
        shaded=False,
        mode="compute_area",
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_trapezoid_mode_compute_height_with_altitude():
    # compute_height mode but not a left-vertical-side case; altitude should still be drawn
    v0, v1, v2, v3 = (0.0, 5.0), (2.0, 5.0), (0.0, 0.0), (5.0, 0.0)
    data = TrapezoidDecomposition(
        units="cm",
        vertices=[list(v0), list(v1), list(v2), list(v3)],
        shaded=False,
        mode="compute_height",
        show_missing_placeholder=True,
        placeholder_text="?",
    )
    path = create_trapezoid_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None
