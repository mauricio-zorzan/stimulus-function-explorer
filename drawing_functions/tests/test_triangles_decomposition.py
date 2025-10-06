import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.triangles_decomposition import (
    TriangleDecomposition,
    TriangleMode,
    create_triangle_decomposition_decimal_only,
)
from matplotlib import pyplot as plt


@pytest.mark.drawing_functions
def test_example_equilateral_draw_only():
    data = TriangleDecomposition(
        title="",
        units="cm",
        vertices=[[0.0, 0.0], [6.0, 0.0], [3.0, 5.2]],
        gridlines=False,
        shaded=False,
        mode=TriangleMode.DRAW_ONLY,
        show_missing_placeholder=False,
        placeholder_text="?",
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_example_isosceles_compute_area():
    data = TriangleDecomposition(
        title="",
        units="in",
        vertices=[[0.0, 0.0], [8.0, 0.0], [4.0, 5.0]],
        gridlines=False,
        shaded=True,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
        placeholder_text="?",
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_obtuse_external_height_visuals():
    """Obtuse triangle where altitude foot is outside the base segment.
    Ensures dashed altitude and right-angle marker render without error."""
    data = TriangleDecomposition(
        title="Triangle Area",
        units="m",
        vertices=[[0.0, 0.0], [4.6, 0.0], [4.2, 3.2]],
        gridlines=False,
        shaded=True,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_example_scalene_compute_height():
    data = TriangleDecomposition(
        title="",
        units="ft",
        vertices=[[0.0, 0.0], [5.5, 0.0], [2.0, 3.2]],
        gridlines=False,
        shaded=False,
        mode=TriangleMode.COMPUTE_HEIGHT,
        show_missing_placeholder=True,
        placeholder_text="?",
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_example_isosceles_compute_base():
    data = TriangleDecomposition(
        title="",
        units="m",
        vertices=[[0.0, 0.0], [7.5, 0.0], [3.75, 4.0]],
        gridlines=False,
        shaded=False,
        mode=TriangleMode.COMPUTE_BASE,
        show_missing_placeholder=True,
        placeholder_text="b?",
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_internal_dashed_height_marker():
    """Acute triangle with internal altitude; verify dashed height/marker draw."""
    data = TriangleDecomposition(
        title="Triangle",
        units="cm",
        vertices=[[0.0, 0.0], [14.0, 0.0], [3.0, 9.5]],
        gridlines=False,
        shaded=True,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_examples_batch():
    examples = [
        {
            "title": "",
            "units": "cm",
            "vertices": [[0.0, 0.0], [6.0, 0.0], [3.0, 5.2]],
            "gridlines": False,
            "shaded": False,
            "mode": TriangleMode.DRAW_ONLY,
            "show_missing_placeholder": False,
            "placeholder_text": "?",
        },
        {
            "title": "",
            "units": "in",
            "vertices": [[0.0, 0.0], [8.0, 0.0], [4.0, 5.0]],
            "gridlines": False,
            "shaded": True,
            "mode": TriangleMode.COMPUTE_AREA,
            "show_missing_placeholder": False,
            "placeholder_text": "?",
        },
        {
            "title": "",
            "units": "ft",
            "vertices": [[0.0, 0.0], [5.5, 0.0], [2.0, 3.2]],
            "gridlines": False,
            "shaded": False,
            "mode": TriangleMode.COMPUTE_HEIGHT,
            "show_missing_placeholder": True,
            "placeholder_text": "?",
        },
        {
            "title": "",
            "units": "m",
            "vertices": [[0.0, 0.0], [7.5, 0.0], [3.75, 4.0]],
            "gridlines": False,
            "shaded": False,
            "mode": TriangleMode.COMPUTE_BASE,
            "show_missing_placeholder": True,
            "placeholder_text": "b?",
        },
    ]

    for ex in examples:
        data = TriangleDecomposition(
            title=ex["title"],
            units=ex["units"],
            vertices=ex["vertices"],
            gridlines=ex["gridlines"],
            shaded=ex["shaded"],
            mode=ex["mode"],
            show_missing_placeholder=ex["show_missing_placeholder"],
            placeholder_text=ex["placeholder_text"],
        )
        path = create_triangle_decomposition_decimal_only(data)
        assert os.path.exists(path)
        assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_vertices_bottom_base_left_apex():
    data = TriangleDecomposition(
        title="",
        units="cm",
        vertices=[[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
        gridlines=False,
        shaded=False,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_vertices_external_altitude_right():
    data = TriangleDecomposition(
        title="",
        units="cm",
        vertices=[[0.0, 0.0], [3.0, 0.0], [5.0, 4.0]],
        gridlines=False,
        shaded=True,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None


@pytest.mark.drawing_functions
def test_right_triangle_marker_only_right_side_left_square():
    """Right triangle where apex x equals the higher base x; marker drawn left of the altitude.
    No dashed vertical altitude should be drawn (visual smoke test)."""
    data = TriangleDecomposition(
        title="",
        units="cm",
        vertices=[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
        gridlines=False,
        shaded=False,
        mode=TriangleMode.COMPUTE_AREA,
        show_missing_placeholder=False,
    )
    path = create_triangle_decomposition_decimal_only(data)
    assert os.path.exists(path)
    assert plt.imread(path) is not None
