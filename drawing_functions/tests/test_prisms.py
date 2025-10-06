import os

import matplotlib.pyplot as plt
import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.rectangular_prisms import (
    draw_multiple_base_area_rectangular_prisms,
    draw_multiple_rectangular_prisms,
    draw_unit_cube_figure,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_prisms import (
    CustomCubeShape,
    Point3d,
    RectangularPrismShape,
    UnitCubeFigure,
)


@pytest.mark.drawing_functions
def test_plot_with_tall_prisms():
    stimulus_description = [
        {"title": "Prism A", "height": 9, "width": 3, "length": 6, "fill": "empty"},
        {"title": "Prism B", "height": 9, "width": 4, "length": 5, "fill": "empty"},
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_single_prism():
    stimulus_description = [
        {"title": "Prism A", "height": 9, "width": 3, "length": 6, "fill": "empty"},
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_multiple_prisms():
    stimulus_description = [
        {"title": "Prism 1", "height": 2, "width": 2, "length": 3, "fill": "empty"},
        {"title": "Prism 2", "height": 5, "width": 5, "length": 1, "fill": "empty"},
        {"title": "Prism 3", "height": 3, "width": 4, "length": 2, "fill": "empty"},
        {"title": "Prism 4", "height": 2, "width": 3, "length": 5, "fill": "empty"},
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_multiple_prisms_full():
    stimulus_description = [
        {"title": "Prism 1", "height": 2, "width": 2, "length": 3, "fill": "full"},
        {"title": "Prism 2", "height": 5, "width": 5, "length": 1, "fill": "full"},
        {"title": "Prism 3", "height": 3, "width": 4, "length": 2, "fill": "full"},
        {"title": "Prism 4", "height": 2, "width": 3, "length": 5, "fill": "full"},
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_rectangular_prisms_medium_level():
    """Test drawing prisms with medium difficulty (dimensions < 20)."""
    stimulus_description = [
        {
            "title": "Medium Prism A",
            "height": 15,
            "width": 12,
            "length": 18,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
        {
            "title": "Medium Prism B",
            "height": 8,
            "width": 16,
            "length": 19,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    print(
        f"\nGenerated medium level prism image at: {filename}"
    )  # Added print statement
    assert os.path.exists(filename)
    assert plt.imread(filename) is not None  # Added image load verification


@pytest.mark.drawing_functions
def test_plot_rectangular_prisms_hard_level():
    """Test drawing prisms with hard difficulty (20 ≤ dimensions ≤ 40)."""
    stimulus_description = [
        {
            "title": "Hard Prism A",
            "height": 25,
            "width": 35,
            "length": 30,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
        {
            "title": "Hard Prism B",
            "height": 38,
            "width": 22,
            "length": 28,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_rectangular_prisms_mixed_difficulty():
    """Test drawing prisms with mixed difficulty levels."""
    stimulus_description = [
        {
            "title": "Medium Prism",
            "height": 15,
            "width": 12,
            "length": 18,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
        {
            "title": "Hard Prism",
            "height": 35,
            "width": 25,
            "length": 30,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_rectangular_prisms_edge_cases():
    """Test drawing prisms with edge case dimensions."""
    stimulus_description = [
        {
            "title": "Medium Edge",
            "height": 19,  # Just under 20
            "width": 19,
            "length": 19,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
        {
            "title": "Hard Edge",
            "height": 20,  # Just at 20
            "width": 40,  # Maximum allowed
            "length": 21,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_single_prism_with_label():
    stimulus_description = [
        {
            "title": "Prism A",
            "height": 6,
            "width": 4,
            "length": 7,
            "fill": "bottom",
            "prism_unit_label": "cm",
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_single_prism_with_label_and_unit_cube_unit_size_and_label():
    stimulus_description = [
        {
            "title": "Storage Box and Wooden Blocks",
            "height": 4,
            "width": 4,
            "length": 4,
            "fill": "empty",
            "prism_unit_label": "ft",
            "unit_cube_unit_size_and_label": "4 inch",
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_minimum_rectangular_prism():
    min_figure = UnitCubeFigure(
        title="Minimum Size Prism",
        shape=RectangularPrismShape(kind="rectangular", length=1, width=1, height=1),
    )
    filename = draw_unit_cube_figure(min_figure)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_maximum_rectangular_prism():
    max_figure = UnitCubeFigure(
        title="Maximum Size Prism",
        shape=RectangularPrismShape(kind="rectangular", length=10, width=10, height=10),
    )
    filename = draw_unit_cube_figure(max_figure)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_tall_rectangular_prism():
    tall_figure = UnitCubeFigure(
        title="Tall Prism",
        shape=RectangularPrismShape(kind="rectangular", length=2, width=3, height=5),
    )
    filename = draw_unit_cube_figure(tall_figure)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_single_cube():
    single_cube = UnitCubeFigure(
        title="Single Cube",
        shape=CustomCubeShape(kind="custom", cubes=[Point3d(x=0, y=0, z=0)]),
    )
    filename = draw_unit_cube_figure(single_cube)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_l_shape():
    l_shape = UnitCubeFigure(
        title="L Shape",
        shape=CustomCubeShape(
            kind="custom",
            cubes=[
                Point3d(x=0, y=0, z=0),
                Point3d(x=1, y=0, z=0),
                Point3d(x=2, y=0, z=0),  # Base row
                Point3d(x=0, y=0, z=1),
                Point3d(x=1, y=0, z=1),  # Middle row
                Point3d(x=0, y=0, z=2),  # Top row
            ],
        ),
    )
    filename = draw_unit_cube_figure(l_shape)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_staircase():
    staircase = UnitCubeFigure(
        title="Staircase",
        shape=CustomCubeShape(
            kind="custom",
            cubes=[
                Point3d(x=0, y=0, z=0),  # Base
                Point3d(x=0, y=0, z=1),
                Point3d(x=1, y=0, z=1),  # First step
                Point3d(x=0, y=0, z=2),
                Point3d(x=1, y=0, z=2),
                Point3d(x=2, y=0, z=2),  # Second step
                Point3d(x=0, y=0, z=3),
                Point3d(x=1, y=0, z=3),
                Point3d(x=2, y=0, z=3),
                Point3d(x=3, y=0, z=3),  # Third step
            ],
        ),
    )
    filename = draw_unit_cube_figure(staircase)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_cross():
    cross = UnitCubeFigure(
        title="Cross",
        shape=CustomCubeShape(
            kind="custom",
            cubes=[
                Point3d(x=0, y=0, z=1),
                Point3d(x=1, y=0, z=1),
                Point3d(x=2, y=0, z=1),  # top bar (3 cubes)
                Point3d(x=1, y=0, z=0),  # stem above
                Point3d(x=1, y=0, z=2),  # stem below
            ],
        ),
    )
    filename = draw_unit_cube_figure(cross)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_multiple_base_area_prisms():
    stimulus_description = [
        {
            "title": "Prism 1",
            "base_area": 6,
            "height": 2,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 2",
            "base_area": 25,
            "height": 2,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 3",
            "base_area": 12,
            "height": 2,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 4",
            "base_area": 15,
            "height": 2,
            "prism_unit_label": "ft",
        },
    ]
    filename = draw_multiple_base_area_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_base_area_prisms_different_heights():
    stimulus_description = [
        {
            "title": "Prism 1",
            "base_area": 6,
            "height": 3,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 2",
            "base_area": 25,
            "height": 5,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 3",
            "base_area": 12,
            "height": 4,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 4",
            "base_area": 15,
            "height": 2,
            "prism_unit_label": "ft",
        },
    ]
    with pytest.raises(ValueError, match="Not all figures have the same height"):
        draw_multiple_base_area_rectangular_prisms(stimulus_description)


@pytest.mark.drawing_functions
def test_plot_with_single_base_area_prism():
    stimulus_description = [
        {
            "title": "Prism A",
            "base_area": 24,
            "height": 5,
            "prism_unit_label": "ft",
        },
    ]
    filename = draw_multiple_base_area_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_with_base_area_prisms_large_area():
    stimulus_description = [
        {
            "title": "Prism 1",
            "base_area": 64,
            "height": 3,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 2",
            "base_area": 81,
            "height": 3,
            "prism_unit_label": "ft",
        },
        {
            "title": "Prism 3",
            "base_area": 100,
            "height": 3,
            "prism_unit_label": "ft",
        },
    ]
    filename = draw_multiple_base_area_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_hide_length_measurement():
    """Test hiding length measurement only."""
    stimulus_description = [
        {
            "title": "Prism A",
            "height": 3,
            "width": 4,
            "length": 5,
            "fill": "empty",
            "show_length": False,
            "show_width": True,
            "show_height": True,
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_hide_width_measurement():
    """Test hiding width measurement only."""
    stimulus_description = [
        {
            "title": "Prism B",
            "height": 2,
            "width": 3,
            "length": 4,
            "fill": "empty",
            "show_length": True,
            "show_width": False,
            "show_height": True,
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_rectangular_prism_hide_height_measurement():
    """Test hiding height measurement only."""
    stimulus_description = [
        {
            "title": "Prism C",
            "height": 6,
            "width": 2,
            "length": 3,
            "fill": "empty",
            "show_length": True,
            "show_width": True,
            "show_height": False,
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_multiple_prisms_different_measurement_visibility():
    """Test multiple prisms with different measurement visibility settings."""
    stimulus_description = [
        {
            "title": "Prism 1",
            "height": 2,
            "width": 3,
            "length": 4,
            "fill": "empty",
            "show_length": True,
            "show_width": True,
            "show_height": False,
        },
        {
            "title": "Prism 2",
            "height": 3,
            "width": 2,
            "length": 5,
            "fill": "empty",
            "show_length": False,
            "show_width": True,
            "show_height": True,
        },
        {
            "title": "Prism 3",
            "height": 4,
            "width": 4,
            "length": 2,
            "fill": "empty",
            "show_length": True,
            "show_width": True,
            "show_height": False,
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_base_area_prism_show_one_measurement_height():
    """Test showing one measurement."""
    stimulus_description = [
        {
            "title": "Base Prism A",
            "base_area": 12,
            "height": 4,
            "prism_unit_label": "ft",
            "show_base_area": False,
            "show_height": True,
        }
    ]
    filename = draw_multiple_base_area_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_base_area_prism_show_one_measurement():
    """Test showing one measurement."""
    stimulus_description = [
        {
            "title": "Base Prism A",
            "base_area": 12,
            "height": 4,
            "prism_unit_label": "ft",
            "show_base_area": True,
            "show_height": False,
        }
    ]
    filename = draw_multiple_base_area_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_plot_rectangular_prism_hard_level_max():
    """Test drawing a single prism with maximum allowed dimensions (40)."""
    stimulus_description = [
        {
            "title": "Maximum Size Prism",
            "height": 40,
            "width": 40,
            "length": 40,
            "fill": "empty",
            "prism_unit_label": "cm",
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    print(
        f"\nGenerated hard level (maximum size) prism image at: {filename}"
    )  # Added print statement
    assert os.path.exists(filename)
    assert plt.imread(filename) is not None  # Added image load verification


@pytest.mark.drawing_functions
def test_plot_rectangular_prism_hard_level_varied():
    """Test drawing a single prism with varied large dimensions (one at max, others > 20)."""
    stimulus_description = [
        {
            "title": "Varied Large Prism",
            "height": 40,  # Maximum height
            "width": 25,  # Medium-large width
            "length": 35,  # Large but not maximum length
            "fill": "empty",
            "prism_unit_label": "cm",
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    print(
        f"\nGenerated hard level (varied dimensions) prism image at: {filename}"
    )  # Added print statement
    assert os.path.exists(filename)
    assert plt.imread(filename) is not None  # Added image load verification
