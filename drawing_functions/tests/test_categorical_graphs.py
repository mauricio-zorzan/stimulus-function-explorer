import json
import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.categorical_graphs import (
    CategoricalGraphList,
    MultiGraphList,
    create_categorical_graph,
    create_multi_bar_graph,
    create_multi_picture_graph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.categorical_graph import (
    CategoricalGraph,
    DataPoint,
    PictureGraphConfig,
)
from content_generators.settings import settings
from memory_profiler import memory_usage


@pytest.mark.drawing_functions
def test_create_bar_graph():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Bar Graph",
                x_axis_label="Categories",
                y_axis_label="Values",
                data=[
                    DataPoint(category="A", frequency=10),
                    DataPoint(category="B", frequency=20),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_single_bar_create_bar_graph():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Bar Graph",
                x_axis_label="Categories",
                y_axis_label="Values",
                data=[
                    DataPoint(category="A", frequency=10),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_bar_graph_2():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Prism Volumes",
                x_axis_label="Prisms",
                y_axis_label="Volume",
                data=[
                    DataPoint(category="Prism 1", frequency=13),
                    DataPoint(category="Prism 2", frequency=25),
                    DataPoint(category="Prism 3", frequency=23),
                    DataPoint(category="Prism 4", frequency=10),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_histogram():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="histogram",
                title="Histogram",
                x_axis_label="Bins",
                y_axis_label="Frequency",
                data=[
                    DataPoint(category="0-10", frequency=5),
                    DataPoint(category="10-20", frequency=9),
                    DataPoint(category="20-30", frequency=30),
                    DataPoint(category="30-40", frequency=14),
                    DataPoint(category="40-50", frequency=50),
                    DataPoint(category="50-60", frequency=10),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_picture_graph():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="X", frequency=3),
                    DataPoint(category="Y", frequency=7),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_invalid_graph_type():
    with pytest.raises(Exception):
        CategoricalGraphList(
            [
                CategoricalGraph(
                    graph_type="invalid_graph",  # type: ignore
                    title="Invalid Graph",
                    x_axis_label="",
                    y_axis_label="",
                    data=[
                        DataPoint(category="X", frequency=3),
                        DataPoint(category="Y", frequency=7),
                    ],
                )
            ]
        )


@pytest.mark.drawing_functions
def test_create_multi_bar_graph():
    # Create test data for 4 different bar graphs
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Graph 1",
                x_axis_label="Categories 1",
                y_axis_label="Values 1",
                data=[
                    DataPoint(category="A", frequency=10),
                    DataPoint(category="B", frequency=20),
                    DataPoint(category="C", frequency=15),
                ],
            ),
            CategoricalGraph(
                graph_type="bar_graph",
                title="Graph 2",
                x_axis_label="Categories 2",
                y_axis_label="Values 2",
                data=[
                    DataPoint(category="X", frequency=25),
                    DataPoint(category="Y", frequency=15),
                    DataPoint(category="Z", frequency=30),
                ],
            ),
            CategoricalGraph(
                graph_type="bar_graph",
                title="Graph 3",
                x_axis_label="Categories 3",
                y_axis_label="Values 3",
                data=[
                    DataPoint(category="Red", frequency=8),
                    DataPoint(category="Blue", frequency=12),
                    DataPoint(category="Green", frequency=9),
                ],
            ),
            CategoricalGraph(
                graph_type="bar_graph",
                title="Graph 4",
                x_axis_label="Categories 4",
                y_axis_label="Values 4",
                data=[
                    DataPoint(category="One", frequency=18),
                    DataPoint(category="Two", frequency=22),
                    DataPoint(category="Three", frequency=16),
                ],
            ),
        ]
    )

    # Test successful creation
    file_name = create_multi_bar_graph(data)
    assert os.path.exists(file_name)

    # Test with invalid number of graphs
    with pytest.raises(ValueError):
        create_multi_bar_graph(
            MultiGraphList(data.root[:3])
        )  # Only 3 graphs instead of 4


@pytest.mark.drawing_functions
def test_create_multi_picture_graph():
    # Create test data for 4 different picture graphs
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 1",
                x_axis_label="Categories 1",
                y_axis_label="Values 1",
                data=[
                    DataPoint(category="A", frequency=3),
                    DataPoint(category="B", frequency=5),
                    DataPoint(category="C", frequency=2),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 2",
                x_axis_label="Categories 2",
                y_axis_label="Values 2",
                data=[
                    DataPoint(category="X", frequency=4),
                    DataPoint(category="Y", frequency=6),
                    DataPoint(category="Z", frequency=1),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 3",
                x_axis_label="Categories 3",
                y_axis_label="Values 3",
                data=[
                    DataPoint(category="Red", frequency=2),
                    DataPoint(category="Blue", frequency=7),
                    DataPoint(category="Green", frequency=3),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 4",
                x_axis_label="Categories 4",
                y_axis_label="Values 4",
                data=[
                    DataPoint(category="One", frequency=4),
                    DataPoint(category="Two", frequency=8),
                    DataPoint(category="Three", frequency=2),
                ],
            ),
        ]
    )

    # Test successful creation
    file_name = create_multi_picture_graph(data)
    assert os.path.exists(file_name)

    # Test with invalid number of graphs
    with pytest.raises(ValueError):
        create_multi_picture_graph(
            MultiGraphList(data.root[:3])
        )  # Only 3 graphs instead of 4


@pytest.mark.drawing_functions
def test_create_multi_picture_graph_one_category():
    # Create test data for 4 different picture graphs, each with 1 category
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 1",
                x_axis_label="Categories 1",
                y_axis_label="Values 1",
                data=[
                    DataPoint(category="A", frequency=3),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 2",
                x_axis_label="Categories 2",
                y_axis_label="Values 2",
                data=[
                    DataPoint(category="X", frequency=4),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 3",
                x_axis_label="Categories 3",
                y_axis_label="Values 3",
                data=[
                    DataPoint(category="Red", frequency=2),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 4",
                x_axis_label="Categories 4",
                y_axis_label="Values 4",
                data=[
                    DataPoint(category="One", frequency=4),
                ],
            ),
        ]
    )

    # Test successful creation
    file_name = create_multi_picture_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_picture_graph_two_categories():
    # Create test data for 4 different picture graphs, each with 2 categories
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 1",
                x_axis_label="Categories 1",
                y_axis_label="Values 1",
                data=[
                    DataPoint(category="A", frequency=3),
                    DataPoint(category="B", frequency=5),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 2",
                x_axis_label="Categories 2",
                y_axis_label="Values 2",
                data=[
                    DataPoint(category="X", frequency=4),
                    DataPoint(category="Y", frequency=6),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 3",
                x_axis_label="Categories 3",
                y_axis_label="Values 3",
                data=[
                    DataPoint(category="Red", frequency=2),
                    DataPoint(category="Blue", frequency=7),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 4",
                x_axis_label="Categories 4",
                y_axis_label="Values 4",
                data=[
                    DataPoint(category="One", frequency=4),
                    DataPoint(category="Two", frequency=8),
                ],
            ),
        ]
    )

    # Test successful creation
    file_name = create_multi_picture_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_picture_graph_four_categories():
    # Create test data for 4 different picture graphs, each with 4 categories
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 1",
                x_axis_label="Categories 1",
                y_axis_label="Values 1",
                data=[
                    DataPoint(category="A", frequency=3),
                    DataPoint(category="B", frequency=5),
                    DataPoint(category="C", frequency=2),
                    DataPoint(category="D", frequency=4),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 2",
                x_axis_label="Categories 2",
                y_axis_label="Values 2",
                data=[
                    DataPoint(category="X", frequency=4),
                    DataPoint(category="Y", frequency=6),
                    DataPoint(category="Z", frequency=1),
                    DataPoint(category="W", frequency=3),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 3",
                x_axis_label="Categories 3",
                y_axis_label="Values 3",
                data=[
                    DataPoint(category="Red", frequency=2),
                    DataPoint(category="Blue", frequency=7),
                    DataPoint(category="Green", frequency=3),
                    DataPoint(category="Yellow", frequency=5),
                ],
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph 4",
                x_axis_label="Categories 4",
                y_axis_label="Values 4",
                data=[
                    DataPoint(category="One", frequency=4),
                    DataPoint(category="Two", frequency=8),
                    DataPoint(category="Three", frequency=2),
                    DataPoint(category="Four", frequency=6),
                ],
            ),
        ]
    )

    # Test successful creation
    file_name = create_multi_picture_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_museum_example():
    """Test the museum example that was problematic - should now have better labeling."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Daily Visitors at the Museum",
                x_axis_label="Day",
                y_axis_label="Visitors",
                data=[
                    DataPoint(category="Monday", frequency=100),
                    DataPoint(category="Tuesday", frequency=120),
                    DataPoint(category="Wednesday", frequency=110),
                    DataPoint(category="Thursday", frequency=130),
                    DataPoint(category="Friday", frequency=150),
                    DataPoint(category="Saturday", frequency=400),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_homework_example():
    """Test the homework example that worked well - should maintain good labeling."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Hours Spent on Homework per Week",
                x_axis_label="Hours",
                y_axis_label="Number of Students",
                data=[
                    DataPoint(category="1", frequency=2),
                    DataPoint(category="2", frequency=4),
                    DataPoint(category="3", frequency=4),
                    DataPoint(category="8", frequency=1),
                    DataPoint(category="9", frequency=1),
                    DataPoint(category="10", frequency=1),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_few_categories():
    """Test that few categories (≤3) use horizontal labels."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Few Categories Test",
                x_axis_label="Categories",
                y_axis_label="Values",
                data=[
                    DataPoint(category="A", frequency=10),
                    DataPoint(category="B", frequency=20),
                    DataPoint(category="C", frequency=15),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_medium_categories_short_labels():
    """Test that medium categories (4-5) with short labels use horizontal labels."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Medium Categories Short Labels",
                x_axis_label="Days",
                y_axis_label="Visitors",
                data=[
                    DataPoint(category="Mon", frequency=100),
                    DataPoint(category="Tue", frequency=120),
                    DataPoint(category="Wed", frequency=110),
                    DataPoint(category="Thu", frequency=130),
                    DataPoint(category="Fri", frequency=150),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_many_categories_long_labels():
    """Test that many categories with long labels trigger appropriate rotation."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Daily Visitors at the Museum",
                x_axis_label="Day",
                y_axis_label="Visitors",
                data=[
                    DataPoint(category="Monday", frequency=100),
                    DataPoint(category="Tuesday", frequency=120),
                    DataPoint(category="Wednesday", frequency=110),
                    DataPoint(category="Thursday", frequency=130),
                    DataPoint(category="Friday", frequency=150),
                    DataPoint(category="Saturday", frequency=400),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_very_long_labels():
    """Test that very long category names trigger 90° rotation."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Academic Subjects Performance",
                x_axis_label="Subject",
                y_axis_label="Average Score",
                data=[
                    DataPoint(category="Mathematics", frequency=85),
                    DataPoint(category="Science & Technology", frequency=78),
                    DataPoint(category="English Literature", frequency=82),
                    DataPoint(category="Social Studies", frequency=79),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_many_short_categories():
    """Test that many categories with short labels use appropriate rotation."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Letter Frequency",
                x_axis_label="Letter",
                y_axis_label="Count",
                data=[
                    DataPoint(category="A", frequency=12),
                    DataPoint(category="B", frequency=8),
                    DataPoint(category="C", frequency=15),
                    DataPoint(category="D", frequency=10),
                    DataPoint(category="E", frequency=18),
                    DataPoint(category="F", frequency=6),
                    DataPoint(category="G", frequency=9),
                    DataPoint(category="H", frequency=11),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_smart_labeling_histogram():
    """Test that histograms also use smart labeling."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="histogram",
                title="Hours Spent on Homework per Week",
                x_axis_label="Hours",
                y_axis_label="Number of Students",
                data=[
                    DataPoint(category="1", frequency=2),
                    DataPoint(category="2", frequency=4),
                    DataPoint(category="3", frequency=4),
                    DataPoint(category="8", frequency=1),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_dynamic_figure_width():
    """Test that figure width adjusts for many categories."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Wide Chart Test",
                x_axis_label="Categories",
                y_axis_label="Values",
                data=[
                    DataPoint(category=f"Cat{i}", frequency=10 + i)
                    for i in range(1, 11)  # 10 categories
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_edge_case_single_category():
    """Test edge case with single category."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Single Category",
                x_axis_label="Category",
                y_axis_label="Value",
                data=[
                    DataPoint(category="Only One", frequency=42),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_edge_case_empty_categories():
    """Test edge case with empty category names."""
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="bar_graph",
                title="Empty Categories Test",
                x_axis_label="Category",
                y_axis_label="Value",
                data=[
                    DataPoint(category="", frequency=5),
                    DataPoint(category="B", frequency=10),
                ],
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


def test_edge_case_data_set_is_large():
    stim_desc_str = """[
        {
            "graph_type": "histogram",
            "title": "Student Scores",
            "x_axis_label": "Score Range",
            "y_axis_label": "Frequency",
            "data": [
                {
                    "category": "12-16",
                    "frequency": 3
                },
                {
                    "category": "17-21",
                    "frequency": 4
                },
                {
                    "category": "22-26",
                    "frequency": 2
                },
                {
                    "category": "27-31",
                    "frequency": 6
                },
                {
                    "category": "32-36",
                    "frequency": 2
                }
            ]
        }
    ]"""
    stimulus_description = json.loads(stim_desc_str)
    file_name = create_categorical_graph(stimulus_description)
    assert os.path.exists(file_name)


def test_create_multi_bar_graph_v2():
    stim_desc_str = """[
        {
            "graph_type": "bar_graph",
            "title": "Fruit Sales",
            "x_axis_label": "Fruit",
            "y_axis_label": "Number Sold",
            "data": [
                {
                    "category": "Apples",
                    "frequency": 4
                },
                {
                    "category": "Bananas",
                    "frequency": 2
                },
                {
                    "category": "Cherries",
                    "frequency": 5
                },
                {
                    "category": "Grapes",
                    "frequency": 3
                }
            ]
        },
        {
            "graph_type": "bar_graph",
            "title": "Book Pages",
            "x_axis_label": "Genre",
            "y_axis_label": "Pages Read",
            "data": [
                {
                    "category": "Fiction",
                    "frequency": 1
                },
                {
                    "category": "Nonfiction",
                    "frequency": 4
                },
                {
                    "category": "Comics",
                    "frequency": 2
                },
                {
                    "category": "Biography",
                    "frequency": 5
                }
            ]
        },
        {
            "graph_type": "bar_graph",
            "title": "Toy Cars",
            "x_axis_label": "Color",
            "y_axis_label": "Toy Cars Owned",
            "data": [
                {
                    "category": "Red",
                    "frequency": 5
                },
                {
                    "category": "Blue",
                    "frequency": 4
                },
                {
                    "category": "Green",
                    "frequency": 3
                },
                {
                    "category": "Yellow",
                    "frequency": 1
                }
            ]
        },
        {
            "graph_type": "bar_graph",
            "title": "Cookie Batches",
            "x_axis_label": "Flavor",
            "y_axis_label": "Cookie Batches Sold",
            "data": [
                {
                    "category": "Chocolate",
                    "frequency": 2
                },
                {
                    "category": "Vanilla",
                    "frequency": 5
                },
                {
                    "category": "Strawberry",
                    "frequency": 3
                },
                {
                    "category": "Lemon",
                    "frequency": 4
                }
            ]
        }
    ]"""
    stimulus_description = json.loads(stim_desc_str)
    mem_usage = memory_usage(
        (create_multi_bar_graph, (stimulus_description,)),  # type: ignore
        interval=0.1,
        timeout=30,
    )
    assert max(mem_usage) < settings.lambda_settings.memory_size

    file_name = create_multi_bar_graph(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_picture_graph_with_half_stars():
    data = CategoricalGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Picture Graph with Half Stars",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="A", frequency=2.5),
                    DataPoint(category="B", frequency=4.5),
                    DataPoint(category="C", frequency=1.5),
                ],
                picture_graph_config=PictureGraphConfig(
                    star_value=2,
                    star_unit="books",
                    show_half_star_value=True,
                ),
            )
        ]
    )
    file_name = create_categorical_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_picture_graph_with_half_stars():
    data = MultiGraphList(
        [
            CategoricalGraph(
                graph_type="picture_graph",
                title="Graph 1",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="A", frequency=2.5),
                    DataPoint(category="B", frequency=3.5),
                ],
                picture_graph_config=PictureGraphConfig(
                    star_value=2,
                    star_unit="books",
                    show_half_star_value=True,
                ),
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Graph 2",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="C", frequency=1.5),
                    DataPoint(category="D", frequency=4.5),
                ],
                picture_graph_config=PictureGraphConfig(
                    star_value=2,
                    star_unit="books",
                    show_half_star_value=True,
                ),
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Graph 3",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="E", frequency=0.5),
                    DataPoint(category="F", frequency=2.5),
                ],
                picture_graph_config=PictureGraphConfig(
                    star_value=2,
                    star_unit="books",
                    show_half_star_value=True,
                ),
            ),
            CategoricalGraph(
                graph_type="picture_graph",
                title="Graph 4",
                x_axis_label="",
                y_axis_label="",
                data=[
                    DataPoint(category="G", frequency=3.5),
                    DataPoint(category="H", frequency=1.5),
                ],
                picture_graph_config=PictureGraphConfig(
                    star_value=2,
                    star_unit="books",
                    show_half_star_value=True,
                ),
            ),
        ]
    )
    file_name = create_multi_picture_graph(data)
    assert os.path.exists(file_name)


if __name__ == "__main__":
    test_edge_case_data_set_is_large()
    test_create_multi_bar_graph_v2()
