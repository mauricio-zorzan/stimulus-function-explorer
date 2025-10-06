import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.categorical_graphs import (
    draw_tree_diagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.tree_diagram import (
    TreeDiagram,
    TreeDiagramNode,
)


def make_example_tree():
    # Matches the user's image: root -> L/M/N/O -> H/T
    return TreeDiagram(
        title="Example Tree Diagram",
        root=TreeDiagramNode(
            label="H",
            left=TreeDiagramNode(
                label="L",
                left=TreeDiagramNode(label="H", left=None, right=None),
                right=TreeDiagramNode(label="T", left=None, right=None),
            ),
            right=TreeDiagramNode(
                label="M",
                left=TreeDiagramNode(label="H", left=None, right=None),
                right=TreeDiagramNode(label="T", left=None, right=None),
            ),
        ),
    )


def make_deep_tree():
    # A tree of depth 4
    return TreeDiagram(
        title="Deep Tree",
        root=TreeDiagramNode(
            label="A",
            left=TreeDiagramNode(
                label="B",
                left=TreeDiagramNode(label="D", left=None, right=None),
                right=TreeDiagramNode(label="E", left=None, right=None),
            ),
            right=TreeDiagramNode(
                label="C",
                left=TreeDiagramNode(label="F", left=None, right=None),
                right=TreeDiagramNode(
                    label="G",
                    left=TreeDiagramNode(label="H", left=None, right=None),
                    right=TreeDiagramNode(label="I", left=None, right=None),
                ),
            ),
        ),
    )


def make_left_skewed_tree():
    # Only left children
    return TreeDiagram(
        title="Left Skewed",
        root=TreeDiagramNode(
            label="A",
            left=TreeDiagramNode(
                label="B",
                left=TreeDiagramNode(
                    label="C",
                    left=TreeDiagramNode(label="D", left=None, right=None),
                    right=None,
                ),
                right=None,
            ),
            right=None,
        ),
    )


def make_right_skewed_tree():
    # Only right children
    return TreeDiagram(
        title="Right Skewed",
        root=TreeDiagramNode(
            label="A",
            left=None,
            right=TreeDiagramNode(
                label="B",
                left=None,
                right=TreeDiagramNode(
                    label="C",
                    left=None,
                    right=TreeDiagramNode(label="D", left=None, right=None),
                ),
            ),
        ),
    )


def make_empty_label_tree():
    # All nodes have empty labels
    return TreeDiagram(
        title="Empty Labels",
        root=TreeDiagramNode(
            label="",
            left=TreeDiagramNode(
                label="",
                left=TreeDiagramNode(label="", left=None, right=None),
                right=TreeDiagramNode(label="", left=None, right=None),
            ),
            right=TreeDiagramNode(
                label="",
                left=TreeDiagramNode(label="", left=None, right=None),
                right=TreeDiagramNode(label="", left=None, right=None),
            ),
        ),
    )


@pytest.mark.drawing_functions
def test_draw_tree_diagram():
    tree = make_example_tree()
    file_path = draw_tree_diagram(tree)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_deep_tree_diagram():
    tree = make_deep_tree()
    file_path = draw_tree_diagram(tree)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_left_skewed_tree_diagram():
    tree = make_left_skewed_tree()
    file_path = draw_tree_diagram(tree)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_right_skewed_tree_diagram():
    tree = make_right_skewed_tree()
    file_path = draw_tree_diagram(tree)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_empty_label_tree_diagram():
    tree = make_empty_label_tree()
    file_path = draw_tree_diagram(tree)
    assert os.path.exists(file_path)
