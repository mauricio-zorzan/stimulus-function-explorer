import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.object_array import (
    draw_object_array,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.common.drawable_items import (
    DrawableItem,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.object_array import (
    ObjectArray,
)


@pytest.mark.drawing_functions
def test_draw_object_array_2x2():
    object_array = ObjectArray(
        object_name=DrawableItem.CAT,
        rows=2,
        columns=2,
    )
    file_name = draw_object_array(object_array)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_object_array_2x3():
    object_array = ObjectArray(
        object_name=DrawableItem.BUTTERFLY,
        rows=2,
        columns=3,
    )
    file_name = draw_object_array(object_array)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_object_array_5x3():
    object_array = ObjectArray(
        object_name=DrawableItem.DOG,
        rows=5,
        columns=3,
    )
    file_name = draw_object_array(object_array)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_object_array_5x7():
    object_array = ObjectArray(
        object_name=DrawableItem.FISH,
        rows=5,
        columns=7,
    )
    file_name = draw_object_array(object_array)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_object_array_7x7():
    object_array = ObjectArray(
        object_name=DrawableItem.SUN,
        rows=7,
        columns=7,
    )
    file_name = draw_object_array(object_array)
    assert os.path.exists(file_name)
