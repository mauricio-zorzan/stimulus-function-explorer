import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_prisms import (
    FillState,
    RectangularPrism,
    RectangularPrismList,
)


def test_check_all_figures_same_fill_success():
    prisms = [
        RectangularPrism.model_construct(
            title="Prism 1", height=2, width=2, length=2, fill=FillState.FULL
        ),
        RectangularPrism.model_construct(
            title="Prism 2", height=3, width=3, length=3, fill=FillState.FULL
        ),
    ]
    prism_list = RectangularPrismList(prisms)
    prism_list.check_all_figures_same_fill()  # type: ignore


def test_check_all_figures_same_fill_failure():
    prisms = [
        RectangularPrism.model_construct(
            title="Prism 1", height=2, width=2, length=2, fill=FillState.FULL
        ),
        RectangularPrism.model_construct(
            title="Prism 2", height=3, width=3, length=3, fill=FillState.EMPTY
        ),
    ]
    with pytest.raises(ValueError, match="Not all figures have the same fill state."):
        RectangularPrismList(prisms)


def test_check_all_figures_same_fill_empty_list():
    with pytest.raises(ValueError, match="No figures found in the list."):
        RectangularPrismList([])


if __name__ == "__main__":
    pytest.main()
