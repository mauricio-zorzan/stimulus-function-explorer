import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_prisms import (
    FillState,
    RectangularPrism,
    RectangularPrismList,
)
from pydantic import ValidationError

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


def test_validate_empty_prisms():
    # Create a list of RectangularPrism instances with fill state EMPTY
    prisms = [
        RectangularPrism.model_construct(
            title="Prism 1", height=2, width=2, length=2, fill=FillState.EMPTY
        ),
        RectangularPrism.model_construct(
            title="Prism 2", height=3, width=3, length=3, fill=FillState.EMPTY
        ),
    ]
    prism_list = RectangularPrismList(prisms)

    # Create a QuestionGeneratorContext with the specific substandard_id
    pipeline_context = MagicMock(
        payload=MagicMock(standard_id="CCSS.MATH.CONTENT.8.G.C.9+2"), question_json={}
    )

    # Validate and expect no assertion error
    prism_list.pipeline_validate(pipeline_context)


def test_validate_non_empty_prisms():
    # Create a list of RectangularPrism instances with different fill states
    prisms = [
        RectangularPrism.model_construct(
            title="Prism 1", height=2, width=2, length=2, fill=FillState.EMPTY
        ),
        RectangularPrism.model_construct(
            title="Prism 2", height=3, width=3, length=3, fill=FillState.FULL
        ),
    ]

    # Create a UnifiedPipelineValidationContext with the specific substandard_id
    pipeline_context: "QuestionGeneratorContext" = MagicMock(
        payload=MagicMock(standard_id="CCSS.MATH.CONTENT.8.G.C.9+2"), question_json={}
    )

    # Validate and expect a validation error
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "Not all figures have the same fill state. Expected FillState.EMPTY, but found FillState.FULL."
        ),
    ):
        prism_list = RectangularPrismList(prisms)
        prism_list.pipeline_validate(pipeline_context)
