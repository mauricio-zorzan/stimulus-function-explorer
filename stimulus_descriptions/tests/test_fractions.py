from unittest.mock import MagicMock

import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionPair,
    FractionPairList,
    FractionShape,
)


def test_check_all_fractions_are_valid():
    fractions = [
        FractionPair(shape=FractionShape.CIRCLE, fractions=["1/4", "2/4"]),
        FractionPair(shape=FractionShape.CIRCLE, fractions=["2/5", "1/5"]),
    ]
    FractionPairList(fractions)


def test_check_all_fractions_are_valid_fail():
    validation_context = MagicMock(
        standard_id="CCSS.MATH.CONTENT.4.NF.B.3.D+1",
    )
    with pytest.raises(
        ValueError, match="All fractions must share the same denominator."
    ):
        fractions = [
            FractionPair(shape=FractionShape.CIRCLE, fractions=["1/3", "2/4"]),
            FractionPair(shape=FractionShape.CIRCLE, fractions=["2/5", "1/5"]),
        ]
        fraction_pairs = FractionPairList(fractions)
        fraction_pairs.pipeline_validate(validation_context)


if __name__ == "__main__":
    pytest.main()
