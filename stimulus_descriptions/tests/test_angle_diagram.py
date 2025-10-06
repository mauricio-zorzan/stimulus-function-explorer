from unittest.mock import MagicMock

from content_generators.additional_content.stimulus_image.stimulus_descriptions.angle_diagram import (
    AngleDiagram,
)


def test_check_all_fractions_are_valid_fail():
    validation_context = MagicMock()

    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 20, "points": ["A", "B", "C"]},
                {"measure": 70, "points": ["C", "B", "D"]},
                {"measure": 25, "points": ["D", "B", "E"]},
                {"measure": 65, "points": ["E", "B", "F"]},
                {"measure": 30, "points": ["F", "B", "G"]},
                {"measure": 60, "points": ["G", "B", "H"]},
                {"measure": 40, "points": ["H", "B", "I"]},
                {"measure": 50, "points": ["I", "B", "A"]},
            ]
        }
    }
    angle_diagram = AngleDiagram.model_validate(stimulus_description)
    angle_diagram.pipeline_validate(validation_context)
