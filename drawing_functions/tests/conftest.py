import os
import sys

from content_generators.settings import settings

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    ),
)


def pytest_configure(config):
    print(
        f"Before patching: {settings.additional_content_settings.image_destination_folder}"
    )
    content_tests_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "..",
            "content",
            "tests",
        )
    )
    settings.additional_content_settings.image_destination_folder = (
        content_tests_directory
    )

    print(
        f"After patching: {settings.additional_content_settings.image_destination_folder}"
    )
