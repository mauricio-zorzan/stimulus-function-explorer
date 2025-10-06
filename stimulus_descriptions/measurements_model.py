import logging
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class Measurements(StimulusDescription):
    """A set of measurement images.
    All measurements support values up to 1000:
    - Liters: 0-100
    - Milliliters: 0-1000
    - Grams: 0-1000
    - Kilograms: 0-1000
    """

    measurement: float = Field(..., description="The measurement value.")
    units: Literal["milliliters", "liters", "grams", "kilograms"] = Field(
        ..., description="The units of measurement."
    )
    color: Optional[
        Literal["red", "lightblue", "green", "yellow", "orange", "purple"]
    ] = Field(None, description="The color associated with the measurement, if any.")

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.units == "liters" and self.measurement > 100:
            raise ValueError("Measurement must be less than 100 when units are liters.")
        if self.units == "grams" and self.measurement > 1000:
            raise ValueError("Measurement must be less than 1000 when units are grams.")
        if self.units == "kilograms" and self.measurement > 1000:
            raise ValueError(
                "Measurement must be less than 1000 when units are kilograms."
            )
        
        # Validate decimal portions
        if self.measurement % 1 != 0:  # If it has decimal places
            decimal_part = round(self.measurement % 1, 2)  # Round to avoid floating point precision issues
            allowed_decimals = [0.25, 0.5, 0.75]
            if decimal_part not in allowed_decimals:
                raise ValueError(
                    f"Measurement decimal portion must be one of 0.25, 0.5, or 0.75. Got {decimal_part}"
                )
        
        return self

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        logging.info(self)
        super().pipeline_validate(pipeline_context)
        if pipeline_context.standard_id in {
            "CCSS.MATH.CONTENT.3.MD.A.2+2",
        }:
            self.check_units(["grams", "kilograms"])

    def check_units(self, valid_units: list) -> None:
        if self.units not in valid_units:
            raise ValueError(
                f"Invalid unit: {self.units}. Must be one of {valid_units}."
            )

    def validate_not_on_minor_tick(self):
        measurement_ml = (
            self.measurement * 1000 if self.units == "liters" else self.measurement
        )

        if measurement_ml <= 100:
            max_scale, major_tick_interval = 100, 20
        elif measurement_ml <= 500:
            max_scale, major_tick_interval = 500, 100
        elif measurement_ml <= 1000:
            max_scale, major_tick_interval = 1000, 100
        else:
            max_scale, major_tick_interval = 5000, 1000

        num_sub_ticks = 1
        for i in range(0, max_scale, major_tick_interval):
            for j in range(1, num_sub_ticks + 2):
                minor_tick = i + j * major_tick_interval / (num_sub_ticks + 1)
                if (
                    abs(measurement_ml - minor_tick) < 1e-6
                ):  # Using a small epsilon for float comparison
                    raise ValueError(
                        f"Measurement {self.measurement} {self.units} falls on a minor tick marker."
                    )

        return self
