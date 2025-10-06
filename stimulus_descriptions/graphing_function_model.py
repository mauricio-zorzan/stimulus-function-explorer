from typing import Literal, Optional

from pydantic import Field, model_validator

from .stimulus_description import StimulusDescription


class GraphingFunction(StimulusDescription):
    function_type: Literal[
        "linear",
        "quadratic",
        "exponential",
        "cubic",
        "square_root",
        "rational",
        "circle",
        "sideways_parabola",
        "hyperbola",
        "ellipse",
    ] = Field(..., description="The type of mathematical relation to be graphed")

    a: float = Field(..., description="Primary coefficient 'a' in the relation")
    b: Optional[float] = Field(
        None, description="Secondary coefficient 'b' in the relation"
    )
    c: Optional[float] = Field(
        None, description="Tertiary coefficient 'c' in the relation"
    )
    d: Optional[float] = Field(
        None, description="Quaternary coefficient 'd' in the relation"
    )

    # Additional parameters for specific relation types
    radius: Optional[float] = Field(
        None, description="Radius parameter for circular relations"
    )
    x_radius: Optional[float] = Field(
        None, description="Horizontal radius for elliptical/hyperbolic relations"
    )
    y_radius: Optional[float] = Field(
        None, description="Vertical radius for elliptical/hyperbolic relations"
    )

    @model_validator(mode="after")
    def validate_required_coefficients(self):
        """Validate that required coefficients are provided for each relation type."""
        if self.function_type == "quadratic" and self.c is None:
            raise ValueError("Coefficient 'c' is required for quadratic relations")

        if self.function_type == "cubic" and self.d is None:
            raise ValueError("Coefficient 'd' is required for cubic relations")

        if self.function_type == "circle" and self.radius is None:
            raise ValueError("Radius parameter is required for circular relations")

        if self.function_type in ["hyperbola", "ellipse"]:
            if self.x_radius is None or self.y_radius is None:
                raise ValueError(
                    f"Both x_radius and y_radius are required for {self.function_type} relations"
                )

        return self

    @model_validator(mode="after")
    def validate_unused_coefficients(self):
        """Validate that unused coefficients are not provided for specific relation types."""
        simple_types = ["linear", "exponential", "rational"]
        if self.function_type in simple_types and self.c is not None:
            raise ValueError(
                f"Coefficient 'c' should not be provided for {self.function_type} relations"
            )

        non_cubic_types = [
            "linear",
            "quadratic",
            "exponential",
            "square_root",
            "rational",
            "sideways_parabola",
        ]
        if self.function_type in non_cubic_types and self.d is not None:
            raise ValueError(
                f"Coefficient 'd' should not be provided for {self.function_type} relations"
            )

        geometric_types = ["circle", "hyperbola", "ellipse"]
        if self.function_type not in geometric_types:
            if self.radius is not None:
                raise ValueError(
                    f"Radius parameter should not be provided for {self.function_type} relations"
                )
            if self.x_radius is not None or self.y_radius is not None:
                raise ValueError(
                    f"Radius parameters should not be provided for {self.function_type} relations"
                )

        return self

    @model_validator(mode="after")
    def validate_coefficient_values(self):
        """Validate coefficient ranges to ensure reasonable graphical output."""
        standard_coeffs = ["a", "b", "c", "d"]
        for coeff in standard_coeffs:
            value = getattr(self, coeff, None)
            if value is not None and abs(value) >= 20:
                raise ValueError(
                    f"Coefficient '{coeff}' must be less than 20 in absolute value"
                )

        # Special validation for geometric parameters
        if self.radius is not None and (self.radius <= 0 or self.radius >= 15):
            raise ValueError("Radius must be positive and less than 15")

        for param in ["x_radius", "y_radius"]:
            value = getattr(self, param, None)
            if value is not None and (value <= 0 or value >= 15):
                raise ValueError(f"{param} must be positive and less than 15")

        return self

    @model_validator(mode="after")
    def validate_function_constraints(self):
        """Validate constraints specific to mathematical relation types."""
        # Rational functions: coefficient 'a' cannot be zero
        if self.function_type == "rational" and self.a == 0:
            raise ValueError("Coefficient 'a' cannot be zero for rational relations")

        # Square root functions: ensure real domain
        if self.function_type == "square_root" and self.b is not None and self.b > 10:
            raise ValueError(
                "Coefficient 'b' for square root relations should not exceed domain limits"
            )

        # Parabolic relations: coefficient 'a' cannot be zero
        if (
            self.function_type in ["quadratic", "sideways_parabola", "cubic"]
            and self.a == 0
        ):
            raise ValueError(
                f"Primary coefficient 'a' cannot be zero for {self.function_type} relations"
            )

        return self


class GraphingFunctionQuadrantOne(StimulusDescription):
    function_type: Literal[
        "linear", "quadratic", "exponential", "cubic", "square_root", "rational"
    ] = Field(
        ...,
        description="The type of function to be graphed in quadrant I (positive domain only)",
    )

    a: float = Field(..., description="Primary coefficient 'a' in the function")
    b: Optional[float] = Field(
        None, description="Secondary coefficient 'b' in the function"
    )
    c: Optional[float] = Field(
        None, description="Tertiary coefficient 'c' in the function"
    )
    d: Optional[float] = Field(
        None, description="Quaternary coefficient 'd' in the function"
    )

    @model_validator(mode="after")
    def validate_required_coefficients(self):
        """Validate that required coefficients are provided for each function type."""
        if self.function_type == "quadratic" and self.c is None:
            raise ValueError("Coefficient 'c' is required for quadratic functions")

        if self.function_type == "cubic" and self.d is None:
            raise ValueError("Coefficient 'd' is required for cubic functions")

        return self

    @model_validator(mode="after")
    def validate_unused_coefficients(self):
        """Validate that unused coefficients are not provided."""
        simple_types = ["linear", "exponential", "rational"]
        if self.function_type in simple_types and self.c is not None:
            raise ValueError(
                f"Coefficient 'c' should not be provided for {self.function_type} functions"
            )

        non_cubic_types = [
            "linear",
            "quadratic",
            "exponential",
            "square_root",
            "rational",
        ]
        if self.function_type in non_cubic_types and self.d is not None:
            raise ValueError(
                f"Coefficient 'd' should not be provided for {self.function_type} functions"
            )

        return self

    @model_validator(mode="after")
    def validate_coefficient_values(self):
        """Validate coefficient ranges for reasonable output."""
        for coeff in ["a", "b", "c", "d"]:
            value = getattr(self, coeff, None)
            if value is not None and abs(value) >= 15:
                raise ValueError(
                    f"Coefficient '{coeff}' must be less than 15 in absolute value for quadrant I"
                )
        return self

    @model_validator(mode="after")
    def validate_quadrant_one_constraints(self):
        """Validate that the function stays in quadrant I for x in [0, 10]."""
        import numpy as np

        # Test points in the range [0, 10]
        x_test = np.linspace(0.1, 10, 100)  # Start from 0.1 to avoid division by zero

        try:
            if self.function_type == "linear":
                y_test = self.a * x_test + (self.b or 0)
            elif self.function_type == "quadratic":
                y_test = self.a * x_test**2 + (self.b or 0) * x_test + (self.c or 0)
            elif self.function_type == "exponential":
                y_test = self.a * np.exp((self.b or 0) * x_test)
            elif self.function_type == "cubic":
                y_test = (
                    self.a * x_test**3
                    + (self.b or 0) * x_test**2
                    + (self.c or 0) * x_test
                    + (self.d or 0)
                )
            elif self.function_type == "square_root":
                domain_shift = self.b or 0
                if np.any(x_test + domain_shift < 0):
                    raise ValueError(
                        "Square root function has invalid domain in quadrant I"
                    )
                y_test = self.a * np.sqrt(x_test + domain_shift) + (self.c or 0)
            elif self.function_type == "rational":
                y_test = self.a / x_test + (self.b or 0)
            else:
                return self

            # Check if any y values are negative in the test range
            if np.any(y_test < 0):
                raise ValueError(
                    "The function produces negative y values in quadrant I. "
                    "Adjust coefficients to ensure y ≥ 0 for x in [0, 10]"
                )
        except (ZeroDivisionError, ValueError) as e:
            if "negative y values" in str(e):
                raise e
            raise ValueError(
                f"Function has mathematical issues in quadrant I: {str(e)}"
            )

        return self
