from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any


class FunctionDefinition(BaseModel):
    """
    Data model describing a callable function and its signature.
    This model defines:
    - the function name,
    - the names and types of its arguments,
    - and its return type.
    It is used to validate function metadata loaded from configuration files.
    """
    fn_name: str = Field(..., min_length=1, description="Function name")
    args_names: list[str] = Field(
        default_factory=list, description="Argument names")
    args_types: dict[str, str] = Field(
        default_factory=dict, description="Argument types")
    return_type: str = Field(..., min_length=1, description="Return type")

    @field_validator("fn_name")
    @classmethod
    def validate_fn_name(cls, v: str) -> str:
        """
        Validate that the function name starts with the prefix 'fn_'.
        Args:
            v: The function name to validate.
        Returns:
            The validated function name.
        Raises:
            ValueError: If the function name does not start with 'fn_'.
        """
        if not v.startswith("fn_"):
            raise ValueError(f"Function name must start with 'fn_': {v}")
        return v

    @model_validator(mode="after")
    def validate_args_consistency(self) -> "FunctionDefinition":
        """
        Validate consistency between argument names and argument types.
        Ensures that every argument listed in ``args_names`` has a
        corresponding entry in ``args_types``.
        Returns:
            The validated FunctionDefinition instance.
        Raises:
            ValueError: If any argument name has no matching type.
        """
        for arg_name in self.args_names:
            if arg_name not in self.args_types:
                raise ValueError(f"Missing type for argument: {arg_name}")
        return self


class PromptInput(BaseModel):
    """
    Data model representing a user input prompt.
    This model wraps a raw user prompt string and ensures that it is not
    empty or composed only of whitespace.
    """
    prompt: str = Field(..., min_length=1, description="User prompt")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """
        Validate that the prompt is not empty or whitespace-only.
        Args:
            v: The prompt string to validate.
        Returns:
            The cleaned prompt string.
        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()


class FunctionCallResult(BaseModel):
    """
    Data model representing the result of a function call decision.
    It contains:
    - the original user prompt,
    - the selected function name,
    - and the extracted arguments for that function.
    """
    prompt: str = Field(..., description="Original prompt")
    fn_name: str = Field(..., description="Selected function name")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Function arguments")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """
        Validate that the stored prompt is not empty.
        Args:
            v: The prompt string to validate.
        Returns:
            The validated prompt string.
        Raises:
            ValueError: If the prompt is empty.
        """
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator("fn_name")
    @classmethod
    def validate_fn_name(cls, v: str) -> str:
        """
        Validate that the function name starts with the prefix 'fn_'.
        Args:
            v: The function name to validate.
        Returns:
            The validated function name.
        Raises:
            ValueError: If the function name does not start with 'fn_'.
        """
        if not v.startswith("fn_"):
            raise ValueError(f"Function name must start with 'fn_': {v}")
        return v
