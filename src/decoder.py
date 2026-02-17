"""
Constrained decoder for validating and normalizing LLM function call results.
This module defines the ConstrainedDecoder class, which converts raw JSON
output from an LLM into a validated FunctionCallResult based on predefined
function definitions.
"""
from .schema import FunctionCallResult, FunctionDefinition
from pydantic import ValidationError
from typing import Any


class ConstrainedDecoder:
    """
    Decode and validate raw LLM outputs into structured function calls.
    This class ensures:
    - The selected function name exists in the provided definitions.
    - All required arguments are present.
    - Argument types are converted and validated.
    - Safe default values are used when conversion fails.
    """
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """
        Initialize the constrained decoder.
        Args:
            functions: A list of FunctionDefinition objects describing
                available functions.
        Raises:
            ValueError: If the functions list is empty.
            TypeError: If any item in the list is not a FunctionDefinition.
        """
        if not functions:
            raise ValueError("Functions list cannot be empty")
        for i, func in enumerate(functions):
            if not isinstance(func, FunctionDefinition):
                raise TypeError(
                    f"Function {i} must be FunctionDefinition instance, "
                    f"got {type(func)}")
        self.functions: dict[str, FunctionDefinition] = {
            f.fn_name: f for f in functions}

    def _get_default_value(self, arg_type: str) -> Any:
        """
        Return a default value for a given argument type.
        Args:
            arg_type: The argument type (e.g., "int", "float", "bool", "str").
        Returns:
            A default value corresponding to the given type.
        """
        if arg_type == "int":
            return 0
        elif arg_type == "float":
            return 0.0
        elif arg_type == "bool":
            return False
        else:
            return ""

    def build_fallback(self, prompt: str) -> FunctionCallResult:
        """
        Build a fallback function call result using the first available
        function and its default argument values.
        This method is used when no function matches the given prompt. It
        selects the first function from the registered functions and
        constructs a `FunctionCallResult` with default argument values.
        Args:
            prompt (str): The user input prompt that triggered the fallback.
        Returns:
            FunctionCallResult: An object containing the prompt, the selected
            function name, and a dictionary of default argument values.
        """
        fn = list(self.functions.values())[0]
        args = {
            name: self._get_default_value(fn.args_types[name])
            for name in fn.args_names}
        return FunctionCallResult(prompt=prompt, fn_name=fn.fn_name, args=args)

    def _convert_and_validate_arg(self, val: Any, arg_type: str, arg_name: str
                                  ) -> Any:
        """
        Convert and validate a single argument value.
        Args:
            val: The raw value to convert.
            arg_type: The expected argument type.
            arg_name: The argument name (used for error reporting).
        Returns:
            The converted and validated argument value.
        Raises:
            TypeError: If the argument type is unknown.
            ValueError: If conversion fails.
        """
        if val is None:
            return self._get_default_value(arg_type)
        if arg_type == "int":
            return int(float(val))
        elif arg_type == "float":
            return float(val)
        elif arg_type == "bool":
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)
        elif arg_type == "str":
            return str(val)
        else:
            raise TypeError(f"Unknown argument type: {arg_type}")

    def decode_prompt(self, prompt: str, raw_json: dict[str, Any]
                      ) -> FunctionCallResult:
        """
        Decode a raw LLM output into a validated FunctionCallResult.
        Args:
            prompt: The original user prompt.
            raw_json: Raw JSON output from the LLM, expected to contain
                "fn_name" and "args" fields.
        Returns:
            A validated FunctionCallResult instance.
        Raises:
            ValueError: If the prompt is empty or no functions are available.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not self.functions:
            raise ValueError("No function definitions available")
        fn_name = raw_json.get("fn_name", "")
        if not fn_name or fn_name not in self.functions:
            fn_name = list(self.functions.keys())[0]
            print(f"[Warning]: Invalid fn_name '{raw_json.get('fn_name')}', "
                  f"using fallback: {fn_name}")
        fn_def = self.functions[fn_name]
        args: dict[str, Any] = {}
        raw_args = raw_json.get("args", {})
        for arg_name in fn_def.args_names:
            val = raw_args.get(arg_name)
            arg_type = fn_def.args_types[arg_name]
            try:
                val = self._convert_and_validate_arg(val, arg_type, arg_name)
            except (ValueError, TypeError) as e:
                print(f"[Warning]: Failed to convert {arg_name}={val} "
                      f"to {arg_type}: {e}")
                val = self._get_default_value(arg_type)
            args[arg_name] = val
        try:
            result = FunctionCallResult(prompt=prompt.strip(), fn_name=fn_name,
                                        args=args)
            return result
        except ValidationError as e:
            print(f"[Error]: Validation failed: {e}")
            return self.build_fallback(prompt)
