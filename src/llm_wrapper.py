"""
LLM-based function and argument selector.
This module provides:
- Vocabulary normalization for BPE tokenizers
- A wrapper around a small LLM model
- Function selection and argument extraction using constrained decoding
"""
import json
import math
import re
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from llm_sdk import Small_LLM_Model


class VocabularyConfig(BaseModel):
    """
    Configuration for vocabulary normalization.
    Attributes:
        replacements: Mapping of special BPE tokens to human-readable strings.
    """
    replacements: dict[str, str] = Field(
        default_factory=lambda: {
            'Ġ': ' ',      # GPT-2/Qwen space
            'Ċ': '\n',     # Newline
            'ĉ': '\t',     # Tab
            '▁': ' ',      # SentencePiece space
            '</w>': '',    # Word boundary
            '<unk>': '',   # Unknown
            '<s>': '',     # Start
            '</s>': '',    # End
            '<pad>': '',   # Padding
            '<mask>': '',  # Mask
            },
        description="BPE token replacements")


class VocaNormalizer:
    """Vocabulary normalizer for BPE/SentencePiece tokens."""
    def __init__(self) -> None:
        """Initialize the vocabulary normalizer."""
        self.config = VocabularyConfig()

    def normalize_vocabulary(self, vocab: dict[str, int]) -> dict[str, int]:
        """
        Normalize special tokens in a vocabulary.
        Args:
            vocab: Original vocabulary mapping token -> token_id.
        Returns:
            A normalized vocabulary mapping token -> token_id.
        Raises:
            ValueError: If vocabulary is empty.
        """
        if not vocab:
            raise ValueError("Vocabulary cannot be empty")
        normalized: dict[str, int] = {}
        for token, token_id in vocab.items():
            normalized_token = token
            for special, replacement in self.config.replacements.items():
                normalized_token = normalized_token.replace(special,
                                                            replacement)
            normalized[normalized_token] = token_id
        return normalized


class LLMConfig(BaseModel):
    """
    Configuration for LLM wrapper.
    Attributes:
        model_name: Name of the underlying LLM model.
    """
    model_name: str = Field(default="Qwen/Qwen3-0.6B")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """
        Validate model name.
        Args:
            v: Model name string.
        Returns:
            Cleaned model name.
        Raises:
            ValueError: If model name is empty.
        """
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class LLMWrapper:
    """
        A wrapper around a small language model (LLM) that provides:
        1. Text encoding and decoding (string ↔ token IDs).
        2. Function selection using constrained decoding over predefined
        function schemas.
        3. Vocabulary loading and normalization for token ↔ id mapping.
        4. Caching of tokenized function schemas to avoid repeated computation.
        The wrapper converts each function schema into a JSON string and then
        into a token ID sequence. During function selection, generation is
        constrained so that only tokens consistent with one of the candidate
        function schemas can be produced. As soon as a unique function match
        is identified, decoding stops early.
        Attributes:
            config (LLMConfig): Model configuration object.
            model_name (str): Name of the underlying LLM model.
            model (Small_LLM_Model): Loaded LLM model instance.
            vocab (dict): Normalized vocabulary loaded from the model.
            token_to_id (dict[str, int]): Mapping from string to token ID.
            id_to_token (dict[int, str]): Mapping from token ID to string.
            visualize (bool): Whether to print intermediate decoding steps.
            _matrix_cache (dict):
                Cache mapping a tuple of function JSON strings to:
                (token_matrix, function_index_map), where:
                    - token_matrix is list[list[int]] of tokenized functions
                    - function_index_map maps function index → function schema
        """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 visualize: bool = False) -> None:
        """
        Initialize the LLM wrapper by loading the model and preparing its
        vocabulary and token mappings.
        This method:
        - Loads the specified LLM model.
        - Reads and normalizes the model's vocabulary file.
        - Builds bidirectional mappings between tokens and token IDs.
        - Initializes a cache for tokenized function schemas.
        Args:
            model_name (str):
                Name or path of the LLM model to load.
                Default is "Qwen/Qwen3-0.6B".
            visualize (bool):
                If True, enables printing of intermediate decoding steps
                during constrained generation (for debugging or analysis).
        """
        self.config = LLMConfig(model_name=model_name)
        self.model_name = self.config.model_name
        self.model = Small_LLM_Model(model_name=self.model_name)
        self._matrix_cache: dict[tuple[
            str, ...], tuple[list[list[int]], dict[int, dict[str, Any]]]] = {}
        self.visualize = visualize
        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        normalizer = VocaNormalizer()
        self.vocab = normalizer.normalize_vocabulary(raw_vocab)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        sample_key = list(self.vocab.keys())[0] if self.vocab else None
        try:
            int(str(sample_key))
            for k, v in self.vocab.items():
                try:
                    token_id = int(k)
                    token = str(v)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            for k, v in self.vocab.items():
                try:
                    token = str(k)
                    token_id = int(v)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                except (ValueError, TypeError):
                    continue
        print(f"[Info]: Loaded {self.model_name} with {len(self.token_to_id)} "
              "tokens")

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        Args:
            text: Input string.
        Returns:
            List of token IDs.
        """
        input_ids_tensor = self.model._encode(text)
        return input_ids_tensor[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back into text.
        Args:
            token_ids: List of token IDs.
        Returns:
            Decoded string.
        """
        return self.model._decode(token_ids)

    def _build_function_matrix(self, functions: list[dict[str, Any]]
                               ) -> tuple[list[list[int]],
                                          dict[int, dict[str, Any]]]:
        """
        Build and cache a tokenized representation of function schemas.
        This method converts each function schema (a Python dict) into a
        JSON string and then encodes it into a sequence of token IDs using the
        model tokenizer. The resulting token sequences are stored as a matrix,
        where each row corresponds to one function.
        To avoid repeated and expensive tokenization, the result is cached
        using the JSON-serialized function schemas as the cache key. If the
        same set of functions is passed again, the cached result is returned
        directly.
        Args:
            functions (list[dict[str, Any]]):
                A list of function schemas. Each schema is a dictionary
                a callable function (e.g., name, parameters, metadata).
        Returns:
            tuple:
                A tuple (token_matrix, function_index_map), where:
                - token_matrix (list[list[int]]):
                    A 2D list of token IDs. Each inner list represents the
                    tokenized JSON form of one function schema.
                - function_index_map (dict[int, dict[str, Any]]):
                    A mapping from function index (row index in token_matrix)
                    to the original function schema dictionary.
        Example:
            Input:
                functions = [
                    {"fn_name": "add", "params": {"a": "int", "b": "int"}},
                    {"fn_name": "delete", "params": {"id": "int"}}]
            Output:
                token_matrix = [
                    [12, 45, 98, 203, ...],   # tokens for "add" schema
                    [12, 47, 66, 150, ...]    # tokens for "delete" schema]
                function_index_map = {
                    0: {"fn_name": "add", "params": {"a": "int", "b": "int"}},
                    1: {"fn_name": "delete", "params": {"id": "int"}}}
        """
        key = tuple(json.dumps(f, sort_keys=True) for f in functions)
        if key in self._matrix_cache:
            return self._matrix_cache[key]
        matrix: list[list[int]] = []
        func_map: dict[int, dict[str, Any]] = {}
        for i, func in enumerate(functions):
            func_str = json.dumps(func, separators=(",", ":"))
            token_ids = self.encode(func_str)
            matrix.append(token_ids)
            func_map[i] = func
        self._matrix_cache[key] = (matrix, func_map)
        return matrix, func_map

    def _get_allowed_tokens(self, generated_ids: list[int],
                            matrix: list[list[int]]) -> list[int]:
        """
        Determine which token IDs are allowed to be generated next, based on
        prefix matching with candidate token sequences.
        This function enforces constrained decoding by only allowing tokens
        that keep the generated sequence consistent with at least one
        candidate function definition.
        Parameters:
            generated_ids (list[int]):
                The sequence of token IDs that has already been generated.
                This represents the current partial output of the model.
                Example: [11, 22]
            matrix (list[list[int]]):
                A list of token sequences, where each inner list represents
                one complete function definition encoded as token IDs.
                Example:
                    [
                        [11, 22, 33, 44],   # function A
                        [11, 22, 77, 88]    # function B
                    ]
        How it works:
            1. For each token sequence in `matrix`, check whether its first
            `len(generated_ids)` tokens are exactly the same as `generated_ids`
            (this is called prefix matching).
            2. If a sequence matches the current prefix, it is still a valid
            candidate.
            3. From all valid candidates, collect their next token
            (the token at position `len(generated_ids)`).
            4. These collected tokens form the set of allowed next tokens.
        Returns:
            list[int]:
                A list of token IDs that are valid choices for the next token.
                The model must choose the next token from this list in order
                to stay consistent with at least one candidate function
                definition.
        Example:
            matrix = [
                [11, 22, 33, 44],
                [11, 22, 77, 88]
            ]
            generated_ids = [11, 22]
            The return value will be:
                [33, 77]
            meaning the next token can be either 33 or 77.
        """
        allowed = set()
        prefix_len = len(generated_ids)
        for tokens in matrix:
            if len(tokens) > prefix_len:
                if tokens[:prefix_len] == generated_ids:
                    allowed.add(tokens[prefix_len])
        return list(allowed)

    def _check_unique_match(self, generated_ids: list[int],
                            matrix: list[list[int]],
                            func_map: dict[int, dict[str, Any]]
                            ) -> tuple[bool, dict[str, Any] | None]:
        """
        Check whether generated tokens uniquely identify one function.
        Stops as soon as only one function still matches the current prefix,
        without requiring the full token sequence to be generated. This is
        the key optimization enabling early termination.
        Args:
            generated_ids: Generated token sequence so far.
            matrix: Function token matrix.
            func_map: Function index map.
        Returns:
            Tuple of (is_unique, matched_function).
            is_unique is True when exactly one function still matches.
        """
        matches = [i for i, tokens in enumerate(matrix)
                   if tokens[:len(generated_ids)] == generated_ids]
        # if len(matches) == 1 and len(matrix[matches[0]]) == len(
        #     generated_ids):
        if len(matches) == 1:
            return True, func_map[matches[0]]
        return False, None

    def _visualize_step(self, generated_ids: list[int],
                        allowed_ids: list[int]) -> None:
        """
        Print the currently generated token sequence and the allowed next
        tokens. This method is used for visualizing the LLM generation process.
        It shows which tokens have been generated so far and which tokens are
        allowed to be generated next.
        Args:
            generated_ids (list[int]): List of token IDs that have been
            generated so far.
            allowed_ids (list[int]): List of token IDs that are allowed as
            the next token.
        """
        print("Generated:", self.decode(generated_ids))
        print("Allowed next tokens:", [self.decode([i]) for i in allowed_ids])

    def _select_function(self, prompt: str, functions: list[dict[str, Any]]
                         ) -> dict[str, Any]:
        """
        Select the most appropriate function from a list of function
        definitions based on a user input prompt, using constrained token
        decoding. This method uses the following approach:
        1. Convert all candidate functions into token sequences (matrix).
        2. Encode the user prompt and a selection instruction into tokens.
        3. Iteratively generate tokens step by step:
            - At each step, only tokens consistent with at least one candidate
            function (allowed_ids) are allowed.
            - The next token is chosen as the one with the highest model score
            (logit) among the allowed tokens.
            - After each token is generated, check whether the generated prefix
            uniquely matches one function. If so, return it immediately.
        4. If no unique match is found after a reasonable number of steps,
        return the first function as a fallback.
        Parameters:
            prompt (str):
                The user's input describing what they want to do.
            functions (list[dict[str, Any]]):
                A list of function schemas (JSON objects) to select from.
                Each schema typically contains:
                    - "fn_name": Name of the function
                    - "params": Parameter definitions
        Returns:
            dict[str, Any]:
                The selected function schema (one of the input `functions`).
        Notes:
            - The method uses "logits" (raw model scores) instead of
            probabilities, because we only need to find the token with the
            highest score among allowed tokens.
            - Constrained decoding ensures that the generated token sequence
            always follows one of the candidate function token sequences.
            - Early termination is applied: as soon as a prefix uniquely
            identifies a function, generation stops.
        Example:
            functions = [
                {"fn_name": "add", "params": {"a": "int", "b": "int"}},
                {"fn_name": "delete", "params": {"id": "int"}}
            ]
            prompt = "Please add two numbers"
            selected_func = _select_function(prompt, functions)
            # selected_func will be {"fn_name": "add", "params": {...}}
        """
        matrix, func_map = self._build_function_matrix(functions)
        func_names = [f["fn_name"] for f in functions]
        func_list = ", ".join(func_names)

        selection_prompt = (
            f"Available functions: {func_list}. "
            f"User request: {prompt}. "
            f"Select best function: ")

        input_ids = self.encode(selection_prompt)
        generated_ids: list[int] = []
        max_tokens = max(len(t) for t in matrix) + 20

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_ids = self._get_allowed_tokens(generated_ids, matrix)
            # self._visualize_step(generated_ids, allowed_ids)
            if not allowed_ids:
                break
            masked_logits = [-math.inf] * len(logits)
            for token_id in allowed_ids:
                if token_id < len(logits):
                    masked_logits[token_id] = float(logits[token_id])
            max_logit = max(masked_logits)
            if max_logit == -math.inf:
                break
            next_token_id = masked_logits.index(max_logit)
            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)
            is_unique, matched_func = self._check_unique_match(
                generated_ids, matrix, func_map)
            if is_unique and matched_func is not None:
                return matched_func
        print("[Warning]: No unique match found, using fallback")
        return functions[0]

    def _extract_candidates(self, prompt: str, fn_name: str, arg_name: str,
                            arg_type: str, used_values:
                            Optional[list[str]] = None) -> list[str]:
        """
        Extract candidate values for a specific argument from the user prompt.
        This function uses regular expressions and rules based on
        function name and argument name to find possible argument values
        mentioned in the prompt.
        Args:
            prompt: User input string.
            fn_name: Name of the selected function.
            arg_name: Name of the argument to extract.
            arg_type: Type of the argument (eg., "int", "float", "bool", "str")
            used_values: List of values already used by previous arguments
                to avoid duplication.
        Returns:
            A list of candidate strings for the argument. If no candidate is
            found, returns a list containing an empty string.
        """
        if used_values is None:
            used_values = []
        candidates = []
        prompt_lower = prompt.lower()

        if arg_type in ["int", "float"]:
            # -?: optional negative sign
            # \d+: one or more digits
            # \.?: optional decimal point
            # \d*: zero or more digits after decimal
            numbers = re.findall(r"-?\d+\.?\d*", prompt)
            return [n for n in numbers if n not in used_values]

        if arg_type == "bool":
            return ["True", "False"]

        if fn_name == "fn_substitute_string_with_regex":
            if arg_name == "source_string":
                # match patterns like "in 'string' with ..." or "with '...'
                # in 'string'"
                match1 = re.search(
                    r"in\s+(?:the\s+)?(?:string\s+)?(['\"])(.*?)\1\s+with",
                    prompt, re.IGNORECASE)
                match2 = re.search(
                    r"with\s+['\"][^'\"]+['\"]\s+in\s+(['\"])(.*?)\1",
                    prompt, re.IGNORECASE)
                if match1:
                    candidates.append(match1.group(2))
                elif match2:
                    candidates.append(match2.group(2))

            elif arg_name == "regex":
                if "digit" in prompt_lower or "number" in prompt_lower:
                    candidates.append(r"\d+")
                elif "vowel" in prompt_lower:
                    candidates.append(r"[aeiouAEIOU]")
                elif "consonant" in prompt_lower:
                    candidates.append(
                        r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")
                elif "space" in prompt_lower or "whitespace" in prompt_lower:
                    candidates.append(r"\s+")
                elif "word" in prompt_lower:
                    word_match = re.search(r"word\s+['\"](\w+)['\"]", prompt,
                                           re.IGNORECASE)
                    if word_match:
                        candidates.append(rf"\b{word_match.group(1)}\b")

            elif arg_name == "replacement":
                # match pattern like "with 'replacement'"
                match = re.search(r"with\s+['\"]([^'\"]+)['\"]", prompt,
                                  re.IGNORECASE)
                if match:
                    candidates.append(match.group(1))
                elif ("asterisk" in prompt_lower or "stars" in prompt_lower
                      or "asterisks" in prompt_lower):
                    candidates.append("*")
                elif "underscore" in prompt_lower:
                    candidates.append("_")
                elif "nothing" in prompt_lower or "empty" in prompt_lower:
                    candidates.append("")

        elif fn_name == "fn_concatenate_strings":
            # pattern 1: "concatenate 's1' and 's2'" / "join 's1' with 's2'"
            # pattern 2: "'s1' and 's2'" or "'s1' with 's2'"
            patterns = [
                r"(?:concatenate|join|combine)\s+['\"]([^'\"]+)['\"]\s+"
                r"(?:and|with)\s+['\"]([^'\"]+)['\"]",
                r"['\"]([^'\"]+)['\"]\s+(?:and|with)\s+['\"]([^'\"]+)['\"]"]
            for pattern in patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    if arg_name == "s1":
                        candidates.append(match.group(1))
                    elif arg_name == "s2":
                        candidates.append(match.group(2))
                    break
            # fallback: take the first two quoted strings from the prompt
            if not candidates:
                all_strings = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
                if arg_name == "s1" and len(all_strings) > 0:
                    candidates.append(all_strings[0])
                elif arg_name == "s2" and len(all_strings) > 1:
                    candidates.append(all_strings[1])

        elif fn_name in ["fn_reverse_string", "fn_to_uppercase",
                         "fn_to_lowercase", "fn_string_length"]:
            all_matches = []
            single_quoted = re.findall(r"'([^']*)'", prompt)
            all_matches.extend(single_quoted)
            double_quoted = re.findall(r'"([^"]*)"', prompt)
            all_matches.extend(double_quoted)
            if all_matches:
                longest = max(all_matches, key=len)
                if longest:
                    candidates.append(longest)

        elif fn_name == "fn_greet":
            match = re.search(r"greet\s+([\w\-]+)", prompt, re.IGNORECASE)
            if match:
                candidates.append(match.group(1))

        else:
            matches = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
            candidates.extend(matches)
        return candidates if candidates else [""]

    def _select_candidate(self, prompt: str, arg_name: str,
                          candidates: list[str]) -> str:
        """
        Select the most appropriate candidate value for a given argument
        using constrained decoding and the context from the user prompt.
        This function is used when multiple candidate values are extracted
        for a function argument. It leverages the LLM to determine which
        candidate best fits the context of the prompt, ensuring that the
        chosen value is both valid and contextually relevant.
        The selection process works as follows:
        1. If there are no candidates, return an empty string.
        2. If there is exactly one candidate, return it directly.
        3. Otherwise, encode each candidate into token IDs and construct
        a token matrix.
        4. Build a selection prompt that instructs the LLM to choose the
        best candidate given the context.
        5. Iteratively generate tokens using constrained decoding, where
        only tokens corresponding to valid candidates are allowed.
        6. Once a generated token sequence matches a candidate exactly,
        return that candidate.
        7. If no exact match is found after the maximum number of tokens,
        return the first candidate as a fallback.
        Args:
            prompt (str): The full user input string providing context.
            arg_name (str): The name of the argument for which a value
                            is being selected.
            candidates (list[str]): A list of possible candidate values
                                    for the argument.
        Returns:
            str: The selected candidate value as a string. If no candidates
                are provided, returns an empty string. If multiple candidates
                exist but no exact match is generated, returns the first
                candidate as a fallback.
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        matrix: list[list[int]] = []
        for candidate in candidates:
            tokens = self.encode(str(candidate))
            matrix.append(tokens)
        cands_str = ", ".join(f'"{c}"' for c in candidates[:5])
        selection_prompt = (
            f"Select {arg_name} from: {cands_str}. "
            f"Context: {prompt}. "
            f"Answer: ")

        input_ids = self.encode(selection_prompt)
        generated_ids: list[int] = []
        max_tokens = max(len(t) for t in matrix) + 5
        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_ids = self._get_allowed_tokens(generated_ids, matrix)
            if not allowed_ids:
                break
            masked_logits = [-math.inf] * len(logits)
            for token_id in allowed_ids:
                if token_id < len(logits):
                    masked_logits[token_id] = float(logits[token_id])
            max_logit = max(masked_logits)
            if max_logit == -math.inf:
                break
            next_token_id = masked_logits.index(max_logit)
            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)
            for i, tokens in enumerate(matrix):
                if tokens == generated_ids:
                    return candidates[i]
        return candidates[0]

    def _extract_arguments(self, prompt: str, func: dict[str, Any]
                           ) -> dict[str, Any]:
        """
        Extract and convert all argument values for a selected function from a
        user prompt.
        This method automates the process of retrieving values for each
        argument of a function based on the user's natural language input.
        1. Iterating over all argument names in the selected function schema.
        2. Using `_extract_candidates` to identify all potential values for
        each argument in the prompt, while avoiding duplicates from previously
        selected arguments.
        3. Selecting the most appropriate candidate using `_select_candidate`
        if multiple candidates are found.
        4. Converting the extracted value to the correct type specified in the
        function schema (int, float, bool, or string).
        5. Returning a dictionary mapping argument names to their final
        extracted values.
        Args:
            prompt (str): The user input text containing information about
            argument values.
            func (dict[str, Any]): The selected function schema, which must
            include:
                - "fn_name": Name of the function.
                - "args_names": List of argument names.
                - "args_types": Dictionary mapping argument names to their
                types ("int", "float", "bool", "str").
        Returns:
            dict[str, Any]: A dictionary where each key is an argument name
            from the function,and each value is the extracted and
            type-converted value. If no candidate is found for an argument,
            a default value is provided based on its type:
                - int → 0
                - float → 0.0
                - bool → False
                - str → ""
        Notes:
            - The order of argument processing matters because previously used
            values are tracked to avoid duplicates.
            - Candidate extraction relies on `_extract_candidates`, which may
            use regex patterns or rules specific to the function name and
            argument.
            - If multiple candidates exist for a parameter, `_select_candidate`
            uses context and constrained decoding to pick the best fit.
            - This function ensures that all arguments are ready for function
            invocation with correctly typed values.
        """
        args: dict[str, Any] = {}
        fn_name = func["fn_name"]
        used_values: list[str] = []

        for arg_name in func["args_names"]:
            arg_type = func["args_types"][arg_name]

            candidates = self._extract_candidates(
                prompt, fn_name, arg_name, arg_type, used_values)

            if len(candidates) == 1:
                value = candidates[0]
            elif len(candidates) > 1:
                value = self._select_candidate(prompt, arg_name, candidates)
            else:
                value = ""
            used_values.append(str(value))
            if arg_type == "int":
                try:
                    args[arg_name] = int(float(value)) if value else 0
                except (ValueError, TypeError):
                    args[arg_name] = 0
            elif arg_type == "float":
                try:
                    args[arg_name] = float(value) if value else 0.0
                except (ValueError, TypeError):
                    args[arg_name] = 0.0
            elif arg_type == "bool":
                args[arg_name] = str(value).lower() == "true"
            else:
                args[arg_name] = str(value)
        return args

    def call(self, prompt: str, functions: list[dict[str, Any]]
             ) -> dict[str, Any]:
        """
        Select a function and extract its arguments from a user prompt.
        1. Chooses the best matching function.
        2. Extracts and types its arguments.
        Args:
            prompt: User input string.
            functions: List of available function schemas.
        Returns:
            A dictionary with:
                - "fn_name": selected function name
                - "args": extracted argument dictionary
        """
        selected_func = self._select_function(prompt, functions)
        args = self._extract_arguments(prompt, selected_func)
        return {"fn_name": selected_func["fn_name"], "args": args}
