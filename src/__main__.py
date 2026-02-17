"""
Main entry point for the function calling system.
This module loads function definitions and test prompts from JSON files,
uses an LLM with constrained decoding to select the best function and
arguments for each prompt, and writes the results to an output file.
"""
from .reader import read_file
from .writer import write_file
from .llm_wrapper import LLMWrapper
from .decoder import ConstrainedDecoder
from .schema import FunctionDefinition, PromptInput, FunctionCallResult
from pydantic import ValidationError
import argparse
import time
from typing import Any


def main() -> None:
    """
    Run the function calling pipeline.
    This function:
    1. Parses command-line arguments.
    2. Loads and validates function definitions.
    3. Initializes the LLM and constrained decoder.
    4. Loads and validates user prompts.
    5. Processes each prompt using the LLM.
    6. Writes the function call results to a JSON file.
    Raises:
        SystemExit: If critical errors occur during initialization.
    """
    parser = argparse.ArgumentParser(description="Function calling system")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests"
                        ".json", help="Input file path")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results"
                        ".json", help="Output file path")
    parser.add_argument("--definitions", type=str,
                        default="data/input/functions_definition"
                        ".json", help="Path to function definitions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name")
    args = parser.parse_args()
    start_time = time.time()

    print(f"[Info]: Reading function definitions from {args.definitions}")
    func_data = read_file(args.definitions)
    if not func_data:
        print("[Error]: No function definitions found!")
        return
    try:
        functions = [FunctionDefinition(**f) for f in func_data]
        print(f"[Info]: Validated {len(functions)} function definitions")
    except ValidationError as e:
        print(f"[Error]: Invalid function definitions: {e}")
        return
    except Exception as e:
        print(f"[Error]: Failed to parse function definitions: {e}")
        return

    print("[Info]: Initializing LLM and decoder")
    decoder = ConstrainedDecoder(functions)
    try:
        llm = LLMWrapper(model_name=args.model)
    except Exception as e:
        print(f"[Error]: Failed to initialize LLM: {e}")
        print("[Info]: Falling back to default model")
        try:
            llm = LLMWrapper()
        except Exception as e2:
            print(f"[Error]: Failed to initialize default model: {e2}")
            return

    print(f"[Info]: Reading test prompts from {args.input}")
    test_data = read_file(args.input)
    if not test_data:
        print("[Warning]: No test prompts found!")
        return

    validated_prompts: list[str] = []
    for i, item in enumerate(test_data, 1):
        try:
            if isinstance(item, dict):
                validated = PromptInput(**item)
            elif isinstance(item, str):
                validated = PromptInput(prompt=item)
            else:
                print(f"[Warning]: Skipping invalid prompt format at {i}")
                continue
            validated_prompts.append(validated.prompt)
        except ValidationError as e:
            print(f"[Warning]: Skipping invalid prompt at {i}: {e}")
        except Exception as e:
            print(f"[Warning]: Error validating prompt at {i}: {e}")
    print(f"[Info]: Validated {len(validated_prompts)}/{len(test_data)} "
          "prompts")

    if not validated_prompts:
        print("[Error]: No valid prompts to process!")
        return
    results: list[dict[str, Any]] = []
    print(f"[Info]: Processing {len(validated_prompts)} prompts")

    for i, prompt in enumerate(validated_prompts, 1):
        item_start = time.time()
        print(f"[{i}/{len(validated_prompts)}] Processing: {prompt}")
        try:
            raw_json = llm.call(prompt, func_data)
            if not isinstance(raw_json, dict):
                raise ValueError("LLM output is not a dict")
            decoded: FunctionCallResult = decoder.decode_prompt(prompt,
                                                                raw_json)
            results.append(decoded.model_dump())
            item_time = time.time() - item_start
            print(f"  → {decoded.fn_name} | Time: {item_time:.2f}s")
        except Exception as e:
            print(f"[Error]: Failed to process prompt: {e}")
            fallback = decoder.build_fallback(prompt)
            results.append(fallback.model_dump())

    total_time = time.time() - start_time
    print(f"\n[Info]: Writing results to {args.output}")
    write_file(args.output, results)
    print("\n[Info]: Done!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average: {total_time/len(validated_prompts):.2f}s per prompt")
    print(f"  Success rate: {len(results)}/{len(validated_prompts)}")


if __name__ == "__main__":
    main()
