import json
from pathlib import Path
from typing import Any


def read_file(file: str) -> list[dict[str, Any]]:
    """
    Read a JSON file and return a validated list of dictionaries.
    The file must contain a JSON array. Each element in the array is processed
    as follows:
    - If the element is a dictionary, it is kept as-is.
    - If the element is a string, it is wrapped into a dictionary
      with the key "prompt".
    - Other element types are ignored with a warning.
    If the file does not exist, is not a file, or contains invalid JSON,
    an empty list is returned and an error message is printed.
    Args:
        file: Path to the JSON file.
    Returns:
        A list of dictionaries extracted and validated from the JSON file.
        Returns an empty list if any error occurs or no valid items are found.
    """
    path = Path(file)
    if not path.exists():
        print(f"[Warning]: {file} not found, returning empty list")
        return []
    if not path.is_file():
        print(f"[Error]: {file} is not a file, returning empty list")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[Error]: {file} must contain a JSON array, got "
                  f"{type(data).__name__}")
            return []
        validated_data: list[dict[str, Any]] = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                validated_data.append(item)
            elif isinstance(item, str):
                validated_data.append({"prompt": item})
            else:
                print(f"[Warning]: Skipping invalid item {i} in {file}: "
                      f"{type(item).__name__}")
        if not validated_data and data:
            print(f"[Warning]: No valid items found in {file}")
        return validated_data
    except json.JSONDecodeError as e:
        print(f"[Error]: Invalid JSON in {file} at line {e.lineno}, "
              f"column {e.colno}: {e.msg}")
        return []
    except UnicodeDecodeError as e:
        print(f"[Error]: Encoding error in {file}: {e}")
        return []
    except OSError as e:
        print(f"[Error]: OS error reading {file}: {e}")
        return []
    except Exception as e:
        print(f"[Error]: Unexpected error reading {file}: {e}")
        return []
