from pathlib import Path
from typing import Any
import json


def write_file(file: str, data: list[dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSON file.
    The target directory will be created automatically if it does not exist.
    Data is written using UTF-8 encoding with pretty-printed indentation.
    Args:
        file: Path to the output JSON file.
        data: A list of dictionaries to be written to the file.
    Raises:
        OSError: If the file cannot be written due to an OS-level error.
        TypeError: If the data contains objects that are not JSON serializable.
    """
    path = Path(file)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[Info]: Successfully wrote {len(data)} items to {file}")
    except OSError as e:
        print(f"[Error]: Cannot write to {file}, reason: {e}")
    except TypeError as e:
        print(f"[Error]: Data contains non-serializable objects, reason: {e}")
