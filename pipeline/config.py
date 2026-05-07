"""
Step 6: Config file support
Reads meloscribe.toml from the project root and returns default values
for all CLI flags. CLI arguments always override config values.
Uses stdlib tomllib (Python 3.11+) with no new dependencies.
"""

import sys
from pathlib import Path
from typing import Any, Dict


def load_config(path: str = 'meloscribe.toml') -> Dict[str, Any]:
    """
    Load configuration from a TOML file.

    Reads the [meloscribe] table if present. Returns an empty dict if the
    file does not exist or contains no [meloscribe] section — callers should
    always treat every key as optional.

    Args:
        path: Path to the TOML config file (default: 'meloscribe.toml')

    Returns:
        Flat dict of config values from the [meloscribe] table.
        Empty dict if file is absent or table is missing.

    Raises:
        ValueError: If the file exists but contains invalid TOML syntax.
    """
    config_path = Path(path)

    if not config_path.exists():
        return {}

    try:
        if sys.version_info >= (3, 11):
            import tomllib
            with open(config_path, 'rb') as f:
                data = tomllib.load(f)
        else:
            try:
                import tomli as tomllib  # pip install tomli on 3.10 and below
                with open(config_path, 'rb') as f:
                    data = tomllib.load(f)
            except ImportError:
                # tomli not available — skip config silently rather than crash
                return {}

    except Exception as e:
        raise ValueError(f"Invalid TOML in {config_path}: {e}") from e

    return data.get('meloscribe', {})
