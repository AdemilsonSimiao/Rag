import sys
from typing import NoReturn

def fail(message: str) -> NoReturn:
    """Print an error message and exit with status 1, without a traceback."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)
