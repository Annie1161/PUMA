"""Add the parent `src` directory to sys.path so that `import tools` resolves (the `tools` package sits next to the agent)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
_s = str(_SRC)
if _s not in sys.path:
    sys.path.insert(0, _s)
