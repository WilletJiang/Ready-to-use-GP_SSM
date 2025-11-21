import sys
from pathlib import Path

# Ensure the repository's src/ directory is importable without installation
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
