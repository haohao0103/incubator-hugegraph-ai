import sys
from pathlib import Path

# Ensure src is on the path for all unit tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
