import sys
from pathlib import Path

# Make the helper module importable as `_legacy_source` from every test file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
