import sys
from pathlib import Path

# scripts/evaluation/tests/conftest.py -> scripts/evaluation
EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
