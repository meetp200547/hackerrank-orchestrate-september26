"""
Evaluation entry point. Runs the project evaluation suite.
"""

import sys
from pathlib import Path

# Add project root and code dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import main as run_eval

if __name__ == "__main__":
    run_eval()
