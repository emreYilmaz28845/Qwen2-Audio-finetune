#!/usr/bin/env python3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.split_validation import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
