#!/usr/bin/env python3
"""CLI entry point for ZipfActivation pipeline."""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zipf_activation.config import config_from_cli
from zipf_activation.pipeline import run_pipeline


def main():
    config = config_from_cli()
    run_pipeline(config)


if __name__ == "__main__":
    main()
