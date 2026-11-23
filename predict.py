"""Predict recorded WESAD windows using the notebook preprocessing contract."""

import sys

from stress_detection.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
