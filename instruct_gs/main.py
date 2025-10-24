#!/usr/bin/env python3
"""
InstructGS - Text-guided 3D Scene Editing with Gaussian Splatting
Main entry point for the application.
"""

import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli.interface import main

if __name__ == "__main__":
    main()