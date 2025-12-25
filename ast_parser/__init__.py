# This file makes the 'ast_parser' directory a Python package.
import sys
import os

# Add the directory containing this __init__.py to the Python path
# This ensures that modules within this package can import each other.
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 