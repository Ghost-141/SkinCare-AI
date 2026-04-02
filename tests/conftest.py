import sys
import os

# Add the project root directory to the Python path
# This ensures that 'from main import app' works regardless of where pytest is run from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
