#!/bin/bash
# Clean all Python cache files and directories

echo "Cleaning Python cache files..."

# Delete all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Delete all .pyc files (compiled Python)
echo "Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null

# Delete all .pyo files (optimized Python)
echo "Removing .pyo files..."
find . -type f -name "*.pyo" -delete 2>/dev/null

# Delete all .pytest_cache directories
echo "Removing .pytest_cache directories..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Delete all .mypy_cache directories
echo "Removing .mypy_cache directories..."
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

echo "✅ Python cache cleanup complete!"
