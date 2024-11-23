#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Current directory: $(pwd)"
echo "Listing contents:"
ls -la

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements-book.txt

# Build the Jupyter Book
jupyter-book build .

# List contents of _build/html to verify output
echo "Contents of _build/html:"
ls -R _build/html

# Move the contents of _build/html to the root directory
mv _build/html/* .

echo "Final contents of root directory:"
ls -la