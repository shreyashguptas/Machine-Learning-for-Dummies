#!/bin/bash

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements-book.txt

# Build the Jupyter Book
jupyter-book build .

# List contents of _build/html to verify output
echo "Contents of _build/html:"
ls -R _build/html

# Create a simple index.html in the output directory
echo "<meta http-equiv=\"refresh\" content=\"0; url=./_build/html/index.html\">" > index.html