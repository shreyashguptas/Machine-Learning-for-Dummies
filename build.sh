#!/bin/bash

python -m pip install --upgrade pip
pip install -r requirements-book.txt
jupyter-book build .
# Add this line to check the contents of the build directory
ls -R _build/html