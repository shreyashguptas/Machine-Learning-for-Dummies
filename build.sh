#!/bin/bash

# Build the book
jupyter-book build . --all

# Copy contents from _build/html to root directory
cp -r _build/html/* .

# Clean up build artifacts
rm -rf _build 