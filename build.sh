#!/bin/bash

set -ex  # Exit immediately if a command exits with a non-zero status and print commands

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

# Create a simple index.html that redirects to the Jupyter Book's index
cat << EOF > index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0;url=./README.html">
    <title>Redirecting...</title>
</head>
<body>
    <p>If you are not redirected automatically, follow this <a href="./README.html">link to the documentation</a>.</p>
</body>
</html>
EOF

echo "Created index.html:"
cat index.html