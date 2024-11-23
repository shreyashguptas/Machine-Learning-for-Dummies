#!/bin/bash

wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

export PATH="$PWD/bin:$PATH"
export MAMBA_ROOT_PREFIX="$PWD/micromamba"

# Initialize Micromamba shell
./bin/micromamba shell init -s bash --no-modify-profile -p $MAMBA_ROOT_PREFIX

# Source Micromamba environment directly
eval "$(./bin/micromamba shell hook -s bash)"

# Activate the Micromamba environment
micromamba create -n jupyterenv python=3.11 -c conda-forge -y
micromamba activate jupyterenv

# install the dependencies
python -m pip install -r requirements-book.txt

# build the book
jupyter-book build . --builder html 