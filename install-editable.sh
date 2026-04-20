#!/bin/bash

source .venv/bin/activate
uv pip install pip # later DeepEP will require pip
uv pip install -r requirements/build.txt
echo "build.txt done"
uv pip install -r requirements/common.txt
echo "common.txt done"

uv pip install -r requirements/cuda.txt
echo "cuda.txt done"

uv pip install -r requirements/lint.txt
echo "lint.txt done"

uv pip install -r requirements/kv_connectors.txt
echo "kv_connectors.txt done"

uv pip install -e . --no-build-isolation

cmake --preset release
