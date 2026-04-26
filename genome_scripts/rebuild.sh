#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

mkdir -p build
mkdir -p cmake-build-release
cmake --build --preset release --target install
