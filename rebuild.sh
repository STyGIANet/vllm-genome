#!/bin/bash

mkdir -p build
mkdir -p cmake-build-release
cmake --build --preset release --target install
