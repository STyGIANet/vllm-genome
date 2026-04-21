#!/bin/bash

if [[ ! -d .venv/ ]];then
	uv venv .venv
fi

source .venv/bin/activate
uv pip install pip # later DeepEP will require pip
uv pip install -r requirements/build.txt
echo "build.txt done"
uv pip install -r requirements/common.txt
echo "common.txt done"

uv pip install -r requirements/cuda.txt
echo "cuda.txt done"

uv pip install -r requirements/cuda-torch.txt
echo "cuda-torch.txt done"

uv pip install -r requirements/lint.txt
echo "lint.txt done"

uv pip install -r requirements/kv_connectors.txt
echo "kv_connectors.txt done"

VLLM_USE_PRECOMPILED=1 uv pip install -e .

uv pip uninstall nvidia-nvshmem-cu12 nvidia-nvshmem-cu13
uv pip install nvidia-nvshmem-cu13==3.6.5
uv pip install packaging
uv pip install pymetis

cd ./DeepEP-SM8x
./install-deepep.sh