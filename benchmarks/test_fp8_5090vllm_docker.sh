#!/usr/bin/env bash

# === Required Env Vars ===
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# TP
# CONC

cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 4096
max-num-batched-tokens: 16
max-model-len: 4096
EOF

export PYTHONNOUSERSITE=1

set -x
vllm serve microsoft/Phi-3-mini-4k-instruct --host=0.0.0.0 --port=8888 \
--config config.yaml \
--gpu-memory-utilization=0.7 \
--tensor-parallel-size=1 \
--max-num-seqs=1  \
--disable-log-requests