#!/usr/bin/env bash

set -e

echo "================================="
echo " RETRIEVAL BENCHMARK EXECUTION "
echo "================================="

python src/cli.py run \
    --config configs/experiments/msmarco_full.yaml

python src/cli.py run \
    --config configs/experiments/scifact_full.yaml

echo "================================="
echo " COMPLETE "
echo "================================="