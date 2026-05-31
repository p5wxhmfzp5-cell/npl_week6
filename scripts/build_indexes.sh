#!/usr/bin/env bash

set -e

METHODS=(
"bm25"
"tfidf"
"dense_minilm"
"dense_m3"
)

for method in "${METHODS[@]}"
do

python src/cli.py index \
    --method "$method" \
    --dataset msmarco

done