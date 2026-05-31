#!/usr/bin/env bash

set -e

python src/cli.py download \
    --dataset msmarco

python src/cli.py download \
    --dataset scifact