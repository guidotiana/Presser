#!/usr/bin/env bash

set -e
source "/Data/alessandroz/myenv/bin/activate"
nohup python -u confront_cms.py --device 2 >& output.out &