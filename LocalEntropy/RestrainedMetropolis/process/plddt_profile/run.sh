#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv_3.9.14_11.7/bin/activate"

mt_num=`sed -n '3p' < inputs.txt`
file_id=num$mt_num

nohup python -u profile_calculation.py >& output_$file_id.out &
