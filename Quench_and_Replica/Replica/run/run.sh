#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv_3.9.14_11.7/bin/activate"

protname=`sed -n '1p' < inputs/inputs.txt`
y=`sed -n '3p' < inputs/inputs.txt`
s=`sed -n '4p' < inputs/inputs.txt`
T=`sed -n '5p' < inputs/inputs.txt`
file_id=$protname-y$y-s$s-T$T

nohup python -u metropolis.py >& outputs/$file_id.out &
