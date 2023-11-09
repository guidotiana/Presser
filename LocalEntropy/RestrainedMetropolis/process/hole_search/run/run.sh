#!/usr/bin/env bash

set -e
source "$HOME/Desktop/myenv/bin/activate"

wtc=`sed -n '11p' < inputs/inputs.txt`
pn=`sed -n '1p' < inputs/inputs.txt`
T=`sed -n '2p' < inputs/inputs.txt`
g=`sed -n '3p' < inputs/inputs.txt`
dtype=`sed -n '4p' < inputs/inputs.txt`
p=`sed -n '5p' < inputs/inputs.txt`
s=`sed -n '6p' < inputs/inputs.txt`
w=`sed -n '7p' < inputs/inputs.txt`
file_id=$wtc-$pn-T$T-g$g-$dtype-p$p-s$s-w$w

nohup python -u main.py >& outputs/$file_id.out &
