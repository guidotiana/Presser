#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv_3.9.14_11.7/bin/activate"

simtype=`sed -n '1p' < inputs/inputs.txt`
protname=`sed -n '2p' < inputs/inputs.txt`
T=`sed -n '10p' < inputs/inputs.txt`
g=`sed -n '11p' < inputs/inputs.txt`
s=`sed -n '14p' < inputs/inputs.txt`

if [ $simtype == 'SM' ]
then
	file_id=$simtype-$protname-T$T-g$g-s$s
elif [ $simtype == 'MM' ]
then
	file_id=$simtype-$protname-T$T-s$s
else
	echo 'Wrong simtype value. Exit.'
	exit 1
fi

nohup python -u metropolis.py >& outputs/$file_id.out &
