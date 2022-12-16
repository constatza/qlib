#!/usr/bin/env bash

NUM_QUBITS=$1
NUM_POINTS=$2
NAME=$3
FILENAME="${NAME}q${NUM_QUBITS}p${NUM_POINTS}"

source $HOME/.virtualenvs/quantum/bin/activate
nohup python experiment.py $NUM_QUBITS $NUM_POINTS &> $FILENAME.out &

