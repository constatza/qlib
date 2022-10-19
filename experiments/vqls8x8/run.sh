#!/usr/bin/env bash

source $HOME/.virtualenvs/quantum/bin/activate
cd $HOME/code/quantum/experiments/vqls8x8/
nohup python experiment1.py > ./results/monitor &
