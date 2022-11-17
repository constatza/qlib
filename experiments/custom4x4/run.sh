#!/usr/bin/env bash

source $HOME/.virtualenvs/quantum/bin/activate
nohup python $1 > ./results/monitor &
