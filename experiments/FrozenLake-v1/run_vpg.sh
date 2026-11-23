#!/usr/bin/env bash

mkdir -p runs/FrozenLake-v1/reinforce

for i in {1..20}
do
    python3 -u -m experiments.FrozenLake-v1.vpg --seed "$i" > "runs/FrozenLake-v1/reinforce/log_${i}.log" 2>&1
done
