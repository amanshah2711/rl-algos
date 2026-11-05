#!/usr/bin/env bash

mkdir -p runs/FrozenLake-v1/DQN

for i in {1..20}
do
    python3 -u -m experiments.FrozenLake-v1.dqn --seed "$i" > "runs/FrozenLake-v1/DQN/log_${i}.log" 2>&1
done
