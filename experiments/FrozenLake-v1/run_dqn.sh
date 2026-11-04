#!/usr/bin/env bash

mkdir -p runs/DQN

for i in {1..20}
do
    python3 -u -m scripts.run_dqn --seed "$i" > "runs/DQN/log_${i}.log" 2>&1
done
