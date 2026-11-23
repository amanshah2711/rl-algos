#!/usr/bin/env bash

mkdir -p runs/Taxi-v4/dqn

for i in {1..20}
do
    python3 -u -m experiments.Taxi-v4.dqn --seed "$i" > "runs/Taxi-v4/dqn/log_${i}.log" 2>&1
done
