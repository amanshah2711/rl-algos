#!/usr/bin/env bash

mkdir -p runs/REINFORCE

for i in {1..20}
do
    python3 -u -m scripts.run_pg --seed "$i" > "runs/REINFORCE/log_${i}.log" 2>&1
done
