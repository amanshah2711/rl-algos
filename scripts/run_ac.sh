
#!/usr/bin/env bash

mkdir -p runs/Actor-Critic

for i in {1..20}
do
    python3 -u -m scripts.run_ac --seed "$i" > "runs/Actor-Critic/log_${i}.log" 2>&1
done
