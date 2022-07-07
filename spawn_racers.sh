#!/bin/bash
trap "kill 0" EXIT

for x in {0..11}; do
    gpu=$(($x % 2))
    python server.py --training_mode --verbose --addr 100.113.4.6 --port $((1330+$x)) >> server_logs$x.txt &
    python network.py --model_type linear --name model_$x --param_search --addr 100.113.4.6 --port $((1330+$x)) --device cuda:$gpu >> training_logs$x.txt &
done

wait