#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

echo "BEB DeepSea Deterministic"

for ds in "${SeaSize[@]}"
do
    iterator=($(seq 0 19))

    echo "Deep Sea size ${ds} ..."
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	python main.py --agent-name="BEB" --env-name="DeepSea" --use-jax --use-normal-gamma-prior --policy-update-interval=10 --seed=${i} --num-environment-steps=$((ds * ds * 50)) --eu-scale=0.25 --dirichlet-param=1e-6 --discount-factor=0.99 --no-instant-reward --beta=1e-4 --deepsea-size=${ds} --no-deepsea-stochastic
    done
done
