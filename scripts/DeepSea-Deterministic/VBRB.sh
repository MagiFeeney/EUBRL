#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

echo "VBRB DeepSea Deterministic"

for ds in "${SeaSize[@]}"
do
    iterator=($(seq 0 19))

    echo "Deep Sea size ${ds} ..."
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	python main.py --agent-name="VBRB" --env-name="DeepSea" --use-jax --use-normal-gamma-prior --policy-update-interval=10 --seed=${i} --num-environment-steps=$((ds * ds * 50)) --eu-scale=0.5 --dirichlet-param=1e-8 --discount-factor=0.99 --eu-type="One-hot" --transition-eu-scale=1 --reward-eu-scale=1 --instant-reward --beta=1e-4 --deepsea-size=${ds} --no-deepsea-stochastic
    done
done
