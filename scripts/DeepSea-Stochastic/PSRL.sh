#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

echo "PSRL DeepSea Stochastic"

for ds in "${SeaSize[@]}"
do
    iterator=($(seq 0 19))

    echo "Deep Sea size ${ds} ..."
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	# PSRL performs better with no instant reward
	python main.py --agent-name="PSRL" --env-name="DeepSea" --use-jax --use-normal-gamma-prior --policy-update-interval=10 --seed=${i} --num-environment-steps=$((ds * ds * 50)) --dirichlet-param=1e-6 --discount-factor=0.99 --eu-type="One-hot" --deepsea-size=${ds} --deepsea-stochastic --no-instant-reward --beta=1e-2
    done
done







