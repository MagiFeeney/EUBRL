#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

echo "BEB DeepSea Stochastic"

for ds in "${SeaSize[@]}"
do
    iterator=($(seq 0 19))

    echo "Deep Sea size ${ds} ..."
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	# No instant reward as the uncertainty quantification doesn't use the posterior information
	python main.py --agent-name="BEB" --env-name="DeepSea" --use-jax --use-normal-gamma-prior --policy-update-interval=10 --seed=${i} --num-environment-steps=$((ds * ds * 50)) --eu-scale=0.25 --dirichlet-param=1e-6 --discount-factor=0.99 --no-instant-reward --beta=5e-2 --deepsea-size=${ds} --deepsea-stochastic
    done
done
