#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

echo "EUBRL+ DeepSea Stochastic"

for ds in "${SeaSize[@]}"
do
    iterator=($(seq 0 19))

    echo "Deep Sea size ${ds} ..."
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	python main.py --agent-name="EUBRL" --env-name="DeepSea" --use-jax --use-normal-gamma-prior --policy-update-interval=10 --use-eubrl-reward --seed=${i} --num-environment-steps=$((ds * ds * 50)) --dirichlet-param=1e-5 --discount-factor=0.99 --eu-type="One-hot" --deepsea-size=${ds} --deepsea-stochastic --instant-reward --beta=1.0 --eu-scale=2.0
    done
done
