#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

echo " --- EUBRL (Chain) Random Test running --- "
for i in {0..500}
do
    echo "Running seed ${i} ..."
    python main.py --agent-name="EUBRL" --env-name="Chain" --no-use-jax --policy-update-interval=10 --use-eubrl-reward --seed=${i} --num-environment-steps=1000 --eu-scale=30 --dirichlet-param=1e-1 --discount-factor=0.95 --eu-type="One-hot" --chain-size=4 --beta=1e-4 --instant-reward --known-reward
done
