#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

numLoops=(2 4 8 16 32)

echo " --- EUBRL (Loop) Random Test running --- "
for nL in "${numLoops[@]}"
do
    iterator=($(seq 0 500))	# number of seeds

    echo " ... Running num. of Loops ${nL} ... "
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	python main.py --agent-name="EUBRL" --env-name="Loop" --no-use-jax --use-normal-gamma-prior --policy-update-interval=1 --use-eubrl-reward --seed=${i} --num-environment-steps=7000 --eu-scale=0.5 --dirichlet-param=1e-8 --discount-factor=0.95 --eu-type="One-hot" --loop-length=5 --beta=1e-4 --instant-reward --num-loops=${nL}
    done
done

