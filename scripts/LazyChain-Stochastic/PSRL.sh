#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

chainSize=(10 20 50 100 200)

diralpha=(1e-5)
beta=(1e-1)

echo " --- PSRL LazyChain Stochastic Benchmark --- "

for ds in "${diralpha[@]}"
do
    for bt in "${beta[@]}"
    do
	echo " ... Running Dir(ɑ) ${ds} | ϐ ${bt} ... "

	for cs in "${chainSize[@]}"
	do
	    echo " ... Chain Size ${cs} ..."

	    iterator=($(seq 0 19))

	    for i in "${iterator[@]}"
	    do
		echo " ... Running seed ${i} ..."
		# PSRL is better without instant reward; same as in DeepSea
		python main.py --agent-name="PSRL" --env-name="LazyChain" --use-jax --use-normal-gamma-prior --policy-update-interval=$((cs)) --seed=${i} --num-environment-steps=$((1000 * cs)) --dirichlet-param="$ds" --discount-factor=0.999 --eu-type="One-hot" --chain-size=${cs} --p-error=0.2 --no-instant-reward --beta=${bt}
	    done
	done
    done
done
