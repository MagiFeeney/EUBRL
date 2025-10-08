#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

chainSize=(10 20 50 100 200)

diralpha=(1e-6)
beta=(1e-1)

echo " --- BEB LazyChain Stochastic Benchmark --- "

for ds in "${diralpha[@]}"
do
    for bt in "${beta[@]}"
    do
	echo " ... Running Dir(ɑ) ${ds} | ϐ ${bt} ... "

	for cs in "${chainSize[@]}"
	do
	    escs=$(echo "0.25 * $cs" | bc -l)
	    echo " ... Chain Size ${cs} with scale $escs ..."

	    iterator=($(seq 0 19))

	    for i in "${iterator[@]}"
	    do
		echo " ... Running seed ${i} ..."
		# BEB uses count-based bonus so no need to have instant reward
		python main.py --agent-name="BEB" --env-name="LazyChain" --use-jax --use-normal-gamma-prior --policy-update-interval=$((cs)) --seed=${i} --num-environment-steps=$((1000 * cs)) --eu-scale=$escs --dirichlet-param="$ds" --discount-factor=0.999 --chain-size=${cs} --p-error=0.2 --no-instant-reward --beta=${bt}
	    done
	done
    done
done
