#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

chainSize=(10 20 50 100 200)

diralpha=(1e-8)
beta=(1e-4)

echo " --- BEB LazyChain Deterministic Benchmark --- "

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
		python main.py --agent-name="BEB" --env-name="LazyChain" --use-jax --use-normal-gamma-prior --policy-update-interval=1 --seed=${i} --num-environment-steps=$((1000 * cs)) --eu-scale=$escs --dirichlet-param="$ds" --discount-factor=0.999 --chain-size=${cs} --p-error=0.0 --no-instant-reward --beta=${bt}
	    done
	done
    done
done
