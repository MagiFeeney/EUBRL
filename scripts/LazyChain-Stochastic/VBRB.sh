#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

chainSize=(10 20 50 100 200)

scale=(0.5)
diralpha=(1e-4)
beta=(5e-2)

echo " --- VBRB LazyChain Stochastic Benchmark --- "

for es in "${scale[@]}"
do
    for ds in "${diralpha[@]}"
    do
	for bt in "${beta[@]}"
	do
	    echo " ... Running Scale ${es} | Dir(ɑ) ${ds} | ϐ ${bt} ... "

	    for cs in "${chainSize[@]}"
	    do
		echo " ... Chain Size ${cs} ..."

		iterator=($(seq 0 19))

		escs=$(echo "$es * $cs" | bc -l)

		for i in "${iterator[@]}"
		do
		    echo " ... Running seed ${i} with scale $escs ..."
		    python main.py --agent-name="VBRB" --env-name="LazyChain" --use-jax --use-normal-gamma-prior --policy-update-interval=$((cs)) --seed=${i} --num-environment-steps=$((1000 * cs)) --eu-scale=$escs --dirichlet-param="$ds" --discount-factor=0.999 --eu-type="One-hot" --chain-size=${cs} --p-error=0.2 --transition-eu-scale=1 --reward-eu-scale=1 --instant-reward --beta=${bt}
		done
	    done
	done
    done
done
