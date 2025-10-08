#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

chainSize=(10 20 50 100 200)

visitCount=(1)

echo " --- RMAX LazyChain Deterministic Benchmark --- "

for vc in "${visitCount[@]}"
do
    echo " ... Running m = ${vc} ... "

    for cs in "${chainSize[@]}"
    do
	echo " ... Chain Size ${cs} ..."

	iterator=($(seq 0 19))

	for i in "${iterator[@]}"
	do
	    echo " ... Running seed ${i} ..."
	    python main.py --agent-name="RMAX" --env-name="LazyChain" --use-jax --seed=${i} --num-environment-steps=$((1000 * cs)) --discount-factor=0.999 --reward-param=$((2 * cs - 1)) --chain-size=${cs} --p-error=0.0 --min-visit-count=${vc}
	done
    done
done
