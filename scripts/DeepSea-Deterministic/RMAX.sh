#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

SeaSize=(10 20 50)

visitCount=(1)

echo " --- RMAX DeepSea Deterministic --- "

for vc in "${visitCount[@]}"
do
    echo " ... Running m = ${vc} ... "

    for ds in "${SeaSize[@]}"
    do
	echo " ... Chain Size ${ds} ..."

	iterator=($(seq 0 19))

	for i in "${iterator[@]}"
	do
	    echo " ... Running seed ${i} ..."
	    python main.py --agent-name="RMAX" --env-name="DeepSea" --use-jax --seed=${i} --num-environment-steps=$((ds * ds * 50)) --discount-factor=0.99 --reward-param=1.0 --min-visit-count=${vc} --deepsea-size=${ds} --no-deepsea-stochastic
	done
    done
done
