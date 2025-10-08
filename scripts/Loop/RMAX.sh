#!/bin/bash

source ~/miniconda3/bin/activate bayesrl

numLoops=(2 4 8 16 32)

echo " --- RMAX (Loop) Random Test running --- "
for nL in "${numLoops[@]}"
do
    iterator=($(seq 0 500))

    echo " ... Running num. of Loops ${nL} ... "
    for i in "${iterator[@]}"
    do
	echo "Running seed ${i} ..."
	# reward-param is the maximum reward; knows the transition and reward immediately experienced them
	python main.py --agent-name="RMAX" --env-name="Loop" --no-use-jax --seed=${i} --num-environment-steps=7000 --discount-factor=0.95 --reward-param=2.0 --loop-length=5 --min-visit-count=1 --num-loops=${nL}
    done
done
