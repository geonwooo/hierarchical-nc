#!/bin/bash
GPU_ID=${1}
#SEEDS=(35 47 251 254 900)
SEEDS=(35)
CFG="configs/mnist/ce_mnist_lenet5.yaml"

GPU_OPT="ddp False dp False mixed_precision False rank ${GPU_ID}"
ARGS="--cfg ${CFG} ${GPU_OPT}"

for SEED in ${SEEDS[@]}
do
    python main/train.py ${ARGS} seed_num ${SEED}
done

