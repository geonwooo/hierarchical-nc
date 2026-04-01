#!/bin/bash
GPU_ID=${1}
DDP="False"
DP="False"
TITLE=${2}
SEED=${3}
CFG="configs/cifar100/ce_cifar100_resnet32.yaml"

if [ ${DDP} = "True" ]
then
    GPU_OPT="ddp True"
elif [ ${DP} = "True" ]
then
    GPU_OPT="ddp False dp True"
else
    GPU_OPT="ddp False dp False rank ${GPU_ID}"
fi

ARGS="
    --cfg ${CFG}
    ${GPU_OPT}
    seed_num ${SEED}
    mixed_precision False
    proctitle ${TITLE}
"


python main/train.py ${ARGS}
